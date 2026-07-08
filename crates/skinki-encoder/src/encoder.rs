//! T2 — the BERT forward pass over a loaded [`EncoderArtifact`].
//!
//! Determinism contract (spec §4): every matrix product runs through
//! `crate::gemm` (fixed left-to-right K order, fused `mul_add` terms);
//! attention scores/context use fixed-order fused f32 accumulation; LayerNorm/softmax statistics accumulate in
//! f64 left-to-right; `exp`/`erf` come from `crate::math` (no `libm`). One
//! sequence is always processed single-threaded — `embed_batch` threads
//! *across* sequences only — so output is byte-identical across runs, thread
//! counts and platforms.
//!
//! Shape notes: activations are row-major `[seq × hidden]`; weights are
//! `[in × out]` (T1 layout), so every projection is one `gemm` with C
//! pre-filled with the broadcast bias (gemm accumulates — the bias add is
//! free and order-fixed).

use std::io;
use std::path::Path;
use std::thread;

use skinki_vector::embed::Embedder;

use crate::format::{EncoderArtifact, Pooling};
use crate::gemm::gemm;
use crate::math::{gelu_slice, layernorm_row, softmax_row};

/// A ready-to-run pure-Rust sentence encoder.
#[derive(Debug)]
pub struct RustEncoder {
    art: EncoderArtifact,
}

impl RustEncoder {
    pub fn load(path: &Path) -> io::Result<Self> {
        Ok(RustEncoder {
            art: EncoderArtifact::load(path)?,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        Ok(RustEncoder {
            art: EncoderArtifact::from_bytes(bytes)?,
        })
    }

    pub fn dim(&self) -> usize {
        self.art.dims.hidden
    }

    /// Ledger staleness wiring (T4): `MethodStamp { id, version }`.
    pub fn method_stamp(&self) -> (u32, u64) {
        self.art.method_stamp()
    }

    /// The model-contract query prefix (e5: `"query: "`, bge: `""` — the
    /// engine never hardcodes a prefix; it is read from the artifact).
    pub fn query_prefix(&self) -> &str {
        &self.art.query_prefix
    }

    /// The model-contract passage prefix (e5: `"passage: "`).
    pub fn passage_prefix(&self) -> &str {
        &self.art.passage_prefix
    }

    /// `[BOS] tok.encode_content(text) [EOS]`, truncated to `max_pos` total
    /// tokens. BOS/EOS are `[CLS]`/`[SEP]` for WordPiece, `<s>`/`</s>` for
    /// Unigram (XLM-R) — both come straight from the artifact's tokenizer.
    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        let d = &self.art.dims;
        let mut content = self.art.tok.encode_content(text);
        content.truncate(d.max_pos - 2);
        let mut ids = Vec::with_capacity(content.len() + 2);
        ids.push(self.art.tok.bos_id());
        ids.extend(content);
        ids.push(self.art.tok.eos_id());
        ids
    }

    /// Embed one text as a **passage** (index-time path): apply the model's
    /// passage prefix, then tokenize → forward → pool → L2-normalize. For
    /// symmetric embedders (bge, where both prefixes are empty) this is
    /// identical to the raw embed.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let prefixed = if self.art.passage_prefix.is_empty() {
            text.to_string()
        } else {
            format!("{}{}", self.art.passage_prefix, text)
        };
        let ids = self.encode_ids(&prefixed);
        self.forward_pooled(&ids)
    }

    /// Embed one text as a **query** (search-time path): apply the model's
    /// query prefix, then the rest. Stage 1C-B D2's finding — a forgotten
    /// query prefix cost bge ~25% recall — is the reason the prefix lives in
    /// the artifact and is applied here, not at a call site.
    pub fn embed_query(&self, text: &str) -> Vec<f32> {
        let prefixed = if self.art.query_prefix.is_empty() {
            text.to_string()
        } else {
            format!("{}{}", self.art.query_prefix, text)
        };
        let ids = self.encode_ids(&prefixed);
        self.forward_pooled(&ids)
    }

    /// Embed many texts, threading **across** sequences (each sequence's
    /// arithmetic stays single-threaded → byte-identical for any `threads`).
    /// Output order matches input order.
    pub fn embed_batch(&self, texts: &[&str], threads: usize) -> Vec<Vec<f32>> {
        if threads <= 1 || texts.len() <= 1 {
            return texts.iter().map(|t| self.embed(t)).collect();
        }
        // Dynamic scheduling: workers pull the next text off a shared atomic
        // counter instead of owning a fixed contiguous band. Text lengths vary
        // wildly (seq² attention), so static bands leave threads idle at the
        // join. Each text's embedding is a pure function of that text alone —
        // output slot `i` is the same no matter which worker computes it or
        // in what order, so determinism is untouched by the scheduling.
        let workers = threads.min(texts.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let slots: Vec<std::sync::OnceLock<Vec<f32>>> = (0..texts.len())
            .map(|_| std::sync::OnceLock::new())
            .collect();
        thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(text) = texts.get(i) else { break };
                    let v = self.embed(text);
                    slots[i].set(v).expect("slot set twice");
                });
            }
        });
        slots
            .into_iter()
            .map(|s| s.into_inner().expect("worker filled every slot"))
            .collect()
    }

    /// Full forward for a framed id sequence, pooled + normalized.
    fn forward_pooled(&self, ids: &[u32]) -> Vec<f32> {
        let hidden = self.hidden_states(ids);
        let d = &self.art.dims;
        let h = d.hidden;
        let seq = ids.len();
        let mut pooled = match d.pooling {
            Pooling::Cls => hidden[..h].to_vec(),
            Pooling::Mean => {
                let mut acc = vec![0.0f64; h];
                for t in 0..seq {
                    for (j, a) in acc.iter_mut().enumerate() {
                        *a += hidden[t * h + j] as f64;
                    }
                }
                acc.into_iter().map(|v| (v / seq as f64) as f32).collect()
            }
        };
        // L2 normalize (f64 sumsq, left-to-right).
        let mut sumsq = 0.0f64;
        for &v in pooled.iter() {
            sumsq += v as f64 * v as f64;
        }
        if sumsq > 1e-24 {
            let inv = 1.0 / sumsq.sqrt();
            for v in pooled.iter_mut() {
                *v = (*v as f64 * inv) as f32;
            }
        }
        pooled
    }

    /// Embeddings + all encoder layers; returns the final `[seq × hidden]`.
    /// Exposed for the layer-golden parity tests (`states_out` captures the
    /// post-embedding-LN state and each layer output when provided).
    pub fn forward_states(
        &self,
        ids: &[u32],
        mut states_out: Option<&mut Vec<Vec<f32>>>,
    ) -> Vec<f32> {
        let d = self.art.dims;
        let h = d.hidden;
        let seq = ids.len();
        assert!(
            seq >= 1 && seq <= d.max_pos,
            "sequence length {seq} out of range"
        );

        // Embeddings: word + (pos + type0), then LayerNorm per row.
        let word = self.art.word_emb();
        let pos = self.art.pos_type_emb();
        let (eg, eb) = self.art.emb_ln();
        let mut x = vec![0.0f32; seq * h];
        for (t, &id) in ids.iter().enumerate() {
            let id = id as usize;
            assert!(id < d.vocab, "token id {id} out of vocab");
            let w = &word[id * h..(id + 1) * h];
            let p = &pos[t * h..(t + 1) * h];
            let row = &mut x[t * h..(t + 1) * h];
            for j in 0..h {
                row[j] = w[j] + p[j];
            }
            layernorm_row(row, eg, eb, d.ln_eps);
        }
        if let Some(states) = states_out.as_deref_mut() {
            states.push(x.clone());
        }

        let heads = d.heads;
        let hd = h / heads;
        let scale = 1.0 / (hd as f64).sqrt() as f32;

        // Reused buffers.
        let mut q = vec![0.0f32; seq * h];
        let mut k = vec![0.0f32; seq * h];
        let mut v = vec![0.0f32; seq * h];
        let mut ctx = vec![0.0f32; seq * h];
        let mut attn_out = vec![0.0f32; seq * h];
        let mut h1 = vec![0.0f32; seq * d.ffn];
        let mut h2 = vec![0.0f32; seq * h];
        // Per-head packed operands so both attention products run through
        // the register-tiled `gemm` instead of strided scalar loops.
        let mut qh = vec![0.0f32; seq * hd];
        let mut kt = vec![0.0f32; hd * seq];
        let mut vh = vec![0.0f32; seq * hd];
        let mut sc_all = vec![0.0f32; seq * seq];
        let mut ctx_h = vec![0.0f32; seq * hd];

        for l in 0..d.layers {
            let lw = self.art.layer(l);

            // Projections: C pre-filled with broadcast bias, gemm accumulates.
            fill_bias(&mut q, lw.bq, seq);
            gemm(seq, h, h, &x, lw.wq, &mut q, 1).expect("gemm q");
            fill_bias(&mut k, lw.bk, seq);
            gemm(seq, h, h, &x, lw.wk, &mut k, 1).expect("gemm k");
            fill_bias(&mut v, lw.bv, seq);
            gemm(seq, h, h, &x, lw.wv, &mut v, 1).expect("gemm v");

            // Attention per head, as two GEMMs on packed per-head panels:
            // scores = Q_head · K_head^T (then ·scale, softmax per row) and
            // ctx = P · V_head. Per element both keep the exact fused
            // ascending-reduction order of the scalar loops they replace
            // (gemm's contract *is* that order), so this is bit-identical —
            // just at microkernel throughput instead of latency-bound
            // per-row dot products.
            for head in 0..heads {
                let off = head * hd;
                for t in 0..seq {
                    qh[t * hd..(t + 1) * hd].copy_from_slice(&q[t * h + off..t * h + off + hd]);
                    vh[t * hd..(t + 1) * hd].copy_from_slice(&v[t * h + off..t * h + off + hd]);
                    for l in 0..hd {
                        kt[l * seq + t] = k[t * h + off + l];
                    }
                }
                sc_all.fill(0.0);
                gemm(seq, seq, hd, &qh, &kt, &mut sc_all, 1).expect("gemm qk");
                for e in sc_all.iter_mut() {
                    *e *= scale;
                }
                for i in 0..seq {
                    softmax_row(&mut sc_all[i * seq..(i + 1) * seq]);
                }
                ctx_h.fill(0.0);
                gemm(seq, hd, seq, &sc_all, &vh, &mut ctx_h, 1).expect("gemm av");
                for t in 0..seq {
                    ctx[t * h + off..t * h + off + hd]
                        .copy_from_slice(&ctx_h[t * hd..(t + 1) * hd]);
                }
            }

            // Output projection + residual + LN1.
            fill_bias(&mut attn_out, lw.bo, seq);
            gemm(seq, h, h, &ctx, lw.wo, &mut attn_out, 1).expect("gemm o");
            for (xi, ai) in x.iter_mut().zip(attn_out.iter()) {
                *xi += ai;
            }
            for t in 0..seq {
                layernorm_row(&mut x[t * h..(t + 1) * h], lw.ln1_g, lw.ln1_b, d.ln_eps);
            }

            // FFN: h1 = gelu(x·W1 + b1); h2 = h1·W2 + b2; residual + LN2.
            fill_bias(&mut h1, lw.b1, seq);
            gemm(seq, d.ffn, h, &x, lw.w1, &mut h1, 1).expect("gemm ffn1");
            gelu_slice(&mut h1);
            fill_bias(&mut h2, lw.b2, seq);
            gemm(seq, h, d.ffn, &h1, lw.w2, &mut h2, 1).expect("gemm ffn2");
            for (xi, hi) in x.iter_mut().zip(h2.iter()) {
                *xi += hi;
            }
            for t in 0..seq {
                layernorm_row(&mut x[t * h..(t + 1) * h], lw.ln2_g, lw.ln2_b, d.ln_eps);
            }

            if let Some(states) = states_out.as_deref_mut() {
                states.push(x.clone());
            }
        }
        x
    }

    fn hidden_states(&self, ids: &[u32]) -> Vec<f32> {
        self.forward_states(ids, None)
    }
}

/// Fill `c` (`seq` rows) with the broadcast `bias` row — the pre-gemm bias
/// trick: gemm accumulates on top, so `out = x·W + b` in one fixed-order pass.
fn fill_bias(c: &mut [f32], bias: &[f32], seq: usize) {
    let n = bias.len();
    debug_assert_eq!(c.len(), seq * n);
    for t in 0..seq {
        c[t * n..(t + 1) * n].copy_from_slice(bias);
    }
}

impl Embedder for RustEncoder {
    fn embed(&self, text: &str) -> Vec<f32> {
        RustEncoder::embed(self, text)
    }
    fn dim(&self) -> usize {
        RustEncoder::dim(self)
    }
    /// Override the default so the model's query prefix is applied at search
    /// time (Stage 1D T1). `SemanticRetriever::search` routes here.
    fn embed_query(&self, text: &str) -> Vec<f32> {
        RustEncoder::embed_query(self, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::toy::build_toy_artifact;

    fn read_u32(bytes: &[u8], r: &mut usize) -> u32 {
        let v = u32::from_le_bytes(bytes[*r..*r + 4].try_into().unwrap());
        *r += 4;
        v
    }

    fn toy() -> RustEncoder {
        RustEncoder::from_bytes(&build_toy_artifact(0xC0DE)).unwrap()
    }

    #[test]
    fn embed_shape_and_unit_norm() {
        let e = toy();
        let v = e.embed("memory engine running tests");
        assert_eq!(v.len(), 16);
        let norm: f64 = v.iter().map(|&x| x as f64 * x as f64).sum();
        assert!((norm.sqrt() - 1.0).abs() < 1e-5, "norm {}", norm.sqrt());
    }

    #[test]
    fn empty_text_embeds_bos_eos_only() {
        let e = toy();
        let ids = e.encode_ids("");
        assert_eq!(ids, vec![e.art.tok.bos_id(), e.art.tok.eos_id()]);
        let v = e.embed("");
        assert_eq!(v.len(), 16);
        // CLS pooling of a real forward — non-zero, normalized.
        let norm: f64 = v.iter().map(|&x| x as f64 * x as f64).sum();
        assert!((norm.sqrt() - 1.0).abs() < 1e-5, "norm {}", norm.sqrt());
    }

    #[test]
    fn truncates_to_max_pos() {
        let e = toy();
        // 100 words >> max_pos 16.
        let long = vec!["memory"; 100].join(" ");
        let ids = e.encode_ids(&long);
        assert_eq!(ids.len(), 16);
        assert_eq!(*ids.last().unwrap(), e.art.tok.eos_id());
        let _ = e.embed(&long); // must not panic
    }

    #[test]
    fn deterministic_across_runs() {
        let e = toy();
        let a = e.embed("the memory engine embeds rust vectors");
        let b = e.embed("the memory engine embeds rust vectors");
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn embed_batch_thread_count_invariant() {
        let e = toy();
        let texts: Vec<String> = (0..9)
            .map(|i| format!("memory engine {i} rust search insight"))
            .collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let one = e.embed_batch(&refs, 1);
        let four = e.embed_batch(&refs, 4);
        assert_eq!(one.len(), four.len());
        for (a, b) in one.iter().zip(four.iter()) {
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "thread-count invariance violated"
            );
        }
        // And batch equals sequential single embeds.
        for (i, t) in refs.iter().enumerate() {
            assert_eq!(one[i], e.embed(t));
        }
    }

    #[test]
    fn different_texts_differ() {
        let e = toy();
        assert_ne!(e.embed("memory engine"), e.embed("rust search"));
    }

    /// Prefix-aware embed: with non-empty query/passage prefixes (built into
    /// the toy artifact), `embed_query` and `embed` must produce different
    /// vectors for the same text — this is the property the e5 query prefix
    /// depends on at search time. Also checks the trait override is wired.
    /// Prefixes are chosen so the toy WordPiece vocab tokenizes them
    /// distinctly (both `memory` and `engine` are in the toy vocab), so the
    /// query/passage id sequences actually differ.
    #[test]
    fn embed_query_uses_query_prefix() {
        use crate::format::toy::build_toy_wordpiece;
        let dims = crate::format::EncoderDims {
            pooling: crate::format::Pooling::Mean,
            ..crate::format::toy::TOY_DIMS
        };
        let bytes = build_toy_wordpiece(dims, 0x5E5, "memory ", "engine ");
        let e = RustEncoder::from_bytes(&bytes).unwrap();
        assert_eq!(e.query_prefix(), "memory ");
        assert_eq!(e.passage_prefix(), "engine ");
        assert_ne!(
            e.embed_query("rust vector"),
            e.embed("rust vector"),
            "query/passage prefixes must yield different embeddings"
        );
        // The trait route hits the same override.
        let boxed: Box<dyn Embedder> = Box::new(RustEncoder::from_bytes(&bytes).unwrap());
        assert_eq!(
            boxed.embed_query("rust vector"),
            e.embed_query("rust vector")
        );
    }

    /// With empty prefixes the artifact is symmetric: query path == passage
    /// path (the bge case, where the engine-side prefix lesson does not apply
    /// inside the artifact).
    #[test]
    fn empty_prefixes_mean_query_equals_passage() {
        let e = toy();
        assert!(e.query_prefix().is_empty());
        assert!(e.passage_prefix().is_empty());
        assert_eq!(e.embed_query("memory engine"), e.embed("memory engine"));
    }

    #[test]
    fn pure_through_trait_object() {
        let e: Box<dyn Embedder> = Box::new(toy());
        assert_eq!(e.embed("graph store"), e.embed("graph store"));
        assert_eq!(e.dim(), 16);
    }

    /// Byte-regression against Rust's own committed toy goldens: any change
    /// to the numerics — intended or not — must show up as a diff here and
    /// be re-blessed via `gen_toy_golden -- --ignored`.
    #[test]
    fn toy_golden_regression() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/encoder_toy_golden.f32"
        );
        let bytes = std::fs::read(path).expect("fixtures/encoder_toy_golden.f32 missing");
        let e = toy();
        let mut r = 0usize;
        let count = read_u32(&bytes, &mut r) as usize;
        let dim = read_u32(&bytes, &mut r) as usize;
        assert_eq!(dim, 16);
        let mut strings = Vec::with_capacity(count);
        for _ in 0..count {
            let len = read_u32(&bytes, &mut r) as usize;
            strings.push(String::from_utf8(bytes[r..r + len].to_vec()).unwrap());
            r += len;
        }
        for s in &strings {
            let got = e.embed(s);
            for (i, g) in got.iter().enumerate() {
                let want = f32::from_le_bytes(bytes[r..r + 4].try_into().unwrap());
                r += 4;
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "toy golden regression: {s:?} dim {i}"
                );
            }
        }
        assert_eq!(r, bytes.len(), "trailing bytes in toy golden");
    }

    const TOY_GOLDEN_STRINGS: &[&str] = &[
        "memory engine",
        "rust vector search",
        "running tests",
        "память поиск",
        "memorys engine, coded.",
        "",
        "the a and of to in is it",
    ];

    #[test]
    #[ignore = "regenerates fixtures/encoder_toy_golden.f32"]
    fn gen_toy_golden() {
        let e = toy();
        let mut out = Vec::new();
        out.extend_from_slice(&(TOY_GOLDEN_STRINGS.len() as u32).to_le_bytes());
        out.extend_from_slice(&16u32.to_le_bytes());
        for s in TOY_GOLDEN_STRINGS {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        for s in TOY_GOLDEN_STRINGS {
            for v in e.embed(s) {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/encoder_toy_golden.f32"
        );
        std::fs::write(path, out).unwrap();
    }

    // -----------------------------------------------------------------
    // Parity vs the torch teacher (real artifact — #[ignore], run after
    // scripts/convert_encoder_to_skenc.py).
    // -----------------------------------------------------------------

    fn fixtures() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
    }

    /// Layer-by-layer parity against the teacher's activation dump: a
    /// numerics bug localizes to the first layer whose max abs diff blows up.
    /// The shared body is invoked once per converted artifact so both the bge
    /// (WordPiece/CLS) and e5 (Unigram/mean) shapes stay covered.
    fn layer_parity_body(artifact: &std::path::Path, golden: &std::path::Path) {
        let e = RustEncoder::load(artifact).unwrap();
        let bytes = std::fs::read(golden).unwrap();
        let mut r = 0usize;
        let n_states = read_u32(&bytes, &mut r) as usize;
        let seq = read_u32(&bytes, &mut r) as usize;
        let hidden = read_u32(&bytes, &mut r) as usize;
        let n_ids = read_u32(&bytes, &mut r) as usize;
        assert_eq!(n_ids, seq);
        assert_eq!(hidden, e.dim());
        let mut ids = Vec::with_capacity(n_ids);
        for _ in 0..n_ids {
            ids.push(read_u32(&bytes, &mut r));
        }
        let mut states: Vec<Vec<f32>> = Vec::with_capacity(n_states);
        let _ = e.forward_states(&ids, Some(&mut states));
        assert_eq!(states.len(), n_states, "state count mismatch");
        for (si, st) in states.iter().enumerate() {
            let mut max_abs = 0.0f32;
            for v in st.iter() {
                let want = f32::from_le_bytes(bytes[r..r + 4].try_into().unwrap());
                r += 4;
                let d = (v - want).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            // Drift grows with depth (12 layers of f32 association diffs vs
            // torch); 5e-3 abs on O(1)-scale activations keeps the test
            // sharp enough to localize real bugs while tolerating summation
            // order. Report per state for the PR record.
            eprintln!("state {si}: max abs diff {max_abs}");
            assert!(max_abs < 5e-3, "state {si} diverged: max abs {max_abs}");
        }
        assert_eq!(r, bytes.len());
    }

    #[test]
    #[ignore = "needs fixtures/encoder_bge_small.skenc — regenerate with scripts/convert_encoder_to_skenc.py"]
    fn layer_parity_vs_teacher_bge() {
        layer_parity_body(
            &fixtures().join("encoder_bge_small.skenc"),
            &fixtures().join("encoder_layer_golden.f32"),
        );
    }

    #[test]
    #[ignore = "needs fixtures/encoder_e5_small.skenc — regenerate with scripts/convert_encoder_to_skenc.py"]
    fn layer_parity_vs_teacher_e5() {
        layer_parity_body(
            &fixtures().join("encoder_e5_small.skenc"),
            &fixtures().join("encoder_e5_layer_golden.f32"),
        );
    }

    /// End-to-end parity: cosine ≥ 0.999 per golden vector (§2 bar).
    fn e2e_parity_body(artifact: &std::path::Path, golden: &std::path::Path) {
        let e = RustEncoder::load(artifact).unwrap();
        let bytes = std::fs::read(golden).unwrap();
        let mut r = 0usize;
        let count = read_u32(&bytes, &mut r) as usize;
        let dim = read_u32(&bytes, &mut r) as usize;
        assert_eq!(dim, e.dim());
        let mut strings = Vec::with_capacity(count);
        for _ in 0..count {
            let len = read_u32(&bytes, &mut r) as usize;
            strings.push(String::from_utf8(bytes[r..r + len].to_vec()).unwrap());
            r += len;
        }
        let mut min_cos = 1.0f64;
        for s in &strings {
            let got = e.embed(s);
            let mut dot = 0.0f64;
            let mut want_sq = 0.0f64;
            let mut got_sq = 0.0f64;
            for g in got.iter() {
                let want = f32::from_le_bytes(bytes[r..r + 4].try_into().unwrap());
                r += 4;
                dot += *g as f64 * want as f64;
                want_sq += (want as f64) * (want as f64);
                got_sq += (*g as f64) * (*g as f64);
            }
            // Zero-vector guard: both sides must agree on degeneracy.
            let cos = if want_sq < 1e-24 || got_sq < 1e-24 {
                if want_sq < 1e-24 && got_sq < 1e-24 {
                    1.0
                } else {
                    0.0
                }
            } else {
                dot / (want_sq.sqrt() * got_sq.sqrt())
            };
            eprintln!("cos({s:?}) = {cos:.7}");
            if cos < min_cos {
                min_cos = cos;
            }
            assert!(cos >= 0.999, "parity failed for {s:?}: cosine {cos}");
        }
        assert_eq!(r, bytes.len());
        eprintln!("min cosine over {count} strings: {min_cos:.7}");
    }

    #[test]
    #[ignore = "needs fixtures/encoder_bge_small.skenc — regenerate with scripts/convert_encoder_to_skenc.py"]
    fn e2e_parity_vs_teacher_bge() {
        e2e_parity_body(
            &fixtures().join("encoder_bge_small.skenc"),
            &fixtures().join("encoder_golden_embeddings.f32"),
        );
    }

    #[test]
    #[ignore = "needs fixtures/encoder_e5_small.skenc — regenerate with scripts/convert_encoder_to_skenc.py"]
    fn e2e_parity_vs_teacher_e5() {
        e2e_parity_body(
            &fixtures().join("encoder_e5_small.skenc"),
            &fixtures().join("encoder_e5_golden_embeddings.f32"),
        );
    }
}
