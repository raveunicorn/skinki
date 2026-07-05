//! `SKENC001` — the Stage 1C-B encoder weight artifact (T1; v2 in Stage 1D).
//!
//! Layout, all little-endian (see `specs/STAGE_1C_B_PURE_RUST_ENCODER.md` §4
//! and `specs/STAGE_1D_RETRIEVAL_QUALITY.md` §3 T1):
//!
//! ```text
//! magic "SKENC001" (8 bytes)
//! u32 version (=2) | u32 arch (1 = BERT post-LN) |
//! u32 layers | u32 hidden | u32 ffn | u32 heads | u32 vocab | u32 max_pos |
//! u32 pooling (0 = CLS, 1 = mean) |
//! u32 tok_kind (0 = WordPiece, 1 = Unigram) |
//! u32 query_prefix_len | u32 passage_prefix_len |
//! [query_prefix UTF-8 bytes] [passage_prefix UTF-8 bytes]   (no NULs)
//! f32 ln_eps
//! tokenizer section (tag-dependent, see below)
//! zero-padding to the next 4-byte boundary
//! tensors, f32 LE, fixed order:
//!   word_emb    [vocab × hidden]
//!   pos_type_emb[max_pos × hidden]   // position emb + token-type-0 row,
//!                                    // pre-summed by the converter
//!   emb_ln_gamma[hidden]  emb_ln_beta[hidden]
//!   per layer L = 0..layers:
//!     wq[hidden × hidden] bq[hidden]     // all W stored [in][out] row-major,
//!     wk[hidden × hidden] bk[hidden]     // so  out = x · W + b  runs through
//!     wv[hidden × hidden] bv[hidden]     // crate::gemm with no transposes
//!     wo[hidden × hidden] bo[hidden]
//!     ln1_gamma[hidden]   ln1_beta[hidden]
//!     w1[hidden × ffn]    b1[ffn]
//!     w2[ffn × hidden]    b2[hidden]
//!     ln2_gamma[hidden]   ln2_beta[hidden]
//! ```
//!
//! **Tokenizer section (Stage 1D, v2):**
//! - `tok_kind=0` (WordPiece, BERT/bge): `vocab × (u32 len | UTF-8 bytes)`,
//!   id = position. `[UNK]`, `[CLS]`, `[SEP]` required. The same byte layout
//!   the v1 reader consumed; the converter just writes it under the new header.
//! - `tok_kind=1` (Unigram, XLM-R/e5): `<u32 sku_size> <sku_size bytes>` — a
//!   complete `SKUNI001` artifact inlined byte-for-byte (the same file
//!   `scripts/dump_unigram_fixtures.py` writes standalone). One file = one
//!   model: the encoder needs no external tokenizer path, preserving the
//!   portability/determinism invariant the FFI and replay gates rely on.
//!
//! **Prefixes (Stage 1D):** `query_prefix` / `passage_prefix` are stored in the
//! header because they are a model contract, not an engine policy. e5 needs
//! `"query: "` / `"passage: "`; bge's query instruction is added by the
//! serving layer (kept empty in the artifact when the engine applies it);
//! a future model will carry different strings. The engine never hardcodes a
//! prefix string — see `RustEncoder::embed` / `embed_query`.
//!
//! **Versioning:** v1 (Stage 1C-B, no `tok_kind`/prefix fields) is rejected by
//! this reader. The real bge artifact was always gitignored and regenerable;
//! re-running the converter under v2 produces the same tensor bytes under the
//! new header. The version mismatch is a loud error, never a silent rehash.
//!
//! The reader is bounds-checked end to end (1B loader lessons): truncation,
//! corrupt length prefixes and dim products that overflow `usize` all surface
//! as `InvalidData`, never as a panic. Tensors are decoded once into a single
//! owned `Vec<f32>` arena — safe Rust cannot reinterpret mmap'd bytes as
//! `&[f32]` without `unsafe`, and the GEMM hot path needs real `&[f32]`; the
//! ~130 MB (bge) / ~472 MB (e5-small) decoded arena is inside the §2 RSS
//! budget and exists only while an encoder is loaded.

use std::io;
use std::path::Path;

use skinki_vector::unigram::UnigramTokenizer;
use skinki_vector::wordpiece::WordPieceTokenizer;

const MAGIC: &[u8; 8] = b"SKENC001";
const FORMAT_VERSION: u32 = 2;
const ARCH_BERT: u32 = 1;

const TOK_WORDPIECE: u32 = 0;
const TOK_UNIGRAM: u32 = 1;

/// `[CLS]` / `[SEP]` — required by the BERT (WordPiece) sequence framing.
pub const CLS: &str = "[CLS]";
pub const SEP: &str = "[SEP]";

/// Method id for "pure-Rust encoder" derivations (per-crate convention,
/// cf. `M_EMBEDDER` in `skinki-vector`). Wired into the ledger in T4.
pub const M_ENCODER: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// First-token pooling (bge family).
    Cls,
    /// Masked mean over the sequence (e5 family).
    Mean,
}

/// Which tokenizer family lives inside the artifact — selects the section
/// layout the reader walks, and the encoder's sequence framing
/// (`[CLS]/[SEP]` for WordPiece, `<s>/</s>` for Unigram/XLM-R).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    WordPiece,
    Unigram,
}

/// Either tokenizer, abstracted to the two operations the encoder needs:
/// content id encoding and BOS/EOS framing ids. The encoder never branches on
/// the family at forward time — only at tokenization time.
pub enum ArtifactTokenizer {
    WordPiece(WordPieceTokenizer),
    Unigram(UnigramTokenizer),
}

impl std::fmt::Debug for ArtifactTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactTokenizer::WordPiece(t) => {
                f.debug_tuple("WordPiece").field(&t.vocab_len()).finish()
            }
            ArtifactTokenizer::Unigram(t) => {
                f.debug_tuple("Unigram").field(&t.vocab_size()).finish()
            }
        }
    }
}

impl ArtifactTokenizer {
    /// Encode `text` to content ids (no specials), using this tokenizer's
    /// Rust-side convention. Mirrors `WordPieceTokenizer::encode` and
    /// `UnigramTokenizer::encode_content` exactly.
    pub fn encode_content(&self, text: &str) -> Vec<u32> {
        match self {
            ArtifactTokenizer::WordPiece(t) => t.encode(text),
            ArtifactTokenizer::Unigram(t) => t.encode_content(text),
        }
    }

    /// Beginning-of-sequence id (`[CLS]` for BERT, `<s>` for XLM-R).
    pub fn bos_id(&self) -> u32 {
        match self {
            ArtifactTokenizer::WordPiece(t) => {
                t.piece_id(CLS).expect("WordPiece vocab has no [CLS]")
            }
            ArtifactTokenizer::Unigram(t) => t.bos_hf_id(),
        }
    }

    /// End-of-sequence id (`[SEP]` for BERT, `</s>` for XLM-R).
    pub fn eos_id(&self) -> u32 {
        match self {
            ArtifactTokenizer::WordPiece(t) => {
                t.piece_id(SEP).expect("WordPiece vocab has no [SEP]")
            }
            ArtifactTokenizer::Unigram(t) => t.eos_hf_id(),
        }
    }

    pub fn kind(&self) -> TokenizerKind {
        match self {
            ArtifactTokenizer::WordPiece(_) => TokenizerKind::WordPiece,
            ArtifactTokenizer::Unigram(_) => TokenizerKind::Unigram,
        }
    }
}

/// Parsed header dimensions.
#[derive(Debug, Clone, Copy)]
pub struct EncoderDims {
    pub layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
    pub vocab: usize,
    pub max_pos: usize,
    pub pooling: Pooling,
    pub ln_eps: f32,
}

/// f32-index offsets of one layer's tensors within the params arena.
#[derive(Debug, Clone, Copy)]
pub struct LayerOffsets {
    pub wq: usize,
    pub bq: usize,
    pub wk: usize,
    pub bk: usize,
    pub wv: usize,
    pub bv: usize,
    pub wo: usize,
    pub bo: usize,
    pub ln1_g: usize,
    pub ln1_b: usize,
    pub w1: usize,
    pub b1: usize,
    pub w2: usize,
    pub b2: usize,
    pub ln2_g: usize,
    pub ln2_b: usize,
}

#[derive(Debug)]
struct Offsets {
    word_emb: usize,
    pos_type_emb: usize,
    emb_ln_g: usize,
    emb_ln_b: usize,
    layers: Vec<LayerOffsets>,
    total: usize,
}

/// A loaded `SKENC001` artifact: dims + tokenizer + one contiguous f32 arena.
pub struct EncoderArtifact {
    pub dims: EncoderDims,
    pub tok: ArtifactTokenizer,
    /// Model-contract prefix applied to queries at search time (e5: `"query: "`).
    pub query_prefix: String,
    /// Model-contract prefix applied to passages at index time (e5: `"passage: "`).
    pub passage_prefix: String,
    params: Vec<f32>,
    off: Offsets,
    version: u64,
}

impl std::fmt::Debug for EncoderArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderArtifact")
            .field("dims", &self.dims)
            .field("tok", &self.tok)
            .field("params_f32", &self.params.len())
            .field("version", &self.version)
            .finish()
    }
}

/// One layer's weight slices, borrowed from the arena.
pub struct LayerView<'a> {
    pub wq: &'a [f32],
    pub bq: &'a [f32],
    pub wk: &'a [f32],
    pub bk: &'a [f32],
    pub wv: &'a [f32],
    pub bv: &'a [f32],
    pub wo: &'a [f32],
    pub bo: &'a [f32],
    pub ln1_g: &'a [f32],
    pub ln1_b: &'a [f32],
    pub w1: &'a [f32],
    pub b1: &'a [f32],
    pub w2: &'a [f32],
    pub b2: &'a [f32],
    pub ln2_g: &'a [f32],
    pub ln2_b: &'a [f32],
}

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Checked offset builder: every size product/add must fit `usize`, or the
/// header is corrupt.
fn build_offsets(d: &EncoderDims) -> io::Result<Offsets> {
    let mut cursor: usize = 0;
    let mut take = |count: Option<usize>| -> io::Result<usize> {
        let count = count.ok_or_else(|| err("tensor size overflows"))?;
        let base = cursor;
        cursor = cursor
            .checked_add(count)
            .ok_or_else(|| err("arena size overflows"))?;
        Ok(base)
    };
    let h = d.hidden;
    let word_emb = take(d.vocab.checked_mul(h))?;
    let pos_type_emb = take(d.max_pos.checked_mul(h))?;
    let emb_ln_g = take(Some(h))?;
    let emb_ln_b = take(Some(h))?;
    let mut layers = Vec::with_capacity(d.layers);
    for _ in 0..d.layers {
        layers.push(LayerOffsets {
            wq: take(h.checked_mul(h))?,
            bq: take(Some(h))?,
            wk: take(h.checked_mul(h))?,
            bk: take(Some(h))?,
            wv: take(h.checked_mul(h))?,
            bv: take(Some(h))?,
            wo: take(h.checked_mul(h))?,
            bo: take(Some(h))?,
            ln1_g: take(Some(h))?,
            ln1_b: take(Some(h))?,
            w1: take(h.checked_mul(d.ffn))?,
            b1: take(Some(d.ffn))?,
            w2: take(d.ffn.checked_mul(h))?,
            b2: take(Some(h))?,
            ln2_g: take(Some(h))?,
            ln2_b: take(Some(h))?,
        });
    }
    Ok(Offsets {
        word_emb,
        pos_type_emb,
        emb_ln_g,
        emb_ln_b,
        layers,
        total: cursor,
    })
}

/// Bounds-checked byte cursor (same discipline as the 1B loader).
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        match end {
            Some(end) => {
                let s = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => Err(err(format!(
                "artifact truncated: read of {n} bytes at offset {} exceeds file size {}",
                self.pos,
                self.bytes.len()
            ))),
        }
    }
    fn u32(&mut self) -> io::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    /// Skip zero-padding to the next 4-byte boundary.
    fn align4(&mut self) -> io::Result<()> {
        let pad = (4 - self.pos % 4) % 4;
        self.take(pad)?;
        Ok(())
    }
    fn string(&mut self, len: usize) -> io::Result<String> {
        let b = self.take(len)?;
        std::str::from_utf8(b)
            .map(str::to_owned)
            .map_err(|e| err(format!("prefix string not UTF-8: {e}")))
    }
}

impl EncoderArtifact {
    pub fn load(path: &Path) -> io::Result<Self> {
        // mmap on unix so the decode pass demand-pages the file instead of
        // double-buffering it; the decoded arena is the only lasting copy.
        #[cfg(unix)]
        let view =
            skinki_vector::store::CodeStore::Mmap(skinki_vector::store::MmapBytes::open(path)?);
        #[cfg(not(unix))]
        let view = skinki_vector::store::CodeStore::Ram(std::fs::read(path)?);
        Self::from_bytes(view.as_slice())
    }

    /// Parse from raw bytes (the loader core; also the test entry point).
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut r = Reader::new(bytes);
        let magic = r.take(8)?;
        if magic != MAGIC {
            return Err(err(format!("bad magic: expected {MAGIC:?}, got {magic:?}")));
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(err(format!(
                "unsupported artifact version {version} (want {FORMAT_VERSION}); v1 artifacts from Stage 1C-B must be regenerated with scripts/convert_encoder_to_skenc.py"
            )));
        }
        let arch = r.u32()?;
        if arch != ARCH_BERT {
            return Err(err(format!("unsupported arch tag {arch} (want 1 = BERT)")));
        }
        let layers = r.u32()? as usize;
        let hidden = r.u32()? as usize;
        let ffn = r.u32()? as usize;
        let heads = r.u32()? as usize;
        let vocab = r.u32()? as usize;
        let max_pos = r.u32()? as usize;
        let pooling = match r.u32()? {
            0 => Pooling::Cls,
            1 => Pooling::Mean,
            other => return Err(err(format!("unknown pooling tag {other}"))),
        };
        let tok_kind = match r.u32()? {
            TOK_WORDPIECE => TokenizerKind::WordPiece,
            TOK_UNIGRAM => TokenizerKind::Unigram,
            other => return Err(err(format!("unknown tok_kind tag {other}"))),
        };
        let qp_len = r.u32()? as usize;
        let pp_len = r.u32()? as usize;
        let query_prefix = r.string(qp_len)?;
        let passage_prefix = r.string(pp_len)?;
        let ln_eps = r.f32()?;
        if layers == 0 || hidden == 0 || ffn == 0 || heads == 0 || vocab == 0 || max_pos == 0 {
            return Err(err("all header dims must be non-zero"));
        }
        if !hidden.is_multiple_of(heads) {
            return Err(err(format!(
                "hidden {hidden} not divisible by heads {heads}"
            )));
        }
        if !(ln_eps.is_finite() && ln_eps > 0.0) {
            return Err(err(format!(
                "ln_eps must be finite and positive, got {ln_eps}"
            )));
        }
        let dims = EncoderDims {
            layers,
            hidden,
            ffn,
            heads,
            vocab,
            max_pos,
            pooling,
            ln_eps,
        };

        // Tokenizer section: tag-dependent. Both paths surface as
        // `ArtifactTokenizer` so the encoder is single code regardless of
        // family.
        let tok = match tok_kind {
            TokenizerKind::WordPiece => {
                // Vocab section, then tokenizer with the required special pieces.
                // `vocab` is untrusted header data: never pre-allocate from it
                // (a corrupt u32::MAX would demand ~96 GiB and abort on Linux,
                // where allocation failure does not unwind). Growth is
                // amortized; a corrupt count dies at the first truncated
                // string read instead.
                let mut pieces = Vec::new();
                for _ in 0..vocab {
                    let len = r.u32()? as usize;
                    let s = std::str::from_utf8(r.take(len)?)
                        .map_err(|e| err(format!("vocab string not UTF-8: {e}")))?
                        .to_owned();
                    pieces.push(s);
                }
                let t = WordPieceTokenizer::from_pieces(pieces)?;
                if t.piece_id(CLS).is_none() {
                    return Err(err(format!("artifact vocab has no '{CLS}' token")));
                }
                if t.piece_id(SEP).is_none() {
                    return Err(err(format!("artifact vocab has no '{SEP}' token")));
                }
                ArtifactTokenizer::WordPiece(t)
            }
            TokenizerKind::Unigram => {
                // Inlined SKUNI001 artifact. The sku_size is untrusted header
                // data; `take` bounds-checks it like every other length.
                let sku_size = r.u32()? as usize;
                let sku_bytes = r.take(sku_size)?;
                match UnigramTokenizer::from_bytes(sku_bytes) {
                    Ok(t) => ArtifactTokenizer::Unigram(t),
                    Err(e) => return Err(err(format!("inlined SKUNI001 parse failed: {e}"))),
                }
            }
        };

        r.align4()?;

        // Tensor section: exact-length check up front, then one decode pass.
        let off = build_offsets(&dims)?;
        let tensor_bytes = off
            .total
            .checked_mul(4)
            .ok_or_else(|| err("tensor bytes overflow"))?;
        let remaining = bytes.len() - r.pos;
        if remaining != tensor_bytes {
            return Err(err(format!(
                "tensor section size mismatch: header implies {tensor_bytes} bytes, file has {remaining}"
            )));
        }
        let raw = r.take(tensor_bytes)?;
        let mut params = Vec::with_capacity(off.total);
        for chunk in raw.chunks_exact(4) {
            params.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        Ok(EncoderArtifact {
            dims,
            tok,
            query_prefix,
            passage_prefix,
            params,
            off,
            version: version as u64,
        })
    }

    /// The artifact version, for ledger staleness wiring
    /// (`MethodStamp { id: M_ENCODER, version }`) in T4.
    pub fn method_stamp(&self) -> (u32, u64) {
        (M_ENCODER, self.version)
    }

    pub fn word_emb(&self) -> &[f32] {
        let n = self.dims.vocab * self.dims.hidden;
        &self.params[self.off.word_emb..self.off.word_emb + n]
    }

    pub fn pos_type_emb(&self) -> &[f32] {
        let n = self.dims.max_pos * self.dims.hidden;
        &self.params[self.off.pos_type_emb..self.off.pos_type_emb + n]
    }

    pub fn emb_ln(&self) -> (&[f32], &[f32]) {
        let h = self.dims.hidden;
        (
            &self.params[self.off.emb_ln_g..self.off.emb_ln_g + h],
            &self.params[self.off.emb_ln_b..self.off.emb_ln_b + h],
        )
    }

    pub fn layer(&self, l: usize) -> LayerView<'_> {
        let o = &self.off.layers[l];
        let h = self.dims.hidden;
        let f = self.dims.ffn;
        let p = &self.params;
        LayerView {
            wq: &p[o.wq..o.wq + h * h],
            bq: &p[o.bq..o.bq + h],
            wk: &p[o.wk..o.wk + h * h],
            bk: &p[o.bk..o.bk + h],
            wv: &p[o.wv..o.wv + h * h],
            bv: &p[o.bv..o.bv + h],
            wo: &p[o.wo..o.wo + h * h],
            bo: &p[o.bo..o.bo + h],
            ln1_g: &p[o.ln1_g..o.ln1_g + h],
            ln1_b: &p[o.ln1_b..o.ln1_b + h],
            w1: &p[o.w1..o.w1 + h * f],
            b1: &p[o.b1..o.b1 + f],
            w2: &p[o.w2..o.w2 + f * h],
            b2: &p[o.b2..o.b2 + h],
            ln2_g: &p[o.ln2_g..o.ln2_g + h],
            ln2_b: &p[o.ln2_b..o.ln2_b + h],
        }
    }
}

// ---------------------------------------------------------------------------
// Toy artifact builder (test-only; the real artifact is converted offline).
// Mirrors the 1B pattern: byte-reproducible from a seed so the committed
// fixture regenerates exactly. Two flavors: WordPiece (covers the bge shape)
// and Unigram (inlines the toy SKUNI from `skinki-vector::unigram::toy`).
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod toy {
    use super::*;
    use skinki_vector::Rng;

    pub(crate) const TOY_VOCAB: &[&str] = &[
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "the",
        "a",
        "and",
        "of",
        "to",
        "in",
        "is",
        "it",
        "that",
        "for",
        "on",
        "with",
        "as",
        "at",
        "by",
        "be",
        "memory",
        "engine",
        "rust",
        "vector",
        "search",
        "recall",
        "sleep",
        "insight",
        "graph",
        "store",
        "embed",
        "compress",
        "happy",
        "sad",
        "running",
        "tests",
        "coded",
        "##ing",
        "##s",
        "##ed",
        "##er",
        "##e",
        "##d",
        "##y",
        "##ment",
        "##tion",
        "память",
        "поиск",
        ",",
        ".",
    ];

    pub(crate) const TOY_DIMS: EncoderDims = EncoderDims {
        layers: 2,
        hidden: 16,
        ffn: 32,
        heads: 2,
        vocab: 50,
        max_pos: 16,
        pooling: Pooling::Cls,
        ln_eps: 1e-12,
    };

    /// Build the SKENC001 v2 byte image for the toy **WordPiece** artifact
    /// (covers the bge shape). Seeded ~N(0, 0.05) weights, LayerNorm
    /// gamma = 1 / beta = 0. Pure and deterministic so the committed
    /// `fixtures/encoder_toy.skenc` is byte-reproducible.
    pub(crate) fn build_toy_artifact(seed: u64) -> Vec<u8> {
        build_toy_wordpiece(TOY_DIMS, seed, "", "")
    }

    /// Build a v2 WordPiece artifact with explicit dims (used by Unigram-via-
    /// wordpiece-dims tests too) and prefix strings.
    pub(crate) fn build_toy_wordpiece(
        dims: EncoderDims,
        seed: u64,
        query_prefix: &str,
        passage_prefix: &str,
    ) -> Vec<u8> {
        let pooling_tag = match dims.pooling {
            Pooling::Cls => 0u32,
            Pooling::Mean => 1u32,
        };
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        for v in [
            FORMAT_VERSION,
            ARCH_BERT,
            dims.layers as u32,
            dims.hidden as u32,
            dims.ffn as u32,
            dims.heads as u32,
            dims.vocab as u32,
            dims.max_pos as u32,
            pooling_tag,
            TOK_WORDPIECE,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        // Prefix lengths + bytes.
        let qb = query_prefix.as_bytes();
        let pb = passage_prefix.as_bytes();
        out.extend_from_slice(&(qb.len() as u32).to_le_bytes());
        out.extend_from_slice(&(pb.len() as u32).to_le_bytes());
        out.extend_from_slice(qb);
        out.extend_from_slice(pb);

        out.extend_from_slice(&dims.ln_eps.to_le_bytes());

        for s in TOY_VOCAB {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        while out.len() % 4 != 0 {
            out.push(0);
        }

        append_toy_tensors(&mut out, dims, seed);
        out
    }

    /// Build a v2 Unigram artifact: same toy dims, but the tokenizer section
    /// is the inlined toy SKUNI bytes (mirroring
    /// `skinki-vector::unigram::toy::build_toy_artifact` byte-for-byte, since
    /// cross-crate `cfg(test)` toys are not visible here — same deliberate
    /// duplication as the encoder/static-embedder toys). `query_prefix` /
    /// `passage_prefix` are exercised by callers.
    pub(crate) fn build_toy_unigram(
        dims: EncoderDims,
        seed: u64,
        query_prefix: &str,
        passage_prefix: &str,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        let pooling_tag = match dims.pooling {
            Pooling::Cls => 0u32,
            Pooling::Mean => 1u32,
        };
        for v in [
            FORMAT_VERSION,
            ARCH_BERT,
            dims.layers as u32,
            dims.hidden as u32,
            dims.ffn as u32,
            dims.heads as u32,
            dims.vocab as u32,
            dims.max_pos as u32,
            pooling_tag,
            TOK_UNIGRAM,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let qb = query_prefix.as_bytes();
        let pb = passage_prefix.as_bytes();
        out.extend_from_slice(&(qb.len() as u32).to_le_bytes());
        out.extend_from_slice(&(pb.len() as u32).to_le_bytes());
        out.extend_from_slice(qb);
        out.extend_from_slice(pb);

        out.extend_from_slice(&dims.ln_eps.to_le_bytes());

        // Inline the toy SKUNI bytes. The Unigram reader path validates this
        // nested artifact, so a corrupt inline blob dies with InvalidData.
        let sku = build_toy_sku();
        out.extend_from_slice(&(sku.len() as u32).to_le_bytes());
        out.extend_from_slice(&sku);

        while out.len() % 4 != 0 {
            out.push(0);
        }

        append_toy_tensors(&mut out, dims, seed);
        out
    }

    /// Build the toy `SKUNI001` byte image — a verbatim mirror of
    /// `skinki-vector::unigram::toy::build_toy_artifact`. Kept in sync by the
    /// `toy_sku_matches_skinki_vector` test below. Public-encoder tests need a
    /// Unigram tokenizer inline but cannot reach `skinki-vector`'s `cfg(test)`
    /// toy module cross-crate.
    fn build_toy_sku() -> Vec<u8> {
        // Constants copied from `skinki-vector/src/unigram.rs` (TOY_PIECES,
        // TOY_CHARSMAP). Same hand-authored vocab + charsmap; same flags.
        const TOY_PIECES: &[(&str, f32, u32)] = &[
            ("<unk>", 0.0, 2),
            ("<s>", 0.0, 3),
            ("</s>", 0.0, 3),
            ("a", -2.0, 1),
            ("b", -2.0, 1),
            ("ab", -3.0, 1),
            ("▁", -1.0, 1),
            ("▁cat", -2.5, 1),
        ];
        const TOY_CHARSMAP: &[(&str, &str)] = &[
            ("\t", " "),
            ("\u{00A0}", " "),
            ("\u{FF21}", "A"),
            ("\u{0065}\u{0301}", "\u{00E9}"),
        ];
        const SKU_MAGIC: &[u8; 8] = b"SKUNI001";
        const SKU_FORMAT_VERSION: u32 = 1;
        const FLAG_ADD_DUMMY_PREFIX: u32 = 1 << 0;
        const FLAG_REMOVE_EXTRA_WHITESPACES: u32 = 1 << 1;
        const FLAG_ESCAPE_WHITESPACES: u32 = 1 << 2;

        let mut out = Vec::new();
        out.extend_from_slice(SKU_MAGIC);
        out.extend_from_slice(&SKU_FORMAT_VERSION.to_le_bytes());
        let vocab_size = TOY_PIECES.len() as u32;
        let sp_unk_id = 0u32;
        let fairseq_offset = 1u32;
        let unk_hf_id = 3u32;
        let bos_hf_id = 0u32;
        let eos_hf_id = 2u32;
        let min_normal = TOY_PIECES
            .iter()
            .filter(|&&(_, _, ty)| ty == 1)
            .map(|&(_, score, _)| score)
            .fold(f32::INFINITY, f32::min);
        let unk_score = min_normal - 10.0;
        let flags = FLAG_ADD_DUMMY_PREFIX | FLAG_REMOVE_EXTRA_WHITESPACES | FLAG_ESCAPE_WHITESPACES;
        let charsmap_entry_count = TOY_CHARSMAP.len() as u32;
        for v in [
            vocab_size,
            sp_unk_id,
            fairseq_offset,
            unk_hf_id,
            bos_hf_id,
            eos_hf_id,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&unk_score.to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&charsmap_entry_count.to_le_bytes());
        for &(piece, score, ty) in TOY_PIECES {
            let b = piece.as_bytes();
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(b);
            out.extend_from_slice(&score.to_le_bytes());
            out.extend_from_slice(&ty.to_le_bytes());
        }
        for &(key, val) in TOY_CHARSMAP {
            let kb = key.as_bytes();
            out.extend_from_slice(&(kb.len() as u32).to_le_bytes());
            out.extend_from_slice(kb);
            let vb = val.as_bytes();
            out.extend_from_slice(&(vb.len() as u32).to_le_bytes());
            out.extend_from_slice(vb);
        }
        out
    }

    fn append_toy_tensors(out: &mut Vec<u8>, d: EncoderDims, seed: u64) {
        let mut rng = Rng::new(seed);
        let mut norm = |out: &mut Vec<u8>, count: usize, scale: f32| {
            for _ in 0..count {
                out.extend_from_slice(&(rng.normal() * scale).to_le_bytes());
            }
        };
        let ones = |out: &mut Vec<u8>, count: usize| {
            for _ in 0..count {
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
        };
        let zeros = |out: &mut Vec<u8>, count: usize| {
            for _ in 0..count {
                out.extend_from_slice(&0.0f32.to_le_bytes());
            }
        };

        let h = d.hidden;
        norm(out, d.vocab * h, 0.1); // word_emb
        norm(out, d.max_pos * h, 0.1); // pos_type_emb
        ones(out, h); // emb_ln_gamma
        zeros(out, h); // emb_ln_beta
        for _ in 0..d.layers {
            for _ in 0..4 {
                // wq/bq, wk/bk, wv/bv, wo/bo
                norm(out, h * h, 0.05);
                norm(out, h, 0.02);
            }
            ones(out, h); // ln1_gamma
            zeros(out, h); // ln1_beta
            norm(out, h * d.ffn, 0.05); // w1
            norm(out, d.ffn, 0.02); // b1
            norm(out, d.ffn * h, 0.05); // w2
            norm(out, h, 0.02); // b2
            ones(out, h); // ln2_gamma
            zeros(out, h); // ln2_beta
        }
    }
}

#[cfg(test)]
mod tests {
    use super::toy::{build_toy_artifact, build_toy_unigram, build_toy_wordpiece, TOY_DIMS};
    use super::*;

    #[test]
    fn toy_round_trips() {
        let bytes = build_toy_artifact(0xC0DE);
        let e = EncoderArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(e.dims.layers, TOY_DIMS.layers);
        assert_eq!(e.dims.hidden, TOY_DIMS.hidden);
        assert_eq!(e.dims.ffn, TOY_DIMS.ffn);
        assert_eq!(e.dims.heads, TOY_DIMS.heads);
        assert_eq!(e.dims.pooling, TOY_DIMS.pooling);
        assert_eq!(e.tok.kind(), TokenizerKind::WordPiece);
        assert_eq!(e.method_stamp(), (M_ENCODER, 2));
        // Special ids resolve and match vocab order.
        assert_eq!(e.tok.bos_id(), 1); // [CLS]
        assert_eq!(e.tok.eos_id(), 2); // [SEP]
                                       // Every accessor slice has the advertised length.
        assert_eq!(e.word_emb().len(), TOY_DIMS.vocab * TOY_DIMS.hidden);
        assert_eq!(e.pos_type_emb().len(), TOY_DIMS.max_pos * TOY_DIMS.hidden);
        let l = e.layer(1);
        assert_eq!(l.w1.len(), TOY_DIMS.hidden * TOY_DIMS.ffn);
        assert_eq!(l.b1.len(), TOY_DIMS.ffn);
        assert_eq!(l.ln2_g.len(), TOY_DIMS.hidden);
    }

    #[test]
    fn toy_unigram_round_trips() {
        // Mean pooling + Unigram tokenizer + prefixes — the e5 shape, on the
        // toy dims. Validates the inlined-SKUNI path end to end.
        let dims = EncoderDims {
            pooling: Pooling::Mean,
            ..TOY_DIMS
        };
        let bytes = build_toy_unigram(dims, 0xBEEF, "query: ", "passage: ");
        let e = EncoderArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(e.dims.pooling, Pooling::Mean);
        assert_eq!(e.tok.kind(), TokenizerKind::Unigram);
        assert_eq!(e.query_prefix, "query: ");
        assert_eq!(e.passage_prefix, "passage: ");
        // bos/eos come from the toy SKUNI (bos_hf_id=0, eos_hf_id=2).
        assert_eq!(e.tok.bos_id(), 0);
        assert_eq!(e.tok.eos_id(), 2);
        // Encode-content must route through the inlined Unigram.
        let ids = e.tok.encode_content("a b");
        assert!(!ids.is_empty());
    }

    #[test]
    fn wordpiece_prefixes_round_trip() {
        let bytes = build_toy_wordpiece(TOY_DIMS, 7, "q:", "p:");
        let e = EncoderArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(e.query_prefix, "q:");
        assert_eq!(e.passage_prefix, "p:");
    }

    #[test]
    fn toy_is_byte_reproducible() {
        assert_eq!(build_toy_artifact(0xC0DE), build_toy_artifact(0xC0DE));
        assert_ne!(build_toy_artifact(0xC0DE), build_toy_artifact(0xC0DF));
    }

    /// The committed fixture must equal the builder output byte-for-byte.
    /// Regenerate with `cargo test -p skinki-encoder gen_toy -- --ignored`.
    #[test]
    fn committed_toy_fixture_matches_builder() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/encoder_toy.skenc"
        );
        let on_disk = std::fs::read(path).expect("fixtures/encoder_toy.skenc missing");
        assert_eq!(
            on_disk,
            build_toy_artifact(0xC0DE),
            "committed toy fixture out of sync with builder"
        );
    }

    #[test]
    #[ignore = "regenerates fixtures/encoder_toy.skenc"]
    fn gen_toy_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/encoder_toy.skenc"
        );
        std::fs::write(path, build_toy_artifact(0xC0DE)).unwrap();
    }

    /// The real converted artifact parses and matches the bge-small shape.
    /// `#[ignore]` — the ~130 MB artifact is not committed; regenerate with
    /// `scripts/convert_encoder_to_skenc.py --teacher BAAI/bge-small-en-v1.5`
    /// first.
    #[test]
    #[ignore = "needs fixtures/encoder_bge_small.skenc — regenerate with scripts/convert_encoder_to_skenc.py"]
    fn real_artifact_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/encoder_bge_small.skenc");
        let e = EncoderArtifact::load(&path).expect("load real artifact");
        assert_eq!(e.dims.layers, 12);
        assert_eq!(e.dims.hidden, 384);
        assert_eq!(e.dims.ffn, 1536);
        assert_eq!(e.dims.heads, 12);
        assert_eq!(e.dims.vocab, 30522);
        assert_eq!(e.dims.max_pos, 512);
        assert_eq!(e.dims.pooling, Pooling::Cls);
        assert_eq!(e.tok.kind(), TokenizerKind::WordPiece);
        // The tokenizer must resolve BERT's canonical special ids.
        assert_eq!(e.tok.bos_id(), 101); // [CLS]
        assert_eq!(e.tok.eos_id(), 102); // [SEP]
    }

    /// The real converted multilingual-e5-small artifact parses.
    /// `#[ignore]` — the ~472 MB artifact is not committed; regenerate with
    /// `scripts/convert_encoder_to_skenc.py --teacher intfloat/multilingual-e5-small
    /// --tokenizer unigram --unigram-sku fixtures/unigram_e5_small.sku
    /// --pooling mean --query-prefix 'query: ' --passage-prefix 'passage: '`.
    #[test]
    #[ignore = "needs fixtures/encoder_e5_small.skenc — regenerate with scripts/convert_encoder_to_skenc.py"]
    fn real_e5_artifact_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/encoder_e5_small.skenc");
        let e = EncoderArtifact::load(&path).expect("load real e5 artifact");
        assert_eq!(e.dims.layers, 12);
        assert_eq!(e.dims.hidden, 384);
        assert_eq!(e.dims.ffn, 1536);
        assert_eq!(e.dims.heads, 12);
        // XLM-R ships 250037 word embeddings: 250000 SentencePiece pieces +
        // the fairseq specials (<s>, <pad>, </s>, <unk>, <mask>).
        assert_eq!(e.dims.vocab, 250_037);
        assert_eq!(e.dims.max_pos, 512);
        assert_eq!(e.dims.pooling, Pooling::Mean);
        assert_eq!(e.tok.kind(), TokenizerKind::Unigram);
        assert_eq!(e.query_prefix, "query: ");
        assert_eq!(e.passage_prefix, "passage: ");
        // XLM-R bos/eos (verified via dump_unigram_fixtures.py).
        assert_eq!(e.tok.bos_id(), 0);
        assert_eq!(e.tok.eos_id(), 2);
    }

    #[test]
    fn rejects_bad_magic_version_arch() {
        let good = build_toy_artifact(1);
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(EncoderArtifact::from_bytes(&bad).is_err());
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&9u32.to_le_bytes()); // version
        assert!(EncoderArtifact::from_bytes(&bad).is_err());
        let mut bad = good.clone();
        bad[12..16].copy_from_slice(&7u32.to_le_bytes()); // arch
        assert!(EncoderArtifact::from_bytes(&bad).is_err());
    }

    /// v1 artifacts (Stage 1C-B) must be rejected loudly — the v1 layout has
    /// no `tok_kind`/prefix fields, so a silent rehash would feed the encoder
    /// garbage header data.
    #[test]
    fn rejects_v1_artifact() {
        let mut v1 = build_toy_artifact(2);
        // Rewrite the version field (offset 8) to 1.
        v1[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert!(EncoderArtifact::from_bytes(&v1).is_err());
    }

    #[test]
    fn rejects_truncation_everywhere() {
        // Every strict prefix must fail with InvalidData, never panic. This
        // sweeps the header, the vocab section and the tensor section in one
        // property (the toy file is ~50 KB; step keeps the test fast).
        let good = build_toy_artifact(2);
        let mut cut = 0;
        while cut < good.len() {
            let r = EncoderArtifact::from_bytes(&good[..cut]);
            assert!(r.is_err(), "prefix of {cut} bytes unexpectedly parsed");
            cut += 997; // prime step: hits header, vocab and tensor offsets
        }
    }

    #[test]
    fn rejects_truncation_everywhere_unigram() {
        // Same property on the Unigram path so the inlined-SKUNI section's
        // bounds checking is exercised too.
        let good = build_toy_unigram(TOY_DIMS, 5, "", "");
        let mut cut = 0;
        while cut < good.len() {
            let r = EncoderArtifact::from_bytes(&good[..cut]);
            assert!(r.is_err(), "prefix of {cut} bytes unexpectedly parsed");
            cut += 997;
        }
    }

    #[test]
    fn rejects_corrupt_vocab_len_and_dim_overflow() {
        let good = build_toy_artifact(3);
        // First vocab length prefix is at byte 60 (8 magic + 11*4 header +
        // 2*4 prefix-len + 0 prefix bytes + 4 eps = 60).
        let mut bad = good.clone();
        bad[60..64].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(EncoderArtifact::from_bytes(&bad).is_err());
        // Huge dims must trip the checked size math, not wrap.
        let mut bad = good.clone();
        bad[20..24].copy_from_slice(&u32::MAX.to_le_bytes()); // hidden
        bad[28..32].copy_from_slice(&u32::MAX.to_le_bytes()); // heads (divides)
        bad[32..36].copy_from_slice(&u32::MAX.to_le_bytes()); // vocab
        assert!(EncoderArtifact::from_bytes(&bad).is_err());
    }

    #[test]
    fn rejects_trailing_garbage() {
        // The tensor section length must match the header exactly — silent
        // trailing bytes would mean a mis-parsed artifact serving wrong rows.
        let mut bytes = build_toy_artifact(4);
        bytes.extend_from_slice(&[0u8; 4]);
        assert!(EncoderArtifact::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_missing_specials() {
        // Clobber "[CLS]" in the vocab bytes: loader must demand it.
        let mut bytes = build_toy_artifact(5);
        let needle = b"\x05\x00\x00\x00[CLS]";
        let at = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .unwrap();
        bytes[at + 4..at + 9].copy_from_slice(b"BANAN");
        assert!(EncoderArtifact::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_corrupt_inline_sku() {
        // A truncated inline SKUNI must surface as InvalidData, not a panic
        // inside the nested parser.
        let mut bytes = build_toy_unigram(TOY_DIMS, 6, "", "");
        // Find the sku_size field (right after ln_eps in the Unigram path) and
        // inflate it without extending the bytes: the nested parse must fail.
        // Easier: corrupt the SKUNI magic.
        let magic_pos = bytes
            .windows(8)
            .position(|w| w == b"SKUNI001")
            .expect("toy inline sku has SKUNI001 magic");
        bytes[magic_pos] = b'X';
        assert!(EncoderArtifact::from_bytes(&bytes).is_err());
    }
}
