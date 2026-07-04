//! Stage 1B — static-distilled semantic embedder (T2: format reader + WordPiece
//! tokenizer + pooling).
//!
//! The artifact (`SKEMB001`) is the rule-3 replay of an offline distillation: a
//! teacher model ran once and dumped a token→vector table; the engine replays
//! those weights forever, deterministically, with no network and no new
//! dependency. Pooling is a Zipf-weighted mean of the per-token rows followed by
//! L2 normalization — the Model2Vec recipe, executed bit-for-bit reproducibly.
//!
//! The bulk of the artifact (the table) is served demand-paged via the existing
//! `store::CodeStore` mmap quarantine; only the small vocab strings and the
//! lookup index live in resident RAM. See [`StaticEmbedder::load`].

use std::io;
use std::path::Path;

use crate::embed::Embedder;
use crate::store::CodeStore;
use crate::wordpiece::WordPieceTokenizer;

/// Method id for "static embedder" derivations (local constant — method ids are
/// per-crate in this repo, cf. `M_INTRO`/`M_REC` in `skinki-graph`). Wired into
/// the derivation ledger when embeddings become derivations in T3.
pub const M_EMBEDDER: u32 = 1;

/// The artifact magic, little-endian throughout: `SKEMB001`.
const MAGIC: &[u8; 8] = b"SKEMB001";
const FORMAT_VERSION: u32 = 1;
/// `flags` bit 0 = WordPiece tokenizer. Reserved bits reserved for a future
/// merges/BPE section.
const FLAG_WORDPIECE: u32 = 1;

/// The WordPiece unknown token (re-exported from `wordpiece`): required in
/// every artifact; its weight is 0 by construction so OOV tokens contribute
/// nothing to the pooled vector.
pub use crate::wordpiece::UNK;

/// A static embedder backed by a `SKEMB001` artifact (mmap'd table + parsed
/// WordPiece vocab). Construct via [`StaticEmbedder::load`].
pub struct StaticEmbedder {
    /// Owns the mmap (unix) or the read-back bytes (non-unix fallback).
    view: CodeStore,
    dim: usize,
    /// Shared BERT-style WordPiece tokenizer (see `crate::wordpiece`).
    tok: WordPieceTokenizer,
    /// Byte offset of the `vocab × dim` f32 table within `view`.
    table_offset: usize,
    /// Byte offset of the `vocab` f32 weights within `view`.
    weights_offset: usize,
    /// The artifact's version, surfaced via [`method_stamp`] for ledger wiring.
    version: u64,
}

impl std::fmt::Debug for StaticEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticEmbedder")
            .field("dim", &self.dim)
            .field("vocab", &self.tok.vocab_len())
            .field("version", &self.version)
            .finish()
    }
}

impl StaticEmbedder {
    /// Load and validate a `SKEMB001` artifact from `path`.
    pub fn load(path: &Path) -> io::Result<Self> {
        #[cfg(unix)]
        let view = CodeStore::Mmap(crate::store::MmapBytes::open(path)?);
        #[cfg(not(unix))]
        let view = CodeStore::Ram(std::fs::read(path)?);

        let bytes = view.as_slice();
        let mut r = Reader::new(bytes);

        let magic = r.take(8)?;
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad magic: expected {:?}, got {:?}", MAGIC, magic),
            ));
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported artifact version {version} (want {FORMAT_VERSION})"),
            ));
        }
        let dim = r.u32()? as usize;
        let vocab_count = r.u32()? as usize;
        let flags = r.u32()?;
        if dim == 0 || vocab_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "dim and vocab must be non-zero",
            ));
        }
        if flags & FLAG_WORDPIECE == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "only WordPiece (flags bit 0) artifacts are supported",
            ));
        }

        // Vocab section: `vocab_count` len-prefixed UTF-8 strings (id = order).
        let mut pieces = Vec::with_capacity(vocab_count);
        for _ in 0..vocab_count {
            let len = r.u32()? as usize;
            let s = std::str::from_utf8(r.take(len)?)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .to_owned();
            pieces.push(s);
        }
        let tok = WordPieceTokenizer::from_pieces(pieces)?;

        let table_offset = r.pos();
        // All size arithmetic is checked: a corrupt header (huge dim/vocab)
        // must yield InvalidData, not a usize overflow that wraps past the
        // bounds check in release mode.
        let (weights_offset, need) = (|| {
            let table_bytes = vocab_count.checked_mul(dim)?.checked_mul(4)?;
            let weights_offset = table_offset.checked_add(table_bytes)?;
            let weights_bytes = vocab_count.checked_mul(4)?;
            let need = weights_offset.checked_add(weights_bytes)?;
            Some((weights_offset, need))
        })()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("artifact header overflows: dim {dim} x vocab {vocab_count}"),
            )
        })?;
        if bytes.len() < need {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "artifact truncated: need {need} bytes (table @ {table_offset} + weights @ {weights_offset}), have {}",
                    bytes.len()
                ),
            ));
        }

        Ok(StaticEmbedder {
            view,
            dim,
            tok,
            table_offset,
            weights_offset,
            version: version as u64,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The artifact version, for staleness wiring (`MethodStamp { id:
    // M_EMBEDDER, version }`) when embeddings become derivations.
    pub fn method_stamp(&self) -> (u32, u64) {
        (M_EMBEDDER, self.version)
    }

    /// Byte slice for token `id`'s row in the mmap'd table.
    fn row_bytes(&self, id: u32) -> &[u8] {
        let base = self.table_offset + id as usize * self.dim * 4;
        &self.view.as_slice()[base..base + self.dim * 4]
    }

    fn weight(&self, id: u32) -> f32 {
        let base = self.weights_offset + id as usize * 4;
        let b = &self.view.as_slice()[base..base + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// WordPiece-encode `text` to a sequence of token ids (no `[CLS]`/`[SEP]`
    /// framing — pooling is over content pieces only). Delegates to the shared
    /// `crate::wordpiece` tokenizer; see its docs for the exact preprocessing.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    /// Pooling: weighted mean of the token rows, L2-normalized.
    /// Empty / all-`[UNK]` input (zero total weight) → the zero vector.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let ids = self.encode(text);
        let mut acc = vec![0.0f32; self.dim];
        let mut wsum = 0.0f32;
        for &id in &ids {
            let w = self.weight(id);
            // Fixed summation order: token order from the tokenizer, f32
            // accumulation left-to-right (rule 2 / invariant §4 bit-determinism).
            let row = self.row_bytes(id);
            for (i, x) in acc.iter_mut().enumerate() {
                let b = &row[i * 4..i * 4 + 4];
                let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                *x += w * v;
            }
            wsum += w;
        }
        if wsum > 1e-12 {
            for x in acc.iter_mut() {
                *x /= wsum;
            }
        }
        // L2-normalize (no-op for the zero vector).
        let n = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-12 {
            for x in acc.iter_mut() {
                *x /= n;
            }
        }
        acc
    }
}

impl Embedder for StaticEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        StaticEmbedder::embed(self, text)
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

// --- A tiny cursor over the mmap'd bytes. ----------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }
    fn pos(&self) -> usize {
        self.pos
    }
    /// Bounds-checked read: a truncated or corrupt artifact (e.g. a vocab
    /// string whose length prefix runs past EOF) must surface as
    /// `InvalidData`, never as a slice panic.
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        match end {
            Some(end) => {
                let s = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "artifact truncated: read of {n} bytes at offset {} exceeds file size {}",
                    self.pos,
                    self.bytes.len()
                ),
            )),
        }
    }
    fn u32(&mut self) -> io::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

// ---------------------------------------------------------------------------
// Toy artifact builder (test-only; the real artifact is distilled offline by T1).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod toy {
    use super::*;
    use crate::{normalize, Rng};

    /// A hand-built ~50-token WordPiece vocab chosen to exercise: common English
    /// words, `##ing`/`##s`/`##ed` continuations, a couple of Cyrillic words
    /// (tokenizer *correctness*, not quality), `[UNK]`, `[CLS]`, `[SEP]`, and the
    /// punctuation pieces BERT carries.
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

    /// Build the SKEMB001 byte image for the toy artifact: dim 16, seeded
    /// unit-norm rows, Zipf-style weights (`1/(rank+2)`, `[UNK]`=0). Pure and
    /// deterministic so the committed `fixtures/static_embed_toy.skemb` is
    /// byte-reproducible from this function.
    pub(crate) fn build_toy_artifact(seed: u64, dim: usize) -> Vec<u8> {
        let vocab = TOY_VOCAB;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(dim as u32).to_le_bytes());
        out.extend_from_slice(&(vocab.len() as u32).to_le_bytes());
        out.extend_from_slice(&FLAG_WORDPIECE.to_le_bytes());
        for s in vocab {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        // Table: seeded normal rows, unit-normalized. [UNK] still gets a row
        // (its weight is what zeroes it out at pooling).
        let mut rng = Rng::new(seed);
        for _ in 0..vocab.len() {
            let mut row = vec![0.0f32; dim];
            for x in row.iter_mut() {
                *x = rng.normal();
            }
            normalize(&mut row);
            for v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        // Weights: Zipf-style, [UNK]=0.
        for (rank, s) in vocab.iter().enumerate() {
            let w = if *s == UNK {
                0.0
            } else {
                1.0 / (rank as f32 + 2.0)
            };
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::toy::{build_toy_artifact, TOY_VOCAB};
    use super::*;
    use crate::dot;
    use std::io::Write;

    /// Write the toy artifact to a fresh per-call temp file and return its
    /// path. Tests run in parallel threads; each call uses a unique counter so
    /// no two tests share a file (rule 2: no flaky CI from shared temp paths).
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALL_SEQ: AtomicU64 = AtomicU64::new(0);
    fn write_toy() -> std::path::PathBuf {
        let seq = CALL_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "skinki_static_embed_{}_{}",
            std::process::id(),
            seq
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("toy.skemb");
        let bytes = build_toy_artifact(0xA11CE, 16);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
        path
    }

    // --- Format reader ------------------------------------------------------

    #[test]
    fn load_toy_succeeds() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        assert_eq!(e.dim(), 16);
        assert_eq!(e.tok.vocab_len(), TOY_VOCAB.len());
        assert_eq!(e.method_stamp(), (M_EMBEDDER, 1));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let dir =
            std::env::temp_dir().join(format!("skinki_static_embed_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("badmagic.skemb");
        std::fs::write(&path, b"BADMAGIC rest of a file that is long enough").unwrap();
        let err = StaticEmbedder::load(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_truncated() {
        let p = write_toy();
        let full = std::fs::read(&p).unwrap();
        // Truncate midway into the table.
        let short = &full[..full.len() / 2];
        let trunc_path = p.with_extension("trunc.skemb");
        std::fs::write(&trunc_path, short).unwrap();
        let err = StaticEmbedder::load(&trunc_path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&trunc_path);
    }

    /// Truncation anywhere before the table — mid-header, mid-vocab, or a
    /// vocab length prefix pointing past EOF — must be `InvalidData`, never a
    /// slice panic (the pre-fix reader panicked on all three).
    #[test]
    fn load_rejects_truncated_header_and_vocab() {
        let full = build_toy_artifact(0xA11CE, 16);
        let dir = std::env::temp_dir().join(format!(
            "skinki_static_embed_trunc_hdr_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Mid-header (5 bytes into the magic) and mid-vocab (a few bytes past
        // the fixed header, inside the first length-prefixed string).
        for (name, cut) in [("hdr", 5usize), ("vocab", 24 + 6)] {
            let path = dir.join(format!("{name}.skemb"));
            std::fs::write(&path, &full[..cut]).unwrap();
            let err = StaticEmbedder::load(&path).unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "cut={cut}");
            let _ = std::fs::remove_file(&path);
        }
        // A corrupt vocab length prefix that runs past EOF: keep the full
        // header, then claim a u32::MAX-byte first string.
        let mut bad = full.clone();
        bad[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        let path = dir.join("badlen.skemb");
        std::fs::write(&path, &bad).unwrap();
        let err = StaticEmbedder::load(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let _ = std::fs::remove_file(&path);
    }

    /// A corrupt header whose dim x vocab product overflows usize must be
    /// `InvalidData` — in release mode an unchecked multiply would wrap and
    /// sail past the length check.
    #[test]
    fn load_rejects_header_size_overflow() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SKEMB001");
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // dim
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // vocab_count
        bytes.extend_from_slice(&1u32.to_le_bytes()); // flags: WordPiece

        // One len-prefixed "[UNK]" so vocab parsing reaches the size math for
        // id 0, then EOF for id 1 — either failure mode must be InvalidData.
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(b"[UNK]");
        let dir = std::env::temp_dir().join(format!(
            "skinki_static_embed_overflow_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overflow.skemb");
        std::fs::write(&path, &bytes).unwrap();
        let err = StaticEmbedder::load(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_missing_unk() {
        // Same shape as the toy but with [UNK] renamed.
        let mut bytes = build_toy_artifact(0xA11CE, 16);
        // Find the literal "[UNK]" vocab entry (len=5 + bytes) and clobber it.
        let needle = b"\x05\x00\x00\x00[UNK]";
        let at = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .unwrap();
        bytes[at + 4..at + 9].copy_from_slice(b"BANAN");
        let dir =
            std::env::temp_dir().join(format!("skinki_static_embed_nounk_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nounk.skemb");
        std::fs::write(&path, &bytes).unwrap();
        let err = StaticEmbedder::load(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let _ = std::fs::remove_file(&path);
    }

    // --- WordPiece tokenizer ------------------------------------------------

    #[test]
    fn encode_known_words() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        // "memory" is in vocab as a whole word.
        let ids = e.encode("memory");
        assert!(!ids.is_empty());
        assert_ne!(ids[0], e.tok.unk_id());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn encode_continuation_piece() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        // "running" -> "running"? No: "running" is not in vocab, but "run" is
        // not either. We planted "running" as a whole word, so it should match.
        // Test the continuation path with a word that splits: "memorys" is not
        // in vocab but "memory" + "##s" should be the greedy split.
        let ids = e.encode("memorys");
        // Expect [memory, ##s], not [UNK].
        let memory_id = e.tok.piece_id("memory").unwrap();
        let cont_s = e.tok.piece_id("##s").unwrap();
        assert_eq!(ids, vec![memory_id, cont_s]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn encode_oov_word_is_unk() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        // "zzqx" has no greedy match at all.
        let ids = e.encode("zzqx");
        assert_eq!(ids, vec![e.tok.unk_id()]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn encode_empty_and_whitespace() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        assert!(e.encode("").is_empty());
        assert!(e.encode("   \t  ").is_empty());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn encode_punctuation_as_own_tokens() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        // "memory," -> [memory, ","] ("," is in vocab).
        let memory_id = e.tok.piece_id("memory").unwrap();
        let comma_id = e.tok.piece_id(",").unwrap();
        let ids = e.encode("memory,");
        assert_eq!(ids, vec![memory_id, comma_id]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn encode_cyrillic_word() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        // "память" is a planted whole-word Cyrillic token.
        let ids = e.encode("память");
        let id = e.tok.piece_id("память").unwrap();
        assert_eq!(ids, vec![id]);
        // Lowercase folding: "ПАМЯТЬ" -> "память".
        assert_eq!(e.encode("ПАМЯТЬ"), vec![id]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn encode_is_case_insensitive() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        assert_eq!(e.encode("MEMORY"), e.encode("memory"));
        assert_eq!(e.encode("Memory"), e.encode("memory"));
        let _ = std::fs::remove_file(&p);
    }

    // --- Pooling & embedding math ------------------------------------------

    #[test]
    fn embed_empty_is_zero_vector() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        let v = e.embed("");
        assert_eq!(v.len(), 16);
        assert!(v.iter().all(|x| *x == 0.0));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn embed_unk_only_is_zero_vector() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        // "zzqx" -> [UNK], weight 0 -> zero vector.
        let v = e.embed("zzqx");
        assert!(v.iter().all(|x| *x == 0.0));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn embed_is_unit_norm() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        for s in ["memory engine", "rust vector search", "happy running tests"] {
            let v = e.embed(s);
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "norm of {s:?} = {n}");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn embed_is_pure() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        for s in [
            "memory engine rust",
            "insight graph store compress",
            "память поиск",
            "memory, engine. search!",
            "",
        ] {
            let a = e.embed(s);
            let b = e.embed(s);
            assert_eq!(a.len(), b.len());
            for (xa, xb) in a.iter().zip(b.iter()) {
                assert_eq!(xa.to_le_bytes(), xb.to_le_bytes(), "impure for {s:?}");
            }
        }
        let _ = std::fs::remove_file(&p);
    }

    /// Golden/parity: the embedder output must equal a *hand-rolled* reference
    /// pooling computed straight from the artifact bytes, byte-for-byte. This is
    /// the same shape as the cross-impl parity test that lands with T1 (where
    /// the reference is the Python teacher's `golden_embeddings.f32`).
    #[test]
    fn embed_matches_reference_pooling() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        let raw = std::fs::read(&p).unwrap();
        let dim = e.dim();

        // Recompute table/weights offsets exactly as load() does.
        let mut header = 8 + 4 * 4;
        let vocab_count = e.tok.vocab_len();
        for _ in 0..vocab_count {
            let len = u32::from_le_bytes(raw[header..header + 4].try_into().unwrap()) as usize;
            header += 4 + len;
        }
        let table_off = header;
        let weights_off = table_off + vocab_count * dim * 4;

        let goldens = [
            "memory engine",
            "rust vector search",
            "the happy insight",
            "память поиск",
            "memorys coded",
            "memory, vector.",
        ];
        for s in goldens {
            let ids = e.encode(s);
            let mut acc = vec![0.0f32; dim];
            let mut wsum = 0.0f32;
            for &id in &ids {
                let w = f32::from_le_bytes(
                    raw[weights_off + id as usize * 4..weights_off + id as usize * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                for d in 0..dim {
                    let base = table_off + id as usize * dim * 4 + d * 4;
                    let v = f32::from_le_bytes(raw[base..base + 4].try_into().unwrap());
                    acc[d] += w * v;
                }
                wsum += w;
            }
            if wsum > 1e-12 {
                for x in acc.iter_mut() {
                    *x /= wsum;
                }
            }
            let n = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 1e-12 {
                for x in acc.iter_mut() {
                    *x /= n;
                }
            }
            let got = e.embed(s);
            assert_eq!(got.len(), acc.len());
            for (i, (g, r)) in got.iter().zip(acc.iter()).enumerate() {
                assert_eq!(
                    g.to_le_bytes(),
                    r.to_le_bytes(),
                    "golden mismatch at dim {i} for {s:?}: got {g}, want {r}"
                );
            }
        }
        let _ = std::fs::remove_file(&p);
    }

    /// Cross-impl parity (T1, invariant §4): the Rust embedder must reproduce
    /// the Python distillation's committed `fixtures/golden_embeddings.f32`
    /// byte-for-byte from the real distilled artifact. The ~30 MB artifact is
    /// model weights and is not committed (see .gitignore), so this test is
    /// `#[ignore]`d in CI; run it manually after every regeneration:
    ///
    /// ```text
    /// python3 scripts/distill_static_embedder.py \
    ///     --teacher BAAI/bge-small-en-v1.5 --dim 256 \
    ///     --out fixtures/static_embed_bge_small_256.skemb \
    ///     --golden-out fixtures/golden_embeddings.f32
    /// cargo test -p skinki-vector golden_parity -- --ignored
    /// ```
    ///
    /// Golden format: u32 count | u32 dim | count x (u32 len | UTF-8 string)
    /// | count x dim f32 LE rows (strings first so this test can iterate).
    #[test]
    #[ignore = "needs fixtures/static_embed_bge_small_256.skemb — regenerate with scripts/distill_static_embedder.py"]
    fn golden_parity_real_artifact() {
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let artifact = fixtures.join("static_embed_bge_small_256.skemb");
        let golden = fixtures.join("golden_embeddings.f32");
        let e = StaticEmbedder::load(&artifact).expect("load distilled artifact");

        let raw = std::fs::read(&golden).expect("read golden fixture");
        let mut r = Reader::new(&raw);
        let count = r.u32().unwrap() as usize;
        let dim = r.u32().unwrap() as usize;
        assert_eq!(dim, e.dim(), "golden dim vs artifact dim");
        let mut strings = Vec::with_capacity(count);
        for _ in 0..count {
            let len = r.u32().unwrap() as usize;
            strings.push(String::from_utf8(r.take(len).unwrap().to_vec()).unwrap());
        }
        for (si, s) in strings.iter().enumerate() {
            let got = e.embed(s);
            assert_eq!(got.len(), dim);
            for (i, g) in got.iter().enumerate() {
                let want = f32::from_le_bytes(r.take(4).unwrap().try_into().unwrap());
                assert_eq!(
                    g.to_le_bytes(),
                    want.to_le_bytes(),
                    "cross-impl mismatch: string {si} {s:?} dim {i}: rust {g}, python {want}"
                );
            }
        }
        // The golden file must end exactly at the last row.
        assert_eq!(r.pos(), raw.len(), "golden fixture has trailing bytes");
    }

    /// Property: cosine(embed(s), embed(s)) == 1 for non-empty, non-UNK-only.
    #[test]
    fn embed_self_cosine_is_one() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        for s in ["memory", "engine rust", "vector search insight"] {
            let v = e.embed(s);
            let c = dot(&v, &v);
            assert!((c - 1.0).abs() < 1e-4, "cos(self,{s:?}) = {c}");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn embedder_trait_object_works() {
        let p = write_toy();
        let e = StaticEmbedder::load(&p).unwrap();
        let boxed: &dyn Embedder = &e;
        assert_eq!(boxed.dim(), 16);
        let v = boxed.embed("memory");
        assert_eq!(v.len(), 16);
        let _ = std::fs::remove_file(&p);
    }

    /// Generator for the committed `fixtures/static_embed_toy.skemb`. Run once,
    /// manually, to (re)generate the fixture:
    ///   `cargo test -p skinki-vector gen_toy_fixture -- --ignored`
    /// The fixture is byte-reproducible from `build_toy_artifact(0xA11CE, 16)`
    /// and exists so out-of-crate consumers (the harness, MCP) can exercise the
    /// reader without depending on the test-only `toy` module.
    #[test]
    #[ignore]
    fn gen_toy_fixture() {
        let bytes = build_toy_artifact(0xA11CE, 16);
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/static_embed_toy.skemb");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        // Round-trip: the committed artifact must load.
        let e = StaticEmbedder::load(&path).unwrap();
        assert_eq!(e.dim(), 16);
        assert_eq!(e.tok.vocab_len(), TOY_VOCAB.len());
        let v = e.embed("memory engine");
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
        println!("wrote {} ({} bytes)", path.display(), bytes.len());
    }
}
