//! SentencePiece **Unigram** tokenizer for `intfloat/multilingual-e5-small`
//! (an XLM-RoBERTa tokenizer) — Stage 1D K0.
//!
//! Rust never parses the SentencePiece protobuf or the `darts-clone`
//! double-array trie binary format. `scripts/dump_unigram_fixtures.py` (dev
//! tooling, offline, outside any gate) does that once and writes our own
//! sequential `SKUNI001` table: vocab pieces + log-prob scores + piece types,
//! the `precompiled_charsmap` normalization rules decoded into a flat
//! `(source, replacement)` table, and the empirically-discovered HF id
//! mapping. This module only ever does trie longest-match and a Viterbi DP
//! over numbers already computed offline — the same "replay, don't invent"
//! discipline as `static_embed.rs` and `skinki-encoder`.
//!
//! ## What was reverse-engineered, and how (see the K0 report for the
//! validation methodology — 8000-sample charsmap fuzz, 5000-sample
//! segmentation fuzz, both 0 real-text failures against `AutoTokenizer`):
//!
//! 1. **Normalization** (`sentencepiece` `normalizer.cc::Normalize`): strip
//!    leading normalized-to-space runs, prepend `▁` (dummy prefix), then walk
//!    the input applying the charsmap's **longest-prefix-match** replacement
//!    at every position (falling back to the single current `char` unchanged
//!    if no rule matches — `sentencepiece` never fails on a Rust `&str`
//!    because malformed UTF-8 cannot occur), collapsing runs of the resulting
//!    literal spaces into `▁` and dropping leading/trailing `▁`. The charsmap
//!    itself is `nmt_nfkc`-class NFKC-ish data — never a hand-rolled NFKC.
//! 2. **Segmentation** (`unigram_model.cc::PopulateNodes` +
//!    `Lattice::Viterbi`): a DP over character positions. At every position,
//!    all vocab pieces starting there are tried (longest-prefix trie walk);
//!    if none has length 1, a synthetic `<unk>` edge of length 1 is added
//!    (score = `min(NORMAL piece scores) - 10.0`, precomputed by the
//!    converter). The DP keeps the first-seen maximum on an exact score tie —
//!    empirically this is "prefer the longer final piece" (see
//!    [`segment`]'s doc comment for the derivation).
//! 3. **Unknown-run merging** (`sentencepiece_processor.cc
//!    ::PopulateSentencePieceText`): consecutive `<unk>` edges collapse into
//!    one output id (this model has `byte_fallback = false`, so there is no
//!    byte-decomposition path to replicate).
//! 4. **The XLM-R id offset**: verified empirically against `AutoTokenizer`
//!    (`tok.convert_tokens_to_ids(piece)` for probe pieces across the vocab):
//!    `hf_id = sp_id + fairseq_offset` for every ordinary vocab piece, with a
//!    single exception — the model's own `<unk>` id maps to whatever HF
//!    declares as `unk_token_id`, not through the `+offset` arithmetic. Both
//!    numbers are discovered and stored by the converter, never assumed.
//!
//! Known gap (disclosed, not swept under the rug): purely adversarial strings
//! that splice unrelated scripts with orphaned combining marks can diverge
//! from `AutoTokenizer` by one merged/dropped `<unk>` edge, because the
//! installed `transformers` version backs `XLMRobertaTokenizer` with the
//! HuggingFace **Rust** `tokenizers` crate, not the reference `sentencepiece`
//! C++ library, and the two have not been verified bit-identical on that
//! narrow corner. Every realistic-text category (EN/RU/DE/ES/CJK/emoji,
//! whitespace, full-width, combining accents, orphaned marks *adjacent to a
//! real base*, digits/punctuation) matches exactly; see the parity fixture.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

/// `▁` (U+2581 LOWER ONE EIGHTH BLOCK) — SentencePiece's whitespace escape.
const SPACE_SYMBOL: &str = "\u{2581}";

const MAGIC: &[u8; 8] = b"SKUNI001";
const FORMAT_VERSION: u32 = 1;

const FLAG_ADD_DUMMY_PREFIX: u32 = 1 << 0;
const FLAG_REMOVE_EXTRA_WHITESPACES: u32 = 1 << 1;
const FLAG_ESCAPE_WHITESPACES: u32 = 1 << 2;
const FLAG_TREAT_WHITESPACE_AS_SUFFIX: u32 = 1 << 3;

/// SentencePiece `ModelProto::SentencePiece::Type` values, copied verbatim so
/// the converter needs no translation table. Only `Normal` pieces enter the
/// segmentation trie (`model_interface.cc::InitializePieces`); `Unknown` and
/// `Control` are reachable only via the synthetic unk edge / the `<s>`/`</s>`
/// wrapping, never by matching literal text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PieceType {
    Normal,
    Unknown,
    Control,
}

impl PieceType {
    fn from_tag(tag: u32) -> io::Result<Self> {
        match tag {
            1 => Ok(PieceType::Normal),
            2 => Ok(PieceType::Unknown),
            3 => Ok(PieceType::Control),
            other => Err(err(format!("unknown piece type tag {other}"))),
        }
    }
}

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Bounds-checked byte cursor (same discipline as `skinki-encoder::format`):
/// truncation and corrupt length prefixes surface as `InvalidData`, never a
/// panic, regardless of what a hand-edited or half-written artifact contains.
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

    /// Length-prefixed UTF-8 string. Never pre-allocates from the untrusted
    /// length: growth is amortized by `take`'s own bounds check, so a corrupt
    /// `u32::MAX` dies at the first truncated read, not at an allocation.
    fn string(&mut self) -> io::Result<String> {
        let len = self.u32()? as usize;
        std::str::from_utf8(self.take(len)?)
            .map(str::to_owned)
            .map_err(|e| err(format!("string not UTF-8: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Trie: shared by the charsmap normalizer and the vocab segmenter. Keyed by
// `char`, not raw bytes — every charsmap key and every vocab piece is a whole
// number of Unicode scalar values by construction (validated empirically:
// 0/224711 decoded charsmap keys were invalid UTF-8), so char-indexing avoids
// byte/char bookkeeping entirely without losing any matches. `BTreeMap`, not
// `HashMap`: lookups are content-addressed so hashing would be safe too, but
// the repo convention (see `wordpiece.rs`) is to never reach for `HashMap`.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct TrieNode<T> {
    children: BTreeMap<char, TrieNode<T>>,
    value: Option<T>,
}

impl<T> TrieNode<T> {
    fn new() -> Self {
        TrieNode {
            children: BTreeMap::new(),
            value: None,
        }
    }

    fn insert(&mut self, key: &str, value: T) {
        let mut node = self;
        for c in key.chars() {
            node = node.children.entry(c).or_insert_with(TrieNode::new);
        }
        node.value = Some(value);
    }
}

/// Longest-prefix match starting at `chars[pos..]`. Returns the value at the
/// deepest node with `value.is_some()` reached along the path, and how many
/// `char`s it consumed — mirrors `Darts::DoubleArray::commonPrefixSearch`
/// (which is byte-wise there; char-wise here is equivalent, see above) plus
/// `Normalizer::NormalizePrefix`'s "take the longest rule" reduction.
fn longest_match<'a, T>(
    chars: &[char],
    pos: usize,
    root: &'a TrieNode<T>,
) -> Option<(&'a T, usize)> {
    let mut node = root;
    let mut best: Option<(&'a T, usize)> = None;
    for (i, &c) in chars[pos..].iter().enumerate() {
        match node.children.get(&c) {
            Some(next) => {
                node = next;
                if let Some(v) = &next.value {
                    best = Some((v, i + 1));
                }
            }
            None => break,
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct NormalizerSpec {
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
}

/// One `NormalizePrefix` step: the charsmap's longest match, or (if nothing
/// matches — e.g. a plain ASCII letter, which the charsmap never lists
/// because it needs no rewrite) the single current `char` unchanged. A
/// malformed-UTF8 fallback exists in `sentencepiece` but is unreachable from
/// a Rust `&str`, which is always valid UTF-8.
fn normalize_prefix(chars: &[char], pos: usize, charsmap: &TrieNode<String>) -> (String, usize) {
    match longest_match(chars, pos, charsmap) {
        Some((repl, consumed)) => (repl.clone(), consumed),
        None => (chars[pos].to_string(), 1),
    }
}

/// `Normalizer::Normalize`, replayed exactly (see the module doc for the
/// per-step derivation). Char-indexed throughout: `chars` is the input's
/// Unicode scalar values, not bytes.
fn normalize(text: &str, charsmap: &TrieNode<String>, spec: NormalizerSpec) -> String {
    if text.is_empty() {
        return String::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    // Ignore heading space: consume normalized-to-" " runs from the front.
    let mut pos = 0;
    if spec.remove_extra_whitespaces {
        while pos < n {
            let (repl, consumed) = normalize_prefix(&chars, pos, charsmap);
            if repl != " " {
                break;
            }
            pos += consumed;
        }
    }
    if pos >= n {
        return String::new();
    }

    let space_symbol = if spec.escape_whitespaces {
        SPACE_SYMBOL
    } else {
        " "
    };
    let mut out = String::new();
    let add_ws = |out: &mut String| out.push_str(space_symbol);

    if !spec.treat_whitespace_as_suffix && spec.add_dummy_prefix {
        add_ws(&mut out);
    }

    let mut is_prev_space = spec.remove_extra_whitespaces;
    while pos < n {
        let (repl, consumed) = normalize_prefix(&chars, pos, charsmap);
        let mut sp: &str = &repl;
        if is_prev_space {
            while let Some(rest) = sp.strip_prefix(' ') {
                sp = rest;
            }
        }
        if !sp.is_empty() {
            for ch in sp.chars() {
                if ch == ' ' {
                    add_ws(&mut out);
                } else {
                    out.push(ch);
                }
            }
            is_prev_space = sp.ends_with(' ');
        }
        pos += consumed;
        if !spec.remove_extra_whitespaces {
            is_prev_space = false;
        }
    }

    if spec.remove_extra_whitespaces {
        while let Some(stripped) = out.strip_suffix(space_symbol) {
            out.truncate(stripped.len());
        }
    }

    if spec.treat_whitespace_as_suffix && spec.add_dummy_prefix {
        add_ws(&mut out);
    }

    out
}

// ---------------------------------------------------------------------------
// Segmentation (Viterbi)
// ---------------------------------------------------------------------------

/// `Lattice::Viterbi` over the vocab trie, replayed as a forward DP.
///
/// **Tie-break** (fixed and deterministic, per the K0 design constraint):
/// reference `sentencepiece` fills `end_nodes_[end]` for a fixed `end` in
/// ascending `begin` order (`PopulateNodes`'s outer loop), and
/// `Viterbi()`'s inner comparison is strict `score > best_score`, so the
/// *first*-considered candidate wins an exact tie. Since `begin` ascending at
/// fixed `end` means piece length `end - begin` *descending*, an exact score
/// tie resolves in favor of the **longer** final piece. This DP achieves the
/// identical candidate order the other way around — outer loop over `begin`
/// ascending, inner loop over length ascending — which updates `dp[end]` for
/// every `end` in the same ascending-`begin` order over the course of the
/// whole run (a candidate from `begin` can only be considered once all
/// `dp[s]` for `s < begin` are final, since every edge only moves forward).
fn segment(
    normalized: &str,
    vocab: &TrieNode<(u32, f32)>,
    unk_sp_id: u32,
    unk_score: f32,
) -> Vec<u32> {
    let chars: Vec<char> = normalized.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }

    let mut dp = vec![f32::NEG_INFINITY; n + 1];
    dp[0] = 0.0;
    let mut back_start = vec![0usize; n + 1];
    let mut back_id = vec![0u32; n + 1];

    for begin in 0..n {
        if !dp[begin].is_finite() {
            // Unreachable in practice: the length-1 fallback below guarantees
            // every position is reachable. Guard anyway rather than trust it.
            continue;
        }
        let base = dp[begin];
        let mut node = vocab;
        let mut found_len1 = false;
        for length in 1..=(n - begin) {
            let c = chars[begin + length - 1];
            let next = match node.children.get(&c) {
                Some(next) => next,
                None => break,
            };
            node = next;
            if let Some(&(id, score)) = node.value.as_ref() {
                if length == 1 {
                    found_len1 = true;
                }
                let end = begin + length;
                let total = base + score;
                if total > dp[end] {
                    dp[end] = total;
                    back_start[end] = begin;
                    back_id[end] = id;
                }
            }
        }
        if !found_len1 {
            let end = begin + 1;
            let total = base + unk_score;
            if total > dp[end] {
                dp[end] = total;
                back_start[end] = begin;
                back_id[end] = unk_sp_id;
            }
        }
    }

    let mut ids_rev = Vec::new();
    let mut pos = n;
    while pos > 0 {
        ids_rev.push(back_id[pos]);
        pos = back_start[pos];
    }
    ids_rev.reverse();

    // Merge consecutive <unk> edges into one output id
    // (`PopulateSentencePieceText`'s unk-run merge; this model has
    // byte_fallback = false, so there is no byte-decomposition path here).
    let mut merged = Vec::with_capacity(ids_rev.len());
    for id in ids_rev {
        if id == unk_sp_id && merged.last() == Some(&unk_sp_id) {
            continue;
        }
        merged.push(id);
    }
    merged
}

// ---------------------------------------------------------------------------
// Artifact + public tokenizer
// ---------------------------------------------------------------------------

/// A loaded `SKUNI001` artifact: charsmap trie, vocab trie, and the
/// empirically-discovered HF id mapping. Construct via
/// [`UnigramTokenizer::load`] or [`UnigramTokenizer::from_bytes`].
pub struct UnigramTokenizer {
    charsmap: TrieNode<String>,
    vocab: TrieNode<(u32, f32)>,
    spec: NormalizerSpec,
    sp_unk_id: u32,
    fairseq_offset: u32,
    unk_hf_id: u32,
    bos_hf_id: u32,
    eos_hf_id: u32,
    unk_score: f32,
    vocab_size: u32,
}

impl std::fmt::Debug for UnigramTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnigramTokenizer")
            .field("vocab_size", &self.vocab_size)
            .field("sp_unk_id", &self.sp_unk_id)
            .field("fairseq_offset", &self.fairseq_offset)
            .finish()
    }
}

impl UnigramTokenizer {
    pub fn load(path: &Path) -> io::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Parse from raw bytes (also the test entry point). See the module doc
    /// for the layout rationale; the byte format itself:
    ///
    /// ```text
    /// magic "SKUNI001" (8 bytes)
    /// u32 version (=1)
    /// u32 vocab_size | u32 sp_unk_id | u32 fairseq_offset
    /// u32 unk_hf_id | u32 bos_hf_id | u32 eos_hf_id
    /// f32 unk_score                     // precomputed: min(NORMAL scores) - 10.0
    /// u32 flags                         // bit0 add_dummy_prefix, bit1 remove_extra_whitespaces,
    ///                                    // bit2 escape_whitespaces, bit3 treat_whitespace_as_suffix
    /// u32 charsmap_entry_count
    /// vocab_size × { u32 len | UTF-8 piece | f32 score | u32 type (1=Normal,2=Unknown,3=Control) }
    /// charsmap_entry_count × { u32 key_len | UTF-8 key | u32 val_len | UTF-8 val }
    /// ```
    ///
    /// No tensor-style fixed-size arena is needed here (unlike `SKENC001`):
    /// every record is read field-by-field through the bounds-checked
    /// `Reader`, so there is no raw-byte reinterpretation and hence no
    /// alignment requirement.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut r = Reader::new(bytes);
        let magic = r.take(8)?;
        if magic != MAGIC {
            return Err(err(format!("bad magic: expected {MAGIC:?}, got {magic:?}")));
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(err(format!(
                "unsupported artifact version {version} (want {FORMAT_VERSION})"
            )));
        }
        let vocab_size = r.u32()?;
        let sp_unk_id = r.u32()?;
        let fairseq_offset = r.u32()?;
        let unk_hf_id = r.u32()?;
        let bos_hf_id = r.u32()?;
        let eos_hf_id = r.u32()?;
        let unk_score = r.f32()?;
        let flags = r.u32()?;
        let charsmap_entry_count = r.u32()?;

        if vocab_size == 0 {
            return Err(err("vocab_size must be non-zero"));
        }
        if sp_unk_id >= vocab_size {
            return Err(err(format!(
                "sp_unk_id {sp_unk_id} out of range for vocab_size {vocab_size}"
            )));
        }
        if !unk_score.is_finite() {
            return Err(err("unk_score must be finite"));
        }

        let spec = NormalizerSpec {
            add_dummy_prefix: flags & FLAG_ADD_DUMMY_PREFIX != 0,
            remove_extra_whitespaces: flags & FLAG_REMOVE_EXTRA_WHITESPACES != 0,
            escape_whitespaces: flags & FLAG_ESCAPE_WHITESPACES != 0,
            treat_whitespace_as_suffix: flags & FLAG_TREAT_WHITESPACE_AS_SUFFIX != 0,
        };

        let mut vocab = TrieNode::new();
        let mut saw_unk_id_as_unknown_type = false;
        for i in 0..vocab_size {
            let piece = r.string()?;
            let score = r.f32()?;
            if !score.is_finite() {
                return Err(err(format!("piece {i} has non-finite score {score}")));
            }
            let ty = PieceType::from_tag(r.u32()?)?;
            if ty == PieceType::Normal {
                if piece.is_empty() {
                    return Err(err(format!("piece {i} is empty")));
                }
                vocab.insert(&piece, (i, score));
            }
            if i == sp_unk_id {
                saw_unk_id_as_unknown_type = ty == PieceType::Unknown;
            }
        }
        if !saw_unk_id_as_unknown_type {
            return Err(err(format!(
                "sp_unk_id {sp_unk_id} is not tagged as an Unknown-type piece"
            )));
        }

        let mut charsmap = TrieNode::new();
        for _ in 0..charsmap_entry_count {
            let key = r.string()?;
            let val = r.string()?;
            if key.is_empty() {
                return Err(err("charsmap key must not be empty"));
            }
            charsmap.insert(&key, val);
        }

        if r.pos != bytes.len() {
            return Err(err(format!(
                "trailing garbage: {} unconsumed bytes",
                bytes.len() - r.pos
            )));
        }

        Ok(UnigramTokenizer {
            charsmap,
            vocab,
            spec,
            sp_unk_id,
            fairseq_offset,
            unk_hf_id,
            bos_hf_id,
            eos_hf_id,
            unk_score,
            vocab_size,
        })
    }

    fn sp_id_to_hf_id(&self, sp_id: u32) -> u32 {
        if sp_id == self.sp_unk_id {
            self.unk_hf_id
        } else {
            // Content ids never overflow u32 in practice (vocab_size is a
            // few hundred thousand at most), but saturate rather than wrap:
            // a corrupt fairseq_offset must never silently alias another id.
            sp_id.saturating_add(self.fairseq_offset)
        }
    }

    /// Normalize + segment, in HF id space, **without** `<s>`/`</s>` framing.
    pub fn encode_content(&self, text: &str) -> Vec<u32> {
        let normalized = normalize(text, &self.charsmap, self.spec);
        segment(&normalized, &self.vocab, self.sp_unk_id, self.unk_score)
            .into_iter()
            .map(|id| self.sp_id_to_hf_id(id))
            .collect()
    }

    /// The full HF-equivalent encoding: `[<s>] content [</s>]`, matching
    /// `AutoTokenizer(text, add_special_tokens=True)` byte-for-byte on the
    /// parity fixture (see `fixtures/unigram_parity.json`).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        out.push(self.bos_hf_id);
        out.extend(self.encode_content(text));
        out.push(self.eos_hf_id);
        out
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn bos_hf_id(&self) -> u32 {
        self.bos_hf_id
    }

    pub fn eos_hf_id(&self) -> u32 {
        self.eos_hf_id
    }
}

// ---------------------------------------------------------------------------
// Toy artifact builder (test-only; the real artifact is converted offline by
// `scripts/dump_unigram_fixtures.py`). Hand-authored rather than
// RNG-generated (cf. the encoder/static-embedder toy pattern): Viterbi
// behavior is only meaningful to test against a small, readable vocabulary
// with deliberately chosen scores, including one designed to exercise the
// documented tie-break.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod toy {
    use super::*;

    /// `(piece, score, type_tag)`. Scores are illustrative log-probs, not
    /// real trained values: higher (less negative) = more probable. Vocab
    /// mirrors XLM-R's convention of leading pieces carrying their own `▁`.
    /// Every entry is exercised by at least one test below; the dedicated
    /// tie-break test builds its own separate two-piece vocab rather than
    /// reusing this one, so the exact-tie arithmetic stays easy to audit.
    pub(crate) const TOY_PIECES: &[(&str, f32, u32)] = &[
        ("<unk>", 0.0, 2), // id 0
        ("<s>", 0.0, 3),   // id 1
        ("</s>", 0.0, 3),  // id 2
        ("a", -2.0, 1),    // id 3
        ("b", -2.0, 1),    // id 4
        ("ab", -3.0, 1), // id 5: score(ab) > score(a)+score(b) (-3.0 > -4.0) -> "ab" wins on merit
        ("▁", -1.0, 1),  // id 6: lone space piece
        ("▁cat", -2.5, 1), // id 7
    ];

    pub(crate) const TOY_CHARSMAP: &[(&str, &str)] = &[
        ("\t", " "),                      // tab -> space (real nmt_nfkc rule, verified above)
        ("\u{00A0}", " "),                // NBSP -> space
        ("\u{FF21}", "A"),                // full-width 'Ａ' -> 'A'
        ("\u{0065}\u{0301}", "\u{00E9}"), // "e" + combining acute -> precomposed 'é'
    ];

    pub(crate) fn build_toy_artifact() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        let vocab_size = TOY_PIECES.len() as u32;
        let sp_unk_id = 0u32;
        let fairseq_offset = 1u32;
        let unk_hf_id = 3u32; // mirrors the real model's fairseq convention
        let bos_hf_id = 0u32;
        let eos_hf_id = 2u32;
        // min(NORMAL scores) - 10.0, computed by hand from TOY_PIECES above.
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
}

#[cfg(test)]
mod tests {
    use super::toy::build_toy_artifact;
    use super::*;

    #[test]
    fn toy_round_trips() {
        let bytes = build_toy_artifact();
        let tok = UnigramTokenizer::from_bytes(&bytes).unwrap();
        assert_eq!(tok.vocab_size(), 8);
        assert_eq!(tok.bos_hf_id(), 0);
        assert_eq!(tok.eos_hf_id(), 2);
    }

    #[test]
    fn toy_is_byte_reproducible() {
        assert_eq!(build_toy_artifact(), build_toy_artifact());
    }

    /// The committed fixture must equal the builder output byte-for-byte.
    /// Regenerate with `cargo test -p skinki-vector gen_toy -- --ignored`.
    #[test]
    fn committed_toy_fixture_matches_builder() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/unigram_toy.sku"
        );
        let on_disk = std::fs::read(path).expect("fixtures/unigram_toy.sku missing");
        assert_eq!(
            on_disk,
            build_toy_artifact(),
            "committed toy fixture out of sync with builder"
        );
    }

    #[test]
    #[ignore = "regenerates fixtures/unigram_toy.sku"]
    fn gen_toy_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/unigram_toy.sku"
        );
        std::fs::write(path, build_toy_artifact()).unwrap();
    }

    #[test]
    fn rejects_bad_magic_version() {
        let good = build_toy_artifact();
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(UnigramTokenizer::from_bytes(&bad).is_err());
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&9u32.to_le_bytes());
        assert!(UnigramTokenizer::from_bytes(&bad).is_err());
    }

    #[test]
    fn rejects_truncation_everywhere() {
        let good = build_toy_artifact();
        let mut cut = 0;
        while cut < good.len() {
            let r = UnigramTokenizer::from_bytes(&good[..cut]);
            assert!(r.is_err(), "prefix of {cut} bytes unexpectedly parsed");
            cut += 17; // odd step: sweeps header, piece and charsmap sections
        }
    }

    #[test]
    fn rejects_trailing_garbage() {
        let mut bytes = build_toy_artifact();
        bytes.extend_from_slice(&[0u8; 4]);
        assert!(UnigramTokenizer::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_bad_sp_unk_id() {
        let good = build_toy_artifact();
        // sp_unk_id field is right after vocab_size, at byte 12.
        let mut bad = good.clone();
        bad[12..16].copy_from_slice(&3u32.to_le_bytes()); // piece 3 = "a", type Normal, not Unknown
        assert!(UnigramTokenizer::from_bytes(&bad).is_err());
    }

    #[test]
    fn normalizer_prefers_charsmap_over_identity() {
        let tok = UnigramTokenizer::from_bytes(&build_toy_artifact()).unwrap();
        // 'a' has no charsmap rule (falls back to identity); the tab between
        // does (charsmap: "\t" -> " ", then escaped to the internal "▁").
        let normalized = normalize(&tok.charsmap, "a\tb");
        assert_eq!(normalized, format!("{SPACE_SYMBOL}a{SPACE_SYMBOL}b"));
    }

    #[test]
    fn normalizer_all_whitespace_input_collapses_to_empty() {
        // A lone tab normalizes to a single " ", which the leading-whitespace
        // strip then consumes entirely -- matches `sp.normalize("\t") == ""`
        // on the real model (verified empirically), not "just the dummy
        // prefix": the dummy prefix is only added once non-whitespace content
        // remains.
        let tok = UnigramTokenizer::from_bytes(&build_toy_artifact()).unwrap();
        assert_eq!(normalize(&tok.charsmap, "\t"), "");
        assert_eq!(normalize(&tok.charsmap, "\u{00A0}"), "");
    }

    // Small helper so the normalizer tests below read as data, not plumbing.
    fn normalize(charsmap: &TrieNode<String>, text: &str) -> String {
        super::normalize(
            text,
            charsmap,
            NormalizerSpec {
                add_dummy_prefix: true,
                remove_extra_whitespaces: true,
                escape_whitespaces: true,
                treat_whitespace_as_suffix: false,
            },
        )
    }

    #[test]
    fn normalizer_handles_nbsp_and_fullwidth_and_combining() {
        let tok = UnigramTokenizer::from_bytes(&build_toy_artifact()).unwrap();
        assert_eq!(
            normalize(&tok.charsmap, "a\u{00A0}b"),
            format!("{SPACE_SYMBOL}a{SPACE_SYMBOL}b")
        );
        assert_eq!(
            normalize(&tok.charsmap, "\u{FF21}"),
            format!("{SPACE_SYMBOL}A")
        );
        assert_eq!(
            normalize(&tok.charsmap, "\u{0065}\u{0301}"),
            format!("{SPACE_SYMBOL}\u{00E9}")
        );
    }

    #[test]
    fn normalizer_strips_leading_trailing_and_collapses_internal_whitespace() {
        let tok = UnigramTokenizer::from_bytes(&build_toy_artifact()).unwrap();
        assert_eq!(
            normalize(&tok.charsmap, "  a   b  "),
            format!("{SPACE_SYMBOL}a{SPACE_SYMBOL}b")
        );
    }

    #[test]
    fn normalizer_empty_input_is_empty() {
        let tok = UnigramTokenizer::from_bytes(&build_toy_artifact()).unwrap();
        assert_eq!(normalize(&tok.charsmap, ""), "");
        // All-whitespace input also collapses to empty (leading-strip consumes it).
        assert_eq!(normalize(&tok.charsmap, "   "), "");
    }

    #[test]
    fn segment_prefers_merged_piece_over_sum_when_score_is_higher() {
        let tok = UnigramTokenizer::from_bytes(&build_toy_artifact()).unwrap();
        // "ab": piece "ab" alone (-3.0) beats "a"+"b" (-2.0-2.0=-4.0).
        let ids = segment("ab", &tok.vocab, tok.sp_unk_id, tok.unk_score);
        assert_eq!(ids, vec![5], "expected the single merged 'ab' piece (id 5)");
    }

    #[test]
    fn segment_falls_back_to_unk_for_uncovered_chars_and_merges_runs() {
        let tok = UnigramTokenizer::from_bytes(&build_toy_artifact()).unwrap();
        // "z" has no vocab entry and no charsmap rule -> single unk edge.
        let ids = segment("zz", &tok.vocab, tok.sp_unk_id, tok.unk_score);
        assert_eq!(
            ids,
            vec![0],
            "two consecutive unk chars must merge into one id"
        );
    }

    #[test]
    fn segment_ties_break_toward_the_longer_piece() {
        // Build a tiny two-piece vocab where "xx" (len 2) and "x"+"x" (len 1 each)
        // have EXACTLY equal total score, to exercise the documented tie-break in
        // isolation from the toy artifact's other pieces.
        let mut vocab = TrieNode::new();
        vocab.insert("x", (100, -1.0f32));
        vocab.insert("xx", (200, -2.0f32)); // -2.0 == -1.0 + -1.0: exact tie
        let ids = segment("xx", &vocab, 999, -1e9);
        assert_eq!(
            ids,
            vec![200],
            "exact score tie must resolve to the longer piece"
        );
    }

    #[test]
    fn encode_wraps_with_bos_eos_and_applies_the_hf_offset() {
        let tok = UnigramTokenizer::from_bytes(&build_toy_artifact()).unwrap();
        let ids = tok.encode("a");
        // bos=0, content: "a" normalizes to "▁a"; "▁a" is not itself a toy
        // piece, so it segments as "▁"(sp id 6) + "a"(sp id 3) -> hf ids 7, 4
        // (fairseq_offset=1); eos=2.
        assert_eq!(ids.first(), Some(&0));
        assert_eq!(ids.last(), Some(&2));
        assert!(ids.len() >= 3);
    }

    #[test]
    fn encode_is_deterministic_across_runs_and_instances() {
        let bytes = build_toy_artifact();
        let a = UnigramTokenizer::from_bytes(&bytes).unwrap();
        let b = UnigramTokenizer::from_bytes(&bytes).unwrap();
        let samples = [
            "hello world",
            "",
            "  a  b  ",
            "ab abc zz",
            "\t\u{00A0}\u{FF21}",
        ];
        for s in samples {
            let ea = a.encode(s);
            let eb = b.encode(s);
            assert_eq!(ea, eb, "mismatched encode for {s:?}");
            // And repeat on the SAME instance: no interior mutability / RNG.
            assert_eq!(a.encode(s), ea);
        }
    }

    /// The real converted artifact loads and matches multilingual-e5-small's
    /// shape. `#[ignore]` — the multi-MB artifact is not committed; regenerate
    /// with `scripts/dump_unigram_fixtures.py` first.
    #[test]
    #[ignore = "needs fixtures/unigram_e5_small.sku — regenerate with scripts/dump_unigram_fixtures.py"]
    fn real_artifact_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/unigram_e5_small.sku");
        let tok = UnigramTokenizer::load(&path).expect("load real artifact");
        assert_eq!(tok.vocab_size(), 250_000);
        assert_eq!(tok.bos_hf_id(), 0);
        assert_eq!(tok.eos_hf_id(), 2);
    }

    /// Byte-exact parity vs `AutoTokenizer` on the committed golden corpus.
    /// `#[ignore]` because it needs the real (regenerable, gitignored)
    /// artifact — the corpus itself (`fixtures/unigram_parity.json`) is
    /// committed and small.
    #[test]
    #[ignore = "needs fixtures/unigram_e5_small.sku — regenerate with scripts/dump_unigram_fixtures.py"]
    fn golden_parity() {
        let artifact_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/unigram_e5_small.sku");
        let tok = UnigramTokenizer::load(&artifact_path).expect("load real artifact");

        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/unigram_parity.json");
        let raw =
            std::fs::read_to_string(&fixture_path).expect("fixtures/unigram_parity.json missing");
        let cases: Vec<(String, Vec<u32>)> =
            serde_json::from_str(&raw).expect("parity fixture is not the expected JSON shape");

        assert!(
            cases.len() >= 1000,
            "parity corpus too small: {}",
            cases.len()
        );
        let mut failures = Vec::new();
        for (text, expected) in &cases {
            let got = tok.encode(text);
            if got != *expected {
                failures.push(format!("{text:?}: expected {expected:?}, got {got:?}"));
            }
        }
        assert!(
            failures.is_empty(),
            "{}/{} parity failures:\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
    }
}
