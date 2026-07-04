//! BERT-style WordPiece tokenization, shared by the Stage-1B static embedder
//! (`static_embed`) and the Stage-1C-B encoder (`skinki-encoder`).
//!
//! Extracted from `static_embed.rs` verbatim (Stage 1C-B T1): lowercase,
//! Unicode-aware pre-tokenization into alphanumeric runs / single punctuation
//! chars, then greedy longest-match WordPiece with `##` continuations and
//! `[UNK]` fallback. Deterministic by construction: `BTreeMap` vocab, no
//! hashing, no ambient state.

use std::collections::BTreeMap;
use std::io;

/// The WordPiece unknown token. Must be present in every vocab.
pub const UNK: &str = "[UNK]";
/// Continuation-piece prefix (BERT WordPiece convention).
pub const CONT: &str = "##";

/// A WordPiece vocabulary + encoder. Construct via
/// [`WordPieceTokenizer::from_pieces`] with the vocab strings in id order.
#[derive(Debug)]
pub struct WordPieceTokenizer {
    /// `piece string -> id`. BTreeMap so iteration is deterministic (rule 2);
    /// lookup order never affects results, but we avoid `HashMap` on principle.
    piece_to_id: BTreeMap<String, u32>,
    unk_id: u32,
}

impl WordPieceTokenizer {
    /// Build from vocab pieces in id order (id = position). Fails with
    /// `InvalidData` if `[UNK]` is absent — every artifact must carry it.
    pub fn from_pieces<I: IntoIterator<Item = String>>(pieces: I) -> io::Result<Self> {
        let mut piece_to_id = BTreeMap::new();
        let mut unk_id = None;
        for (id, s) in pieces.into_iter().enumerate() {
            let id = u32::try_from(id)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "vocab exceeds u32 ids"))?;
            if s == UNK {
                unk_id = Some(id);
            }
            piece_to_id.insert(s, id);
        }
        let unk_id = unk_id.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("vocab has no '{UNK}' token"),
            )
        })?;
        Ok(WordPieceTokenizer {
            piece_to_id,
            unk_id,
        })
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Number of pieces in the vocab.
    pub fn vocab_len(&self) -> usize {
        self.piece_to_id.len()
    }

    /// Id of an exact piece string, if present (e.g. `[CLS]`, `[SEP]`).
    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }

    /// WordPiece-encode `text` to a sequence of token ids.
    ///
    /// Preprocessing matches the uncased-BERT recipe the Stage-1B/1C-B
    /// artifacts are built for: lowercase (NFC is the conversion script's
    /// job; Rust lowercasing is byte-stable here), pre-tokenize into
    /// whitespace/punctuation-separated words, then greedy longest-match
    /// WordPiece with `##` continuations and `[UNK]` fallback. No `[CLS]`
    /// / `[SEP]` are added here — sequence framing is the caller's job.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let lowered = text.to_lowercase();
        let mut ids = Vec::new();
        for word in pretokenize(&lowered) {
            self.wordpiece(word, &mut ids);
        }
        ids
    }

    /// Greedy longest-match WordPiece for a single pre-token. On any full-miss
    /// the whole word collapses to a single `[UNK]`.
    fn wordpiece(&self, word: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = word.chars().collect();
        if chars.is_empty() {
            return;
        }
        let mut start = 0;
        let mut sub_tokens = Vec::new();
        let mut is_bad = false;
        while start < chars.len() {
            let mut end = chars.len();
            let mut cur_id = None;
            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let key = if start > 0 {
                    // Continuation piece: stored with the `##` prefix.
                    format!("{CONT}{substr}")
                } else {
                    substr
                };
                if let Some(&id) = self.piece_to_id.get(&key) {
                    cur_id = Some(id);
                    break;
                }
                end -= 1;
            }
            match cur_id {
                Some(id) => {
                    sub_tokens.push(id);
                    start = end;
                }
                None => {
                    is_bad = true;
                    break;
                }
            }
        }
        if is_bad {
            out.push(self.unk_id);
        } else {
            out.extend(sub_tokens);
        }
    }
}

/// BERT-style pre-tokenization: split into runs of alphanumeric chars (Unicode
/// aware) or single punctuation chars; whitespace separates and is dropped.
/// Deterministic and order-preserving. Returns `&str` slices into the input.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut word_start: Option<usize> = None;
    for (idx, c) in text.char_indices() {
        let end = idx + c.len_utf8();
        if c.is_alphanumeric() {
            // Extend the current alphanumeric run.
            if word_start.is_none() {
                word_start = Some(idx);
            }
        } else {
            // Flush any in-flight alphanumeric word.
            if let Some(start) = word_start.take() {
                out.push(&text[start..idx]);
            }
            // Punctuation is its own token; whitespace is dropped entirely.
            if !c.is_whitespace() {
                out.push(&text[idx..end]);
            }
        }
    }
    if let Some(start) = word_start {
        out.push(&text[start..]);
    }
    out
}
