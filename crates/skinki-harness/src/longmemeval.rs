//! LongMemEval adapter — real long-term chat-assistant memory benchmark
//! (ICLR 2025, https://github.com/xiaowu0162/LongMemEval).
//!
//! Unlike LoCoMo (one conversation, many queries), LongMemEval gives each of
//! 500 questions its OWN compiled haystack of ~40 (LongMemEval_S) or ~500
//! (LongMemEval_M) timestamped user/assistant sessions. The retrieval task is
//! per-instance: given a question, surface the evidence turns from THAT
//! question's haystack. We therefore build a fresh [`Corpus`] per instance and
//! average the per-instance scores — the official benchmark semantics.
//!
//! Five core abilities are tested, encoded in `question_type`:
//!   `single-session-user`, `single-session-assistant`,
//!   `single-session-preference`, `multi-session` (the multi-hop analogue —
//!   the regime where BM25 is expected to fail and a graph could earn its
//!   place), `temporal-reasoning`, `knowledge-update`. Questions whose
//! `question_id` ends with `_abs` are abstention instances (no evidence) and
//! are skipped for retrieval scoring, per the benchmark's own convention.
//!
//! ## Why this benchmark (the Stage-3 real-data follow-up)
//!
//! On LoCoMo the typed-fact graph did not beat BM25 on multi-hop because the
//! multi-hop gap did not exist there (BM25 cat-2 recall = 0.784 vs 0.075 on
//! synthetic). LongMemEval's `multi-session` category is explicitly designed
//! to require joining information across distant sessions — the regime where
//! BM25 is expected to leave a real gap. This adapter is the measurement
//! instrument that decides whether the gap exists; if it does, the typed-fact
//! graph (PR #3) gets a real regime to earn its keep on.
//!
//! Deliberately minimal: we navigate `serde_json::Value` rather than model the
//! full LongMemEval schema (which carries `question_date`, per-session
//! metadata, and other fields irrelevant to retrieval eval). Only the
//! retrieval-relevant fields are read.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use skinki_corpus::{
    Corpus, CorpusMeta, Difficulty, Entry, EntryId, EntryKind, GroundTruth, RecallQuery,
};

/// The LongMemEval question types. `multi-session` is the multi-hop analogue
/// — the category a graph retriever is meant to help on.
pub const QUESTION_TYPES: &[&str] = &[
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
];

/// One LongMemEval evaluation instance: a question plus its own haystack of
/// sessions. Built once from the JSON; converted to a per-instance [`Corpus`]
/// on demand during scoring.
pub struct LongMemEvalInstance {
    /// Kept for future per-instance reporting / artifact-log keying; not read
    /// by the v0 BM25-only eval loop.
    #[allow(dead_code)]
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub answer: String,
    /// Sessions in the order they appear in `haystack_sessions`. Each session
    /// is a vec of turns; each turn is `(role, content, has_answer)`.
    pub sessions: Vec<Vec<(String, String, bool)>>,
    /// Indices into `sessions` that contain evidence (from `answer_session_ids`).
    pub answer_session_idxs: Vec<usize>,
}

impl LongMemEvalInstance {
    /// Build a Stage-0 [`Corpus`] from this instance's haystack: each turn
    /// becomes an [`Entry`] (id = global turn index across all sessions, in
    /// session order; `day` = session index; `text` = `"{role}: {content}"`).
    /// `relevant_entries` = turns with `has_answer: true` (the per-turn
    /// evidence label LongMemEval provides); if none, fall back to all turns
    /// in the `answer_session_ids` sessions.
    pub fn to_corpus(&self) -> Corpus {
        let mut entries: Vec<Entry> = Vec::new();
        let mut relevant: Vec<EntryId> = Vec::new();
        let mut fallback_relevant: Vec<EntryId> = Vec::new();

        for (s_idx, session) in self.sessions.iter().enumerate() {
            let is_answer_session = self.answer_session_idxs.contains(&s_idx);
            for (t_idx, (role, content, has_answer)) in session.iter().enumerate() {
                let id = entries.len() as EntryId;
                entries.push(Entry {
                    id,
                    day: s_idx as u32,
                    date: String::new(),
                    kind: EntryKind::Text,
                    text: format!("{role}: {content}"),
                });
                if *has_answer {
                    relevant.push(id);
                }
                if is_answer_session {
                    fallback_relevant.push(id);
                }
                let _ = t_idx; // unused — id is the global turn index
            }
        }

        if relevant.is_empty() {
            relevant = fallback_relevant;
        }

        let num_entries = entries.len();
        Corpus {
            meta: CorpusMeta {
                seed: 0,
                years: 0,
                num_entries,
                difficulty: Difficulty::V2,
            },
            entries,
            ground_truth: GroundTruth {
                recall: vec![RecallQuery {
                    id: 0,
                    question: self.question.clone(),
                    answer: self.answer.clone(),
                    relevant_entries: relevant,
                }],
                ..Default::default()
            },
        }
    }
}

/// Load LongMemEval instances from `path` (a `longmemeval_s_cleaned.json` /
/// `longmemeval_m_cleaned.json` / `longmemeval_oracle.json` file). Optionally
/// filter by `question_type` (one of [`QUESTION_TYPES`]); `None` = all types.
/// `limit` caps the number of instances returned (testing). Abstention
/// questions (`question_id` ends with `_abs`) are always skipped — they have
/// no evidence to retrieve, per the benchmark's retrieval-eval convention.
pub fn load_longmemeval(
    path: &Path,
    type_filter: Option<&str>,
    limit: Option<usize>,
) -> Result<Vec<LongMemEvalInstance>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading LongMemEval dataset at {}", path.display()))?;
    let data: Vec<Value> = serde_json::from_str(&raw).context("parsing LongMemEval JSON")?;

    if let Some(qt) = type_filter {
        if !QUESTION_TYPES.contains(&qt) {
            bail!(
                "unknown question_type '{qt}'; expected one of: {}",
                QUESTION_TYPES.join(", ")
            );
        }
    }

    let mut out: Vec<LongMemEvalInstance> = Vec::new();
    for v in &data {
        let question_id = v
            .get("question_id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        // Skip abstention questions: they have no evidence to retrieve.
        if question_id.ends_with("_abs") {
            continue;
        }

        let question_type = v
            .get("question_type")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        if let Some(want) = type_filter {
            if question_type != want {
                continue;
            }
        }

        let question = v
            .get("question")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let answer = match v.get("answer") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => String::new(),
        };

        // haystack_sessions: a list of sessions; each session is a list of
        // turns {"role": "user"/"assistant", "content": "...", optionally
        // "has_answer": true}.
        let mut sessions: Vec<Vec<(String, String, bool)>> = Vec::new();
        if let Some(arr) = v.get("haystack_sessions").and_then(|x| x.as_array()) {
            for session in arr {
                let mut turns: Vec<(String, String, bool)> = Vec::new();
                if let Some(turns_arr) = session.as_array() {
                    for turn in turns_arr {
                        let role = turn
                            .get("role")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let content = turn
                            .get("content")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let has_answer = turn
                            .get("has_answer")
                            .and_then(|x| x.as_bool())
                            .unwrap_or(false);
                        turns.push((role, content, has_answer));
                    }
                }
                sessions.push(turns);
            }
        }

        // answer_session_ids: session ids (as strings in the JSON) that
        // contain evidence. We map them to indices into `sessions` by parsing
        // trailing integers — LongMemEval session ids look like "session_42".
        let mut answer_session_idxs: Vec<usize> = Vec::new();
        if let Some(arr) = v.get("answer_session_ids").and_then(|x| x.as_array()) {
            for sid in arr {
                if let Some(s) = sid.as_str() {
                    if let Some(idx) = parse_session_idx(s, sessions.len()) {
                        answer_session_idxs.push(idx);
                    }
                }
            }
        }

        out.push(LongMemEvalInstance {
            question_id,
            question_type,
            question,
            answer,
            sessions,
            answer_session_idxs,
        });

        if let Some(limit) = limit {
            if out.len() >= limit {
                break;
            }
        }
    }

    Ok(out)
}

/// Parse a LongMemEval session id like "session_42" into an index. Returns
/// None if the id doesn't parse or is out of range — those are silently
/// skipped (the `has_answer` per-turn label is the primary evidence source).
fn parse_session_idx(sid: &str, n_sessions: usize) -> Option<usize> {
    let n = sid.rsplit('_').next()?.parse::<usize>().ok()?;
    (n < n_sessions).then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        json!([{
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "What did the user decide to cook after talking to Alex?",
            "answer": "pasta",
            "haystack_session_ids": ["session_0", "session_1"],
            "haystack_dates": ["2024-01-01", "2024-01-08"],
            "haystack_sessions": [
                [{"role": "user", "content": "I talked to Alex today."},
                 {"role": "assistant", "content": "What did Alex say?", "has_answer": true}],
                [{"role": "user", "content": "I decided to cook pasta.", "has_answer": true}]
            ],
            "answer_session_ids": ["session_1"]
        }, {
            "question_id": "q2_abs",
            "question_type": "multi-session",
            "question": "Did the user ever mention Brazil?",
            "answer": "No, the user never mentioned Brazil.",
            "haystack_sessions": [[{"role": "user", "content": "I went to Peru."}]],
            "answer_session_ids": []
        }, {
            "question_id": "q3",
            "question_type": "single-session-user",
            "question": "What is the user's favorite color?",
            "answer": "blue",
            "haystack_sessions": [
                [{"role": "user", "content": "My favorite color is blue.", "has_answer": true}]
            ],
            "answer_session_ids": ["session_0"]
        }])
    }

    fn write_fixture() -> tempfile_path::TempFile {
        tempfile_path::TempFile::new(&fixture())
    }

    // Same minimal temp-file helper style as locomo.rs, but with an atomic
    // nonce (per llm_graph.rs) so parallel tests never collide on the same
    // nanosecond timestamp.
    mod tempfile_path {
        use super::*;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NONCE: AtomicU64 = AtomicU64::new(0);

        pub struct TempFile {
            pub path: PathBuf,
        }

        impl TempFile {
            pub fn new(value: &Value) -> Self {
                let path = std::env::temp_dir().join(format!(
                    "skinki_longmemeval_test_{}_{}.json",
                    std::process::id(),
                    NONCE.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
                TempFile { path }
            }
        }

        impl Drop for TempFile {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }

    #[test]
    fn loads_instances_and_skips_abstention() {
        let f = write_fixture();
        let instances = load_longmemeval(&f.path, None, None).unwrap();
        // q2_abs is skipped -> 2 instances.
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].question_id, "q1");
        assert_eq!(instances[1].question_id, "q3");
    }

    #[test]
    fn filters_by_question_type() {
        let f = write_fixture();
        let multi = load_longmemeval(&f.path, Some("multi-session"), None).unwrap();
        assert_eq!(multi.len(), 1);
        assert_eq!(multi[0].question_type, "multi-session");

        let single = load_longmemeval(&f.path, Some("single-session-user"), None).unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].question_id, "q3");
    }

    #[test]
    fn rejects_unknown_question_type() {
        let f = write_fixture();
        assert!(load_longmemeval(&f.path, Some("bogus"), None).is_err());
    }

    #[test]
    fn limit_caps_instance_count() {
        let f = write_fixture();
        let limited = load_longmemeval(&f.path, None, Some(1)).unwrap();
        assert_eq!(limited.len(), 1);
    }

    #[test]
    fn to_corpus_builds_turns_and_relevant_entries() {
        let f = write_fixture();
        let instances = load_longmemeval(&f.path, Some("multi-session"), None).unwrap();
        let inst = &instances[0];
        let corpus = inst.to_corpus();

        // 2 sessions × (2 + 1) turns = 3 turns total.
        assert_eq!(corpus.entries.len(), 3);
        assert_eq!(corpus.entries[0].text, "user: I talked to Alex today.");
        assert_eq!(corpus.entries[1].text, "assistant: What did Alex say?");
        assert_eq!(corpus.entries[2].text, "user: I decided to cook pasta.");
        // day = session index.
        assert_eq!(corpus.entries[0].day, 0);
        assert_eq!(corpus.entries[2].day, 1);

        // relevant_entries = turns with has_answer: true (entries 1 and 2).
        let q = &corpus.ground_truth.recall[0];
        assert_eq!(q.question, inst.question);
        assert_eq!(q.answer, "pasta");
        assert_eq!(q.relevant_entries, vec![1, 2]);
    }

    #[test]
    fn to_corpus_falls_back_to_answer_session_ids_when_no_has_answer() {
        // If no turn has has_answer: true, relevant_entries = all turns in the
        // answer_session_ids sessions.
        use std::sync::atomic::{AtomicU64, Ordering};
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let v = json!([{
            "question_id": "q",
            "question_type": "multi-session",
            "question": "?",
            "answer": "x",
            "haystack_sessions": [
                [{"role": "user", "content": "filler"}],
                [{"role": "user", "content": "evidence"}]
            ],
            "answer_session_ids": ["session_1"]
        }]);
        let path = std::env::temp_dir().join(format!(
            "skinki_longmemeval_fallback_{}_{}.json",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
        let instances = load_longmemeval(&path, None, None).unwrap();
        let _ = std::fs::remove_file(&path);
        let corpus = instances[0].to_corpus();
        // entry 1 is in session_1 (the answer session) -> fallback relevant.
        assert_eq!(corpus.ground_truth.recall[0].relevant_entries, vec![1]);
    }

    #[test]
    fn parse_session_idx_handles_longmemeval_format() {
        assert_eq!(parse_session_idx("session_0", 5), Some(0));
        assert_eq!(parse_session_idx("session_4", 5), Some(4));
        assert_eq!(parse_session_idx("session_5", 5), None); // out of range
        assert_eq!(parse_session_idx("bogus", 5), None);
    }
}
