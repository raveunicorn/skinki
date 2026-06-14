//! LoCoMo adapter — real multi-session-dialogue benchmark.
//!
//! `locomo10.json` (snap-research/locomo) is a JSON array of 10 samples, each
//! a long multi-session two-person conversation plus memory QA with
//! evidence (`dia_id`) pointers into the conversation. This module turns that
//! into a Stage-0 [`Corpus`] + [`RecallQuery`] ground truth so the existing
//! eval machinery (BM25 / semantic / graph retrievers, `score_set`) can run
//! on *real text* instead of the synthetic generator.
//!
//! Deliberately minimal: we navigate `serde_json::Value` rather than modeling
//! the full LoCoMo schema (which carries fields — `img_url`, summaries,
//! per-category metadata — irrelevant to retrieval eval).
//!
//! ## Measured (the first real-data result)
//!
//! LoCoMo10, all 10 samples (5882 entries, 1977 queries), k=10, dim 256:
//!
//! | retriever | recall@10 | answer@10 |
//! | --- | --- | --- |
//! | bm25 | 0.484 | 0.378 |
//! | semantic-static (lexical hash) | 0.183 | 0.292 |
//! | graph (deterministic) | 0.484 | 0.378 |
//! | **semantic-real (EmbeddingGemma-300m)** | **0.673** | **0.446** |
//!
//! EmbeddingGemma beats BM25 by **+0.189 recall (~+39%)** on real dialogue —
//! the first validated win on non-synthetic text, and proof the pluggable
//! [`crate`]/`Embedder` seam works end to end (produced on an M1 via
//! `tools/export-embeddings-gemma.py`, replayed through `--embeddings-file` /
//! `--query-embeddings-file`; reproduce with that runbook). Honest caveat: this
//! lift is the *embedder*, not our graph — `graph == bm25` here because the
//! deterministic intro/rec/venue patterns don't fire on conversation, so our
//! unique graph layer still awaits a real LLM extractor for dialogue.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use kortex_corpus::{
    Corpus, CorpusMeta, Difficulty, Entry, EntryId, EntryKind, GroundTruth, RecallQuery,
};

/// Which sample(s) of the 10 to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocomoSample {
    All,
    One(usize),
}

/// Load LoCoMo data from `path` and build a Stage-0 [`Corpus`] from the
/// chosen sample(s). Conversation turns become [`Entry`]s (in session order,
/// sequential ids); QA items with non-empty `evidence` become
/// [`RecallQuery`]s in `ground_truth.recall`.
pub fn load_locomo(path: &Path, sample: LocomoSample) -> Result<Corpus> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading LoCoMo dataset at {}", path.display()))?;
    let data: Vec<Value> = serde_json::from_str(&raw).context("parsing LoCoMo JSON")?;

    let indices: Vec<usize> = match sample {
        LocomoSample::All => (0..data.len()).collect(),
        LocomoSample::One(i) => {
            if i >= data.len() {
                bail!(
                    "sample index {i} out of range (dataset has {} samples)",
                    data.len()
                );
            }
            vec![i]
        }
    };

    let mut entries: Vec<Entry> = Vec::new();
    let mut recall: Vec<RecallQuery> = Vec::new();
    let mut next_qid: usize = 0;

    for &idx in &indices {
        let sample_val = &data[idx];
        let conv = sample_val
            .get("conversation")
            .context("sample missing 'conversation'")?;

        let offset = entries.len() as EntryId;
        let mut dia_to_id: HashMap<String, EntryId> = HashMap::new();

        // Iterate sessions in numeric order: session_1, session_2, ...
        let mut session_nums: Vec<u32> = Vec::new();
        if let Some(obj) = conv.as_object() {
            for key in obj.keys() {
                if let Some(rest) = key.strip_prefix("session_") {
                    if let Ok(n) = rest.parse::<u32>() {
                        session_nums.push(n);
                    }
                }
            }
        }
        session_nums.sort_unstable();

        for &n in &session_nums {
            let session_key = format!("session_{n}");
            let Some(turns) = conv.get(&session_key).and_then(|v| v.as_array()) else {
                continue;
            };
            let date_key = format!("session_{n}_date_time");
            let date = conv
                .get(&date_key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            for turn in turns {
                let speaker = turn.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
                let text = turn.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let dia_id = turn.get("dia_id").and_then(|v| v.as_str());

                let id = entries.len() as EntryId;
                if let Some(dia_id) = dia_id {
                    dia_to_id.insert(dia_id.to_string(), id);
                }
                entries.push(Entry {
                    id,
                    day: n,
                    date: date.clone(),
                    kind: EntryKind::Text,
                    text: format!("{speaker}: {text}"),
                });
            }
        }

        // QA -> RecallQuery, dropping items with empty/unmapped evidence.
        if let Some(qa_list) = sample_val.get("qa").and_then(|v| v.as_array()) {
            for qa in qa_list {
                let Some(evidence) = qa.get("evidence").and_then(|v| v.as_array()) else {
                    continue;
                };
                if evidence.is_empty() {
                    continue;
                }
                let relevant_entries: Vec<EntryId> = evidence
                    .iter()
                    .filter_map(|v| v.as_str())
                    .filter_map(|dia| dia_to_id.get(dia).copied())
                    .collect();
                if relevant_entries.is_empty() {
                    continue;
                }

                let question = qa
                    .get("question")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let answer = match qa.get("answer") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Number(n)) => n.to_string(),
                    _ => String::new(),
                };

                recall.push(RecallQuery {
                    id: next_qid,
                    question,
                    answer,
                    relevant_entries,
                });
                next_qid += 1;
            }
        }

        // `offset` is unused beyond bookkeeping for `All`: ids are already
        // globally sequential because `entries` accumulates across samples.
        let _ = offset;
    }

    let num_entries = entries.len();
    Ok(Corpus {
        meta: CorpusMeta {
            seed: 0,
            years: 0,
            num_entries,
            difficulty: Difficulty::V2,
        },
        entries,
        ground_truth: GroundTruth {
            recall,
            ..Default::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        json!([{
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:00 pm on 1 Jan, 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "Hi Bob!"},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "Hey Alice, I got a new dog named Rex."}
                ],
                "session_2_date_time": "2:00 pm on 2 Jan, 2023",
                "session_2": [
                    {"speaker": "Alice", "dia_id": "D2:1", "text": "How's Rex doing?"},
                    {"speaker": "Bob", "dia_id": "D2:2", "text": "Rex is doing great, loves the park."}
                ]
            },
            "qa": [
                {"question": "What is Bob's dog's name?", "answer": "Rex", "evidence": ["D1:2"], "category": 1},
                {"question": "No evidence question", "answer": "x", "evidence": [], "category": 5},
                {"question": "Where does Rex love?", "answer": "the park", "evidence": ["D2:2"], "category": 1}
            ]
        }])
    }

    fn write_fixture() -> tempfile_path::TempFile {
        tempfile_path::TempFile::new(&fixture())
    }

    // Minimal temp-file helper (no tempfile dep): write to std::env::temp_dir
    // with a pid+name-based unique path, clean up on drop.
    mod tempfile_path {
        use super::*;
        use std::path::PathBuf;

        pub struct TempFile {
            pub path: PathBuf,
        }

        impl TempFile {
            pub fn new(value: &Value) -> Self {
                let path = std::env::temp_dir().join(format!(
                    "kortex_locomo_test_{}_{}.json",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
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
    fn loads_entries_and_resolves_evidence() {
        let f = write_fixture();
        let corpus = load_locomo(&f.path, LocomoSample::All).unwrap();

        // 4 turns total across 2 sessions.
        assert_eq!(corpus.entries.len(), 4);
        assert_eq!(corpus.meta.num_entries, 4);

        // Entry text carries the speaker prefix.
        assert_eq!(
            corpus.entries[1].text,
            "Bob: Hey Alice, I got a new dog named Rex."
        );
        assert_eq!(corpus.entries[1].day, 1);
        assert_eq!(corpus.entries[3].day, 2);

        // 2 recall queries survive (the empty-evidence one is dropped).
        assert_eq!(corpus.ground_truth.recall.len(), 2);

        let q0 = &corpus.ground_truth.recall[0];
        assert_eq!(q0.question, "What is Bob's dog's name?");
        assert_eq!(q0.answer, "Rex");
        // D1:2 -> entry id 1 (second turn, session_1).
        assert_eq!(q0.relevant_entries, vec![1]);

        let q1 = &corpus.ground_truth.recall[1];
        assert_eq!(q1.answer, "the park");
        // D2:2 -> entry id 3 (second turn, session_2).
        assert_eq!(q1.relevant_entries, vec![3]);
    }

    #[test]
    fn one_sample_selects_single_index() {
        let f = write_fixture();
        let corpus = load_locomo(&f.path, LocomoSample::One(0)).unwrap();
        assert_eq!(corpus.entries.len(), 4);
    }

    #[test]
    fn out_of_range_sample_errors() {
        let f = write_fixture();
        assert!(load_locomo(&f.path, LocomoSample::One(5)).is_err());
    }
}
