//! LLM-extracted entity graph over **real** text (the Stage-3 dialogue path).
//!
//! Where the synthetic corpus is read by hand-written intro/rec/venue patterns
//! ([`kortex_graph::RelationRetriever`]), real dialogue needs a model. This
//! retriever **rebuilds** a graph deterministically from the append-only
//! artifact log produced by `tools/extract-graph-llm.py` (one JSON object per
//! turn: `{"entry": i, "entities": [...], "facts": [...]}`) and ranks entries by
//! **entity co-mention** — a 1-hop walk from the query's entities through the
//! turns that share them — optionally fused with BM25 via RRF.
//!
//! This is the consume side of AGENTS.md rule 3: the LLM run (`produce`) is not
//! bit-reproducible, but `rebuild(log)` here is fully deterministic
//! (`BTreeMap`/sorted iteration, ascending-id tie-breaks).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::Context;
use kortex_baseline::Bm25;
use kortex_corpus::{Corpus, EntryId};
use kortex_eval::RetrievalSystem;

/// Entities mentioned in more than this many turns are hubs (speakers, "today",
/// generic nouns the model emits); expanding through them floods candidates, so
/// the hop step skips them. Mirrors `RelationRetriever`'s guard.
const MAX_BRIDGE_DF: usize = 64;
/// Weight of a hop relative to the seed entry's score.
const HOP_LAMBDA: f64 = 0.5;
/// Reciprocal Rank Fusion smoothing constant.
const RRF_C: f64 = 60.0;

/// Canonical entity key: trimmed + lowercased. (Coreference like "Mel" vs
/// "Melanie" is NOT merged — a documented v0 limitation; most names still
/// link.)
fn normalize(e: &str) -> String {
    e.trim().to_lowercase()
}

pub struct LlmGraphRetriever {
    /// normalized entity -> turns that mention it (ascending, deduped).
    entity_postings: BTreeMap<String, Vec<EntryId>>,
    /// turn -> the entities it mentions (normalized).
    entry_entities: BTreeMap<EntryId, Vec<String>>,
    /// sorted entity vocabulary, for matching entities inside a query string.
    vocab: Vec<String>,
    n_entries: usize,
    bm25: Bm25,
    fuse_bm25: bool,
}

impl LlmGraphRetriever {
    /// Rebuild from a JSON-lines artifact log. Each line's `entry` field indexes
    /// the dumped texts, which equals the `EntryId` in a single-sample corpus
    /// (`--sample <n>`); use the SAME sample for the dump, the extraction, and
    /// the eval so the ids line up.
    pub fn from_artifacts(path: &Path, fuse_bm25: bool) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading artifact log {}", path.display()))?;
        let mut entity_postings: BTreeMap<String, Vec<EntryId>> = BTreeMap::new();
        let mut entry_entities: BTreeMap<EntryId, Vec<String>> = BTreeMap::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_str(line).context("malformed artifact-log line")?;
            let entry = v
                .get("entry")
                .and_then(|x| x.as_u64())
                .context("artifact-log line missing integer `entry`")?
                as EntryId;

            let mut ents: BTreeSet<String> = BTreeSet::new();
            if let Some(arr) = v.get("entities").and_then(|x| x.as_array()) {
                for e in arr {
                    if let Some(s) = e.as_str() {
                        let n = normalize(s);
                        if n.len() >= 2 {
                            ents.insert(n);
                        }
                    }
                }
            }
            for e in &ents {
                entity_postings.entry(e.clone()).or_default().push(entry);
            }
            entry_entities.insert(entry, ents.into_iter().collect());
        }

        for post in entity_postings.values_mut() {
            post.sort_unstable();
            post.dedup();
        }
        let vocab: Vec<String> = entity_postings.keys().cloned().collect();

        Ok(LlmGraphRetriever {
            entity_postings,
            entry_entities,
            vocab,
            n_entries: 0,
            bm25: Bm25::new(),
            fuse_bm25,
        })
    }

    fn idf(&self, df: usize) -> f64 {
        ((self.n_entries.max(1) as f64) / (df.max(1) as f64)).ln() + 1.0
    }

    /// Entities from the vocabulary that appear (as a substring) in `query`.
    /// `>= 3` chars avoids matching tiny tokens that collide with everything.
    fn query_entities(&self, query_lower: &str) -> BTreeSet<String> {
        self.vocab
            .iter()
            .filter(|e| e.len() >= 3 && query_lower.contains(e.as_str()))
            .cloned()
            .collect()
    }

    /// Seed + 1-hop co-mention scores, sorted (score desc, id asc).
    fn graph_ranked(&self, qents: &BTreeSet<String>) -> Vec<(EntryId, f64)> {
        let mut score: BTreeMap<EntryId, f64> = BTreeMap::new();
        for qe in qents {
            if let Some(post) = self.entity_postings.get(qe) {
                let w = self.idf(post.len());
                for &e in post {
                    *score.entry(e).or_default() += w;
                }
            }
        }
        let seeds: Vec<(EntryId, f64)> = score.iter().map(|(e, s)| (*e, *s)).collect();
        for (s, sscore) in &seeds {
            let Some(ents) = self.entry_entities.get(s) else {
                continue;
            };
            for e2 in ents {
                if qents.contains(e2) {
                    continue;
                }
                let df = self.entity_postings.get(e2).map_or(0, |p| p.len());
                if df == 0 || df > MAX_BRIDGE_DF {
                    continue;
                }
                let w2 = self.idf(df);
                for &t in &self.entity_postings[e2] {
                    if t == *s {
                        continue;
                    }
                    *score.entry(t).or_default() += HOP_LAMBDA * w2 * sscore;
                }
            }
        }
        let mut ranked: Vec<(EntryId, f64)> = score.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        ranked
    }
}

impl RetrievalSystem for LlmGraphRetriever {
    fn name(&self) -> &str {
        if self.fuse_bm25 {
            "llm-graph+bm25"
        } else {
            "llm-graph"
        }
    }

    fn index(&mut self, corpus: &Corpus) {
        self.n_entries = corpus.entries.len();
        self.bm25 = Bm25::new();
        self.bm25.index(corpus);
    }

    fn search(&self, query: &str, k: usize) -> Vec<EntryId> {
        let ql = query.to_lowercase();
        let qents = self.query_entities(&ql);
        let graph_ranked = self.graph_ranked(&qents);

        if !self.fuse_bm25 {
            return graph_ranked.into_iter().take(k).map(|(id, _)| id).collect();
        }

        let bm = self.bm25.search(query, (k * 5).max(50));
        let mut fused: BTreeMap<EntryId, f64> = BTreeMap::new();
        for (r, (e, _)) in graph_ranked.iter().enumerate() {
            *fused.entry(*e).or_default() += 1.0 / (RRF_C + r as f64);
        }
        for (r, e) in bm.iter().enumerate() {
            *fused.entry(*e).or_default() += 1.0 / (RRF_C + r as f64);
        }
        let mut ranked: Vec<(EntryId, f64)> = fused.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        ranked.into_iter().take(k).map(|(id, _)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kortex_corpus::{CorpusMeta, Difficulty, Entry, EntryKind, GroundTruth};
    use std::io::Write;

    fn tiny_corpus(texts: &[&str]) -> Corpus {
        let entries = texts
            .iter()
            .enumerate()
            .map(|(i, t)| Entry {
                id: i as EntryId,
                day: 0,
                date: String::new(),
                kind: EntryKind::Text,
                text: t.to_string(),
            })
            .collect();
        Corpus {
            meta: CorpusMeta {
                seed: 0,
                years: 0,
                num_entries: texts.len(),
                difficulty: Difficulty::V2,
            },
            entries,
            ground_truth: GroundTruth::default(),
        }
    }

    fn write_log(lines: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("kortex_llmgraph_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("g.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        path
    }

    #[test]
    fn entity_walk_links_turns_sharing_an_entity() {
        // turn 0 mentions {Caroline, support group}; turn 2 mentions {Caroline};
        // a query naming "Caroline" should surface both via the entity index.
        let corpus = tiny_corpus(&[
            "Caroline: I went to the support group.",
            "Mel: nice weather today.",
            "Caroline: the group meets Tuesdays.",
        ]);
        let log = write_log(&[
            r#"{"entry":0,"entities":["Caroline","support group"],"facts":[]}"#,
            r#"{"entry":1,"entities":["Mel"],"facts":[]}"#,
            r#"{"entry":2,"entities":["Caroline"],"facts":[]}"#,
        ]);
        let mut r = LlmGraphRetriever::from_artifacts(&log, false).unwrap();
        r.index(&corpus);
        let got = r.search("When does Caroline's support group meet?", 3);
        let _ = std::fs::remove_dir_all(log.parent().unwrap());
        assert!(got.contains(&0), "should retrieve turn 0: {got:?}");
        assert!(
            got.contains(&2),
            "should retrieve turn 2 (shares Caroline): {got:?}"
        );
    }

    #[test]
    fn deterministic_rebuild() {
        let corpus = tiny_corpus(&["Caroline: hi", "Caroline: bye"]);
        let log = write_log(&[
            r#"{"entry":0,"entities":["Caroline"],"facts":[]}"#,
            r#"{"entry":1,"entities":["Caroline"],"facts":[]}"#,
        ]);
        let build = || {
            let mut r = LlmGraphRetriever::from_artifacts(&log, true).unwrap();
            r.index(&corpus);
            r.search("Caroline", 2)
        };
        assert_eq!(build(), build());
        let _ = std::fs::remove_dir_all(log.parent().unwrap());
    }
}
