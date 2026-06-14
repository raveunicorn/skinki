#![forbid(unsafe_code)]
//! A classic BM25 lexical retriever.
//!
//! This is intentionally "dumb": no embeddings, no graph, no synthesis. It is
//! the yardstick. We expect it to do reasonably on single-entry recall (the
//! answer shares words with the question) and to *fail* on multi-hop and
//! insight discovery — which is exactly how we know the harness measures the
//! right things. Every future engine (Stages 1-5) must beat this.

use kortex_corpus::{Corpus, EntryId};
use kortex_eval::RetrievalSystem;
use std::collections::HashMap;

const K1: f64 = 1.2;
const B: f64 = 0.75;

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_lowercase())
        .collect()
}

#[derive(Default)]
pub struct Bm25 {
    /// term -> list of (entry id, term frequency in that entry)
    postings: HashMap<String, Vec<(EntryId, u32)>>,
    doc_len: Vec<u32>,
    avgdl: f64,
    num_docs: usize,
}

impl Bm25 {
    pub fn new() -> Self {
        Bm25::default()
    }

    fn idf(&self, df: usize) -> f64 {
        let n = self.num_docs as f64;
        (((n - df as f64 + 0.5) / (df as f64 + 0.5)) + 1.0).ln()
    }
}

impl RetrievalSystem for Bm25 {
    fn name(&self) -> &str {
        "bm25-lexical"
    }

    fn index(&mut self, corpus: &Corpus) {
        self.num_docs = corpus.entries.len();
        self.doc_len = vec![0; self.num_docs];
        self.postings.clear();

        let mut total_len: u64 = 0;
        for entry in &corpus.entries {
            let tokens = tokenize(&entry.text);
            self.doc_len[entry.id as usize] = tokens.len() as u32;
            total_len += tokens.len() as u64;

            let mut tf: HashMap<&str, u32> = HashMap::new();
            for tok in &tokens {
                *tf.entry(tok.as_str()).or_insert(0) += 1;
            }
            for (tok, count) in tf {
                self.postings
                    .entry(tok.to_string())
                    .or_default()
                    .push((entry.id, count));
            }
        }
        self.avgdl = if self.num_docs == 0 {
            0.0
        } else {
            total_len as f64 / self.num_docs as f64
        };
    }

    fn search(&self, query: &str, k: usize) -> Vec<EntryId> {
        let mut scores: HashMap<EntryId, f64> = HashMap::new();
        for term in tokenize(query) {
            let Some(postings) = self.postings.get(&term) else {
                continue;
            };
            let idf = self.idf(postings.len());
            for &(id, tf) in postings {
                let dl = self.doc_len[id as usize] as f64;
                let denom = tf as f64 + K1 * (1.0 - B + B * dl / self.avgdl.max(1.0));
                let contribution = idf * (tf as f64 * (K1 + 1.0)) / denom;
                *scores.entry(id).or_insert(0.0) += contribution;
            }
        }
        let mut ranked: Vec<(EntryId, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        ranked.into_iter().take(k).map(|(id, _)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kortex_corpus::{generate, GenConfig};

    fn recall_hit_rate(difficulty: kortex_corpus::Difficulty) -> f64 {
        let corpus = generate(&GenConfig {
            seed: 11,
            years: 3,
            entries_per_day: 2,
            difficulty,
        });
        let mut bm25 = Bm25::new();
        bm25.index(&corpus);
        let mut found = 0;
        for q in &corpus.ground_truth.recall {
            let res = bm25.search(&q.question, 10);
            if res.iter().any(|id| q.relevant_entries.contains(id)) {
                found += 1;
            }
        }
        found as f64 / corpus.ground_truth.recall.len() as f64
    }

    /// The yardstick contract, both directions: the legacy corpus (V1) is
    /// lexically saturated — BM25 solves recall — while the hardened corpus
    /// (V2: paraphrases + distractors) must NOT be solvable by lexical overlap
    /// alone. If V2 creeps back toward 1.0, the hardening has regressed and
    /// the benchmark stops discriminating semantic systems from grep.
    #[test]
    fn v1_recall_is_lexically_saturated_v2_is_not() {
        let v1 = recall_hit_rate(kortex_corpus::Difficulty::V1);
        let v2 = recall_hit_rate(kortex_corpus::Difficulty::V2);
        assert!(v1 > 0.9, "V1 should be lexically easy, got {v1:.2}");
        assert!(v2 < 0.5, "V2 must not saturate lexically, got {v2:.2}");
        assert!(v2 < v1, "hardening must strictly reduce BM25 recall");
    }

    #[test]
    fn tokenizer_drops_punctuation() {
        let toks = tokenize("Hello, world! It's RUST.");
        assert!(toks.contains(&"hello".to_string()));
        assert!(toks.contains(&"world".to_string()));
        assert!(toks.contains(&"rust".to_string()));
    }
}
