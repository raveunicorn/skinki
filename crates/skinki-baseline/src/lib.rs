#![forbid(unsafe_code)]
//! A classic BM25 lexical retriever.
//!
//! This is intentionally "dumb": no embeddings, no graph, no synthesis. It is
//! the yardstick. We expect it to do reasonably on single-entry recall (the
//! answer shares words with the question) and to *fail* on multi-hop and
//! insight discovery — which is exactly how we know the harness measures the
//! right things. Every future engine (Stages 1-5) must beat this.

use skinki_corpus::{Corpus, EntryId};
use skinki_eval::RetrievalSystem;
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

use skinki_vector::dot;
use skinki_vector::embed::{Embedder, StaticHashEmbedder};

/// How the production semantic retriever embeds text. `Hash` is the legacy
/// deterministic hash-of-tokens embedder (zero deps, byte-reproducible);
/// `Static` loads a `SKEMB001` artifact (Stage 1B's distilled token→vector
/// table); `Encoder` loads a `SKENC001` artifact (Stage 1C-B / 1D) and runs
/// the pure-Rust forward — bge-small (WordPiece / CLS) or multilingual-e5-small
/// (Unigram / mean). The query/passage prefixes are read from the artifact,
/// never hardcoded: the encoder's `embed` / `embed_query` apply them.
/// `Hash` remains the default so existing benchmarks are not silently
/// perturbed until D1 freezes the static bars.
#[derive(Debug, Clone)]
pub enum EmbedderSpec {
    Hash,
    /// `static:<path>` — load a `SKEMB001` artifact at `path`.
    Static {
        path: std::path::PathBuf,
    },
    /// `encoder:<path>` — load a `SKENC001` v2 artifact (Stage 1C-B / 1D) at
    /// `path`. Query/passage prefixes come from the artifact header, so the
    /// CLI surface stays a bare path (no query-string).
    Encoder {
        path: std::path::PathBuf,
    },
}

impl EmbedderSpec {
    /// Parse the `--embedder` flag value: `hash`, `static:<path>` or
    /// `encoder:<path>`. Anything else is a loud error — a typo silently
    /// falling back to `hash` would mislabel a benchmark column, which is
    /// worse than failing.
    pub fn parse(flag: &str) -> Result<Self, String> {
        if flag == "hash" {
            return Ok(EmbedderSpec::Hash);
        }
        if let Some(rest) = flag.strip_prefix("static:") {
            if rest.is_empty() {
                return Err("--embedder static: needs a path (static:<path>)".to_string());
            }
            return Ok(EmbedderSpec::Static {
                path: std::path::PathBuf::from(rest),
            });
        }
        if let Some(rest) = flag.strip_prefix("encoder:") {
            if rest.is_empty() {
                return Err("--embedder encoder: needs a path (encoder:<path>)".to_string());
            }
            return Ok(EmbedderSpec::Encoder {
                path: std::path::PathBuf::from(rest),
            });
        }
        Err(format!(
            "invalid --embedder '{flag}': expected 'hash', 'static:<path>' or 'encoder:<path>'"
        ))
    }

    /// Construct the embedder described by this spec. `Hash` costs nothing;
    /// `Static` and `Encoder` mmap their artifacts (fallible).
    pub fn build(&self) -> std::io::Result<Box<dyn Embedder>> {
        match self {
            EmbedderSpec::Hash => Ok(Box::new(StaticHashEmbedder::new(256))),
            EmbedderSpec::Static { path } => Ok(Box::new(
                skinki_vector::static_embed::StaticEmbedder::load(path)?,
            )),
            EmbedderSpec::Encoder { path } => {
                Ok(Box::new(skinki_encoder::RustEncoder::load(path)?))
            }
        }
    }
}

/// A cosine-similarity nearest-neighbor retriever over a fixed set of
/// per-entry embeddings, generic-free over a boxed [`Embedder`]. Vectors
/// produced by [`StaticHashEmbedder`] are L2-normalized so cosine == dot; the
/// `SKEMB001` static path also yields unit-norm rows. This is the single
/// production semantic retriever shared by the harness and `skinki-mcp`
/// (Stage 1B T3 consolidation — previously duplicated in three crates).
pub struct SemanticRetriever {
    embedder: Box<dyn Embedder>,
    vectors: Vec<Vec<f32>>,
    ids: Vec<EntryId>,
    name: String,
}

impl SemanticRetriever {
    /// Build from an already-constructed embedder with a custom column name.
    pub fn new(embedder: Box<dyn Embedder>, name: &str) -> Self {
        SemanticRetriever {
            embedder,
            vectors: Vec::new(),
            ids: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Build from an [`EmbedderSpec`] (`hash` or `static:<path>`) with a
    /// column name; mmaps the artifact if `Static`. Convenience constructor
    /// for the harness / MCP paths.
    pub fn from_spec(spec: &EmbedderSpec, name: &str) -> std::io::Result<Self> {
        Ok(SemanticRetriever::new(spec.build()?, name))
    }

    /// The embedder's dimensionality (constant for an instance).
    pub fn dim(&self) -> usize {
        self.embedder.dim()
    }
}

impl RetrievalSystem for SemanticRetriever {
    fn name(&self) -> &str {
        &self.name
    }

    fn index(&mut self, corpus: &Corpus) {
        self.vectors.clear();
        self.ids.clear();
        self.vectors.reserve(corpus.entries.len());
        self.ids.reserve(corpus.entries.len());
        for e in &corpus.entries {
            self.vectors.push(self.embedder.embed(&e.text));
            self.ids.push(e.id);
        }
    }

    fn search(&self, query: &str, k: usize) -> Vec<EntryId> {
        // Search-time path: `embed_query`, so an asymmetric embedder (e5 query
        // prefix) applies here while `index` keeps the passage-side `embed`.
        // For symmetric embedders the trait's default makes this identical to
        // `embed`, so no behavior change for `Hash` / `Static`.
        let qv = self.embedder.embed_query(query);
        let mut scored: Vec<(f32, EntryId)> = self
            .vectors
            .iter()
            .zip(self.ids.iter())
            .map(|(v, &id)| (dot(&qv, v), id))
            .collect();
        // Sort by score descending, tie-break by ascending id for determinism.
        scored.sort_by(|a, b| match b.0.partial_cmp(&a.0) {
            Some(std::cmp::Ordering::Equal) | None => a.1.cmp(&b.1),
            Some(ord) => ord,
        });
        scored.truncate(k);
        scored.into_iter().map(|(_, id)| id).collect()
    }
}

/// The standard reciprocal-rank-fusion constant (Cormack et al. 2009). Large
/// enough that a few ranks' difference in one list cannot drown the other
/// list's signal; small enough that top ranks still dominate.
pub const RRF_K: f64 = 60.0;

/// Stage 1B T8: deterministic reciprocal-rank fusion over two
/// **already-indexed** retrievers — the cheap architecture change *around*
/// the embedder. BM25 and the static semantic retriever have disjoint
/// failure modes (exact-term misses vs common-word crowding); RRF fuses
/// rankings without needing comparable scores:
/// `score(d) = Σ_i 1/(RRF_K + rank_i(d))` over each system's top-`depth`.
///
/// Borrows its components (they are indexed and owned by the caller), so
/// [`RetrievalSystem::index`] is a documented no-op here — constructing a
/// fusion over un-indexed retrievers is a caller bug and will simply search
/// empty indexes.
pub struct RrfFusion<'a> {
    a: &'a dyn RetrievalSystem,
    b: &'a dyn RetrievalSystem,
    depth: usize,
    name: String,
}

impl<'a> RrfFusion<'a> {
    /// `depth` is how deep each component list goes before fusing (T8 uses
    /// 100: deep enough to rescue docs one system ranks poorly, shallow
    /// enough to stay O(depth log depth) per query).
    pub fn new(a: &'a dyn RetrievalSystem, b: &'a dyn RetrievalSystem, depth: usize) -> Self {
        RrfFusion {
            a,
            b,
            depth,
            name: format!("rrf({}+{})", a.name(), b.name()),
        }
    }
}

impl RetrievalSystem for RrfFusion<'_> {
    fn name(&self) -> &str {
        &self.name
    }

    /// No-op: the fused components are indexed by their owners (see the
    /// struct docs).
    fn index(&mut self, _corpus: &Corpus) {}

    fn search(&self, query: &str, k: usize) -> Vec<EntryId> {
        // BTreeMap: deterministic iteration (rule 2). Each id accumulates at
        // most two additions, in the fixed order a-then-b.
        let mut score: std::collections::BTreeMap<EntryId, f64> = std::collections::BTreeMap::new();
        for list in [
            self.a.search(query, self.depth),
            self.b.search(query, self.depth),
        ] {
            for (rank, id) in list.into_iter().enumerate() {
                *score.entry(id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
            }
        }
        let mut scored: Vec<(f64, EntryId)> = score.into_iter().map(|(id, s)| (s, id)).collect();
        // Sort by fused score descending, tie-break by ascending id.
        scored.sort_by(|x, y| match y.0.partial_cmp(&x.0) {
            Some(std::cmp::Ordering::Equal) | None => x.1.cmp(&y.1),
            Some(ord) => ord,
        });
        scored.truncate(k);
        scored.into_iter().map(|(_, id)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skinki_corpus::{generate, GenConfig};

    /// T8: reciprocal-rank fusion — pure ranking arithmetic, tested against
    /// fixed fake rankings so every property is exact, not statistical.
    mod rrf_fusion {
        use super::*;

        /// A fake retriever serving a fixed ranking (best first).
        struct FixedList {
            ids: Vec<EntryId>,
            name: &'static str,
        }
        impl RetrievalSystem for FixedList {
            fn name(&self) -> &str {
                self.name
            }
            fn index(&mut self, _corpus: &Corpus) {}
            fn search(&self, _query: &str, k: usize) -> Vec<EntryId> {
                self.ids.iter().copied().take(k).collect()
            }
        }

        #[test]
        fn doc_in_both_lists_beats_single_list_docs() {
            // 30 is rank 2 in both lists; 10/20 are rank 1 in only one.
            // 2/(60+2) > 1/(60+1), so consensus wins.
            let a = FixedList {
                ids: vec![10, 30],
                name: "a",
            };
            let b = FixedList {
                ids: vec![20, 30],
                name: "b",
            };
            let f = RrfFusion::new(&a, &b, 100);
            assert_eq!(f.search("q", 3), vec![30, 10, 20]);
        }

        #[test]
        fn equal_scores_tie_break_by_ascending_id() {
            // 10 and 20 each appear at rank 1 of exactly one list: same score.
            let a = FixedList {
                ids: vec![20],
                name: "a",
            };
            let b = FixedList {
                ids: vec![10],
                name: "b",
            };
            let f = RrfFusion::new(&a, &b, 100);
            assert_eq!(f.search("q", 2), vec![10, 20]);
        }

        #[test]
        fn depth_caps_each_component_list() {
            // With depth 1 only the top-1 of each list is fused; 30 (rank 2
            // in both) disappears entirely.
            let a = FixedList {
                ids: vec![10, 30],
                name: "a",
            };
            let b = FixedList {
                ids: vec![20, 30],
                name: "b",
            };
            let f = RrfFusion::new(&a, &b, 1);
            assert_eq!(f.search("q", 10), vec![10, 20]);
        }

        #[test]
        fn deterministic_and_named() {
            let a = FixedList {
                ids: vec![1, 2, 3],
                name: "bm25",
            };
            let b = FixedList {
                ids: vec![3, 2, 9],
                name: "semantic-static",
            };
            let f = RrfFusion::new(&a, &b, 100);
            assert_eq!(f.name(), "rrf(bm25+semantic-static)");
            assert_eq!(f.search("q", 10), f.search("q", 10));
        }

        #[test]
        fn k_truncates_fused_output() {
            let a = FixedList {
                ids: vec![1, 2, 3, 4],
                name: "a",
            };
            let b = FixedList {
                ids: vec![5, 6, 7, 8],
                name: "b",
            };
            let f = RrfFusion::new(&a, &b, 100);
            assert_eq!(f.search("q", 3).len(), 3);
        }
    }

    /// Mark: tests for the new T3 embedder-selection seam live alongside the
    /// baseline yardstick so a regression here is caught by `cargo test -p
    /// skinki-baseline` (the same CI step that gates BM25).
    mod embedder_spec {
        use super::*;

        #[test]
        fn parse_hash() {
            assert!(matches!(
                EmbedderSpec::parse("hash"),
                Ok(EmbedderSpec::Hash)
            ));
        }

        #[test]
        fn parse_static_path() {
            match EmbedderSpec::parse("static:/tmp/model.skemb") {
                Ok(EmbedderSpec::Static { path }) => {
                    assert_eq!(path, std::path::PathBuf::from("/tmp/model.skemb"));
                }
                other => panic!("expected Static, got {other:?}"),
            }
        }

        #[test]
        fn parse_encoder_path() {
            match EmbedderSpec::parse("encoder:/tmp/model.skenc") {
                Ok(EmbedderSpec::Encoder { path }) => {
                    assert_eq!(path, std::path::PathBuf::from("/tmp/model.skenc"));
                }
                other => panic!("expected Encoder, got {other:?}"),
            }
        }

        /// A typo must be a loud error, never a silent fall-back to `hash` —
        /// a mislabeled benchmark column is worse than a failed run.
        #[test]
        fn parse_rejects_typos_and_empty() {
            assert!(EmbedderSpec::parse("").is_err());
            assert!(EmbedderSpec::parse("nope").is_err());
            assert!(EmbedderSpec::parse("statik:/tmp/m.skemb").is_err());
            assert!(EmbedderSpec::parse("Hash").is_err());
            // `static:` / `encoder:` with no path are errors too, not the hash default.
            assert!(EmbedderSpec::parse("static:").is_err());
            assert!(EmbedderSpec::parse("encoder:").is_err());
        }

        #[test]
        fn build_encoder_missing_file_errors() {
            let spec = EmbedderSpec::Encoder {
                path: std::path::PathBuf::from("/nonexistent/skinki-encoder-missing.skenc"),
            };
            assert!(
                spec.build().is_err(),
                "missing artifact must error, not panic"
            );
        }

        #[test]
        fn build_hash_yields_working_embedder() {
            let e = EmbedderSpec::Hash.build().expect("hash never fails");
            let v = e.embed("hello world");
            assert_eq!(v.len(), 256);
            assert_eq!(e.dim(), 256);
        }

        #[test]
        fn build_static_missing_file_errors() {
            let spec = EmbedderSpec::Static {
                path: std::path::PathBuf::from("/nonexistent/skinki-static-missing.skemb"),
            };
            assert!(
                spec.build().is_err(),
                "missing artifact must error, not panic"
            );
        }

        #[test]
        fn from_spec_hash_indexes_and_searches() {
            let corpus = generate(&GenConfig {
                seed: 7,
                years: 1,
                entries_per_day: 1,
                difficulty: skinki_corpus::Difficulty::V1,
            });
            let mut r = SemanticRetriever::from_spec(&EmbedderSpec::Hash, "semantic-static")
                .expect("hash build");
            assert_eq!(r.name(), "semantic-static");
            r.index(&corpus);
            let hits = r.search("project", 5);
            // Hits must be valid ids and deduplicated (a cosine retriever can
            // never return the same id twice).
            let mut seen = std::collections::BTreeSet::new();
            for &id in &hits {
                assert!(seen.insert(id), "duplicate id {id} in results");
            }
        }

        /// `search` routes through `embed_query` (Stage 1D T1): a fake embedder
        /// that records which side its query method was called on proves an
        /// asymmetric embedder (e5 query prefix) is honored at search time
        /// without changing the index path.
        #[test]
        fn search_uses_embed_query_not_embed() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static EMBED: AtomicUsize = AtomicUsize::new(0);
            static QUERY: AtomicUsize = AtomicUsize::new(0);
            struct Spy;
            impl Embedder for Spy {
                fn embed(&self, _text: &str) -> Vec<f32> {
                    EMBED.fetch_add(1, Ordering::SeqCst);
                    vec![1.0, 0.0]
                }
                fn dim(&self) -> usize {
                    2
                }
                fn embed_query(&self, _text: &str) -> Vec<f32> {
                    QUERY.fetch_add(1, Ordering::SeqCst);
                    vec![1.0, 0.0]
                }
            }
            EMBED.store(0, Ordering::SeqCst);
            QUERY.store(0, Ordering::SeqCst);
            let corpus = generate(&GenConfig {
                seed: 1,
                years: 1,
                entries_per_day: 1,
                difficulty: skinki_corpus::Difficulty::V1,
            });
            let mut r = SemanticRetriever::new(Box::new(Spy), "spy");
            r.index(&corpus);
            // Index used `embed` once per entry, never `embed_query`.
            assert_eq!(
                QUERY.load(Ordering::SeqCst),
                0,
                "index must use embed (passage), not embed_query"
            );
            assert!(EMBED.load(Ordering::SeqCst) > 0);
            let pre = EMBED.load(Ordering::SeqCst);
            let _ = r.search("anything", 3);
            assert_eq!(
                QUERY.load(Ordering::SeqCst),
                1,
                "search must use embed_query (the asymmetric query path)"
            );
            assert_eq!(
                EMBED.load(Ordering::SeqCst),
                pre,
                "search must not re-call embed"
            );
        }

        /// The same string embeds to identical bytes (rule-2 determinism),
        /// via the boxed trait object — this is the property that lets a
        /// boxed `dyn Embedder` replace the old generic code path without
        /// changing any downstream golden.
        #[test]
        fn hash_embedder_is_pure_through_box() {
            let e = EmbedderSpec::Hash.build().unwrap();
            let a = e.embed("skinki static embedder");
            let b = e.embed("skinki static embedder");
            assert_eq!(a, b, "embed must be pure through Box<dyn Embedder>");
        }

        /// Cosine-self == 1 for any non-empty text (the normalization
        /// contract every consumer of `SemanticRetriever` relies on).
        #[test]
        fn hash_embeddings_are_unit_norm() {
            let e = EmbedderSpec::Hash.build().unwrap();
            let v = e.embed("the quick brown fox");
            let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-4,
                "expected unit-norm embedding, got magnitude {mag}"
            );
        }
    }

    fn recall_hit_rate(difficulty: skinki_corpus::Difficulty) -> f64 {
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
        let v1 = recall_hit_rate(skinki_corpus::Difficulty::V1);
        let v2 = recall_hit_rate(skinki_corpus::Difficulty::V2);
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
