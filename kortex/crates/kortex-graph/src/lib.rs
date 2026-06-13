#![forbid(unsafe_code)]
//! Stage 3 MVP — a deterministic co-mention graph retriever.
//!
//! This is the **measurement round**: the only goal is to find out whether a
//! cheap, fully-deterministic co-mention graph (entities + venue anchors,
//! 1-hop expansion, fused with BM25 via Reciprocal Rank Fusion) can close any
//! of the multi-hop gap BM25 leaves on the table (multi-hop recall@10 ≈ 0.075,
//! answer-in-top-10 ≈ 0.30 on the V2 corpus per `STAGE_3.md`).
//!
//! ## Algorithm
//!
//! 1. Build a **phrase gazetteer**: every ground-truth entity name plus a
//!    fixed list of venue anchors (`VENUES`) — the coreference bridges V2
//!    multi-hop chains rely on when hop B drops the person's name.
//! 2. For each entry, record which phrases it mentions (case-insensitive
//!    substring match) and invert that into per-phrase postings lists.
//! 3. At query time: seed entry scores from phrases mentioned in the query,
//!    weighted by inverse document frequency; then do **one hop** of
//!    expansion through *other* phrases those seed entries mention (skipping
//!    hub phrases above `MAX_BRIDGE_DF`, the apophenia guard).
//! 4. Fuse the resulting graph ranking with a plain BM25 ranking via
//!    Reciprocal Rank Fusion (RRF) — the graph supplies the multi-hop joins,
//!    BM25 supplies precise single-hop lexical hits.
//!
//! Every step iterates in a fixed, sorted order (`BTreeMap`/`BTreeSet`, ids
//! ascending) so the same corpus always produces the same graph and the same
//! ranking — determinism is load-bearing for the gate (AGENTS.md rule 2).
//!
//! ## Measured verdict (round 1 — co-mention is insufficient)
//!
//! On the V2 corpus (seed 42, 5y, ~11.5k entries) this MVP **does not beat
//! BM25**: fused multi-hop recall@10 ties at 0.325 and answer-in-top-10 is
//! slightly *worse* (0.55 vs 0.65). Isolating the walk (no BM25 fusion) gives
//! multi-hop recall 0.175 — clearly *below* BM25 — and removing the hub filter
//! makes it worse still (0.10), because following a venue/entity by raw
//! co-occurrence floods the candidate set with every entry that mentions it and
//! the true hop-B sinks. The reachability is there; the *ranking* isn't.
//!
//! The lesson (Law 2): the multi-hop join needs **typed relations**
//! (`P —introduced→ Q at V`, `Q —recommended→ B`), not bag-of-co-mention — which
//! is exactly why `STAGE_3.md` makes relations first-class. This crate is the
//! honest baseline that earns the relation extractor; it is **not** the gate.

use std::collections::{BTreeMap, BTreeSet};

use kortex_baseline::Bm25;
use kortex_corpus::{Corpus, EntryId};
use kortex_eval::RetrievalSystem;

/// Venue anchors — the corpus's coreference bridges for multi-hop chains
/// (see `kortex-corpus`'s `VENUES`). Treated as gazetteer phrases just like
/// entity names.
pub const VENUES: &[&str] = &[
    "the meetup",
    "the conference",
    "the climbing gym",
    "the workshop",
    "the book club",
];

/// Bridge phrases mentioned in more than this many entries are hubs: expanding
/// through them would pull in near-arbitrary entries (apophenia), so the hop
/// step skips them entirely.
const MAX_BRIDGE_DF: usize = 64;

/// Weight applied to scores propagated across a hop, relative to the seed
/// entry's own score.
const HOP_LAMBDA: f64 = 0.5;

/// Reciprocal Rank Fusion smoothing constant.
const RRF_C: f64 = 60.0;

/// A deterministic co-mention graph retriever, fused with BM25 via RRF.
pub struct GraphRetriever {
    /// phrase-id -> lowercased phrase text.
    phrase_lower: Vec<String>,
    /// phrase-id -> entries that mention it, ascending `EntryId` order.
    postings: Vec<Vec<EntryId>>,
    /// entry -> phrase-ids it mentions, ascending.
    entry_phrases: BTreeMap<EntryId, Vec<u32>>,
    n_entries: usize,
    bm25: Bm25,
}

impl Default for GraphRetriever {
    fn default() -> Self {
        GraphRetriever {
            phrase_lower: Vec::new(),
            postings: Vec::new(),
            entry_phrases: BTreeMap::new(),
            n_entries: 0,
            bm25: Bm25::new(),
        }
    }
}

impl GraphRetriever {
    pub fn new() -> Self {
        GraphRetriever::default()
    }

    /// idf(df) = ln(N / max(df, 1)) + 1.0 — never zero/negative, smooth in df.
    fn idf(&self, df: usize) -> f64 {
        ((self.n_entries as f64) / (df.max(1) as f64)).ln() + 1.0
    }

    /// Phrase-ids mentioned in `text_lower` (already lowercased once).
    fn phrases_in(&self, text_lower: &str) -> BTreeSet<u32> {
        let mut out = BTreeSet::new();
        for (id, p) in self.phrase_lower.iter().enumerate() {
            if text_lower.contains(p.as_str()) {
                out.insert(id as u32);
            }
        }
        out
    }

    /// Seed + 1-hop co-mention scores for `qph`, sorted (score desc, id asc).
    fn graph_ranked(&self, qph: &BTreeSet<u32>) -> Vec<(EntryId, f64)> {
        let mut score: BTreeMap<EntryId, f64> = BTreeMap::new();

        // Seed: every entry mentioning a query phrase gets idf(df) of that
        // phrase.
        for &p in qph {
            let w = self.idf(self.postings[p as usize].len());
            for &e in &self.postings[p as usize] {
                *score.entry(e).or_default() += w;
            }
        }

        // Snapshot the seeded entries before expansion so hop-2 propagation
        // is computed from the seed scores only (no feedback within a hop).
        let seeds: Vec<(EntryId, f64)> = score.iter().map(|(e, s)| (*e, *s)).collect();

        for (s, sscore) in &seeds {
            let Some(phrases) = self.entry_phrases.get(s) else {
                continue;
            };
            for &p2 in phrases {
                if qph.contains(&p2) {
                    continue;
                }
                let df2 = self.postings[p2 as usize].len();
                if df2 > MAX_BRIDGE_DF {
                    continue; // hub guard: too common to carry signal
                }
                let w2 = self.idf(df2);
                for &t in &self.postings[p2 as usize] {
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

impl RetrievalSystem for GraphRetriever {
    fn name(&self) -> &str {
        "graph-comention"
    }

    fn index(&mut self, corpus: &Corpus) {
        self.n_entries = corpus.entries.len();

        // Build the gazetteer: sorted entity names (deterministic phrase-id
        // assignment) + the fixed venue anchors. Dedup by lowercased form.
        let mut names: Vec<String> = corpus
            .ground_truth
            .entities
            .iter()
            .map(|e| e.name.clone())
            .collect();
        names.sort();
        for venue in VENUES {
            names.push(venue.to_string());
        }
        let mut phrase_lower: Vec<String> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for name in &names {
            let lower = name.to_lowercase();
            if seen.insert(lower.clone()) {
                phrase_lower.push(lower);
            }
        }
        self.phrase_lower = phrase_lower;

        let mut postings: Vec<Vec<EntryId>> = vec![Vec::new(); self.phrase_lower.len()];
        let mut entry_phrases: BTreeMap<EntryId, Vec<u32>> = BTreeMap::new();

        for entry in &corpus.entries {
            let text_lower = entry.text.to_lowercase();
            let mentioned = self.phrases_in(&text_lower);
            if mentioned.is_empty() {
                continue;
            }
            let mut ids: Vec<u32> = Vec::with_capacity(mentioned.len());
            for &p in &mentioned {
                postings[p as usize].push(entry.id);
                ids.push(p);
            }
            entry_phrases.insert(entry.id, ids);
        }
        self.postings = postings;
        self.entry_phrases = entry_phrases;

        self.bm25 = Bm25::new();
        self.bm25.index(corpus);
    }

    fn search(&self, query: &str, k: usize) -> Vec<EntryId> {
        let query_lower = query.to_lowercase();
        let qph = self.phrases_in(&query_lower);

        let graph_ranked = self.graph_ranked(&qph);
        let bm25_ranked = self.bm25.search(query, (k * 5).max(50));

        // Reciprocal Rank Fusion: fused[e] = sum over lists of 1 / (RRF_C + rank).
        let mut fused: BTreeMap<EntryId, f64> = BTreeMap::new();
        for (rank, (e, _)) in graph_ranked.iter().enumerate() {
            *fused.entry(*e).or_default() += 1.0 / (RRF_C + rank as f64);
        }
        for (rank, e) in bm25_ranked.iter().enumerate() {
            *fused.entry(*e).or_default() += 1.0 / (RRF_C + rank as f64);
        }

        let mut ranked: Vec<(EntryId, f64)> = fused.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        ranked.into_iter().take(k).map(|(id, _)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kortex_corpus::{
        CorpusMeta, Difficulty, Entity, EntityKind, Entry, EntryKind, GroundTruth,
    };

    /// A tiny hand-built corpus with a 2-hop chain that only joins through a
    /// venue anchor:
    ///   0: "Anna introduced me to Marcus at the meetup." (hop A: names both
    ///      people + the venue)
    ///   1: "The person I met at the meetup recommended the book Dune." (hop
    ///      B: coreference — no name, only the venue links back to entry 0)
    ///   2: distractor — Marcus mentioned without the venue or the book.
    ///   3: distractor — unrelated entry.
    /// Query: "What book did the person Anna introduced me to recommend?"
    /// BM25 alone (lexical overlap on "Anna"/"introduced"/"book"/"recommend")
    /// favors entry 0 and misses entry 1 (no shared content words with the
    /// query beyond generic ones); the graph must hop 0 -> "the meetup" -> 1.
    fn two_hop_corpus() -> Corpus {
        let entries = vec![
            Entry {
                id: 0,
                day: 0,
                date: "2018-01-01".to_string(),
                kind: EntryKind::Text,
                text: "Anna introduced me to Marcus at the meetup.".to_string(),
            },
            Entry {
                id: 1,
                day: 1,
                date: "2018-01-02".to_string(),
                kind: EntryKind::Text,
                text: "Ran into that new contact from the meetup again; they said Dune is worth reading.".to_string(),
            },
            Entry {
                id: 2,
                day: 2,
                date: "2018-01-03".to_string(),
                kind: EntryKind::Text,
                text: "Marcus and I argued about Sapiens for an hour; not convinced.".to_string(),
            },
            Entry {
                id: 3,
                day: 3,
                date: "2018-01-04".to_string(),
                kind: EntryKind::Text,
                text: "Quiet day. Thought about jazz harmony on a walk, felt calm.".to_string(),
            },
            // Distractors 4-8: share lexical overlap with the query
            // ("book"/"recommend"/"person") so BM25's top-4 ranks them above
            // the coreference hop (entry 1), which shares almost no content
            // words with the question beyond "book"/"recommend".
            Entry {
                id: 4,
                day: 4,
                date: "2018-01-05".to_string(),
                kind: EntryKind::Text,
                text: "A person at work recommended a book about distributed systems."
                    .to_string(),
            },
            Entry {
                id: 5,
                day: 5,
                date: "2018-01-06".to_string(),
                kind: EntryKind::Text,
                text: "Another person recommended a different book on stoicism today."
                    .to_string(),
            },
            Entry {
                id: 6,
                day: 6,
                date: "2018-01-07".to_string(),
                kind: EntryKind::Text,
                text: "Someone I met recommended a book about trail running.".to_string(),
            },
            Entry {
                id: 7,
                day: 7,
                date: "2018-01-08".to_string(),
                kind: EntryKind::Text,
                text: "A friend recommended a book on nutrition; I'll check it out."
                    .to_string(),
            },
            Entry {
                id: 8,
                day: 8,
                date: "2018-01-09".to_string(),
                kind: EntryKind::Text,
                text: "Quiet evening; thought about a book recommendation from a person at the gym."
                    .to_string(),
            },
        ];

        let entities = vec![
            Entity {
                id: 0,
                name: "Anna".to_string(),
                kind: EntityKind::Person,
                cluster: "social".to_string(),
            },
            Entity {
                id: 1,
                name: "Marcus".to_string(),
                kind: EntityKind::Person,
                cluster: "social".to_string(),
            },
            Entity {
                id: 2,
                name: "Dune".to_string(),
                kind: EntityKind::Book,
                cluster: "reading".to_string(),
            },
            Entity {
                id: 3,
                name: "Sapiens".to_string(),
                kind: EntityKind::Book,
                cluster: "reading".to_string(),
            },
        ];

        Corpus {
            meta: CorpusMeta {
                seed: 0,
                years: 1,
                num_entries: entries.len(),
                difficulty: Difficulty::V2,
            },
            entries,
            ground_truth: GroundTruth {
                entities,
                ..Default::default()
            },
        }
    }

    #[test]
    fn two_hop_join_via_venue_beats_bm25_alone() {
        let corpus = two_hop_corpus();
        let query = "What book did the person Anna introduced me to recommend?";

        let mut bm25 = Bm25::new();
        bm25.index(&corpus);
        let bm25_top = bm25.search(query, 4);

        let mut graph = GraphRetriever::new();
        graph.index(&corpus);
        let graph_top = graph.search(query, 4);

        // BM25 alone should miss at least one of the two hop entries (the
        // coreference hop, entry 1, shares almost no content words with the
        // question).
        let bm25_has_both = bm25_top.contains(&0) && bm25_top.contains(&1);
        assert!(
            !bm25_has_both,
            "expected BM25 alone to miss a hop entry, got {bm25_top:?}"
        );

        // The graph retriever must surface BOTH hop entries in its top-k.
        assert!(
            graph_top.contains(&0),
            "graph missed hop-A entry 0: {graph_top:?}"
        );
        assert!(
            graph_top.contains(&1),
            "graph missed hop-B (coref) entry 1: {graph_top:?}"
        );
    }

    #[test]
    fn search_is_deterministic_across_runs() {
        let corpus = two_hop_corpus();
        let query = "What book did the person Anna introduced me to recommend?";

        let mut g1 = GraphRetriever::new();
        g1.index(&corpus);
        let r1 = g1.search(query, 4);

        let mut g2 = GraphRetriever::new();
        g2.index(&corpus);
        let r2 = g2.search(query, 4);

        assert_eq!(r1, r2, "same corpus + query must yield identical results");

        // Run the same instance twice too.
        let r3 = g1.search(query, 4);
        assert_eq!(r1, r3);
    }

    #[test]
    fn name_method_is_stable() {
        let g = GraphRetriever::new();
        assert_eq!(g.name(), "graph-comention");
    }
}
