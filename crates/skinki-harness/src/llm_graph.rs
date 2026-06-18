//! LLM-extracted entity graph over **real** text (the Stage-3 dialogue path).
//!
//! Where the synthetic corpus is read by hand-written intro/rec/venue patterns
//! ([`skinki_graph::RelationRetriever`]), real dialogue needs a model. This
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
use skinki_baseline::Bm25;
use skinki_corpus::{Corpus, EntryId};
use skinki_eval::RetrievalSystem;

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
    /// `>= 3` chars avoids matching tiny tokens that collide with everything;
    /// entities mentioned in more than a quarter of all turns are conversational
    /// hubs (the speakers in a 2-person dialogue, generic nouns) — they carry no
    /// retrieval signal and only flood, so they are dropped as query anchors.
    fn query_entities(&self, query_lower: &str) -> BTreeSet<String> {
        let hub_df = (self.n_entries / 4).max(1);
        self.vocab
            .iter()
            .filter(|e| {
                e.len() >= 3
                    && query_lower.contains(e.as_str())
                    && self.entity_postings.get(*e).map_or(0, |p| p.len()) <= hub_df
            })
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

// ---------------------------------------------------------------------------
// FactsGraphRetriever — typed-fact graph over the LLM artifact log
// ---------------------------------------------------------------------------
//
// Why this exists. `LlmGraphRetriever` above is a **co-mention** walk: it hops
// through every entity the model emitted in the `entities` field. That is the
// same shape that already lost to BM25 on the *synthetic* corpus in round 1
// (`skinki_graph::GraphRetriever`): raw co-occurrence floods candidates and the
// true hop-B sinks. The synthetic win in round 2 came from **typed relations**
// (`P —introduced→ Q at V`), not bag-of-co-mention — but on real text that
// retriever is never run: its patterns are templated to the synthetic corpus.
//
// This retriever closes that gap. `tools/extract-graph-llm.py` already emits a
// `facts: [[subject, relation, object], ...]` field per turn — which
// `LlmGraphRetriever` silently discards. Here the typed edges ARE the graph:
// every fact becomes a `FactEdge { subj, rel, obj, entry }`, and the multi-hop
// walk hops only through an entity that appears as a **fact endpoint** (not any
// mentioned token), joining entry A to entry B when they share a typed-edge
// endpoint. That is the direct real-text analogue of the synthetic
// `RelationRetriever`'s intro→rec join, with the relation types supplied by the
// model instead of hand-written patterns.
//
// Two further fixes for failure modes measured in `LlmGraphRetriever`:
//
// * **Coreference (Mel vs Melanie).** `normalize` only trimmed/lowercased; the
//   unresolved coref was named in `locomo.rs` as a noise source. Here a
//   deterministic prefix-merge canonicalizes short surface forms to the longest
//   seen variant sharing that prefix (`Mel` -> `Melanie`), so a fact asserted
//   under one alias is reachable from a query naming another. The merge rule is
//   conservative (short len >= 3, longer strictly longer by >= 2, short is a
//   prefix of longer, after stripping `'s`); false merges like `car` ->
//   `caroline` are a documented v0 risk, acceptable on this dialogue domain.
//
// * **Structural gate (no single-hop regression).** `LlmGraphRetriever` fuses
//   graph scores into every query, which is why it regresses temporal / open-
//   domain categories. Here the graph contributes **only hop-joined entries**
//   — entries that are 2-hop reachable from a query entity through a typed fact
//   endpoint and are NOT already a seed. Seed entries (direct entity matches)
//   are left to BM25, so a non-multi-hop query gets pure BM25 with zero graph
//   noise. The gate is structural, not lexical-cue-based: it doesn't need
//   "introduced"/"recommended" cue words that don't exist in real questions.
//
// Determinism (AGENTS rule 2): all structures are `BTreeMap`/`Vec` built in
// ascending entry order; canonicalization is a pure function of the sorted
// entity vocab; ranking ties break on ascending `EntryId`. `rebuild(log)` is
// byte-identical.

/// Minimum length for a short surface form to be prefix-merged into a longer
/// one. Below this the merge is too aggressive (e.g. "al" -> "alice" risks
/// colliding with unrelated 2-letter tokens the model may emit).
const COREF_MIN_LEN: usize = 3;
/// A short form must be at least this much shorter than the long form for a
/// prefix-merge, so "caroline" does not merge with "caroline's" (already
/// handled by `'s` stripping) and near-identical variants stay distinct.
const COREF_MIN_DELTA: usize = 2;

/// Weight of a 2-hop join relative to a seed entry's idf score.
const FACT_HOP_LAMBDA: f64 = 0.5;

/// One typed edge extracted by the LLM: `subj —rel→ obj`, asserted by `entry`.
/// All three strings are canonicalized (see [`FactsGraphRetriever::canon`]).
struct FactEdge {
    subj: String,
    /// Kept for a measured follow-up: per-relation weighting (a "attended"
    /// edge may carry a different hop weight than a "felt" edge). The v0 walk
    /// treats all relations uniformly, so this is unread for now — silencing
    /// dead_code, not deleting it, is the honest choice.
    #[allow(dead_code)]
    rel: String,
    obj: String,
    entry: EntryId,
}

/// Canonicalize a raw entity string: trim, lowercase, strip a trailing `'s`
/// possessive and any trailing non-alphanumerics. Pure and order-independent.
fn normalize_entity(raw: &str) -> String {
    let mut s = raw.trim().to_lowercase();
    if s.ends_with("'s") {
        s.truncate(s.len() - 2);
    }
    while let Some(c) = s.chars().last() {
        if c.is_alphanumeric() {
            break;
        }
        s.pop();
    }
    s
}

/// Build the canonicalization map over a sorted vocab of normalized entity
/// strings: for each short form `a`, if some longer form `b` has `b` starting
/// with `a` (and the lengths differ by at least [`COREF_MIN_DELTA`]), map
/// `a -> b`. When several `b` qualify, pick the longest, tie-broken by the
/// lexicographically smallest (deterministic). Returns `canonical_of[form]`.
///
/// This is the cheap, deterministic coref pass: `Mel` -> `Melanie` (both
/// normalized) merges because "melanie" starts with "mel". It is deliberately
/// conservative — see the module docs for the documented false-merge risk.
fn build_canon_map(sorted_vocab: &[String]) -> BTreeMap<String, String> {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for (i, a) in sorted_vocab.iter().enumerate() {
        if a.len() < COREF_MIN_LEN {
            continue;
        }
        // Scan only later (longer-or-equal) vocab entries; because the vocab is
        // sorted ascending, a prefix match must be lexicographically >= `a`.
        // Collect qualifying longer forms and pick longest, then lex-smallest.
        let mut best: Option<&String> = None;
        for b in &sorted_vocab[i + 1..] {
            if b.len() < a.len() + COREF_MIN_DELTA {
                continue;
            }
            if b.starts_with(a.as_str()) {
                match best {
                    None => best = Some(b),
                    Some(cur) => {
                        // Prefer longer; tie-break lexicographically smaller.
                        if b.len() > cur.len() || (b.len() == cur.len() && *b < *cur) {
                            best = Some(b);
                        }
                    }
                }
            }
        }
        if let Some(b) = best {
            map.insert(a.clone(), b.clone());
        }
    }
    map
}

pub struct FactsGraphRetriever {
    /// canonical entity -> entries whose `entities` field mentions it (asc,
    /// deduped). Seeds come from here.
    entity_postings: BTreeMap<String, Vec<EntryId>>,
    /// All typed fact edges, in ascending entry order.
    fact_edges: Vec<FactEdge>,
    /// canonical entity (as subject) -> indices into `fact_edges` (asc).
    by_subject: BTreeMap<String, Vec<usize>>,
    /// canonical entity (as object) -> indices into `fact_edges` (asc).
    by_object: BTreeMap<String, Vec<usize>>,
    /// sorted canonical entity vocab, for substring-matching a query.
    vocab: Vec<String>,
    n_entries: usize,
    bm25: Bm25,
    fuse_bm25: bool,
}

impl FactsGraphRetriever {
    /// Rebuild from a JSON-lines artifact log (same format
    /// `tools/extract-graph-llm.py` writes, and that `LlmGraphRetriever`
    /// consumes). Both `entities` AND `facts` are read here — the `facts`
    /// field is the typed-edge source the co-mention retriever discards.
    pub fn from_artifacts(path: &Path, fuse_bm25: bool) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading artifact log {}", path.display()))?;

        // Pass 1: collect raw (entities, facts) per entry in ascending entry
        // order. We sort by entry id so every downstream structure is built in
        // a deterministic order regardless of the log's on-disk ordering.
        //
        // `EntryExtraction` factors out what would otherwise be a complex
        // `BTreeMap<EntryId, (BTreeSet<String>, Vec<(String, String, String)>)>`
        // (clippy type_complexity): the per-entry entity set + raw fact triples.
        struct EntryExtraction {
            entities: BTreeSet<String>,
            facts: Vec<(String, String, String)>,
        }
        let mut per_entry: BTreeMap<EntryId, EntryExtraction> = BTreeMap::new();
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
                        let n = normalize_entity(s);
                        if n.len() >= 2 {
                            ents.insert(n);
                        }
                    }
                }
            }
            let mut facts: Vec<(String, String, String)> = Vec::new();
            if let Some(arr) = v.get("facts").and_then(|x| x.as_array()) {
                for f in arr {
                    if let Some(t) = f.as_array() {
                        if t.len() == 3 {
                            let triple: Vec<String> = t
                                .iter()
                                .filter_map(|x| x.as_str())
                                .map(normalize_entity)
                                .collect();
                            if triple.len() == 3 && !triple[0].is_empty() && !triple[2].is_empty() {
                                facts.push((
                                    triple[0].clone(),
                                    triple[1].clone(),
                                    triple[2].clone(),
                                ));
                            }
                        }
                    }
                }
            }
            per_entry.insert(
                entry,
                EntryExtraction {
                    entities: ents,
                    facts,
                },
            );
        }

        // Build the canonicalization map over the full normalized entity vocab
        // (both `entities` mentions and fact endpoints), sorted ascending.
        let mut vocab_set: BTreeSet<String> = BTreeSet::new();
        for ex in per_entry.values() {
            for e in &ex.entities {
                vocab_set.insert(e.clone());
            }
            for (s, _r, o) in &ex.facts {
                vocab_set.insert(s.clone());
                vocab_set.insert(o.clone());
            }
        }
        let sorted_vocab: Vec<String> = vocab_set.into_iter().collect();
        let canon_map = build_canon_map(&sorted_vocab);
        let canon =
            |s: &str| -> String { canon_map.get(s).cloned().unwrap_or_else(|| s.to_string()) };

        // Pass 2: build canonicalized indexes in ascending entry order.
        let mut entity_postings: BTreeMap<String, Vec<EntryId>> = BTreeMap::new();
        let mut fact_edges: Vec<FactEdge> = Vec::new();
        for (entry, ex) in per_entry.iter() {
            let mut seen_ents: BTreeSet<String> = BTreeSet::new();
            for e in &ex.entities {
                seen_ents.insert(canon(e));
            }
            for e in &seen_ents {
                entity_postings.entry(e.clone()).or_default().push(*entry);
            }
            for (s, r, o) in &ex.facts {
                fact_edges.push(FactEdge {
                    subj: canon(s),
                    rel: r.clone(),
                    obj: canon(o),
                    entry: *entry,
                });
            }
        }
        for post in entity_postings.values_mut() {
            post.sort_unstable();
            post.dedup();
        }

        let mut by_subject: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut by_object: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, e) in fact_edges.iter().enumerate() {
            by_subject.entry(e.subj.clone()).or_default().push(i);
            by_object.entry(e.obj.clone()).or_default().push(i);
        }

        let vocab: Vec<String> = entity_postings.keys().cloned().collect();

        Ok(FactsGraphRetriever {
            entity_postings,
            fact_edges,
            by_subject,
            by_object,
            vocab,
            n_entries: 0,
            bm25: Bm25::new(),
            fuse_bm25,
        })
    }

    fn idf(&self, df: usize) -> f64 {
        ((self.n_entries.max(1) as f64) / (df.max(1) as f64)).ln() + 1.0
    }

    /// Canonical entities from the vocab that appear (as a substring) in the
    /// query, dropping conversational hubs (mentioned in > 1/4 of all turns).
    /// Same hub guard as `LlmGraphRetriever` — speakers and generic nouns
    /// carry no retrieval signal and only flood.
    fn query_entities(&self, query_lower: &str) -> BTreeSet<String> {
        let hub_df = (self.n_entries / 4).max(1);
        self.vocab
            .iter()
            .filter(|e| {
                e.len() >= 3
                    && query_lower.contains(e.as_str())
                    && self.entity_postings.get(*e).map_or(0, |p| p.len()) <= hub_df
            })
            .cloned()
            .collect()
    }

    /// Seed + typed 2-hop join scores. Seeds are entries mentioning a query
    /// entity (scored by idf). The hop joins to OTHER entries that share a
    /// typed-fact endpoint with a seed's fact, weighted by the bridge entity's
    /// idf and `FACT_HOP_LAMBDA`. Seeds themselves are excluded from the hop
    /// contribution — the graph injects **only joins**, never re-ranking the
    /// lexical hits BM25 already has (the structural no-regression gate).
    fn graph_ranked(&self, qents: &BTreeSet<String>) -> Vec<(EntryId, f64)> {
        // Seed scores: entries whose `entities` mention a query entity.
        let mut seed_score: BTreeMap<EntryId, f64> = BTreeMap::new();
        for qe in qents {
            if let Some(post) = self.entity_postings.get(qe) {
                let w = self.idf(post.len());
                for &e in post {
                    *seed_score.entry(e).or_default() += w;
                }
            }
        }

        // For each query entity, walk the typed edges that have it as an
        // endpoint. The OTHER endpoint is the bridge. Every OTHER entry
        // asserting a typed fact about that bridge is 2-hop reachable:
        //   query_entity —(fact in entry A)→ bridge —(fact in entry B)→ …
        // This is the typed analogue of the synthetic intro→rec join, with the
        // relation type supplied by the model. Hopping only through fact
        // endpoints (not any mentioned token) is the (A) difference from
        // co-mention: it can't flood through every co-mentioned generic noun.
        let mut hop_score: BTreeMap<EntryId, f64> = BTreeMap::new();
        for qe in qents {
            // Edges where the query entity is the subject; bridge = object.
            let mut bridges: BTreeSet<String> = BTreeSet::new();
            if let Some(idxs) = self.by_subject.get(qe) {
                for &i in idxs {
                    bridges.insert(self.fact_edges[i].obj.clone());
                }
            }
            // Edges where the query entity is the object; bridge = subject.
            if let Some(idxs) = self.by_object.get(qe) {
                for &i in idxs {
                    bridges.insert(self.fact_edges[i].subj.clone());
                }
            }
            for bridge in bridges {
                // Hub guard for the typed walk: count entries that assert a
                // fact with `bridge` as an endpoint (NOT bare `entities`
                // mentions — a bridge may legitimately appear only in facts).
                // This is the join fanout: a bridge endpoint in 100 facts
                // joins 100 entries and floods; one in 2 facts is a tight join.
                let mut reachable: BTreeSet<EntryId> = BTreeSet::new();
                if let Some(idxs) = self.by_subject.get(&bridge) {
                    for &i in idxs {
                        reachable.insert(self.fact_edges[i].entry);
                    }
                }
                if let Some(idxs) = self.by_object.get(&bridge) {
                    for &i in idxs {
                        reachable.insert(self.fact_edges[i].entry);
                    }
                }
                let df = reachable.len();
                if df == 0 || df > MAX_BRIDGE_DF {
                    continue; // hub guard: a bridge endpoint in too many facts joins everything
                }
                let w = self.idf(df) * FACT_HOP_LAMBDA;
                for e in reachable {
                    // Exclude seeds: the graph contributes ONLY joins, so it
                    // never re-ranks a lexical hit (structural gate).
                    if seed_score.contains_key(&e) {
                        continue;
                    }
                    *hop_score.entry(e).or_default() += w;
                }
            }
        }

        // Combine: seeds first (by idf), then hop joins. The ordering within
        // each group is score-desc / id-asc; seeds rank above joins because a
        // direct entity match is stronger evidence than a 2-hop walk.
        let mut ranked: Vec<(EntryId, f64)> = Vec::new();
        for (e, s) in seed_score {
            ranked.push((e, s));
        }
        for (e, s) in hop_score {
            ranked.push((e, s));
        }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        ranked
    }
}

impl RetrievalSystem for FactsGraphRetriever {
    fn name(&self) -> &str {
        if self.fuse_bm25 {
            "llm-facts+bm25"
        } else {
            "llm-facts"
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
    use skinki_corpus::{CorpusMeta, Difficulty, Entry, EntryKind, GroundTruth};
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
        // Unique dir per call (pid + an atomic nonce) so parallel tests never
        // share a path or remove each other's files mid-read.
        use std::sync::atomic::{AtomicU64, Ordering};
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "skinki_llmgraph_{}_{}",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ));
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

    // --- FactsGraphRetriever ------------------------------------------------
    //
    // These pin the three things that distinguish it from the co-mention
    // `LlmGraphRetriever`: (A) the walk goes through typed-fact endpoints, not
    // any mentioned entity; (B) prefix-merge coref links aliases; (В) the
    // structural gate means a query with no resolvable entity gets pure BM25.

    #[test]
    fn fact_walk_links_entries_via_a_shared_fact_endpoint() {
        // entry 0 asserts (Caroline, attended, support group); entry 2 asserts
        // (support group, helped, caroline). A query naming "Caroline" seeds
        // entry 0, and the typed walk reaches entry 2 through the bridge
        // "support group" — a genuine 2-hop join that co-mention would also
        // make, but here ONLY because the bridge is a fact endpoint.
        let corpus = tiny_corpus(&[
            "Caroline: I went to the support group.",
            "Mel: nice weather today.",
            "Mel: the support group really helped.",
        ]);
        let log = write_log(&[
            r#"{"entry":0,"entities":["Caroline"],"facts":[["Caroline","attended","support group"]]}"#,
            r#"{"entry":1,"entities":["Mel"],"facts":[]}"#,
            r#"{"entry":2,"entities":["Mel"],"facts":[["support group","helped","caroline"]]}"#,
        ]);
        let mut r = FactsGraphRetriever::from_artifacts(&log, false).unwrap();
        r.index(&corpus);
        let got = r.search("What did Caroline do?", 3);
        let _ = std::fs::remove_dir_all(log.parent().unwrap());
        assert!(got.contains(&0), "seed entry 0 must be present: {got:?}");
        assert!(
            got.contains(&2),
            "hop-joined entry 2 (shares the 'support group' fact endpoint) must be present: {got:?}"
        );
    }

    #[test]
    fn hop_only_through_fact_endpoints_not_bare_mentions() {
        // entry 0 and entry 1 both mention "weather" in `entities`, but neither
        // asserts a fact about it. The (A) difference: the typed walk must NOT
        // join them, because "weather" is never a fact endpoint. The
        // co-mention retriever would link them.
        let corpus = tiny_corpus(&["Caroline: lovely weather today.", "Mel: weather is nice."]);
        let log = write_log(&[
            r#"{"entry":0,"entities":["Caroline","weather"],"facts":[["Caroline","liked","weather"]]}"#,
            r#"{"entry":1,"entities":["Mel","weather"],"facts":[]}"#,
        ]);
        let mut r = FactsGraphRetriever::from_artifacts(&log, false).unwrap();
        r.index(&corpus);
        let got = r.search("Caroline", 3);
        let _ = std::fs::remove_dir_all(log.parent().unwrap());
        // entry 0 is a seed (mentions Caroline). entry 1 shares only a bare
        // mention of "weather", never a fact endpoint, so it must NOT be
        // hop-joined.
        assert!(got.contains(&0), "seed entry 0 must be present: {got:?}");
        assert!(
            !got.contains(&1),
            "entry 1 shares no fact endpoint, must not be hop-joined: {got:?}"
        );
    }

    #[test]
    fn coref_prefix_merge_links_mel_to_melanie() {
        // Coref value at the BRIDGE level (the regime that matters on real
        // dialogue, where speaker names are hubs and get filtered as query
        // entities): entry 0 asserts (Caroline, met, Mel); entry 2 asserts
        // (Melanie, likes, painting). Without the prefix-merge, "mel" and
        // "melanie" are distinct bridge nodes and entry 2 is unreachable from
        // a "Caroline" query. With the merge, both canonicalize to "melanie"
        // and the typed walk joins entry 0 -> entry 2 through the shared node.
        let corpus = tiny_corpus(&[
            "Caroline: I met Mel.",
            "Filler: nothing here.",
            "Narrator: painting is Melanie's outlet.",
        ]);
        let log = write_log(&[
            r#"{"entry":0,"entities":["Caroline"],"facts":[["Caroline","met","Mel"]]}"#,
            r#"{"entry":1,"entities":["Filler"],"facts":[]}"#,
            r#"{"entry":2,"entities":["painting"],"facts":[["Melanie","likes","painting"]]}"#,
        ]);
        let mut r = FactsGraphRetriever::from_artifacts(&log, false).unwrap();
        r.index(&corpus);
        let got = r.search("What did Caroline do?", 3);
        let _ = std::fs::remove_dir_all(log.parent().unwrap());
        assert!(got.contains(&0), "seed entry 0 must be present: {got:?}");
        assert!(
            got.contains(&2),
            "entry 2 must be hop-joined via the merged 'melanie' bridge (coref): {got:?}"
        );
    }

    #[test]
    fn gate_no_regression_when_query_has_no_resolvable_entity() {
        // A query that names no graph entity must get the BM25 ranking
        // unchanged: the structural gate means the graph contributes no
        // candidates (no seeds, no hops). This is the no-regression guarantee
        // `LlmGraphRetriever` lacks.
        let corpus = tiny_corpus(&[
            "Caroline: I went to the support group.",
            "Mel: nice weather today.",
            "Mel: the group meets Tuesdays.",
        ]);
        let log = write_log(&[
            r#"{"entry":0,"entities":["Caroline","support group"],"facts":[["Caroline","attended","support group"]]}"#,
            r#"{"entry":1,"entities":["Mel"],"facts":[]}"#,
            r#"{"entry":2,"entities":["Mel"],"facts":[]}"#,
        ]);
        let mut r = FactsGraphRetriever::from_artifacts(&log, true).unwrap();
        r.index(&corpus);
        let graph_result = r.search("Tuesdays", 3);

        // Pure BM25 on the same corpus:
        let mut bm25 = Bm25::new();
        bm25.index(&corpus);
        let bm25_result = bm25.search("Tuesdays", 3);
        let _ = std::fs::remove_dir_all(log.parent().unwrap());
        assert_eq!(
            graph_result, bm25_result,
            "no query entity -> graph must reduce to BM25 exactly (structural gate)"
        );
    }

    #[test]
    fn facts_deterministic_rebuild() {
        let corpus = tiny_corpus(&[
            "Caroline: I went to the support group.",
            "Mel: the group meets Tuesdays.",
        ]);
        let log = write_log(&[
            r#"{"entry":0,"entities":["Caroline"],"facts":[["Caroline","attended","support group"]]}"#,
            r#"{"entry":1,"entities":["Mel"],"facts":[["support group","meets","tuesdays"]]}"#,
        ]);
        let build = || {
            let mut r = FactsGraphRetriever::from_artifacts(&log, true).unwrap();
            r.index(&corpus);
            r.search("Caroline support group", 3)
        };
        assert_eq!(build(), build(), "rebuild(log) must be byte-identical");
        let _ = std::fs::remove_dir_all(log.parent().unwrap());
    }

    #[test]
    fn canon_map_prefix_merge_rules() {
        // Direct unit test of the canonicalization: "mel" merges into "melanie"
        // (prefix, len 3 -> 7, delta >= 2); "caroline" does NOT merge into
        // "caroline's" because the `'s` is stripped first (both normalize to
        // "caroline"); "al" is too short to merge.
        let vocab: Vec<String> = vec![
            "al".into(),
            "caroline".into(),
            "mel".into(),
            "melanie".into(),
        ];
        let sorted: Vec<String> = {
            let mut s = vocab.clone();
            s.sort();
            s
        };
        let map = build_canon_map(&sorted);
        assert_eq!(map.get("mel"), Some(&"melanie".to_string()));
        assert!(
            !map.contains_key("caroline"),
            "no longer variant to merge into"
        );
        assert!(!map.contains_key("al"), "below COREF_MIN_LEN");
    }
}
