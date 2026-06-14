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
//! is exactly why `STAGE_3.md` makes relations first-class. `GraphRetriever` is
//! the honest baseline that earned the relation extractor below.
//!
//! ## Measured verdict (round 2 — typed relations win, gated)
//!
//! [`RelationRetriever`] extracts `IntroEdge`/`RecEdge` typed edges
//! (introduction-at-venue, recommendation-by/at) and walks the *exact* planted
//! chain: from a query person `P`, take `P`'s introduction edges to reach `{Q,
//! V}`, then the recommendation edges bridged by the **person** `Q` (the precise
//! non-coref case) or by the **venue** `V` with a **temporal-proximity** weight
//! (the coref case, where the venue is shared across chains so recency
//! disambiguates). Relation expansion only fires when the query carries an
//! intro/rec cue, so plain single-hop questions reduce to BM25 (no regression).
//!
//! On the V2 corpus this **decisively beats BM25** and the gap *widens* with
//! scale (BM25 degrades, the typed walk holds):
//!
//! | corpus | metric | bm25 | co-mention | **relation** |
//! | --- | --- | --- | --- | --- |
//! | ~11.5k | multi-hop recall@10 | 0.325 | 0.325 | **0.800** |
//! | ~11.5k | multi-hop ans@10    | 0.650 | 0.550 | **0.900** |
//! | ~29.6k | multi-hop recall@10 | 0.172 | 0.156 | **0.422** |
//! | ~29.6k | multi-hop ans@10    | 0.219 | 0.594 | **0.656** |
//!
//! Single-hop recall never drops below BM25. Gated by
//! `graph-eval --assert-gate` (multi-hop recall@10 >= 0.50, ans@10 >= 0.60, no
//! single-hop regression). The deterministic tier clears the gate alone.
//!
//! ## Measured verdict (round 3 — the LLM tier is *not earned* on this corpus)
//!
//! T6/D2 add the two-tier machinery: an append-only [`ArtifactLog`] (replay
//! contract, AGENTS rule 3), a deterministic [`RelationRetriever::selects_for_llm_tier`]
//! policy (a unit goes to tier-1 iff it is an ambiguous coreference rec — a rec
//! cue + a venue + *no* person), and [`RelationRetriever::index_with_artifacts`]
//! to fold replayed resolutions back in. To measure the tier's *ceiling* without
//! a model, `graph-eval` builds a **ground-truth oracle** artifact log (the
//! perfect LLM) and replays it.
//!
//! The honest result: the oracle tier-1 selects only ~0.1% of units (well under
//! the 5% budget) but **does not reliably help** — multi-hop recall@10 lift is
//! **-0.125 at ~11.5k** and only **+0.031 at ~29.6k**. Resolving the
//! coreference to a *person name* re-indexes that hop under `rec_by_person`, and
//! because the corpus reuses person names, this injects cross-chain noise the
//! deterministic **venue+temporal** bridge had avoided. In other words: even a
//! *perfect* model adds nothing here once the substrate disambiguates by time —
//! a direct vindication of the project's first law, **intelligence in the
//! memory, not the model**. So we *do not adopt* the LLM tier: the gate prints
//! the lift as informational and does not require it. The replay+selection
//! machinery stays, ready to be re-measured on a regime where the oracle ceiling
//! actually pays (e.g. genuinely unrecoverable coref, or relation types the
//! deterministic patterns can't see).

use std::collections::{BTreeMap, BTreeSet};

use skinki_baseline::Bm25;
use skinki_corpus::{Corpus, EntityKind, EntryId};
use skinki_eval::RetrievalSystem;
use skinki_ledger::{ContentHash, Derivation, Ledger, MethodStamp};

/// Venue anchors — the corpus's coreference bridges for multi-hop chains
/// (see `skinki-corpus`'s `VENUES`). Treated as gazetteer phrases just like
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

// ---------------------------------------------------------------------------
// RelationRetriever — typed-relation graph (round 2)
// ---------------------------------------------------------------------------
//
// `GraphRetriever` above measured that raw co-mention is insufficient for the
// planted multi-hop chains (Law 2 verdict in the module docs). This retriever
// extracts two *typed* edges per entry — `IntroEdge` ("P introduced me to Q at
// V") and `RecEdge` ("Q recommended book B", possibly coreferenced only via a
// venue) — and joins them explicitly: hop A (intro) -> hop B (rec), bridged
// either by the introduced person's name or, for the coreference form, by the
// shared venue with a temporal-proximity weight (hop B is planted within ~90
// days after hop A). The graph ranking is then RRF-fused with BM25, exactly
// like `GraphRetriever`.
//
// Determinism: all indexes are `BTreeMap`/`Vec` built in corpus entry order;
// all iteration during `search` is over sorted keys / vecs in insertion order,
// and the final ranking ties break on ascending `EntryId`.

/// Cues that signal an "introduction" sentence (hop A).
const INTRO_CUES: &[&str] = &["introduced me to", "introduced", "through", "brought"];

/// Cues that signal a "recommendation" sentence (hop B).
const REC_CUES: &[&str] = &[
    "recommended",
    "told me to read",
    "worth reading",
    "suggested",
    "recommendation",
];

/// Weight for a hop-A (intro) edge matching the query's person.
const W_HOPA: f64 = 3.0;
/// Weight for a hop-B (rec) edge reached via a hop-A bridge.
const W_HOPB: f64 = 3.0;
/// Weight for a rec edge directly naming the query's person as recommender.
const W_DIRECT: f64 = 2.0;
/// Maximum days between hop A and a venue-bridged hop B for the join to fire.
const MAX_DT_DAYS: u32 = 90;
/// Reciprocal Rank Fusion smoothing constant (shared with `GraphRetriever`).
const REL_RRF_C: f64 = 60.0;

/// Extractor version stamped onto every edge's derivation. Bump it when the
/// extraction rules change: the ledger then flags every edge this extractor
/// produced for re-derivation, even if no source text moved (AGENTS rule 3 —
/// the "how" changed). Method ids distinguish the two relation kinds.
const EXTRACTOR_VERSION: u64 = 1;
const M_INTRO: u32 = 1;
const M_REC: u32 = 2;

/// Method id for an LLM-tier coreference resolution applied to a `RecEdge`
/// (Stage 3 D2). The "model version" is carried per-record in
/// [`LlmExtraction::model_version`] so the ledger can flag every LLM-tier edge
/// for re-derivation if the (simulated) model changes, exactly like
/// `EXTRACTOR_VERSION` does for the deterministic tier.
pub const M_LLM_REC: u32 = 3;

// ---------------------------------------------------------------------------
// T6 — replay artifact log
// ---------------------------------------------------------------------------
//
// Every LLM-tier output that feeds the graph is appended to a flat,
// append-only JSON-lines log before it is folded in. `index_with_artifacts`
// never calls an LLM directly: it only consumes records that were produced
// once (by the oracle in this measurement, or a real model in Stage 5+) and
// replayed back through this log, so the graph build stays
// `rebuild(log)`-deterministic (AGENTS.md: LLM outputs are replayable, not
// bit-deterministic, but everything downstream of the log is).

/// One LLM-tier extraction: the coreference `RecEdge` it resolved, and the
/// (lowercased) recommender name(s) it inferred for that edge's `by` field.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct LlmExtraction {
    /// The coref rec entry the LLM resolved.
    pub entry: EntryId,
    /// Lowercased recommender person name(s) it inferred.
    pub resolved_by: Vec<String>,
    /// Simulated model version, stamped into the ledger's `MethodStamp` so a
    /// model upgrade can be detected as staleness (AGENTS.md rule 3).
    pub model_version: u64,
}

/// Thin namespace for the artifact-log replay contract (T6). Methods are
/// associated functions — there is no per-instance state, only the file.
pub struct ArtifactLog;

impl ArtifactLog {
    /// Append one record as a JSON line to `path` (create-or-append).
    pub fn append(path: &std::path::Path, x: &LlmExtraction) -> std::io::Result<()> {
        use std::io::Write;
        let line = serde_json::to_string(x).map_err(std::io::Error::other)?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    /// Replay all records (JSON-lines) in file order. Deterministic.
    pub fn replay(path: &std::path::Path) -> std::io::Result<Vec<LlmExtraction>> {
        let contents = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for line in contents.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let x: LlmExtraction = serde_json::from_str(line).map_err(std::io::Error::other)?;
            out.push(x);
        }
        Ok(out)
    }
}

/// "P introduced me to Q (and maybe others) at venue V on day `day`."
struct IntroEdge {
    persons: Vec<String>,
    venue: String,
    entry: EntryId,
    day: u32,
}

/// "Person(s) `by` recommended a book, possibly only identifiable via `venue`,
/// on day `day`." `by` is empty for the coreference form ("the person I met at
/// V recommended B").
struct RecEdge {
    by: Vec<String>,
    venue: Option<String>,
    entry: EntryId,
    day: u32,
}

/// A deterministic typed-relation graph retriever, fused with BM25 via RRF.
pub struct RelationRetriever {
    /// Lowercased ground-truth person names, sorted ascending.
    persons: Vec<String>,
    /// Lowercased venue strings (from `VENUES`), sorted ascending.
    venues: Vec<String>,
    intro: Vec<IntroEdge>,
    rec: Vec<RecEdge>,
    /// person name (lowercase) -> indices into `intro`.
    intro_by_person: BTreeMap<String, Vec<usize>>,
    /// person name (lowercase) -> indices into `rec`.
    rec_by_person: BTreeMap<String, Vec<usize>>,
    /// venue (lowercase) -> indices into `rec`.
    rec_by_venue: BTreeMap<String, Vec<usize>>,
    bm25: Bm25,
    /// Derivation ledger: one record per extracted edge, pinning the source
    /// entry's content hash + the extractor's method/version. This is what makes
    /// the graph incrementally re-extractable and staleness-aware (Stage 3 T7):
    /// when a unit changes (or `EXTRACTOR_VERSION` bumps), `ledger.stale_closure`
    /// flags exactly the edges that must be rebuilt.
    ledger: Ledger,
}

impl Default for RelationRetriever {
    fn default() -> Self {
        RelationRetriever {
            persons: Vec::new(),
            venues: Vec::new(),
            intro: Vec::new(),
            rec: Vec::new(),
            intro_by_person: BTreeMap::new(),
            rec_by_person: BTreeMap::new(),
            rec_by_venue: BTreeMap::new(),
            bm25: Bm25::new(),
            ledger: Ledger::new(),
        }
    }
}

impl RelationRetriever {
    pub fn new() -> Self {
        RelationRetriever::default()
    }

    /// Returns (persons present, venues present) in `text_lower`, each as a
    /// `BTreeSet` for deterministic ordering. `text_lower` must already be
    /// lowercased.
    fn entities_in(&self, text_lower: &str) -> (BTreeSet<String>, BTreeSet<String>) {
        let mut persons = BTreeSet::new();
        for p in &self.persons {
            if text_lower.contains(p.as_str()) {
                persons.insert(p.clone());
            }
        }
        let mut venues = BTreeSet::new();
        for v in &self.venues {
            if text_lower.contains(v.as_str()) {
                venues.insert(v.clone());
            }
        }
        (persons, venues)
    }

    fn has_any_cue(text_lower: &str, cues: &[&str]) -> bool {
        cues.iter().any(|c| text_lower.contains(c))
    }

    /// Content hash of a source entry's text — the provenance "input" an edge
    /// derives from. (A sentence-unit hash would be finer; entry-level is the
    /// right grain for this corpus, where one entry asserts the relation.)
    fn entry_hash(text: &str) -> ContentHash {
        ContentHash::of(text.as_bytes())
    }

    /// Canonical, rebuild-stable content hash of an intro / rec edge.
    fn intro_output_hash(entry: EntryId, persons: &[String], venue: &str) -> ContentHash {
        ContentHash::of(format!("intro|{entry}|{}|{venue}", persons.join(",")).as_bytes())
    }
    fn rec_output_hash(entry: EntryId, by: &[String], venue: Option<&str>) -> ContentHash {
        ContentHash::of(format!("rec|{entry}|{}|{}", by.join(","), venue.unwrap_or("")).as_bytes())
    }

    /// The derivation ledger backing this graph — one record per extracted edge,
    /// pinning the source entry's content hash and the extractor's method
    /// version. Drives incremental re-extraction and staleness (Stage 3 T7).
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// True iff this unit is an AMBIGUOUS coreference recommendation — a REC
    /// cue + a venue mentioned but NO person named. This is exactly the unit
    /// whose recommender the LLM tier (D2) must resolve: the deterministic
    /// tier can only bridge it via the noisy venue+temporal walk.
    ///
    /// Pure function of `text` (any case) and the gazetteer built by `index`.
    pub fn selects_for_llm_tier(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        if !Self::has_any_cue(&text_lower, REC_CUES) {
            return false;
        }
        let (persons, venues) = self.entities_in(&text_lower);
        !venues.is_empty() && persons.is_empty()
    }

    /// Fraction of `corpus.entries` selected for the LLM tier by
    /// [`Self::selects_for_llm_tier`] — the "tier-1 share" the Stage 3 gate
    /// caps (cheap deterministic tier handles the rest).
    pub fn llm_tier_share(&self, corpus: &Corpus) -> f64 {
        let n = corpus.entries.len();
        if n == 0 {
            return 0.0;
        }
        let selected = corpus
            .entries
            .iter()
            .filter(|e| self.selects_for_llm_tier(&e.text))
            .count();
        selected as f64 / n as f64
    }

    /// Like [`RetrievalSystem::index`], but after deterministic extraction,
    /// fold in replayed LLM-tier resolutions (T6/D2): for each
    /// [`LlmExtraction`], find the `RecEdge` at that entry and merge its
    /// `resolved_by` persons into the edge's `by` (dedup, sorted), then rebuild
    /// `rec_by_person`. Each applied resolution is also recorded in the ledger
    /// as a `Derivation` (input = the entry's content hash, output = the
    /// upgraded rec edge's hash, method = `MethodStamp{ id: M_LLM_REC, version:
    /// model_version }`), so the LLM tier is replay- and staleness-tracked too.
    ///
    /// Deterministic and idempotent on the same `(corpus, artifacts)`:
    /// `artifacts` is processed in its given (file) order, `by` is
    /// deduped+sorted before hashing, and `rec_by_person` is rebuilt from
    /// scratch in ascending `rec` index order.
    pub fn index_with_artifacts(&mut self, corpus: &Corpus, artifacts: &[LlmExtraction]) {
        self.index(corpus);

        // entry -> index into `self.rec`, for O(1) lookup while applying
        // artifacts. Built once; rec entries are unique per EntryId by
        // construction (one REC-cue check per corpus entry).
        let mut rec_by_entry: BTreeMap<EntryId, usize> = BTreeMap::new();
        for (i, e) in self.rec.iter().enumerate() {
            rec_by_entry.insert(e.entry, i);
        }

        for art in artifacts {
            let Some(&ri) = rec_by_entry.get(&art.entry) else {
                continue; // no rec edge at this entry — nothing to upgrade
            };
            if art.resolved_by.is_empty() {
                continue;
            }

            let edge = &mut self.rec[ri];
            let src = Self::entry_hash(&corpus.entries[art.entry as usize].text);
            let before = Self::rec_output_hash(edge.entry, &edge.by, edge.venue.as_deref());

            // Merge: dedup + sort for determinism, idempotent on repeats.
            let mut by: BTreeSet<String> = edge.by.iter().cloned().collect();
            for p in &art.resolved_by {
                by.insert(p.clone());
            }
            edge.by = by.into_iter().collect();

            let after = Self::rec_output_hash(edge.entry, &edge.by, edge.venue.as_deref());
            self.ledger.record(Derivation::new(
                after,
                vec![src, before],
                MethodStamp::new(M_LLM_REC, art.model_version),
            ));
        }

        // Rebuild rec_by_person from scratch (ascending rec index order ->
        // deterministic posting order, same as `index`).
        let mut rec_by_person: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, e) in self.rec.iter().enumerate() {
            for p in &e.by {
                rec_by_person.entry(p.clone()).or_default().push(i);
            }
        }
        self.rec_by_person = rec_by_person;
    }

    /// Approximate resident bytes of the **graph** structures that scale with
    /// corpus size — typed edges, the person/venue indexes, and the derivation
    /// ledger (Stage 3 T8). Excludes the BM25 fusion component (the lexical /
    /// Stage-1 surrogate, budgeted separately) and the gazetteer. Each
    /// `String`/`Vec` is counted as its payload plus a 24-byte (ptr,len,cap)
    /// header, with a small per-`BTreeMap`-entry node overhead — honest enough
    /// to project a 5M budget, not a precise allocator readout.
    pub fn graph_resident_bytes(&self) -> usize {
        const HDR: usize = 24; // String / Vec header
        let strs = |v: &[String]| v.iter().map(|s| s.len() + HDR).sum::<usize>();

        let intro_bytes: usize = self
            .intro
            .iter()
            .map(|e| std::mem::size_of::<IntroEdge>() + strs(&e.persons) + e.venue.len() + HDR)
            .sum();
        let rec_bytes: usize = self
            .rec
            .iter()
            .map(|e| {
                std::mem::size_of::<RecEdge>()
                    + strs(&e.by)
                    + e.venue.as_ref().map_or(0, |v| v.len() + HDR)
            })
            .sum();

        let map_bytes = |m: &BTreeMap<String, Vec<usize>>| -> usize {
            m.iter()
                .map(|(k, v)| {
                    k.len() + HDR + v.len() * std::mem::size_of::<usize>() + HDR + 48
                    // node ovhd
                })
                .sum::<usize>()
        };
        let index_bytes = map_bytes(&self.intro_by_person)
            + map_bytes(&self.rec_by_person)
            + map_bytes(&self.rec_by_venue);

        // One ledger record ~ output(16) + one input(16) + Vec hdr(24) + method(12).
        let ledger_bytes = self.ledger.len() * 68;

        intro_bytes + rec_bytes + index_bytes + ledger_bytes
    }
}

impl RetrievalSystem for RelationRetriever {
    fn name(&self) -> &str {
        "graph-relation"
    }

    fn index(&mut self, corpus: &Corpus) {
        // Gazetteer: lowercased, deduped, sorted person names; venues from the
        // fixed VENUES list (already lowercase-friendly phrases).
        let mut persons: BTreeSet<String> = BTreeSet::new();
        for e in &corpus.ground_truth.entities {
            if e.kind == EntityKind::Person {
                persons.insert(e.name.to_lowercase());
            }
        }
        self.persons = persons.into_iter().collect();

        let mut venues: BTreeSet<String> = BTreeSet::new();
        for v in VENUES {
            venues.insert(v.to_lowercase());
        }
        self.venues = venues.into_iter().collect();

        let mut intro: Vec<IntroEdge> = Vec::new();
        let mut rec: Vec<RecEdge> = Vec::new();
        // One derivation per edge, in entry order (deterministic): input = the
        // source entry's content hash, output = the edge's canonical hash.
        let mut ledger = Ledger::new();

        for entry in &corpus.entries {
            let text_lower = entry.text.to_lowercase();
            let (entry_persons, entry_venues) = self.entities_in(&text_lower);
            let src = Self::entry_hash(&entry.text);

            if Self::has_any_cue(&text_lower, INTRO_CUES)
                && !entry_venues.is_empty()
                && !entry_persons.is_empty()
            {
                // entry_venues / entry_persons are BTreeSets -> already sorted;
                // "first" is deterministic.
                let venue = entry_venues.iter().next().unwrap().clone();
                let persons: Vec<String> = entry_persons.iter().cloned().collect();
                ledger.record(Derivation::new(
                    Self::intro_output_hash(entry.id, &persons, &venue),
                    vec![src],
                    MethodStamp::new(M_INTRO, EXTRACTOR_VERSION),
                ));
                intro.push(IntroEdge {
                    persons,
                    venue,
                    entry: entry.id,
                    day: entry.day,
                });
            }

            if Self::has_any_cue(&text_lower, REC_CUES) {
                let venue = entry_venues.iter().next().cloned();
                let by: Vec<String> = entry_persons.iter().cloned().collect();
                ledger.record(Derivation::new(
                    Self::rec_output_hash(entry.id, &by, venue.as_deref()),
                    vec![src],
                    MethodStamp::new(M_REC, EXTRACTOR_VERSION),
                ));
                rec.push(RecEdge {
                    by,
                    venue,
                    entry: entry.id,
                    day: entry.day,
                });
            }
        }
        self.ledger = ledger;

        let mut intro_by_person: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, e) in intro.iter().enumerate() {
            for p in &e.persons {
                intro_by_person.entry(p.clone()).or_default().push(i);
            }
        }

        let mut rec_by_person: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut rec_by_venue: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, e) in rec.iter().enumerate() {
            for p in &e.by {
                rec_by_person.entry(p.clone()).or_default().push(i);
            }
            if let Some(v) = &e.venue {
                rec_by_venue.entry(v.clone()).or_default().push(i);
            }
        }

        self.intro = intro;
        self.rec = rec;
        self.intro_by_person = intro_by_person;
        self.rec_by_person = rec_by_person;
        self.rec_by_venue = rec_by_venue;

        self.bm25 = Bm25::new();
        self.bm25.index(corpus);
    }

    fn search(&self, query: &str, k: usize) -> Vec<EntryId> {
        let query_lower = query.to_lowercase();
        let (mut qpersons, _qvenues) = self.entities_in(&query_lower);

        // Only expand along typed relations when the query is actually
        // relational (asks about an introduction / recommendation). For a plain
        // single-hop question we drop the seeds, so fusion reduces to BM25 and a
        // multi-hop candidate never displaces a lexical single-hop answer.
        if !(Self::has_any_cue(&query_lower, INTRO_CUES)
            || Self::has_any_cue(&query_lower, REC_CUES))
        {
            qpersons.clear();
        }

        let mut score: BTreeMap<EntryId, f64> = BTreeMap::new();

        for p in &qpersons {
            // Direct: the query names the recommender outright.
            if let Some(idxs) = self.rec_by_person.get(p) {
                for &ri in idxs {
                    *score.entry(self.rec[ri].entry).or_default() += W_DIRECT;
                }
            }

            // Hop A: "p introduced me to Q at V".
            if let Some(idxs) = self.intro_by_person.get(p) {
                for &ei in idxs {
                    let e = &self.intro[ei];
                    *score.entry(e.entry).or_default() += W_HOPA;

                    // Person-bridged hop B: the *other* person(s) named in the
                    // intro edge (the specific, non-coreference case).
                    for q in e.persons.iter().filter(|n| !qpersons.contains(*n)) {
                        if let Some(ridxs) = self.rec_by_person.get(q) {
                            for &ri in ridxs {
                                if self.rec[ri].entry != e.entry {
                                    *score.entry(self.rec[ri].entry).or_default() += W_HOPB;
                                }
                            }
                        }
                    }

                    // Venue-bridged hop B (the coreference case): same venue,
                    // dated on/after hop A within MAX_DT_DAYS, weighted by
                    // temporal proximity.
                    if let Some(ridxs) = self.rec_by_venue.get(&e.venue) {
                        for &ri in ridxs {
                            let r = &self.rec[ri];
                            if r.entry == e.entry {
                                continue;
                            }
                            if r.day >= e.day {
                                let dt = r.day - e.day;
                                if dt <= MAX_DT_DAYS {
                                    let w = W_HOPB * (1.0 / (1.0 + (dt as f64) / 30.0));
                                    *score.entry(r.entry).or_default() += w;
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut graph_ranked: Vec<(EntryId, f64)> = score.into_iter().collect();
        graph_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));

        let bm25_ranked = self.bm25.search(query, (k * 5).max(50));

        // Reciprocal Rank Fusion, identical scheme to `GraphRetriever::search`.
        let mut fused: BTreeMap<EntryId, f64> = BTreeMap::new();
        for (rank, (e, _)) in graph_ranked.iter().enumerate() {
            *fused.entry(*e).or_default() += 1.0 / (REL_RRF_C + rank as f64);
        }
        for (rank, e) in bm25_ranked.iter().enumerate() {
            *fused.entry(*e).or_default() += 1.0 / (REL_RRF_C + rank as f64);
        }

        let mut ranked: Vec<(EntryId, f64)> = fused.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        ranked.into_iter().take(k).map(|(id, _)| id).collect()
    }
}

// ---------------------------------------------------------------------------
// Stage 3C — context assembler
// ---------------------------------------------------------------------------
//
// The L5 surface a small model verbalizes from: not "top-k chunks" but a
// small, dense, CITED, dated package that fits a fixed token budget. This is
// generic over `RetrievalSystem` so the same assembler builds both the graph
// package (the system under test) and a naive top-k "dump" baseline (BM25 at
// the same budget) for an apples-to-apples context-sufficiency comparison.

/// A single cited, dated fact in an assembled context package.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CitedFact {
    pub entry: EntryId,
    pub date: String,
    pub text: String,
}

/// A budgeted, cited context package (the L5 surface a small model verbalizes).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ContextPackage {
    pub facts: Vec<CitedFact>,
    pub est_tokens: usize,
}

impl ContextPackage {
    /// Does any cited fact contain `answer` (case-insensitive substring)? The
    /// context-sufficiency proxy on the synthetic corpus.
    pub fn contains_answer(&self, answer: &str) -> bool {
        let needle = answer.to_lowercase();
        self.facts
            .iter()
            .any(|f| f.text.to_lowercase().contains(&needle))
    }
}

/// Rough token estimate of a string: ~4 chars/token + a small per-fact overhead.
pub fn est_tokens(text: &str) -> usize {
    text.chars().count() / 4 + 2
}

/// Assemble a budgeted context package for `query` from ANY retrieval system:
/// take its top entries in rank order, add each as a [`CitedFact`] (date +
/// text) while the running token estimate stays within `token_budget`; stop
/// before exceeding it.
///
/// Deterministic: `ids` come from `system.search` (already deterministic per
/// AGENTS rule 2), the entry lookup is a `BTreeMap` built once from the
/// corpus's entries (ascending `EntryId`), and iteration order over `ids` is
/// preserved (rank order) — no `HashMap`/unordered iteration anywhere.
pub fn assemble_context(
    system: &dyn RetrievalSystem,
    corpus: &Corpus,
    query: &str,
    token_budget: usize,
) -> ContextPackage {
    let by_id: BTreeMap<EntryId, &skinki_corpus::Entry> =
        corpus.entries.iter().map(|e| (e.id, e)).collect();

    let ids = system.search(query, 64);

    let mut facts = Vec::new();
    let mut running = 0usize;
    for id in ids {
        let Some(entry) = by_id.get(&id) else {
            continue;
        };
        let f = est_tokens(&entry.date) + est_tokens(&entry.text);
        if running + f > token_budget {
            break;
        }
        facts.push(CitedFact {
            entry: id,
            date: entry.date.clone(),
            text: entry.text.clone(),
        });
        running += f;
    }

    ContextPackage {
        facts,
        est_tokens: running,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skinki_corpus::{
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

    /// Two independent intro->rec chains that share the SAME venue ("the
    /// meetup") but at different times, both using the coreference form for
    /// hop B (no person name in hop B). The venue-bridge join must pick the
    /// hop-B entry that is closest in time *after* the matching hop-A, not
    /// just any entry mentioning the venue.
    ///
    ///   0: "Anna introduced me to Marcus at the meetup." (day 0)
    ///   1: "The person I met at the meetup recommended the book Dune."
    ///      (day 10 -- close to entry 0)
    ///   2: "Carol introduced me to Diane at the meetup." (day 100)
    ///   3: "That new acquaintance from the meetup told me to read Sapiens."
    ///      (day 105 -- close to entry 2, far from entry 0)
    fn two_chain_corpus() -> Corpus {
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
                day: 10,
                date: "2018-01-11".to_string(),
                kind: EntryKind::Text,
                text: "The person I met at the meetup recommended the book Dune.".to_string(),
            },
            Entry {
                id: 2,
                day: 100,
                date: "2018-04-11".to_string(),
                kind: EntryKind::Text,
                text: "Carol introduced me to Diane at the meetup.".to_string(),
            },
            Entry {
                id: 3,
                day: 105,
                date: "2018-04-16".to_string(),
                kind: EntryKind::Text,
                text: "That new acquaintance from the meetup told me to read Sapiens.".to_string(),
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
                name: "Carol".to_string(),
                kind: EntityKind::Person,
                cluster: "social".to_string(),
            },
            Entity {
                id: 3,
                name: "Diane".to_string(),
                kind: EntityKind::Person,
                cluster: "social".to_string(),
            },
            Entity {
                id: 4,
                name: "Dune".to_string(),
                kind: EntityKind::Book,
                cluster: "reading".to_string(),
            },
            Entity {
                id: 5,
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
    fn venue_bridge_picks_temporally_closest_hop_b() {
        let corpus = two_chain_corpus();

        let mut rel = RelationRetriever::new();
        rel.index(&corpus);

        // Query about Anna's introduction should surface entry 0 (hop A) and
        // entry 1 (the temporally-closest hop B via "the meetup"), not entry
        // 3 (which belongs to the Carol/Diane chain).
        let q_anna = "What book was recommended by the person Anna introduced me to?";
        let top_anna = rel.search(q_anna, 4);
        assert!(
            top_anna.contains(&0),
            "missing hop-A entry 0 for Anna query: {top_anna:?}"
        );
        assert!(
            top_anna.contains(&1),
            "missing hop-B entry 1 (Dune) for Anna query: {top_anna:?}"
        );

        // Query about Carol's introduction should surface entry 2 (hop A) and
        // entry 3 (the temporally-closest hop B via "the meetup"), not entry
        // 1 (which belongs to the Anna/Marcus chain).
        let q_carol = "What book was recommended by the person Carol introduced me to?";
        let top_carol = rel.search(q_carol, 4);
        assert!(
            top_carol.contains(&2),
            "missing hop-A entry 2 for Carol query: {top_carol:?}"
        );
        assert!(
            top_carol.contains(&3),
            "missing hop-B entry 3 (Sapiens) for Carol query: {top_carol:?}"
        );
    }

    #[test]
    fn relation_search_is_deterministic_across_runs() {
        let corpus = two_chain_corpus();
        let query = "What book was recommended by the person Anna introduced me to?";

        let mut r1 = RelationRetriever::new();
        r1.index(&corpus);
        let out1 = r1.search(query, 4);

        let mut r2 = RelationRetriever::new();
        r2.index(&corpus);
        let out2 = r2.search(query, 4);

        assert_eq!(
            out1, out2,
            "same corpus + query must yield identical results"
        );

        let out3 = r1.search(query, 4);
        assert_eq!(out1, out3);
    }

    #[test]
    fn relation_name_method_is_stable() {
        let r = RelationRetriever::new();
        assert_eq!(r.name(), "graph-relation");
    }

    // --- T7: ledger-backed incremental re-extraction + staleness ------------

    #[test]
    fn changing_an_entry_flags_exactly_its_edges() {
        use skinki_ledger::score_staleness;
        let corpus = two_chain_corpus();
        let mut r = RelationRetriever::new();
        r.index(&corpus);
        assert!(
            !r.ledger().is_empty(),
            "edges must be recorded in the ledger"
        );

        // Supersede entry 0 (a hop-A intro entry): its content hash moves.
        let h0 = RelationRetriever::entry_hash(&corpus.entries[0].text);
        let changed: BTreeSet<ContentHash> = [h0].into_iter().collect();

        // Independent oracle: outputs whose derivation cites entry 0's hash.
        let truth: BTreeSet<ContentHash> = r
            .ledger()
            .records()
            .iter()
            .filter(|d| d.inputs.contains(&h0))
            .map(|d| d.output)
            .collect();
        assert!(!truth.is_empty(), "entry 0 should assert >= 1 edge");

        let flagged = r.ledger().stale_closure(&changed, &BTreeMap::new());
        let s = score_staleness(&flagged, &truth);
        assert_eq!(
            s.invalidation_recall, 1.0,
            "must catch every dependent edge"
        );
        assert_eq!(
            s.over_invalidation, 0.0,
            "must not flag any other entry's edges"
        );
        assert_eq!(flagged, truth, "exactly entry 0's edges, nothing else");
    }

    #[test]
    fn bumping_extractor_version_flags_all_its_edges() {
        let corpus = two_chain_corpus();
        let mut r = RelationRetriever::new();
        r.index(&corpus);

        // Pretend the intro extractor moved to a new version: every intro edge
        // must be flagged for re-derivation, with no source text changed.
        let current = BTreeMap::from([(M_INTRO, EXTRACTOR_VERSION + 1)]);
        let flagged = r.ledger().stale_closure(&BTreeSet::new(), &current);

        let intro_outputs: BTreeSet<ContentHash> = r
            .ledger()
            .records()
            .iter()
            .filter(|d| d.method.id == M_INTRO)
            .map(|d| d.output)
            .collect();
        assert!(!intro_outputs.is_empty(), "expected intro edges to exist");
        assert_eq!(flagged, intro_outputs, "all intro edges, no rec edges");
    }

    // --- T6/D2: LLM-tier artifact log + selection policy --------------------

    #[test]
    fn artifact_log_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "skinki_graph_artifact_log_test_{}_{}",
            std::process::id(),
            "roundtrip"
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("artifacts.jsonl");
        let _ = std::fs::remove_file(&path);

        let a = LlmExtraction {
            entry: 1,
            resolved_by: vec!["anna".to_string()],
            model_version: 1,
        };
        let b = LlmExtraction {
            entry: 3,
            resolved_by: vec!["carol".to_string(), "diane".to_string()],
            model_version: 1,
        };
        ArtifactLog::append(&path, &a).unwrap();
        ArtifactLog::append(&path, &b).unwrap();

        let replayed = ArtifactLog::replay(&path).unwrap();
        assert_eq!(replayed, vec![a, b]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn llm_tier_upgrades_coref_recedge() {
        let corpus = two_chain_corpus();

        // Entry 1 is the coreference rec edge: "The person I met at the
        // meetup recommended the book Dune." -> hop A (entry 0) names Anna and
        // Marcus; the LLM resolves "the person I met" to Marcus.
        let mut rel = RelationRetriever::new();
        rel.index(&corpus);
        assert!(
            rel.selects_for_llm_tier(&corpus.entries[1].text),
            "entry 1 (coref rec) should be selected for the LLM tier"
        );
        assert!(
            !rel.selects_for_llm_tier(&corpus.entries[0].text),
            "entry 0 (names both people) should NOT be selected"
        );

        let artifacts = vec![LlmExtraction {
            entry: 1,
            resolved_by: vec!["marcus".to_string()],
            model_version: 1,
        }];

        let mut rel_llm = RelationRetriever::new();
        rel_llm.index_with_artifacts(&corpus, &artifacts);

        let ri = rel_llm
            .rec
            .iter()
            .position(|e| e.entry == 1)
            .expect("rec edge at entry 1 must exist");
        assert_eq!(
            rel_llm.rec[ri].by,
            vec!["marcus".to_string()],
            "RecEdge.by must now contain the LLM-resolved recommender"
        );
        assert!(
            rel_llm
                .rec_by_person
                .get("marcus")
                .is_some_and(|v| v.contains(&ri)),
            "rec_by_person must index the upgraded edge"
        );

        // Ledger gained an M_LLM_REC derivation for this upgrade.
        assert!(
            rel_llm
                .ledger()
                .records()
                .iter()
                .any(|d| d.method.id == M_LLM_REC && d.method.version == 1),
            "expected an M_LLM_REC derivation"
        );
    }

    #[test]
    fn index_with_artifacts_is_deterministic() {
        let corpus = two_chain_corpus();
        let artifacts = vec![LlmExtraction {
            entry: 1,
            resolved_by: vec!["marcus".to_string()],
            model_version: 1,
        }];
        let query = "What book was recommended by the person Anna introduced me to?";

        let mut r1 = RelationRetriever::new();
        r1.index_with_artifacts(&corpus, &artifacts);
        let out1 = r1.search(query, 4);

        let mut r2 = RelationRetriever::new();
        r2.index_with_artifacts(&corpus, &artifacts);
        let out2 = r2.search(query, 4);

        assert_eq!(
            out1, out2,
            "same corpus + artifacts must yield identical search output"
        );

        // Idempotent: applying the same artifacts again (fresh index + apply
        // twice via two separate calls is not the API, so just re-run on a
        // fresh instance) yields the same edge state.
        assert_eq!(
            r1.rec[r1.rec.iter().position(|e| e.entry == 1).unwrap()].by,
            r2.rec[r2.rec.iter().position(|e| e.entry == 1).unwrap()].by
        );
    }

    // --- Stage 3C: context assembler ----------------------------------------

    #[test]
    fn assembler_includes_answer_bearing_entry_within_budget() {
        let corpus = two_hop_corpus();
        let query = "What book did the person Anna introduced me to recommend?";

        let mut graph = GraphRetriever::new();
        graph.index(&corpus);

        let pkg = assemble_context(&graph, &corpus, query, 512);
        assert!(
            pkg.contains_answer("Dune"),
            "expected the assembled package to contain the answer 'Dune': {pkg:?}"
        );
        assert!(pkg.est_tokens <= 512);
        assert!(!pkg.facts.is_empty());
    }

    #[test]
    fn tiny_budget_admits_only_top_fact() {
        let corpus = two_hop_corpus();
        let query = "What book did the person Anna introduced me to recommend?";

        let mut graph = GraphRetriever::new();
        graph.index(&corpus);

        // Budget large enough for exactly one fact (the top entry's est_tokens
        // is well under this), but too small for two.
        let top_id = graph.search(query, 1)[0];
        let top_entry = corpus.entries.iter().find(|e| e.id == top_id).unwrap();
        let one_fact_tokens = est_tokens(&top_entry.date) + est_tokens(&top_entry.text);

        let pkg = assemble_context(&graph, &corpus, query, one_fact_tokens);
        assert_eq!(pkg.facts.len(), 1, "expected exactly the top fact: {pkg:?}");
        assert!(pkg.est_tokens <= one_fact_tokens);
        assert_eq!(pkg.facts[0].entry, top_id);
    }
}
