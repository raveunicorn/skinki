# Stage 3 — Incremental local GraphRAG (SPEC)

- **Status:** in-progress (deterministic tier done + gated; **T7 ledger, T8
  telemetry, T6 replay + D2 selection all done**; live-LLM integration deferred —
  the oracle ceiling shows the tier isn't earned on this corpus)

> **Measurement log — round 1 (co-mention MVP).** A deterministic entity+venue
> **co-mention** graph (1-hop + RRF, `crates/kortex-graph::GraphRetriever`)
> **does not beat BM25**: fused multi-hop recall@10 ties at 0.325; the walk alone
> is *below* BM25 (0.175) — raw co-occurrence floods candidates and the true
> hop-B sinks. The join is *reachable* but not *rankable* by co-mention →
> earned the typed-relation extractor.
>
> **Measurement log — round 2 (typed relations — PASS).**
> `RelationRetriever` extracts `IntroEdge`/`RecEdge` and walks the planted chain
> (person bridge for the precise case; venue + temporal-proximity for the coref
> case), with expansion gated on a query intro/rec cue (no single-hop
> regression). Measured, V2 corpus:
>
> | corpus | multi-hop recall@10 | multi-hop ans@10 |
> | --- | --- | --- |
> | ~11.5k | **0.800** (bm25 0.325) | **0.900** (bm25 0.650) |
> | ~29.6k | **0.422** (bm25 0.172) | **0.656** (bm25 0.219) |
>
> The deterministic tier **clears the gate alone** (recall@10 0.800 ≥ 0.50,
> ans@10 0.900 ≥ 0.60 at default scale) and the relative win *widens* with scale.
> recall@10 falls off at larger N (coref hops sharing a venue) — the residual the
> LLM tier (D2) targets. `graph-eval --assert-gate` is wired and in CI.
>
> **Measurement log — round 3 (LLM tier T6/D2 — measured, NOT earned).** Added
> the replay machinery (`ArtifactLog`), the deterministic selection policy
> (`selects_for_llm_tier`: rec-cue + venue + no person → ambiguous coref), and
> `index_with_artifacts`. Measured the tier's *ceiling* with a ground-truth
> **oracle** (perfect LLM) replayed through the log — no live model:
>
> | corpus | tier-1 share | multi-hop recall@10 lift (rel+llm − relation) |
> | --- | --- | --- |
> | ~11.5k | 0.10% | **−0.125** |
> | ~29.6k | 0.04% | **+0.031** |
>
> Even a *perfect* model doesn't reliably help: resolving coref to a reused
> person name re-indexes the hop under `rec_by_person` and injects cross-chain
> noise the deterministic **venue+temporal** bridge avoided. **Verdict: the LLM
> tier is not earned on this corpus** — a vindication of "intelligence in the
> memory, not the model." The gate prints the lift as informational and does not
> require it; the machinery stays for re-measurement on a regime where the oracle
> ceiling actually pays. Live-LLM integration deferred to Stage 6/7.
- **Owner of the design (frontier/human):** frontier — the graph schema, the
  retrieval algorithm, the tier-0/tier-1 split, the replay contract, and the
  ledger wiring are decided here. Heavy review on every algorithm-core PR.
- **Delegatable to (cheaper model):** **yes** for the impl tickets (gazetteer
  matcher, pattern extractor, graph/CSR plumbing, CLI/gate, golden tests, ledger
  wiring). **No** for the two design tickets marked frontier (selection policy
  calibration; the retrieval ranking core).

> Read [`../../AGENTS.md`](../../AGENTS.md) and the compute arithmetic in
> [`STAGE_3_BUDGET.md`](STAGE_3_BUDGET.md) — extraction is **two-tier by
> construction**; all LLM outputs are **replayable** (rule 3). Determinism is law
> for everything that selects/structures (rule 2). No new deps without approval.

## 1. Hypothesis

A **knowledge graph built incrementally from memory units** — entities and
typed, provenance-bearing relations — lets a small model answer **multi-hop**
questions that pure lexical or pure vector retrieval cannot, by **traversing the
join** instead of hoping it appears in one chunk. Concretely: BM25 scores
multi-hop **recall@10 ≈ 0.075 / answer-in-top-10 ≈ 0.30** on the V2 corpus
because ~40% of second hops drop the entity name and join only through a
**venue anchor**. A graph that links entries sharing an entity/venue and walks
1–2 hops should recover most of that gap **within the M1 Air budget**, with the
**deterministic tier doing the bulk of the work** and the LLM tier earning its
keep only on the ambiguous-coreference minority.

Falsifiable: if a deterministic graph can't beat BM25 multi-hop by a wide,
measured margin, the graph approach is wrong for this corpus; if it can, the LLM
tier must *measurably* lift the residual (coref hops) to justify its cost.

## 2. Budgets / fitness function (the gate)

Measured by `kortex graph-eval` over the V2 corpus (deterministic; the LLM tier
is **replayed** from a checked-in artifact-log fixture — never inferred in CI).

| Metric | Budget | How measured |
| --- | --- | --- |
| **Multi-hop recall@10** | **≥ 0.50** (stretch 0.70) | `recall_at_k` over `ground_truth.multi_hop`, vs BM25 0.075 |
| **Multi-hop answer-in-top-10** | **≥ 0.60** | `answer_in_entries` aggregate, vs BM25 0.30 |
| Single-hop recall@10 (no regression) | **≥ BM25** (0.138 floor) | `recall` set; graph must not hurt plain recall |
| Tier-0 determinism | byte-identical | same corpus → identical graph (golden hash) |
| Replay determinism | byte-identical | `rebuild(log)` twice → identical graph (rule 3) |
| Tier-1 backfill share | **≤ 5%** of units | selection policy reports it on a 5M-projected count |
| Graph RAM @5M | **≤ ~120 MB** (within the 250 idle budget alongside the index) | projected from bytes/(node+edge) |
| Extraction never runs in the gate | enforced | gate loads replayed artifacts only |

`graph-eval --assert-gate` exits non-zero on any miss. The two recall numbers are
the headline "impossible task"; the rest guard the laws and the budget.

> The ≥0.50 / ≥0.60 bars are the *first* honest target (≫ BM25). After T4 lands,
> the deterministic-tier measurement may raise them (Stage 1 likewise mapped its
> own ceiling before the gate was finalized) — never lower them without sign-off.

## 3. Public interface

New crate **`kortex-graph`** (`#![forbid(unsafe_code)]`; deps: `serde`,
`serde_json`, internal `kortex-corpus`/`-store`/`-eval`/`-ledger`).

```rust
// --- Extraction (tiered; tier is DATA, not code) -------------------------
pub type NodeId = u32;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tier { Deterministic, Llm }

/// One unit's extracted content: the entities it mentions and the typed
/// relations it asserts, each with the unit as provenance.
pub struct UnitExtraction {
    pub unit: UnitId,
    pub entry: EntryId,                 // for harness scoring / provenance
    pub entities: Vec<EntityRef>,       // name + kind (kind may be Unknown)
    pub relations: Vec<RelationAssert>, // typed edges this unit asserts
    pub tier: Tier,
}

pub struct EntityRef { pub name: String, pub kind: EntityKind }

/// A typed relation between two named entities, asserted by one unit. The
/// venue/anchor is first-class because it carries the multi-hop join.
pub enum Relation {
    IntroducedAt { person_a: String, person_b: String, venue: String },
    RecommendedBy { item: String, by: Option<String>, venue: Option<String> },
    CoMention,                          // generic co-occurrence in a unit
    TemporalLead { lead: String, trail: String },
}
pub struct RelationAssert { pub rel: Relation }

/// The extractor contract. Tier-0 is deterministic over EVERY unit; tier-1 is
/// the REPLAY of a checked-in artifact log (never live inference in tests/gate).
pub trait Extractor {
    fn extract(&self, unit: UnitId, entry: EntryId, text: &str) -> UnitExtraction;
}

/// Deterministic selection: which units are worth the LLM tier. Pure function of
/// unit features (novel-entity candidates, ambiguous coref markers, topic
/// entropy). Seeded; no wall clock; testable. Reports the selected share.
pub trait SelectionPolicy {
    fn select(&self, f: &UnitFeatures) -> bool;
}
pub struct UnitFeatures { /* deterministic, derived from text + prior tiers */ }

// --- Graph + retrieval ---------------------------------------------------
pub struct KnowledgeGraph { /* CSR adjacency, typed edges, provenance */ }

impl KnowledgeGraph {
    /// Build (or incrementally extend) the graph from a stream of extractions.
    /// Deterministic in input order; idempotent on replay.
    pub fn build(extractions: &[UnitExtraction]) -> Self;
    pub fn extend(&mut self, more: &[UnitExtraction]);

    /// Multi-hop entry retrieval: seed nodes from the query, expand ≤2 hops
    /// along typed edges (venue/person/item), rank entries by joined evidence.
    pub fn search_entries(&self, query: &str, k: usize) -> Vec<EntryId>;

    pub fn resident_bytes(&self) -> usize;
}

/// Plugs the graph into the Stage-0 eval harness. May optionally seed from the
/// Stage-1 vector index (hybrid) behind the same trait.
pub struct GraphRetriever { /* graph + gazetteer + optional vector seed */ }
// impl kortex_eval::RetrievalSystem for GraphRetriever { name/index/search/... }

// --- Replay (AGENTS rule 3) ----------------------------------------------
/// Append-only artifact log of LLM-tier outputs. `rebuild` is byte-deterministic.
pub struct ArtifactLog { /* JSON-lines of LlmExtraction { unit, entities, relations, model_version } */ }
impl ArtifactLog {
    pub fn append(&mut self, x: &LlmExtraction) -> std::io::Result<()>;
    pub fn replay(path: &std::path::Path) -> std::io::Result<Vec<UnitExtraction>>;
}
```

## 4. Invariants (must always hold)

- **Tier-0 is fully deterministic** (rule 2): same corpus → byte-identical graph.
  No `HashMap` iteration in anything that affects edge order; sort by `(NodeId,
  Relation, EntryId)`.
- **Tier-1 is replayable, not bit-deterministic** (rule 3): live inference writes
  `LlmExtraction` records to the `ArtifactLog`; `rebuild(log)` is deterministic;
  the gate consumes a **checked-in fixture log**, never an inference call.
- **Selection is a pure function** of `UnitFeatures` (rule 2): seeded, testable;
  it decides *which* units go to tier-1 but never *what* tier-1 returns.
- **Provenance preserved end to end:** every edge carries the `EntryId`(s) /
  `UnitId`(s) that assert it; every retrieved entry is traceable to source bytes.
- **Ledger-wired:** every edge is recorded as a `kortex_ledger::Derivation`
  (inputs = the asserting units' content hashes; method = the extractor's
  `MethodStamp{ id, version }`), so re-extraction is incremental and staleness
  propagates for free (a changed unit / bumped extractor version flags its
  edges). The graph is `rebuild(ledger + artifact log)`-deterministic.
- **No `unsafe`.** New crate keeps `#![forbid(unsafe_code)]`.

## 5. Test plan

- **Unit:** gazetteer matches known entity names (case/punct-robust); each
  `Relation` pattern fires on its templated surface forms and *not* on
  distractors; venue-anchored join links hop-A and hop-B entries.
- **Property:** `build` is order-deterministic; `extend` then `build` from
  scratch yield the same graph (incremental == batch); `replay(log)` round-trips.
- **Golden:** a fixed small corpus → locked graph hash; a fixed artifact-log
  fixture → locked replayed graph hash.
- **Metric:** `graph-eval` reproduces multi-hop recall@10 / answer-in-top-10 and
  asserts the §2 budgets; prints the BM25 baseline alongside for contrast.
- **Gate command:**
  `cargo run --release -p kortex-harness -- graph-eval --assert-gate`

## 6. Task decomposition

Build order is **deterministic-tier first, measure, then earn the LLM tier** —
the project's Law 2 applied inside the stage.

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| **D1**: retrieval ranking core (seed → ≤2-hop expansion → entry scoring; PPR vs plain traversal decided by measurement) | design | **frontier** | rationale + measured multi-hop recall; algorithm doc |
| **D2**: tier-1 selection policy + its calibration (which features, what share) | design | **frontier** | ≤5% backfill share with measured residual lift |
| T1: `kortex-graph` crate skeleton + types/traits above | impl | sonnet | builds; `forbid(unsafe)`; trait objects work |
| T2: deterministic gazetteer NER (entity-name matcher over `entry.text`) | impl | sonnet | unit tests: matches names, robust to case/punct |
| T3: deterministic relation pattern extractor (the 4 `Relation`s) | impl | sonnet | each pattern's golden surface forms fire; distractors don't |
| T4: `KnowledgeGraph` (CSR adjacency, typed edges, provenance) + `build`/`extend` + `search_entries` per D1 | impl | sonnet (frontier reviews D1 core) | order-deterministic golden; multi-hop gate |
| T5: `GraphRetriever` impl `RetrievalSystem`; wire into `graph-eval` CLI + `--assert-gate` | impl | sonnet | gate runs, prints BM25 contrast |
| T6: `ArtifactLog` append/replay + a checked-in fixture log + replay golden | impl | sonnet | `rebuild(log)` byte-identical twice |
| T7: ledger wiring — emit a `Derivation` per edge; an incremental re-extract test showing staleness propagation | impl | sonnet (frontier reviews) | changed unit flags exactly its edges |
| T8: telemetry — graph `resident_bytes` + bytes/(node+edge) projection to 5M; build-cost timing | impl | sonnet | RAM projection in report; within budget |

## 7. Definition of done

- [ ] `graph-eval --assert-gate` green; multi-hop recall@10 ≥ 0.50 and
      answer-in-top-10 ≥ 0.60, single-hop no regression, both determinism golden
      hashes stable, tier-1 share ≤ 5%, graph RAM@5M within budget.
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [ ] CI: add a `graph gate` step (replay only; no inference).
- [ ] Docs: ROADMAP Stage 3 → done with measured numbers; README graph section;
      this spec Status → done.
- [ ] Decision recorded: did the deterministic tier alone clear the gate, or did
      the LLM tier earn its cost — with the measured residual lift either way.

## 8. Out of scope (deferred)

- **Communities (Leiden) + RAPTOR hierarchical summaries** and **PPR at scale** —
  Stage **3B** once plain traversal's ceiling is measured (D1 may pull a thin PPR
  in if traversal plateaus, but full community/summary machinery is deferred).
- **The context assembler** (budgeted, pre-joined package + context-sufficiency
  metric) — Stage **3C**; it consumes this graph but is a separate fitness fn.
- **Insight discovery over the graph** — Stage **5** (the keystone); this stage
  only builds the substrate insights will run on.
- **Real on-device LLM inference / EmbeddingGemma wiring** — the engine consumes
  a *replayed* artifact log here; live inference + the model harness land with
  Stage 6/7. This stage proves the graph and the replay contract, not the model.
