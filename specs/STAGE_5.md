# Stage 5 — Insight Engine (keystone, anti-hallucination) (SPEC)

- **Status:** **draft** — design tickets **D1–D3 open** (frontier); impl tickets
  ready behind them. This is the project's "soul": **frontier-owned, heavy review
  on every algorithm-core PR, no blind hand-off** (per `CLAUDE.md`). Mechanical
  plumbing may be delegated *only* behind a frozen interface, exactly as Stage 3
  did.
- **Owner of the design (frontier/human):** **frontier** — the surprise/effect
  metric, the FDR calibration, the per-detector candidate generation, and the
  cite-or-silence contract are decided here. These are the anti-hallucination
  core; reviewer taste does not get a vote, the gate does.
- **Delegatable to (cheaper model):** **no** for D1–D3 (the discovery + validation
  + narration cores). **Yes**, narrowly, for the mechanical impl tickets (crate
  skeleton, BH-FDR arithmetic, `ArtifactLog` replay, ledger wiring, CLI/gate,
  golden tests) once D1–D3 are frozen.

> Read [`../AGENTS.md`](../AGENTS.md) first (it is law) and
> [`STAGE_3.md`](STAGE_3.md) round 4 (the real-text close-out that hands Stage 5
> a *substrate*, not a retriever). Determinism is law for everything that
> discovers, scores, or selects (rule 2); the LLM **narration** is replayable,
> not bit-deterministic (rule 3) — it never runs in a gate. No new deps without
> approval.

## 0. What Stage 3 hands us (the premise)

The real-text close-out proved the graph is **not** a better retriever than a
semantic embedder. What it *is* good at is **structure**: a provenance-bearing
knowledge graph + the derivation ledger (hash-pinned premises, deterministic
staleness propagation). Stage 5 is where that structure finally pays: insight
discovery is a **structural / statistical** problem, not a ranking problem, so
it plays to the substrate's actual strength. This is Law 1 in its purest form —
**the intelligence is in the memory; the model only verbalizes.**

## 1. Hypothesis

**Deterministic discovery over the graph + ledger, gated by statistical
validation, surfaces genuinely non-obvious, *grounded* connections** —
structural-hole bridges, temporal lead/lag patterns, and contradictions — **above
an insight-precision threshold and below a hard false-insight threshold**, while
**staying silent on apophenia traps** (hub entities that co-occur everywhere but
mean nothing). The small model **narrates only what survives validation, and only
with citations** ("cite-or-silence"), so the count of uncited claims is **zero by
construction**.

Falsifiable two ways. (a) If a naive co-occurrence detector matches the
statistical one on the planted bridges, the "surprise/FDR" machinery is not
earned — keep the naive one. (b) If the validated detector cannot beat the false
-insight budget on the V2 apophenia traps (`negative_bridges`), the discovery
approach is wrong and must be redesigned, not shipped. Either is a clean Law-2
verdict, recorded.

## 2. Budgets / fitness function (the gate)

Measured by `skinki insight-eval` over the V2 synthetic corpus, whose
ground truth plants exactly the needles this stage must find (`InsightBridge`
with a `surprise` score) and the traps it must refuse (`NegativeBridge`
apophenia hubs). Discovery + validation are deterministic; **narration is
replayed from a checked-in artifact-log fixture — never inferred in CI.**

| Metric | Budget | How measured |
| --- | --- | --- |
| **False-insight rate** | **< 0.05** (hard, from `CLAUDE.md`) | `score_insights().false_insight_rate` |
| **Certified-false hits (apophenia)** | **= 0** | `score_insights().negative_hits` on V2 `negative_bridges` — the keystone |
| **Uncited claims** | **= 0** (hard, from `CLAUDE.md`) | every `DiscoveredInsight.supporting_entries` non-empty ∧ ⊆ corpus; enforced structurally |
| **Insight precision** | **≥ 0.70** (stretch 0.85) | `score_insights().precision` |
| **Insight recall** (planted bridges found) | **≥ 0.50** (first honest target) | `score_insights().recall` over `ground_truth.insights` |
| Temporal lead/lag recovered | **≥ 0.50** at lag within ±1 planted day | per `TemporalPattern` (leading→trailing, `lag_days`) |
| Contradiction surfaced | **≥ 0.80** | per `Contradiction` via the ledger's before/after staleness flag |
| Discovery determinism | byte-identical | same corpus → identical surfaced set + ranking (golden hash) |
| Narration replay determinism | byte-identical | `rebuild(narration log)` twice → identical output (rule 3) |
| Discovery RAM/cost @5M | within idle budget; completes in one sleep window | projected from bytes/(candidate) + per-detector cost; runs as a Stage-4 `Job`, never on the query path |

`insight-eval --assert-gate` exits non-zero on any miss. The first two rows are
the anti-hallucination keystone; precision/recall are the "find real insights"
headline; the rest guard the laws and the budget.

> The ≥0.70 / ≥0.50 bars are the *first* honest target. As Stage 1 and Stage 3
> did, the **first measurement may raise them** — never lower them without
> human sign-off. The `< 0.05` false-insight and `= 0` uncited budgets are
> fixed by `CLAUDE.md` and are **not** negotiable.

## 3. Public interface

New crate **`skinki-insight`** (`#![forbid(unsafe_code)]`; deps: `serde`,
`serde_json`, internal `skinki-corpus` / `-graph` / `-ledger` / `-eval`). It
produces `skinki_eval::DiscoveredInsight` so the existing `score_insights`
keystone metric scores it unchanged.

```rust
use skinki_corpus::{Corpus, EntityId, EntryId};
use skinki_eval::DiscoveredInsight;
use skinki_graph::KnowledgeGraph;
use skinki_ledger::Ledger;

pub type CandidateId = u64;

/// The family of structural/statistical pattern a detector proposes. Each maps
/// to a planted ground-truth type the gate scores against.
pub enum InsightKind { StructuralBridge, TemporalLead, Contradiction, Changepoint }

/// A raw, PRE-validation candidate. `evidence` is provenance and MUST be
/// non-empty — a candidate with no citable support never enters the pipeline.
pub struct InsightCandidate {
    pub id: CandidateId,
    pub kind: InsightKind,
    pub entities: Vec<EntityId>,
    pub evidence: Vec<EntryId>,   // non-empty invariant; ⊆ corpus
    pub stat: Statistic,
}

/// The per-candidate test result, BEFORE multiple-hypothesis correction.
/// `surprise` is the apophenia discriminator (rarity / PMI-lift), `support` the
/// minimum-evidence guard, `p_value` the input to BH-FDR.
pub struct Statistic { pub effect: f64, pub p_value: f64, pub support: u32, pub surprise: f64 }

/// A detector proposes candidates DETERMINISTICALLY from the substrate. Pure
/// function of (graph, ledger, corpus); no wall clock, no HashMap iteration order.
pub trait Detector {
    fn name(&self) -> &str;
    fn propose(&self, g: &KnowledgeGraph, l: &Ledger, c: &Corpus) -> Vec<InsightCandidate>;
}

/// Statistical gate: Benjamini–Hochberg FDR over `p_value` + an effect-size /
/// `surprise` floor + a min-`support` guard. Pure and deterministic; returns the
/// accepted ids in a stable (effect-desc, id-asc) order. This is where apophenia
/// hubs are rejected — they have high co-occurrence but low surprise.
pub fn validate(cands: &[InsightCandidate], cfg: &ValidationCfg) -> Vec<CandidateId>;
pub struct ValidationCfg { pub fdr_q: f64, pub min_surprise: f64, pub min_support: u32 }

/// Cite-or-silence narration (rule 3 — REPLAYABLE). Live inference appends a
/// record to an `ArtifactLog`; `rebuild(log)` is byte-deterministic; the gate
/// consumes a checked-in fixture log, never an inference call. `None` = the model
/// chose silence (low confidence / can't ground the claim).
pub trait Narrator {
    fn narrate(&self, c: &InsightCandidate, corpus: &Corpus) -> Option<NarratedInsight>;
}
/// `citations` MUST be non-empty and ⊆ `c.evidence`; a record violating this is
/// DROPPED before scoring — the `= 0` uncited budget is a check, not a hope.
pub struct NarratedInsight { pub text: String, pub citations: Vec<EntryId> }

pub struct InsightEngine { /* detectors + ValidationCfg + Narrator + ArtifactLog */ }
impl InsightEngine {
    /// Full pipeline: propose → validate (FDR) → narrate (cite-or-silence) →
    /// emit. Every emitted insight is validated, narrated, and cited; each is
    /// also recorded as a `skinki_ledger::Derivation` so a changed premise flags
    /// it stale. Deterministic given the substrate + a fixed narration log.
    pub fn discover(&self, g: &KnowledgeGraph, l: &Ledger, c: &Corpus) -> Vec<DiscoveredInsight>;
    pub fn resident_bytes(&self) -> usize;
}
```

## 4. Invariants (must always hold)

- **Discovery + validation are fully deterministic** (rule 2): candidate
  generation, statistics, FDR, and selection sort by stable keys (`(effect
  desc, CandidateId asc)`); no `HashMap` iteration order anywhere that affects
  the surfaced set. Same corpus → byte-identical surfaced set **and** ranking
  (golden hash).
- **Narration is replayable, not bit-deterministic** (rule 3): live inference
  writes `NarrationRecord`s to an append-only `ArtifactLog`; `rebuild(log)` is
  deterministic; the gate replays a **checked-in fixture log**, never an
  inference call.
- **Cite-or-silence is structural, not aspirational:** a `NarratedInsight` with
  empty citations, or citations ⊄ the candidate's evidence, is **dropped before
  scoring**. The `= 0` uncited-claims budget therefore cannot regress silently.
- **Provenance end to end:** every surfaced insight's `supporting_entries` are
  real corpus entries, traceable to source bytes; the bridge entity (if any) is
  a real `EntityId`.
- **Apophenia safety is earned by statistics, not heuristics:** ranking is by
  **surprise/rarity + FDR-controlled significance + min-support**, never raw
  co-occurrence. The gate proves this on `negative_bridges` (`negative_hits = 0`).
- **Ledger-wired** (reuse Stage 3 / the Derivation Ledger): each surfaced insight
  is a `Derivation` (inputs = evidence content hashes; method = the detector +
  validation `MethodStamp{ id, version }`). A changed premise or a bumped
  detector version flags exactly the dependent insights stale — staleness
  propagation comes for free, and `discover` is `rebuild(graph + ledger +
  narration log)`-deterministic.
- **Runs offline, never on the query path:** discovery is a Stage-4 `Job`
  (interruptible, resumable, idle+power-gated). The query path only *reads*
  already-surfaced, already-cited insights.
- **No `unsafe`.** New crate keeps `#![forbid(unsafe_code)]`.

## 5. Test plan

- **Unit (per detector):** the structural-bridge detector surfaces each
  `InsightBridge` entity and **stays silent on every `NegativeBridge` hub**; the
  temporal detector recovers `leading→trailing` at the planted `lag_days` within
  ±1 day and rejects spurious lags; the contradiction detector flags
  `entry_before`/`entry_after` via the ledger and not stable beliefs.
- **Property:** `validate` is monotone in `fdr_q` (looser q ⇒ superset); BH-FDR
  controls the empirical false-discovery fraction on a synthetic null; discovery
  is order-deterministic (shuffle input unit order ⇒ identical surfaced set);
  `rebuild(narration log)` round-trips.
- **Golden:** a fixed-seed corpus → locked surfaced-insight set hash; a fixed
  narration artifact-log fixture → locked narrated-output hash.
- **Metric / contrast:** `insight-eval` reports precision / recall /
  false_insight_rate / negative_hits and asserts §2; it prints the **naive
  co-occurrence detector** (the one that *does* fire on apophenia hubs) beside
  the validated engine — the measured proof the statistics are earned (Law 2).
- **Gate command:**
  `cargo run --release -p skinki-harness -- insight-eval --assert-gate`

## 6. Task decomposition

Build order is **deterministic discovery + statistical validation first, measure
against apophenia, then add narration** — Law 2 inside the stage. The narration
layer cannot mask a discovery that doesn't clear `negative_hits = 0`.

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| **D1**: the surprise/effect metric + BH-FDR calibration (the apophenia discriminator) | design | **frontier** | `negative_hits = 0` ∧ false-insight `< 0.05` on V2, with measured separation from the naive detector |
| **D2**: per-detector candidate generation (structural-hole over the graph; temporal lag estimation; contradiction via ledger) + ranking/selection | design | **frontier** | each detector clears its §2 recall row; algorithm doc + rationale |
| **D3**: the cite-or-silence contract + calibration (when to stay silent; how citations are verified ⊆ evidence) | design | **frontier** | `= 0` uncited by construction; silence rate measured, not guessed |
| T1: `skinki-insight` crate skeleton + the types/traits above | impl | sonnet | builds; `forbid(unsafe)`; traits object-safe where needed |
| T2: BH-FDR + min-surprise + min-support `validate()` | impl | sonnet | property tests: monotonicity, FDR control on synthetic null |
| T3: detector impls behind D2 (structural / temporal / contradiction) | impl | sonnet (frontier reviews D2 core) | per-detector unit + golden tests |
| T4: `ArtifactLog` for narration + a checked-in fixture log + replay golden | impl | sonnet | `rebuild(log)` byte-identical twice |
| T5: ledger wiring — a `Derivation` per surfaced insight + a staleness test | impl | sonnet (frontier reviews) | changed premise flags exactly its insights |
| T6: `insight-eval` CLI + `--assert-gate` + the naive-detector contrast column | impl | sonnet | gate runs, prints contrast, exits non-zero on any §2 miss |
| T7: telemetry — `resident_bytes` + bytes/candidate projection to 5M; wrap discovery as a Stage-4 `Job` | impl | sonnet | RAM projection in report; runs interruptible/resumable |

## 7. Definition of done

- [ ] `insight-eval --assert-gate` green: false-insight `< 0.05`, `negative_hits
      = 0`, uncited `= 0`, precision `≥ 0.70`, recall `≥ 0.50`, temporal/
      contradiction rows met, discovery + narration-replay determinism goldens
      stable, discovery RAM@5M within budget.
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [ ] CI: add an `insight gate` step (replay only; **no inference** in CI).
- [ ] Docs: ROADMAP Stage 5 → done with measured numbers; README "honest status"
      row; this spec Status → done.
- [ ] Decision recorded: did deterministic discovery + statistics alone clear the
      keystone (apophenia + false-insight), or did we have to invent — and did
      the naive detector get beaten, with the measured margin either way.

## 8. Out of scope (deferred)

- **Live on-device LLM narration inference** — narration is *replayed* from an
  artifact log here (rule 3); the live model harness lands with Stage 6/7. This
  stage proves discovery + validation + the cite-or-silence contract, not the
  model.
- **The retrieval multi-hop gap** (Stage 3 round 4: EmbeddingGemma still misses
  ~71% of evidence turns). That is a *retrieval* problem, not an insight one; the
  earned candidates are **query-focused summarization** and **iterative/multi-step
  retrieval**, **not** the LLM-entity graph. Tracked separately; do not conflate
  it with insight discovery.
- **Cross-user / federated insights**, and any networked discovery — violates the
  0-bytes-network law.
- **The macOS pop-up / mascot surfacing UI** — Stage 7. This stage emits cited
  insights; how they are shown to a human is product work.
- **Changepoint detection** beyond a minimal stub — `InsightKind::Changepoint`
  is reserved in the interface but only pursued if D2 measurement shows the
  planted temporal patterns need it; otherwise deferred.
