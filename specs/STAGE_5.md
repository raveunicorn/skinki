# Stage 5 — Insight Engine (keystone, anti-hallucination) (SPEC)

- **Status:** **infrastructure built + gated; detection blocked on a measured
  corpus finding (D0).** The frozen interface, the FDR validation core, the
  cite-or-silence enforcement, the apophenia-safe reference detector, the naive
  contrast, and the `insight-eval --assert-gate` instrument all landed in
  `crates/skinki-insight` + the harness and are green. The measurement campaign
  (round 1 below) then surfaced an honest blocker: **on the current V2 corpus the
  planted insight signal is statistically undetectable** because bridge entities
  are not rare. Earning recall requires **D0** (co-design the insight ground
  truth) before a detector can be earned — a clean Law-2 result, not a failure.
- **Owner of the design (frontier/human):** **frontier** — the fairness boundary,
  the FDR core, the cite-or-silence contract, and the apophenia discrimination are
  decided and implemented here. Heavy review on every algorithm-core PR; the gate
  decides correctness, not reviewer taste.
- **Delegatable to (cheaper model):** **no** for D0 (corpus co-design) and the
  surprise/effect calibration once D0 lands. **Yes**, behind the frozen
  interface, for the mechanical impl tickets (T-series below): the temporal /
  contradiction detectors, the LLM-narration replay log, ledger wiring.

> Read [`../AGENTS.md`](../AGENTS.md) first (law) and [`STAGE_3.md`](STAGE_3.md)
> round 4 (the real-text close-out that hands Stage 5 a *substrate*, not a
> retriever). Determinism is law for everything that discovers, scores, or
> selects (rule 2); LLM **narration** is replayable, not bit-deterministic (rule
> 3) — never in a gate. No new deps without approval.

## 0. What Stage 3 hands us (the premise)

Stage 3's close-out proved the graph is not a better *retriever* than a semantic
embedder. What it *is* good at is **structure**: a provenance-bearing graph + the
derivation ledger (hash-pinned premises, deterministic staleness). Stage 5 is
where that pays — insight discovery is a **structural / statistical** problem,
not a ranking one. Law 1 in its purest form: **the intelligence is in the
memory; the model only verbalizes.**

## 1. Hypothesis

**Deterministic discovery over the corpus + ledger, gated by statistical
validation, surfaces genuinely non-obvious, *grounded* connections** —
structural-hole bridges, temporal lead/lag, contradictions — **above an
insight-precision threshold and below a hard false-insight threshold**, while
**staying silent on apophenia traps** (hub entities that co-occur everywhere but
mean nothing). The small model narrates only what survives validation, and only
with citations, so the uncited-claim count is **zero by construction**.

Falsifiable two ways, both observed in round 1: (a) if the naive co-occurrence
detector matches the validated one, the statistics aren't earned; (b) if no
detector can clear the false-insight budget while keeping recall, either the
approach or the **measurement corpus** is wrong. Round 1 found (b)'s second horn:
the corpus, not (yet) the approach.

## 2. Budgets / fitness function (the gate)

Measured by `skinki insight-eval` over V2, whose ground truth plants the needles
(`InsightBridge` with a `surprise` score) and the traps (`NegativeBridge`
apophenia hubs). Discovery + validation are deterministic; narration is replayed
from a checked-in fixture — never inferred in CI. The gate runs on **two
independent seeds** (overfit-resistance: no tuning to one draw).

**EARNED now — asserted by `insight-eval --assert-gate` (green, CI-able):**

| Metric | Budget | How measured |
| --- | --- | --- |
| Discovery determinism | byte-identical | `discover` twice → identical surfaced set |
| **Uncited claims** | **= 0** (hard, `CLAUDE.md`) | every emitted `supporting_entries` non-empty (structural) |
| **Apophenia-safety (reference)** | **`negative_hits = 0`** | `score_insights` on V2 `negative_bridges` |
| Gate has teeth (naive contrast) | naive `negative_hits > 0` | the naive detector *does* fire on traps |

**NOT YET EARNED — reported informational until D0 lands (Stage-5 "done"):**

| Metric | Target | Status (round 1, both seeds) |
| --- | --- | --- |
| Insight recall | **≥ 0.50** | **0.00** — signal undetectable (see round 1) |
| Insight precision | **≥ 0.70** | n/a (nothing surfaced) |
| False-insight rate | **< 0.05** (hard) | n/a |
| Temporal lead/lag recovered | ≥ 0.50 @ ±1 day | detector not built (T-ticket) |
| Contradiction surfaced | ≥ 0.80 | detector not built (T-ticket) |

> The EARNED gate is in CI **now** — it locks the infrastructure (interface, FDR,
> cite-or-silence, apophenia-safety) and fails if any regresses. Promoting the
> recall/precision rows to asserted is **the definition of Stage-5 done**, and is
> blocked on D0. The `< 0.05` false-insight and `= 0` uncited budgets are fixed
> by `CLAUDE.md` and never negotiable.

> **Measurement log — round 1 (infrastructure + the corpus blocker).** The
> reference `StructuralBridgeDetector` profiles each entity's spread over topic
> clusters (a 2-cluster bridge is surprising; a 4-cluster hub is apophenia),
> scores `surprise = concentration / (spread − 1)` + a binomial-null `p_value`,
> and feeds BH-FDR. Measured on V2 (seeds 42 & 7, ~11.5k entries, 5 planted
> insights, 5 traps):
>
> | engine | surfaced | recall | precision | apophenia hits |
> | --- | --- | --- | --- | --- |
> | naive co-mention (contrast) | 22 | **1.000** | 0.14 | **5 / 5** |
> | reference (FDR-validated) | 0 | 0.000 | — | 0 / 5 |
>
> The naive detector finds *every* planted bridge — but also *every* apophenia
> hub and ~12 other false positives (precision 0.14). The validated detector
> rejects the traps (apophenia-safe) but, at any threshold that does so, also
> surfaces nothing: **the planted bridges are statistically indistinguishable
> from hubs.** Diagnosed cause — bridge entities are **not rare**: their names are
> reused as distractors **~160×/11.5k across all 5 clusters**, so the planted
> 2-cluster signal (4–6 entries) is drowned. There is **no surprise/p threshold**
> that separates positives from negatives because, at the entity-mention level,
> they are the same distribution. **Verdict: the instrument works; the V2 insight
> ground-truth is not yet a detectable benchmark.** Unblock = **D0** below.

## 3. Public interface (as built — `crates/skinki-insight`)

`#![forbid(unsafe_code)]`; deps: `serde`, `serde_json`, internal `skinki-corpus`
/ `skinki-eval`. Produces `skinki_eval::DiscoveredInsight` so the existing
`score_insights` keystone metric scores it unchanged.

```rust
pub type CandidateId = u64;
pub enum InsightKind { StructuralBridge, TemporalLead, Contradiction }

pub struct Statistic { pub effect: f64, pub p_value: f64, pub support: u32, pub surprise: f64 }
pub struct InsightCandidate {
    pub id: CandidateId, pub kind: InsightKind,
    pub entities: Vec<EntityId>, pub evidence: Vec<EntryId>,  // evidence non-empty
    pub stat: Statistic, pub claim: String,
}

/// The ONLY view a detector may see — never the planted answer key. `from_corpus`
/// is the single audited seam (drops ground_truth.{insights,negative_bridges,...}).
/// Fairness as a TYPE guarantee, not a convention.
pub struct InsightInput<'a> { pub entries: &'a [Entry], pub vocab: &'a [Entity] }

pub trait Detector { fn name(&self) -> &str; fn propose(&self, input: &InsightInput) -> Vec<InsightCandidate>; }

/// FDR core (frontier-owned, provably correct): support+surprise floors, then
/// Benjamini–Hochberg at `cfg.fdr_q`. Pure, deterministic.
pub fn validate(cands: &[InsightCandidate], cfg: &ValidationCfg) -> Vec<CandidateId>;
pub struct ValidationCfg { pub fdr_q: f64, pub min_surprise: f64, pub min_support: u32 }

/// Cite-or-silence narration (rule 3 — REPLAYABLE). `None` = silence; a returned
/// record MUST cite non-empty ⊆ evidence or `discover` drops it.
pub trait Narrator { fn narrate(&self, c: &InsightCandidate, input: &InsightInput) -> Option<NarratedInsight>; }
pub struct NarratedInsight { pub text: String, pub citations: Vec<EntryId> }

pub struct InsightEngine { /* detectors + cfg + narrator */ }
impl InsightEngine {
    pub fn structural() -> Self;  // reference (FDR-validated) — apophenia-safe
    pub fn naive() -> Self;       // the Law-2 contrast that fails apophenia
    pub fn discover(&self, input: &InsightInput) -> Vec<DiscoveredInsight>; // validated+narrated+cited
}
```

Built-in detectors: `StructuralBridgeDetector` (reference, frontier-owned),
`CoMentionDetector` (naive contrast). `ExtractiveNarrator` is the deterministic
cite-or-silence reference; the live LLM narrator (replayed) is a T-ticket.

## 4. Invariants (must always hold)

- **Discovery + validation fully deterministic** (rule 2): stable sort keys,
  `BTreeMap` only, no `HashMap` iteration order. Same input → identical surfaced
  set + ranking.
- **Narration replayable, not bit-deterministic** (rule 3): live inference writes
  to an append-only artifact log; `rebuild(log)` deterministic; the gate replays a
  fixture, never an inference call.
- **Cite-or-silence is structural:** a narrated record with empty citations, or
  citations ⊄ the candidate's evidence, is dropped before scoring. The `= 0`
  uncited budget cannot regress silently. (Enforced + unit-tested.)
- **Fairness is a type guarantee:** detectors see only `InsightInput`; the answer
  key is unreachable, not merely "don't read it".
- **Apophenia safety is earned by statistics, not heuristics:** surprise/FDR/
  min-support, never raw co-occurrence; proved on `negative_bridges`.
- **Ledger-wired (T-ticket):** each surfaced insight becomes a
  `skinki_ledger::Derivation` (inputs = evidence hashes; method = detector +
  validation `MethodStamp`), so a changed premise flags it stale.
- **Runs offline, never on the query path:** discovery is a Stage-4 `Job`.
- **No `unsafe`** (`#![forbid(unsafe_code)]`).

## 5. Test plan

- **Unit (shipped):** BH-FDR monotone in q; binomial upper-tail basics;
  `discover` deterministic; uncited narrations dropped; reference apophenia-safe
  while naive fires (teeth).
- **After D0:** each detector surfaces its planted type and not distractors; the
  structural detector hits the planted bridges and *not* the hubs; temporal
  recovers `lead→trail` at `lag_days ± 1`; contradiction via the ledger.
- **Property:** FDR controls the empirical false-discovery fraction on a synthetic
  null; `rebuild(narration log)` round-trips.
- **Golden:** fixed-seed corpus → locked surfaced-set hash; fixed narration log
  → locked narrated-output hash.
- **Gate command:** `cargo run --release -p skinki-harness -- insight-eval --assert-gate`

## 6. Task decomposition

Build order is **infrastructure → corpus co-design (D0) → detectors → narration**.
The infrastructure is done; the rest is gated by the instrument it provides.

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| ✅ **INF**: crate, `InsightInput`, `validate` (BH-FDR), cite-or-silence, reference + naive detectors, `insight-eval --assert-gate` | done | frontier | EARNED gate green on 2 seeds |
| **D0**: co-design the insight ground truth so it is *detectable* — give `InsightBridge` entities **unique, rare names** (own pool, not shared with distractors) and bound distractor reuse, so a 2-cluster bridge is separable from a 4-cluster hub. (Touches `skinki-corpus` generation → will move golden hashes; **confirm scope with the human** before changing generation.) | design | **frontier + human** | reference detector reaches recall ≥ 0.50 ∧ `negative_hits = 0` on 2 seeds |
| **D1**: surprise/effect + FDR calibration *after D0* — set `ValidationCfg` so recall ≥ 0.50, precision ≥ 0.70, false-insight < 0.05; beat the naive contrast with a measured margin | design | **frontier** | the informational §2 rows promoted to asserted |
| T2: temporal lead/lag detector (cross-correlation of entity mention series; null = shuffled lags) → `InsightKind::TemporalLead` | impl | DeepSeek (frontier reviews) | recovers planted `TemporalPattern` at `lag_days ± 1`, ≥ 0.50 |
| T3: contradiction detector — adapt `skinki-ledger` staleness output into `DiscoveredInsight` → `InsightKind::Contradiction` | impl | DeepSeek (frontier reviews) | ≥ 0.80 of planted `Contradiction` surfaced |
| T4: LLM-narration artifact log (append/replay) + a checked-in fixture + replay golden; wire as a `Narrator` | impl | DeepSeek | `rebuild(log)` byte-identical twice; gate replays, no inference |
| T5: ledger wiring — a `Derivation` per surfaced insight + a staleness test | impl | DeepSeek (frontier reviews) | changed premise flags exactly its insights |
| T6: telemetry — `resident_bytes` + bytes/candidate to 5M; wrap discovery as a Stage-4 `Job` | impl | DeepSeek | RAM projection in report; interruptible/resumable |

## 7. Definition of done

- [ ] D0 landed; `insight-eval --assert-gate` **promotes** recall ≥ 0.50,
      precision ≥ 0.70, false-insight < 0.05 to asserted (kept `negative_hits = 0`,
      `0` uncited, determinism golden).
- [ ] Temporal + contradiction rows asserted.
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean; CI runs the
      insight gate (replay only).
- [ ] Docs: ROADMAP Stage 5 → done with measured numbers; README honest-status
      row; this spec Status → done.
- [ ] Decision recorded: did deterministic discovery + statistics alone clear the
      keystone (apophenia + false-insight + recall), or did we invent — and by
      what measured margin over the naive contrast.

## 8. Out of scope (deferred)

- **Live on-device LLM narration inference** — replayed here (rule 3); the model
  harness lands with Stage 6/7.
- **The retrieval multi-hop gap** (Stage 3 round 4) — a *retrieval* problem; see
  [`STAGE_3B_MULTIHOP.md`](STAGE_3B_MULTIHOP.md). Do not conflate with insight
  discovery.
- **Cross-user / federated insights**, any networked discovery — violates the
  0-bytes-network law.
- **The macOS surfacing UI** — Stage 7. This stage emits cited insights; how a
  human sees them is product work.
