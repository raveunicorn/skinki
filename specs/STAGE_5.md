# Stage 5 — Insight Engine (keystone, anti-hallucination) (SPEC)

- **Status:** **all three detectors earned + gated; only the live (replayed) LLM
  narrator remains.** The frozen interface, BH-FDR validation, cite-or-silence,
  the structural / temporal / contradiction detectors, the naive contrast, the
  narration replay log (T4), ledger wiring (T5), telemetry + sleep `Job` (T6),
  and `insight-eval --assert-gate` all landed in `crates/skinki-insight` + the
  harness. Round 1 found the planted signal undetectable; **round 2 (D0)** made
  bridges rare (RNG-neutral, V1 golden held); **round 3** added T2–T6 but the
  detectors were recall-only (false-insight 0.78 / 0.57); **round 4** fixed their
  precision. The gate now asserts, on two seeds: structural recall/precision
  1.000, temporal recall 0.800, contradiction recall 1.000, **every detector's
  false-insight 0.000**, apophenia 0, 0 uncited, deterministic. Remaining for
  "done": swap the extractive reference narrator for a live (replayed) LLM
  narrator (Stage 6/7) — the only delegated ticket left.
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

**EARNED now — asserted by `insight-eval --assert-gate` (green in CI, both seeds):**

| Metric | Budget | Measured (seeds 42 & 7) |
| --- | --- | --- |
| Discovery determinism | byte-identical | ✓ |
| **Uncited claims** | **= 0** (hard, `CLAUDE.md`) | 0 (structural) |
| **Apophenia hits (reference)** | **`negative_hits = 0`** | 0 / 5 |
| **Structural insight recall** | **≥ 0.50** | **1.000** |
| **Structural insight precision** | **≥ 0.70** | **1.000** |
| **Temporal recall** | **≥ 0.50** | **0.800** |
| **Contradiction recall** | **≥ 0.80** | **1.000** |
| **False-insight rate (every detector)** | **< 0.05** (hard) | **0.000** |
| Gate has teeth (naive contrast) | naive `negative_hits > 0` | 5 / 5 (precision 0.14–0.19) |

**NOT YET EARNED:**

| Metric | Target | Status |
| --- | --- | --- |
| Live narration (replayed LLM) | replay golden stable | extractive + replay-log reference shipped; a live LLM narrator (Stage 6/7) replaces it behind the same trait |

> The keystone gate is in CI **now** and asserts the anti-hallucination budgets
> (every detector's recall + the hard false-insight/apophenia/0-uncited/
> determinism bars) on two seeds. The `< 0.05` false-insight and `= 0` uncited
> budgets are fixed by `CLAUDE.md` and never negotiable.

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
>
> **Measurement log — round 2 (D0 landed — detection earned).** Bridge entities
> now draw from a dedicated rare name pool (`BRIDGE_PEOPLE`, disjoint from the
> `PEOPLE` distractor pool), scoped to V2 and **RNG-neutral** (the legacy
> `PEOPLE` pick is still consumed and still seeds `bridge_names`, so the V2 RNG
> stream — and the V1 byte-frozen golden — are unchanged; only the bridge
> entities' names + their planted entries differ). A planted bridge is now a
> *rare 2-cluster* entity; an apophenia hub stays a *common 4-cluster* one. The
> reference engine, re-measured (seeds 42 & 7, k=all):
>
> | engine | surfaced | recall | precision | apophenia hits |
> | --- | --- | --- | --- | --- |
> | naive co-mention (contrast) | 27 | 1.000 | 0.19 | 5 / 5 |
> | reference (FDR-validated) | 5 | **1.000** | **1.000** | **0 / 5** |
>
> **Verdict: the StructuralBridge keystone is earned** — full recall, zero false
> insights, zero apophenia, on both seeds, deterministically and with every claim
> cited. `insight-eval --assert-gate` now asserts these budgets (promoted from
> informational). The naive contrast still fails (precision 0.19, all 5 traps),
> proving the FDR/surprise validation — not the corpus — does the work.
>
> **Measurement log — round 3 (T2–T6 landed; structural gated, temporal/
> contradiction recall-only).** PR #6 added the temporal + contradiction
> detectors, the LLM-narration replay log (T4), ledger wiring (T5), and telemetry
> + the sleep `Job` (T6). T4/T5/T6 are sound (round-trip + replay-determinism
> tests; `resident_bytes` ≈ 0.33 MB @5M). But the two new **detectors meet recall
> and miss precision** — they blow the hard false-insight budget:
>
> | detector | recall | precision | false-insight (< 0.05 bar) |
> | --- | --- | --- | --- |
> | temporal (T2) | 0.800 ✓ | 0.22 | **0.78** ✗ |
> | contradiction (T3) | 1.000 ✓ | 0.40 | **0.57** ✗ |
>
> Root causes: T2's binomial-null cross-correlation isn't tight enough for the
> synthetic temporal noise; T3 keys on sentiment cues and sets `p=0, surprise=1`,
> **bypassing `validate`** (so nothing is statistically gated) and ignoring the
> ledger the ticket called for. **Verdict: T2/T3 are recall-only prototypes, kept
> *informational* in `insight-eval` (correctly NOT asserted) and NOT done.**
> Promoting them needs the precision work in the follow-up tickets
> (`HANDOFF_DEEPSEEK.md`). The structural keystone gate is unaffected.
>
> **Measurement log — round 4 (T2/T3 precision rework — promoted to the gate).**
> Diagnosing the false positives turned up three concrete causes, each fixed:
> (1) **substring matching** counted a short name like "rust" inside "trust"/
> "frustrated" (1763 phantom mentions → ubiquitous-entity false leads) → fixed
> with **word-boundary matching**; (2) the temporal **null ignored the trail
> entity's frequency**, so a rare lead aligned with a *common* trail (a person
> mentioned 165×) looked significant → fixed with a **density-corrected binomial
> null** (`p ≈ n_b·(2·tol+1)/max_day`) plus a Bonferroni correction for the
> 91-lag search; (3) the measurement engines **bundled the structural detector**,
> so its insights counted as temporal/contradiction false positives → fixed by
> measuring each detector in isolation. For T3, sentiment cues were **mis-
> attributed** ("rust was a mistake. go is clearly better" flagged *go*) → fixed
> with **name-anchored** stance attribution (the entity must be the cue's subject/
> object), dropping Y-referring cues, and emitting **one entity-level candidate**
> citing all its endorse/regret entries (instead of an all-pairs cross product).
> Re-measured (seeds 42 & 7):
>
> | detector | recall | precision | false-insight |
> | --- | --- | --- | --- |
> | temporal (T2) | 0.800 | 1.000 | **0.000** |
> | contradiction (T3) | 1.000 | 1.000 | **0.000** |
>
> **Verdict: all three detectors clear the keystone bar.** `insight-eval
> --assert-gate` now asserts temporal recall ≥ 0.50 and contradiction recall
> ≥ 0.80, each at false-insight < 0.05, alongside the structural keystone — on
> both seeds. (T3 is an exact name-anchored matcher, deterministic, so its
> `p=0` is sound: there is no multiple-testing search to FDR-correct — and with
> correct attribution it produces zero false positives.)

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
| ✅ **INF**: crate, `InsightInput`, `validate` (BH-FDR), cite-or-silence, reference + naive detectors, `insight-eval --assert-gate` | done | frontier | gate green on 2 seeds |
| ✅ **D0**: rare/unique bridge names (`BRIDGE_PEOPLE`, V2-scoped, RNG-neutral so the V1 golden holds) so a 2-cluster bridge is separable from a 4-cluster hub | done | frontier | reference recall 1.000 ∧ `negative_hits = 0` on 2 seeds |
| ✅ **D1**: surprise/effect + FDR calibration (`ValidationCfg` default: q=0.05, min_surprise=0.60, min_support=2) | done | frontier | recall/precision/false-insight rows promoted to asserted; beats naive (precision 1.00 vs 0.19) |
| ✅ T2: temporal lead/lag detector → `InsightKind::TemporalLead` | done (impl DeepSeek; precision rework frontier) | recall 0.800, precision 1.000, false-insight 0.000 — **gated** |
| ✅ T3: contradiction detector → `InsightKind::Contradiction` | done (impl DeepSeek; precision rework frontier) | recall 1.000, precision 1.000, false-insight 0.000 — **gated** |
| ✅ T4: LLM-narration artifact log (append/replay) + fixture + replay golden | done | DeepSeek | `rebuild(log)` byte-identical (tested) |
| ✅ T5: ledger wiring — `record_insight_derivations` (a `Derivation` per insight) | done | DeepSeek | staleness propagation tested |
| ✅ T6: telemetry (`resident_bytes`) + `InsightJob` (Stage-4 `Job`) | done | DeepSeek | ≈ 0.33 MB @5M; interruptible |

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

- **Real-data validation of this keystone** — everything here is gated on the
  **synthetic** V2 corpus, and the detectors are coupled to its structure (topic
  lexicon, exact templates). Whether the win transfers to real dialogue is the
  explicit next step, instrumented in
  [`STAGE_5B_REAL_INSIGHT.md`](STAGE_5B_REAL_INSIGHT.md). Treat the synthetic
  numbers as "earned on synthetic", not "earned" — Stage 3 is the cautionary tale.
- **Live on-device LLM narration inference** — replayed here (rule 3); the model
  harness lands with Stage 6/7.
- **The retrieval multi-hop gap** (Stage 3 round 4) — a *retrieval* problem; see
  [`STAGE_3B_MULTIHOP.md`](STAGE_3B_MULTIHOP.md). Do not conflate with insight
  discovery.
- **Cross-user / federated insights**, any networked discovery — violates the
  0-bytes-network law.
- **The macOS surfacing UI** — Stage 7. This stage emits cited insights; how a
  human sees them is product work.
