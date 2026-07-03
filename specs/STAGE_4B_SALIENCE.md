# Stage 4B — Salience, reinforcement, and forgetting-as-compaction (SPEC)

> Batch 7 of the 2026-07 review (`REVIEW_FRONTIER_2026_07.md` §6). The
> substrate stores, compresses, links, and invalidates — but does not
> **prioritize** or **forget**. Stage 3's test list named
> "salience/reconsolidation (use counts + recency feed ranking)" and it was
> never built. For a 10-year corpus this is the difference between a memory
> and an archive. The skinki-shaped version: usage is an append-only event
> log; the ranking formula is pinned and **versioned via the ledger's own
> `MethodStamp`** (the ledger eats its own dogfood); every behavior is gated
> by a deterministic protocol test, no ground-truth changes needed.

- **Status:** draft — build **after** 1B/6B (it modulates their retriever and
  write path); last in the 2026-07 batch.
- **Owner of the design (frontier/human):** frontier — the salience formula
  and its failure modes (feedback loops, rich-get-richer) are the subtle part;
  locked below with an explicit anti-runaway bound.
- **Delegatable to (cheaper model):** **yes** for T1–T4 behind the frozen
  formula; D1 (formula verdict after measurement) stays frontier.

> Read [`../AGENTS.md`](../AGENTS.md). Rule 2: *logical* time only — the
> usage log carries caller-supplied dates / monotonic tick counters, never
> wall clock inside logic. 0 network; no new deps.

## 1. Hypothesis

A deterministic salience signal — reinforcement on retrieval-use with
exponential recency decay, bounded so semantic relevance always dominates —
**improves ranking on repeated-topic workloads without degrading one-shot
recall**: on the protocol tests below, reinforced entries outrank
equal-similarity unreinforced ones for related queries, decayed entries fall
back, and every existing retrieval gate stays green (the floor: salience must
never *hurt*). Falsifiable by the same protocol: if the no-regression rows
fail at any α that produces a measurable reinforcement effect, salience as
designed is net-negative — a recorded Law-2 verdict.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Reinforcement | after `R = 5` recorded uses of entry e, a related query ranks e above an equal-similarity twin in ≥ 95% of protocol trials | new protocol test (deterministic, seeded) |
| Decay | after `Δt = 180` logical days with no use, the twin ordering reverts | protocol test |
| **No regression (hard)** | all existing retrieval gates unchanged: `graph-eval --assert-gate`, LongMemEval per-type ≥ pre-salience numbers with an empty usage log | gates + eval rerun |
| Bounded influence | with any usage log, `rank_score ≤ (1 + α_max) × semantic_score`, α_max = 0.5 | property test |
| Determinism | same (index, usage log, query) → identical ranking | golden |
| Usage-log cost | ≤ 16 B/event on disk; replay of 1M events ≤ 1 s | bench report |
| Near-dup consolidation (T4) | on planted paraphrase-duplicates: ≥ 90% merged, 0 false merges across different GT tags | new corpus-protocol test |

## 3. Public interface

```rust
// skinki-store — usage events ride the existing append-only machinery.
/// (entry, logical_day) — one record per retrieval hit the caller chose to
/// reinforce (the MCP server reinforces the top-k it returned; explicit
/// `remember` premises reinforce their cited entries).
pub struct UseEvent { pub entry: EntryId, pub day: u32 }
// append via the existing segmented store; replay in file order.

// skinki-baseline (SemanticRetriever) — the pinned formula, v1:
/// salience(e, now) = Σ_{uses u of e} exp(-(now - u.day) / TAU)   TAU = 90.0
/// rank_score(e)    = semantic(e) × (1 + α × min(salience(e), S_CAP) / S_CAP)
///                    α = 0.25, S_CAP = 5.0
/// MethodStamp{ id: M_SALIENCE, version: 1 } — bumping the version flags all
/// cached rankings stale through the ledger, like any other method change.
pub struct SalienceCfg { pub alpha: f64, pub tau_days: f64, pub cap: f64 }
impl SemanticRetriever {
    pub fn with_salience(self, log: &[UseEvent], now_day: u32, cfg: SalienceCfg) -> Self;
}
```

Design notes, locked:

- **Anti-runaway:** the multiplicative bonus is capped (`S_CAP`) and bounded
  (`α`), so no amount of self-reinforcement can promote an irrelevant entry —
  semantic similarity keeps veto power. This kills the rich-get-richer loop by
  construction, not by tuning.
- **Reinforce on *use*, not on *retrieval*:** the MCP server records a
  `UseEvent` only for hits actually returned to the agent (top-k), and
  `remember`-premises count double (weight 2 as two events) — cited-as-premise
  is the strongest use signal the system sees.
- **Forgetting = demotion, never deletion:** L0 is sacred. T5 (optional)
  demotes entries with salience 0 and age > 2 years to a cold IVF tier
  (excluded from `nprobe` unless the hot pass under-fills k). Measured by RAM
  and recall deltas; adopt only if recall holds.

Near-dup consolidation (T4) — a Stage-4 `Job`:

```rust
/// Offline sleep job: within a ±W-day window, entries with cosine ≥ TAU_DUP
/// (Stage-1B embeddings; TAU_DUP = 0.92, W = 30) merge into the earliest
/// entry as canonical; the merged set becomes a provenance union (all ids
/// remain resolvable; retrieval returns the canonical + cites the union).
pub struct ConsolidateJob { /* embedder + store + cursor */ }
```

## 4. Invariants (must always hold)

- Empty usage log ⇒ ranking byte-identical to pre-salience (the formula's
  `salience = 0` fixed point) — this is what keeps every old gate green.
- Usage log is append-only, replayed in order; salience is a pure function of
  (log, now_day).
- Consolidation never deletes L0 bytes; merges are new derivations
  (`MethodStamp{M_CONSOLIDATE}`) so the ledger can invalidate a merge if a
  member entry changes.
- Rule 2: no wall clock; `now_day` is always caller-supplied.
- No `unsafe`; no new deps.

## 5. Test plan

- **Protocol (the gate):** build a synthetic corpus with twin entries (equal
  embeddings by construction: identical text, different ids); record R uses
  of one twin; assert ordering flips; advance `now_day` by Δt; assert revert.
  Run over 20 seeded twin pairs; ≥ 95% row.
- **Property:** bounded-influence inequality under random logs (seeded);
  empty-log fixed point (byte-identical rankings).
- **Golden:** ranking hash for a fixed (corpus, log, queries) triple.
- **Consolidation protocol:** plant paraphrase duplicates via the V3 pipeline
  (Stage 0B machinery) on a small corpus; assert merge coverage + 0 cross-tag
  false merges + provenance union resolvable.
- **Regression:** rerun `graph-eval --assert-gate` and the LongMemEval local
  gate with salience wired but log empty.
- **Gate command:** `cargo test -p skinki-baseline salience` +
  `cargo run --release -p skinki-harness -- salience-eval --assert-gate`
  (new subcommand running the protocol + regression rows).

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `UseEvent` on the store's segmented log + replay | impl | cheaper | round-trip + cost budgets |
| T2 salience formula in `SemanticRetriever` + property/golden/protocol tests | impl (formula frozen) | cheaper | §2 reinforcement/decay/bounded rows green |
| T3 MCP wiring: record top-k `UseEvent`s + premise double-weight; `salience-eval` subcommand | impl | cheaper | protocol green through the server path |
| T4 `ConsolidateJob` (near-dup merge, provenance union, ledger derivations) | impl | cheaper (frontier reviews merge semantics) | consolidation protocol green |
| T5 (optional, measure-first) cold-tier demotion in IVF | impl | cheaper | RAM delta reported; recall holds, else rejected |
| **D1** verdict: does salience improve any *real* workload (LongMemEval reruns with simulated usage; the 5D ablation hook) — keep α=0.25 or re-tune once, never silently | design | **frontier** | numbers + decision recorded |

## 7. Definition of done

- [ ] `salience-eval --assert-gate` green in CI; all pre-existing gates green.
- [ ] `cargo test`, clippy, fmt clean.
- [ ] ROADMAP gains the salience row with measured numbers; README "memory
      dynamics" honest-status.
- [ ] Decision recorded (D1): net effect on real workloads, formula kept or
      revised (version bumped through the ledger).

## 8. Out of scope

- Learned/trainable salience (formula is pinned; changes are versioned).
- Cross-session user modeling, priors, or any personalization beyond usage.
- Deleting anything, ever (L0 append-only law).
- Spaced-repetition-style *surfacing* (a product feature over this substrate;
  Stage 7 territory).
