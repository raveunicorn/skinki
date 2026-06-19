# Hand-off to DeepSeek — independent, gate-arbitrated tickets

This file is the work queue for delegated implementation. The architecture, the
fitness functions (gates), and the frozen interfaces are already decided by the
frontier owner; **the gate decides correctness, not reviewer taste.** Your job is
to make the gate go green without weakening it.

## Working agreement (read first)

1. **One ticket = one branch off `main`.** Do **not** stack branches and do
   **not** copy another branch's code to fake a base. (A prior batch shipped a
   fake "stacked" topology by copying files instead of branching off the parent —
   it made the PR base misleading. Branch each ticket directly from `main`.)
2. **Never weaken a gate.** If a budget seems wrong, stop and ask the human — do
   not lower a threshold to pass. (`AGENTS.md`, `CLAUDE.md`.)
3. **Determinism is law (rule 2).** Seeded PRNG only, `BTreeMap`/sorted, no
   `HashMap` iteration order in anything that affects results. Any LLM output is
   **replayed** from an append-only artifact log (rule 3) — never inferred in a
   gate.
4. **A change is correct iff:** `cargo build` + `cargo test` +
   `cargo clippy --all-targets -- -D warnings` + `cargo fmt --check` + the
   ticket's named gate all pass. Put the gate command in the PR body with its
   output.
5. **Minimal deps:** only `serde`, `serde_json`, `clap`, `libc`, `anyhow`, and
   internal `skinki-*`. Any new third-party dep needs human approval.
6. Frontier reviews every PR touching an algorithm core (marked ⚠ below) — a fast
   diff review, not a rewrite.

## Stage 5 — Insight Engine (`crates/skinki-insight`, spec: `STAGE_5.md`)

The keystone is built and gated: `insight-eval --assert-gate` is green and now
asserts the full anti-hallucination budgets (recall 1.000, precision 1.000,
false-insight 0, apophenia 0, 0 uncited, deterministic, on two seeds) for the
**structural-bridge** detector. The D0 corpus fix is done. These tickets add the
remaining detectors + narration. The frozen interface is in
`crates/skinki-insight/src/lib.rs` — implement against it, don't change it; copy
the `StructuralBridgeDetector` shape (propose → `Statistic` → `validate`).

> Do **not** touch `skinki-corpus` generation (it moves golden hashes —
> frontier+human only). **Status after PR #6:** T4/T5/T6 ✅ done. T2/T3 landed
> but are **recall-only** (false-insight 0.78 / 0.57 vs the < 0.05 keystone bar)
> — kept *informational*, not gated. The open tickets below fix their precision.

| Ticket | Branch | Files | Do | Gate / DoD |
| --- | --- | --- | --- | --- |
| ✅ **T2** temporal detector | merged (PR #6) | — | recall 0.800; precision ✗ — see T2-precision below | — |
| ✅ **T3** contradiction detector | merged (PR #6) | — | recall 1.000; precision ✗, bypasses `validate`, no ledger — see T3-rework | — |
| ✅ **T4** narration replay log | merged (PR #6) | — | round-trip + replay-determinism tested | — |
| ✅ **T5** ledger wiring | merged (PR #6) | — | `record_insight_derivations`, staleness tested | — |
| ✅ **T6** telemetry + sleep job | merged (PR #6) | — | `resident_bytes` ≈ 0.33 MB @5M; `InsightJob` | — |

### Open follow-ups (precision — required before T2/T3 can be promoted to the gate)

> The keystone is **anti-hallucination**: a detector that hits recall but floods
> false insights is worse than none. These tickets bring T2/T3 to the hard
> `false-insight < 0.05` bar so they can be **asserted** in `insight-eval`.

| Ticket | Branch | Files | Do | Gate / DoD |
| --- | --- | --- | --- | --- |
| **T2-precision** | `feat/insight-temporal-precision` | `skinki-insight/src/lib.rs` | The cross-correlation surfaces ~7 false leads per 2 real ones (false-insight 0.78). Tighten the null / effect floor / BH-FDR so only genuine lead→trail pairs survive. Do **not** weaken `score_temporal`. | `insight-eval` informational temporal row reaches recall ≥ 0.50 **and** false-insight < 0.05 on both seeds; then promote it into `--assert-gate`. |
| **T3-rework** | `feat/insight-contradiction-rework` | `skinki-insight/src/lib.rs`, `skinki-ledger` | Current detector keys on sentiment cues and sets `p=0, surprise=1` (bypasses `validate`) — it overfits templates and floods false positives (0.57). Rebuild it to (a) produce a **real statistic** that flows through `validate`, and (b) use the **ledger** staleness signal the ticket intended, not surface cues. | recall ≥ 0.80 **and** false-insight < 0.05 on both seeds; flows through `validate`; then promote into `--assert-gate`. |
| **PROC** non-bundled PRs | n/a | — | Future work: **one ticket = one branch/PR** (PR #6 bundled T2–T6 + 3B + pipeline into 1328 lines — hard to review/revert). | reviewers can land/revert each ticket independently. |

## Stage 3B — multi-hop retrieval gap (spec: `STAGE_3B_MULTIHOP.md`)

Instrument already merged (`longmemeval-eval`). **Not** the entity graph.

| Ticket | Branch | Files | Do | Gate / DoD |
| --- | --- | --- | --- | --- |
| **T0** embedder ablation | `chore/semantic-real-fulldim-ablation` | (measurement, maybe a flag) | Re-run `semantic-real` at full embedding dim on `multi-session`; record whether the 0.291 ceiling is partly compression. | Numbers recorded in the PR; re-baseline if it moves. |
| **T1** iterative retriever | `feat/iterative-retrieval` | `skinki-harness` (new module) | Extractive, deterministic 2–3-round `IterativeRetriever` over semantic-real (retrieve → pick uncovered facets → re-query). | `longmemeval-eval --pooled --question-type multi-session` recall@10 > 0.291 with no single-session regression; clean. |
| **T2** session pooling | `feat/session-summary-pooling` | `skinki-harness`, `skinki-sleep` | Coarse-to-fine: per-session summaries (a Stage-4 job) → retrieve sessions → drill to turns. | Measured lift vs T1; summaries don't drop the evidence turn; clean. |

## How to verify locally (copy/paste)

```bash
cd skinki
cargo build && cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --check
# Stage 5 infra gate (must stay green):
cargo run --release -p skinki-harness -- insight-eval --assert-gate
# Stage 3B measurement (needs the dataset; see STAGE_3B / PR runbooks):
cargo run --release -p skinki-harness -- longmemeval-eval --pooled --question-type multi-session
```
