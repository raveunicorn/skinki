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

> All tickets below are independent and can proceed now. Do **not** touch
> `skinki-corpus` generation (it moves golden hashes — frontier+human only).

| Ticket | Branch | Files | Do | Gate / DoD |
| --- | --- | --- | --- | --- |
| **T2** temporal detector | `feat/insight-temporal-detector` | `skinki-insight/src/lib.rs` (+a `temporal.rs` if you like) | Implement a `Detector` for `InsightKind::TemporalLead`: cross-correlate each entity's mention-day series against candidate trailing events; null = shuffled lags; emit `InsightCandidate`s feeding the existing `validate`. | New unit tests recover planted `TemporalPattern` at `lag_days ± 1` with recall ≥ 0.50; `cargo test`/clippy/fmt clean. ⚠ frontier reviews the statistic. |
| **T3** contradiction detector | `feat/insight-contradiction-detector` | `skinki-insight/src/lib.rs`, uses `skinki-ledger` | Adapt the `skinki_ledger` staleness output into `DiscoveredInsight`s (`InsightKind::Contradiction`). The ledger already catches planted `Contradiction`s; wire it through the engine + cite-or-silence. | Unit test: ≥ 0.80 of planted `Contradiction`s surfaced, each cited; clean. |
| **T4** narration replay log | `feat/insight-narration-replay` | `skinki-insight/src/lib.rs` | Add an append-only artifact log + replay for the `Narrator` (rule 3): a live narrator appends `NarrationRecord`s, `rebuild(log)` is byte-identical, a checked-in fixture drives the gate. Implement the `Narrator` trait around it. | Golden: `rebuild(log)` byte-identical twice; gate replays, no inference; clean. |
| **T5** ledger wiring | `feat/insight-ledger-derivations` | `skinki-insight/src/lib.rs`, `skinki-ledger` | Emit a `skinki_ledger::Derivation` per surfaced insight (inputs = evidence content hashes; method = detector+validation `MethodStamp`). | Unit test: a changed premise flags exactly its insights stale; clean. ⚠ frontier reviews. |
| **T6** telemetry + sleep job | `feat/insight-telemetry-job` | `skinki-insight/src/lib.rs`, `skinki-sleep` | Add `resident_bytes` + bytes/candidate projection to 5M; wrap `discover` as an interruptible/resumable Stage-4 `Job`. | Report shows RAM projection; job resumes losslessly (mirror `sleep-sim`); clean. |

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
