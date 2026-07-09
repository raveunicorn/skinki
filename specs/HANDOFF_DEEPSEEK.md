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
| ✅ **T2-precision** | done (frontier) | `skinki-insight/src/lib.rs` | Word-boundary matching + density-corrected binomial null + Bonferroni over the lag search + measure the detector in isolation. | recall 0.800, false-insight 0.000 on both seeds — **asserted in the gate**. |
| ✅ **T3-rework** | done (frontier) | `skinki-insight/src/lib.rs` | Name-anchored stance attribution (entity is the cue's subject/object), dropped Y-referring cues, one entity-level candidate citing all endorse/regret entries. Kept exact-match (p=0 is sound — no multiple-testing search). | recall 1.000, false-insight 0.000 on both seeds — **asserted in the gate**. |
| **PROC** non-bundled PRs | n/a | — | Standing rule: **one ticket = one branch/PR** (PR #6 bundled T2–T6 + 3B + pipeline into 1328 lines — hard to review/revert). | reviewers can land/revert each ticket independently. |

> **Note:** the temporal/contradiction *precision* rework was kept on the
> frontier (it was the subtle anti-hallucination statistics, not mechanical
> plumbing). The ledger-based contradiction detector the ticket originally
> imagined is moot — the detector is now a precise exact-match; T5 already wires
> insight→`Derivation` staleness separately.

## Stage 3B — multi-hop retrieval gap (spec: `STAGE_3B_MULTIHOP.md`)

Instrument already merged (`longmemeval-eval`). **Not** the entity graph.

| Ticket | Branch | Files | Do | Gate / DoD |
| --- | --- | --- | --- | --- |
| **T0** embedder ablation | `chore/semantic-real-fulldim-ablation` | (measurement, maybe a flag) | Re-run `semantic-real` at full embedding dim on `multi-session`; record whether the 0.291 ceiling is partly compression. | Numbers recorded in the PR; re-baseline if it moves. |
| **T1** iterative retriever | `feat/iterative-retrieval` | `skinki-harness` (new module) | Extractive, deterministic 2–3-round `IterativeRetriever` over semantic-real (retrieve → pick uncovered facets → re-query). | `longmemeval-eval --pooled --question-type multi-session` recall@10 > 0.291 with no single-session regression; clean. |
| ✅ **T2** session pooling (coarse-to-fine) | merged (PR #6) | — | LongMemEval multi-session 0.291 → 0.438 (+46%), measured not gated | add a `--assert-gate` once the dataset is in the runbook |

> 3B status: coarse-to-fine (T2) shipped and is the win; iterative expansion was
> a negative (0.017). Open: gate the 0.438 number reproducibly (dataset in
> runbook + `--assert-gate`).

## Stage 5B — validate the Insight Engine on REAL data (spec: `STAGE_5B_REAL_INSIGHT.md`)

The synthetic keystone is gated; this proves whether it transfers. Build the
instrument first (it will likely show the synthetic detectors don't fire on real
text — that's the point). **Reuse the Stage-3 LLM extraction artifacts** as the
real entity source; the **LLM oracle judge is replayed** from a fixture (rule 3).

| Ticket | Branch | Files | Do | Gate / DoD |
| --- | --- | --- | --- | --- |
| **T0** real-insight eval skeleton | `feat/insight-eval-real` | `skinki-harness`, `skinki-insight` | `insight-eval --real`: real corpus + replayed Stage-3 extraction → `RealInsightInput`; run current detectors; report what surfaces. | runs end to end; honestly reports the (likely ~zero) transfer of synthetic detectors. |
| **T1** knowledge-update → contradiction GT | `feat/insight-real-recall` | `skinki-harness`, `skinki-eval` | Remap LongMemEval `knowledge-update` to `Contradiction` ground truth; measure real contradiction recall. | recall measured; unit-tested remap. |
| **T2** oracle-judge replay seam | `feat/insight-oracle-judge` | `skinki-insight` | `dump_manifest` (insight + cited text) + `JudgmentLog` replay + `score_real_false_insight` + a small fixture. | `rebuild(log)` byte-identical; false-insight from replay; ⚠ frontier reviews the contract. |
| **T4** wire real-signal detectors | `feat/insight-real-detectors` | `skinki-insight` | Behind T3's frontier design: structural via embedding communities, contradiction via stance/extraction, temporal over extracted day-series. | false-insight < 0.05 (oracle) + recall bars on the instrument; ⚠ frontier owns the cores (T3). |

## Batch 2026-07 (next up — see `specs/README.md` for the full order)

The frontier review (`REVIEW_FRONTIER_2026_07.md`) produced seven new specs.
Work them **in order**; every ticket table inside marks its tier. Start here:

1. `STAGE_5C_HARDENING.md` — bug fixes + scale gate + store soak (all tickets
   delegatable; T1/T3/T4 get a frontier diff review). **Status: merged (PR #8).**
2. `STAGE_1B_STATIC_EMBEDDER.md` — static embedder + IVF serving (T1–T5
   delegatable behind the parity gate; D1 frontier).
   - **T2 done** — `StaticEmbedder` Rust core landed (`feat/1b-static-embedder`):
     `SKEMB001` reader + WordPiece tokenizer + Zipf pooling, toy artifact at
     `fixtures/static_embed_toy.skemb`, 18 tests green, `cargo test -p skinki-vector
     static_embed` is the gate.
   - **T1 done** — `scripts/distill_static_embedder.py` distills
     `BAAI/bge-small-en-v1.5` → `SKEMB001` (30.2 MB ≤ 48 MB budget; **not
     committed** — model weights, regenerate with the script) +
     `fixtures/golden_embeddings.f32` (32 strings, committed). **Cross-impl
     parity verified**: the `#[ignore]` `golden_parity` test reproduces all 32
     golden embeddings byte-for-byte; rerun it after every regeneration.
   - **T3 done** — `EmbedderSpec { Hash, Static { path } }` +
     `SemanticRetriever` consolidated into `skinki-baseline` (single source;
     previously duplicated in mcp/harness). `--embedder hash|static:<path>` on
     `loco-eval` / `longmemeval-eval` / `skinki-mcp`; typos are a loud parse
     error, never a silent hash fallback. Default `hash`.
   - **D1 done — HYPOTHESIS FALSIFIED.** LongMemEval multi-session pooled
     recall@10 = 0.090 for the distilled static artifact vs BM25 0.134 and the
     §2 bar ≥ 0.22. Root cause + full verdict table recorded in the spec (§6).
     No `--assert-gate` bar frozen (an unreachable bar would be dishonest);
     `semantic-static` stays a measurement instrument; the hash embedder
     remains the served default. T4/T5/T7 de-prioritized pending a
     discriminative base artifact. **Stage 1B closed.**
   - **D1 addendum (2026-07-04)** — ready-made Model2Vec artifacts measured
     via `scripts/convert_model2vec_to_skemb.py`: potion-base-8M (30 MB)
     0.116, potion-retrieval-32M (130 MB, deliberately over budget) 0.086 on
     the full pool — bigger decays *faster*; the failure is architectural
     (context-free token vectors), not budget. Static closed in all forms.
   - **T8 done (2026-07-04)** — hybrid RRF fusion (`skinki_baseline::RrfFusion`,
     BM25 + potion-8M static, depth = k): recall@10 **0.145** vs BM25 0.134 —
     first config to beat the yardstick on the D1 row; ndcg +12%; answer@10
     −7%. NOT flipped to served default (margin within noise on 121 queries;
     artifact not shipped); `hybrid-rrf` column ships as an instrument.
     The successor path went through 1C-B and is now 1D; fusion is the pattern
     to re-apply once a discriminative encoder base lands.
2b. `STAGE_1C_B_PURE_RUST_ENCODER.md` — **closed/trend-closed.** The pure-Rust
   encoder, `SKENC001`, converter, goldens, `encoder-embed`, `EmbedderSpec::Encoder`,
   query/passage prefix seam, and bge-small trend verdict are done. bge-small is
   not the served model; the sidecar fallback was rejected as the product shape.
   Keep the machinery and continue in 1D.
2c. `STAGE_1D_RETRIEVAL_QUALITY.md` — **e5-small D2 closed negative.**
   - Done: K0 Unigram tokenizer; T1 multilingual-e5-small converter/artifact
     path; T2 trend-row eval (41q/201k: e5 `rrf` 0.423 vs bge 0.411); T6
     doc2query replay instrument with a preliminary negative 0.5B signal; full
     594k/121q e5 D2 replay.
   - Perf landed: Fable PR #27 and follow-up #28 cut the cold encoder/indexing
     path enough that the full D1-row e5 replay is now practical. See
     `specs/PERF_COLD_INDEX_10X.md`.
   - **D2 result:** full row `semantic-real` recall@10 0.152,
     `rrf(bm25+real)` 0.160 vs the ≥0.30 bar. e5-small is not the served
     default.
   - Do **not** spend SDOT/VDOT int8 on e5-small. PERF records safe-Rust int8
     as only ~1.2× over the optimized f32 kernel, and unsafe SDOT is only worth
     considering after a stronger model/strategy clears quality.
3. `STAGE_6B_AGENT_MEMORY.md` — `remember` / staleness / `memory_asof`
   (all delegatable; T2 semantics reviewed).

Same standing rules: one ticket = one branch = one PR; never weaken a gate;
put the gate command + output in the PR body.

## How to verify locally (copy/paste)

```bash
cd skinki
cargo build && cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --check
# Stage 5 infra gate (must stay green):
cargo run --release -p skinki-harness -- insight-eval --assert-gate
# Stage 3B measurement (needs the dataset; see STAGE_3B / PR runbooks):
cargo run --release -p skinki-harness -- longmemeval-eval --pooled --question-type multi-session
```
