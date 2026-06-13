# Skinki Exocortex — Roadmap

This is not a list of "ready solutions" but a map of **hypothesis-driven
stages**. Each stage is a small "impossible task" with a hard budget (a fitness
function) and a **kill-or-keep gate**: if the best existing approach can't fit
the M1 Air 8 GB budget, that is the license to invent our own. The primary
artifact is the headless Rust engine (`kortex`); the macOS app is the Stage 7
wrapper.

Design target: a worst-case "extreme" user — **~10 years, tens of GB of raw
input, ~5M memory units.** See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
layered design and budgets.

## Stage status

| Stage | Focus | Status |
| --- | --- | --- |
| 0 | Eval harness + synthetic corpus (the measuring stick) | **Done** |
| 1 | Memory compression PoC (RaBitQ + Model2Vec, two-stage, mmap) | **Done** (re-validated at scale; IVF closes the scan-cost gap, gated) |
| 2 | Storage substrate (Lance/Cozo) + append-only L0; maybe a `.kx` codec | **Done** (+ 2B durability) |
| 3 | Incremental local GraphRAG (deterministic-first, two-tier; venue-anchored multi-hop) | **In progress** — deterministic tier done + gated (multi-hop 2.5–3× BM25); LLM tier (D2) next ([`STAGE_3.md`](kortex/specs/STAGE_3.md)) |
| 4 | "Sleep" consolidation engine (idle + on-power background jobs) | **Done** (policy proven in simulation; real jobs land at Stage 3/5) |
| 5 | Insight Engine (deterministic discovery + FDR + cite-or-silence) | Planned |
| 6 | Portable `kortex` crate (C-ABI/FFI + CLI, Swift/Python bindings) | Planned |
| 7 | Skinki macOS product integration (the wrapper) | Planned |

---

## Stage 0 — Eval harness & synthetic corpus (the yardstick) — Done

- **Hypothesis:** we can generate "years of thoughts" with *planted ground
  truth* (recall facts, multi-hop links, leading temporal patterns, temporal
  contradictions, insight "needles") as an honest proxy for real cognition.
- **Built:** deterministic synthetic generator + metrics (recall@k, nDCG,
  multi-hop QA, **insight-precision** and **false-insight rate**) + latency/RAM
  telemetry + a BM25 baseline. See [`kortex/README.md`](kortex/README.md).
- **Gate (met):** the harness reproducibly measures the budgets and both
  dimensions (recall + insight), and cleanly separates recall (BM25 solves it)
  from multi-hop and insight discovery (BM25 fails them) — establishing the
  targets for Stages 1-5.
- **Hardened (V2, default):** paraphrase banks, coreference in multi-hop
  chains, planted *negative* bridges (apophenia traps with a `neg-hits`
  metric), lexical distractors, and per-year topic drift. BM25 recall@10 drops
  from 1.000 (V1) to 0.138 at ~1M entries, so no metric is solvable by lexical
  overlap alone. The legacy generator remains behind `--difficulty v1`, pinned
  byte-for-byte by a golden-hash test.

## Stage 1 — Memory compression PoC (first "impossible task") — Done

- **Hypothesis:** 5M units are searchable within the RAM/latency budget via
  **RaBitQ + Model2Vec first-pass**, with recall >= 95% vs full-precision.
- **Built:** from-scratch codecs in [`kortex/crates/kortex-vector`](kortex/crates/kortex-vector)
  — int8 scalar, Product Quantization (per-subspace k-means), and RaBitQ (random
  rotation via sign-flip + Walsh-Hadamard, then 1-bit/multi-bit codes with the
  unbiased estimator) — plus a two-stage retriever, an mmap-backed code store,
  and an exact float32 baseline. Benchmarked via `kortex compress-bench`.
- **Gate (met at small N, then re-validated at scale):** the winning config —
  **Matryoshka-truncation to 256 dims + two-stage (1-bit RaBitQ popcount scan
  -> float rerank of candidates from mmap)** — measured by `kortex scale-bench`
  on real streamed indexes (not projections): **1M: recall@10 = 1.000, p95
  32.9 ms, 38 MB resident. 5M: recall 1.000 at p95 119 ms / 190.7 MB resident
  on mild cluster geometry; on adversarial geometry (78k-point clusters) recall
  1.000 costs p95 157.7 ms — ~5% over the 150 ms budget.** Cold open + first
  query < 300 ms. The re-validation surfaced and fixed a real latency bug (a
  full per-query sort of all candidates) and corrected the resident accounting
  to 40 B/vec (was 36).
- **Invent? Not yet — iterate, and the iteration landed.** Existing blocks
  clear the budget except on adversarial geometry at 5M, where 1-bit codes
  against a *global* centroid can't rank within a dense cluster, and the flat
  O(n) scan eats ~120 ms of the 150 ms budget at 5M regardless of geometry. The
  earned next move — **IVF-style partitioning with per-list centroids** (the
  standard RaBitQ deployment) — is now built (`kortex-vector/src/ivf.rs`) and
  gated. Measured at 1M on the realistic (mild) geometry it holds **recall@10
  1.000 at p95 2.6 ms** (vs flat's 32.9 ms — a ~12x scan-cost cut) for ~8 extra
  B/vec, projecting to 229 MB at 5M (budget 250). A small mild-geometry
  `scale-bench --index ivf --assert-gate` run guards it in CI; the 5M RAM
  projection is split (per-vec linear + sqrt(n) centroids) so the verdict is
  N-independent. IVF does **not** rescue the synthetic adversarial extreme (huge
  blobs k-means can't sub-partition) — that stays the stress ceiling and a
  Stage-3 co-design item (multi-bit residuals / OPQ), not a Stage-1 blocker.
  Real-EmbeddingGemma validation remains the other open follow-up.

## Stage 2 — Storage substrate (pure-Rust, gate passed) — Done

- **Hypothesis:** a pure-Rust, mmap-backed append-only log + content-addressed
  unit store can hold years of capture within budget without pulling in a heavy
  embedded DB.
- **Built:** `kortex-store` crate — compact binary encoding (lossless
  `created_utc_secs`, `text_len` derived from framing), segmented append-only
  files, 128-bit FNV1a dedup, zero-copy mmap reads, `derive_units` sentence
  splitter. Benchmarked via `kortex store-bench`.
- **Gate (met):** two metrics checked — **content overhead 1.21x** (budget
  1.25x) and **index bytes/unit 20.0** (budget 24). Random-access p95 = 0.3 us
  (budget < 1 ms). Ingest throughput 2.5M units/s (budget >= 50k).
- **Invent? No.** The pure-Rust substrate clears all budgets, so D1 (Lance/Cozo
  vs pure-Rust) resolves to *keep pure Rust*. D2 (`.kx` codec) stays deferred —
  no budget broke. All existing `cargo test`/clippy/fmt gates green.
- **2B — durability (done, gate extended).** Size-based segment rotation
  (64 MiB) + write-through appends with fsync on `sync()`, torn-tail crash
  recovery, and a persisted sorted-run dedup index → reopen scans at most one
  segment tail instead of the whole history. New budgets met: durable ingest
  (fsync per event) ~240 events/s (budget ≥ 100); cold reopen 80 ms at ~894k
  units (budget < 1 s). See [`kortex/specs/STAGE_2B.md`](kortex/specs/STAGE_2B.md).

## Stage 3 — Graph & retrieval quality

- **Hypothesis:** a local incremental GraphRAG (LightRAG extraction + HippoRAG 2
  PPR) on **Gemma 4B** gives multi-hop quality close to big-model RAG within the
  cost budget.
- **Pre-design constraint (do the math first):** full-LLM extraction of a 5M-unit
  backfill is infeasible by ~100x on the M1 Air — extraction is **two-tier by
  construction** (deterministic tier over everything; LLM tier only for a
  selected ≤~5% on backfill), and all LLM outputs go to an append-only artifact
  log (replayable; AGENTS.md rule 3). Arithmetic and ticket implications:
  [`kortex/specs/STAGE_3_BUDGET.md`](kortex/specs/STAGE_3_BUDGET.md).
- **Tests:** 4B entity/relation extraction (quality + speed, both tiers);
  incremental updates without rebuild; PPR vs traversal vs PathRAG-pruning;
  hybrid vector+graph; RAPTOR summaries; **salience/reconsolidation** (use
  counts + recency feed retrieval ranking; reinforced-on-use links); the
  **context assembler** — a budgeted package (cited facts with dates,
  pre-joined multi-hop chains, community summary, flagged contradictions)
  measured by a **context-sufficiency** metric: can the answer be produced
  from the assembled package alone within a 1-2k-token budget (on an M1 Air,
  prefill speed makes every context token expensive — small dense packages
  beat top-k chunk dumps).
- **Gate:** target multi-hop accuracy on synthetic + LongMemEval/LoCoMo; cost
  per MB within the battery budget; context-sufficiency above threshold at the
  token budget.

## Stage 4 — "Sleep" engine (consolidation)

- **Hypothesis:** all expensive work (extraction, communities, summaries, PPR,
  dedup) runs as **interruptible incremental background jobs only on idle + on
  power**, with no realtime or battery regression.
- **Tests:** scheduler on macOS power/thermal/idle signals; incremental Leiden;
  resumable queue; latency during/after; drain over a simulated week.
- **Gate:** zero perceptible realtime impact + within the battery budget.
- **Result (policy, simulated):** `kortex-sleep` ships the scheduler — priority
  queue, signal-gated `tick` (runs iff power ∧ idle ∧ thermal), crash-safe
  checkpoint/restore, and a deterministic week-long simulator. `sleep-sim
  --assert-gate` passes all six metrics; a locked golden-trace hash pins the
  policy byte-for-byte and `resume_is_lossless` proves a mid-run crash resumes
  draining the identical backlog with nothing lost. The real consolidation jobs
  (Leiden / RAPTOR / PPR) plug in as `Job`s at Stage 3/5; on-hardware battery
  draw is measured at Stage 7. (PR #1)

## Stage 5 — Insight Engine (keystone, anti-hallucination)

- **Hypothesis:** deterministic **discovery + statistical validation** finds
  genuinely non-obvious *grounded* links / leading indicators above an
  insight-precision threshold and below a false-insight threshold; the LLM
  narrates only with citations and calibration.
- **Tests:** link prediction, structural-hole bridges, temporal co-occurrence
  lags, changepoints, contradiction detection; FDR / effect size; "cite-or-
  silence" narration; measure fraction of planted insights found vs false ones;
  hallucination audit.
- **Gate:** >= X% of planted insights at <= Y% false; **0** uncited claims.

## Stage 6 — Portable engine (primary artifact, "FFmpeg")

- **Hypothesis:** everything above packages into a Rust crate `kortex` with a
  stable C-ABI/FFI + CLI, headless and embeddable anywhere; `no_std`-friendly in
  hot paths where possible.
- **Tests:** Swift bindings (for Skinki) and Python bindings (for CI/eval);
  third-party embedding; reproducible benchmarks; an **MCP server** over the
  query/insight surface — the cheapest distribution channel ("memory for
  agents": Claude Code, Cursor, any MCP host) and a forcing function for a
  clean headless API. (The Stage 1 `RaBitQ::save/load` index format is the
  seed of what `kx_open` consumes.)
- **Gate:** documented, benchmarked, embeddable by a third party.

## Stage 7 — Skinki product integration (secondary)

- **Hypothesis:** the engine delivers a consumer-ready experience on an M1 Air
  end-to-end within all budgets.
- **Tests:** capture (multilingual voice STT + text), agent interface,
  calibrated pop-up insights, mascot reactions; onboarding/privacy; `.dmg`
  (codesign + notarize).
- **Gate:** a `.dmg` that meets every budget on a live M1 Air. The parked
  scaffolding lives in [`apps/skinki-macos/`](apps/skinki-macos/).

## Open questions / risks

- Exact insight-precision / false-insight thresholds (set on Stage 0 synthetic).
- ~~Embedding dimensionality vs quality vs RAM~~ **resolved on real vectors**
  (nomic-embed-text-v1.5, an MRL stand-in for EmbeddingGemma): Matryoshka-256
  keeps the fidelity gate green (recall 0.987-1.000) within the RAM budget;
  real geometry needs only `refine ≈ 1%` of n (vs 16%+ on adversarial
  synthetic). Re-confirm on EmbeddingGemma when the product embedder lands.
- IVF-style partitioning (per-list centroids) vs flat scan at 5M — the Stage 1
  at-scale verdict makes this the default candidate for the Stage 3 index.
- STT choice for Russian (Whisper large/distil vs Parakeet vs Apple Speech).
- The "invent a format vs use Lance/Cozo" line is decided by Stage 2 data, not
  in advance.
- **Staleness-aware memory (v0 built + gated):** store the *reasoning chain*
  behind a derived fact and hash-pin its premises, so a changed premise breaks
  the link and the conclusion is flagged for re-evaluation — deterministic
  propagation of contradictions over a content-addressed Merkle DAG (Git/Nix
  shape, not a literal blockchain). `crates/kortex-ledger` + `ledger-bench
  --assert-gate` reach **invalidation-recall 1.000 at 0 over-invalidation** on
  the corpus's planted `Contradiction` ground truth, vs a provenance-free
  baseline's **0.000**. Cross-cuts L0 provenance, Stage 3 graph, and the Stage 5
  keystone. Remaining: durable persistence (on `kortex-store`) and Stage-3
  integration. See
  [`kortex/specs/DERIVATION_LEDGER.md`](kortex/specs/DERIVATION_LEDGER.md).
- External validation of the architecture's core pattern: DeepSeek V4's
  CSA/HCA attention (query-dependent selection over compressed entries +
  heavily-compressed global context) is the same "compress + select" shape we
  build *outside* the model — RAPTOR summaries ≈ the compressed global view,
  two-stage retrieval + context assembler ≈ the query-dependent selection.
  Inspiration for the assembler's structure; not a component to adopt.
