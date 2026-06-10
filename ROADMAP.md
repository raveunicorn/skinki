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
| 1 | Memory compression PoC (RaBitQ + Model2Vec, two-stage, mmap) | **Done** (re-validated at scale; flat-scan limit mapped) |
| 2 | Storage substrate (Lance/Cozo) + append-only L0; maybe a `.kx` codec | **Done** |
| 3 | Incremental local GraphRAG (LightRAG + HippoRAG 2 PPR + RAPTOR) | Next |
| 4 | "Sleep" consolidation engine (idle + on-power background jobs) | Planned |
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
- **Invent? Not yet — iterate.** Existing blocks clear the budget except on
  adversarial geometry at 5M, where 1-bit codes against a *global* centroid
  can't rank within a dense cluster. The earned next move is **IVF-style
  partitioning with per-list centroids** (standard RaBitQ deployment): it fixes
  within-cluster discrimination and cuts scan cost 10-50x — co-designed with
  Stage 3's index work. Real-EmbeddingGemma validation is the other open
  follow-up.

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

## Stage 3 — Graph & retrieval quality

- **Hypothesis:** a local incremental GraphRAG (LightRAG extraction + HippoRAG 2
  PPR) on **Gemma 4B** gives multi-hop quality close to big-model RAG within the
  cost budget.
- **Tests:** 4B entity/relation extraction (quality + speed); incremental
  updates without rebuild; PPR vs traversal vs PathRAG-pruning; hybrid
  vector+graph; RAPTOR summaries.
- **Gate:** target multi-hop accuracy on synthetic + LongMemEval/LoCoMo; cost
  per MB within the battery budget.

## Stage 4 — "Sleep" engine (consolidation)

- **Hypothesis:** all expensive work (extraction, communities, summaries, PPR,
  dedup) runs as **interruptible incremental background jobs only on idle + on
  power**, with no realtime or battery regression.
- **Tests:** scheduler on macOS power/thermal/idle signals; incremental Leiden;
  resumable queue; latency during/after; drain over a simulated week.
- **Gate:** zero perceptible realtime impact + within the battery budget.

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
  third-party embedding; reproducible benchmarks.
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
- Embedding dimensionality vs quality vs RAM (co-designed in Stage 1).
- STT choice for Russian (Whisper large/distil vs Parakeet vs Apple Speech).
- The "invent a format vs use Lance/Cozo" line is decided by Stage 2 data, not
  in advance.
