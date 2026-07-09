# skinki — Roadmap

This is not a list of "ready solutions" but a map of **hypothesis-driven
stages**. Each stage is a small "impossible task" with a hard budget (a fitness
function) and a **kill-or-keep gate**: if the best existing approach can't fit
the M1 Air 8 GB budget, that is the license to invent our own. This repo is the
headless engine; a macOS consumer product is a planned Stage 7.

What's gated-and-validated vs. validated *only on synthetic* (and what failed on
real data) is tracked honestly in the
[README](README.md#honest-status-read-this-first) — read that for the unvarnished
state. This roadmap is the plan and the per-stage detail.

Design target: a worst-case "extreme" user — **~10 years, tens of GB of raw
input, ~5M memory units.** See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
layered design and budgets.

## Stage status

| Stage | Focus | Status |
| --- | --- | --- |
| 0 | Eval harness + synthetic corpus (the measuring stick) | **Done** |
| 1 | Compression + retrieval-quality substrate | **Compression done; embedder-quality track still open.** RaBitQ/IVF is gated. 1B static distillation is closed/falsified; 1C-B pure-Rust encoder is built and trend-closed (bge-small ceiling); 1D multilingual-e5-small is closed negative on the full row (`rrf` 0.160 vs 0.30 bar). Next: **Stage 1E — base-class encoder (768-dim, MIT/Apache) through the proven machinery** ([`STAGE_1E`](specs/STAGE_1E_BASE_CLASS_ENCODER.md)); the 1D lesson (trend row is a cheap abort, never a pass) is encoded as an invariant. |
| 2 | Storage substrate (Lance/Cozo) + append-only L0; maybe a `.kx` codec | **Done** (+ 2B durability) |
| 3 | Incremental local GraphRAG (deterministic-first, two-tier; venue-anchored multi-hop) | **Closed — graph is substrate, not a retriever (honest).** 2.5–3× BM25 on synthetic, ledger-wired + 3C assembler, gated; but on **two** real benchmarks (LoCoMo, LongMemEval) the graph does **not** beat BM25, and a semantic embedder (EmbeddingGemma) is SOTA (+51% over BM25 on LongMemEval multi-session). Default retriever → EmbeddingGemma; graph retained for structure (ledger/provenance/staleness) into Stage 5. Multi-hop gap remains open ([`STAGE_3.md`](specs/STAGE_3.md)) |
| 4 | "Sleep" consolidation engine (idle + on-power background jobs) | **Done** (policy proven in simulation; real jobs land at Stage 3/5) |
| 5 | Insight Engine (deterministic discovery + FDR + cite-or-silence) | **Synthetic keystone gated across three detector families.** Structural bridges, temporal lead/lag, and contradictions all run through cite-or-silence with 0 uncited claims and 0 false-insight/apophenia on the asserted seeds. 5C core hardening landed; next frontier is real-data transfer (5B) and Law-1 end-to-end QA (5D). |
| 6 | Portable `skinki` (C-ABI/FFI + Python binding; MCP server) | **Done** (C-ABI + Python parity gated; `skinki-mcp` ships search + context-assembler to agents; Swift → Stage 7) |
| 7 | skinki macOS product integration (the wrapper) | Planned |

---

## Stage 0 — Eval harness & synthetic corpus (the yardstick) — Done

- **Hypothesis:** we can generate "years of thoughts" with *planted ground
  truth* (recall facts, multi-hop links, leading temporal patterns, temporal
  contradictions, insight "needles") as an honest proxy for real cognition.
- **Built:** deterministic synthetic generator + metrics (recall@k, nDCG,
  multi-hop QA, **insight-precision** and **false-insight rate**) + latency/RAM
  telemetry + a BM25 baseline. See [`README.md`](README.md).
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

## Stage 1 — Compression done; retrieval-quality track active

- **Hypothesis:** 5M units are searchable within the RAM/latency budget via
  **RaBitQ + Model2Vec first-pass**, with recall >= 95% vs full-precision.
- **Built:** from-scratch codecs in [`crates/skinki-vector`](crates/skinki-vector)
  — int8 scalar, Product Quantization (per-subspace k-means), and RaBitQ (random
  rotation via sign-flip + Walsh-Hadamard, then 1-bit/multi-bit codes with the
  unbiased estimator) — plus a two-stage retriever, an mmap-backed code store,
  and an exact float32 baseline. Benchmarked via `skinki compress-bench`.
- **Gate (met at small N, then re-validated at scale):** the winning config —
  **Matryoshka-truncation to 256 dims + two-stage (1-bit RaBitQ popcount scan
  -> float rerank of candidates from mmap)** — measured by `skinki scale-bench`
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
  standard RaBitQ deployment) — is now built (`skinki-vector/src/ivf.rs`) and
  gated. Measured at 1M on the realistic (mild) geometry it holds **recall@10
  1.000 at p95 2.6 ms** (vs flat's 32.9 ms — a ~12x scan-cost cut) for ~8 extra
  B/vec, projecting to 229 MB at 5M (budget 250). A small mild-geometry
  `scale-bench --index ivf --assert-gate` run guards it in CI; the 5M RAM
  projection is split (per-vec linear + sqrt(n) centroids) so the verdict is
  N-independent. IVF does **not** rescue the synthetic adversarial extreme (huge
  blobs k-means can't sub-partition) — that stays the stress ceiling and a
  Stage-3 co-design item (multi-bit residuals / OPQ), not a Stage-1 blocker.
  Real-model retrieval validation moved into the Stage-1 quality track below.

### Stage 1 quality track — 1B / 1C-B / 1D / 1E

The compression/index layer is done; the open Stage-1 work is the embedder
quality and serving path that feeds the index.

- **1B static embedder — closed/falsified.** `SKEMB001`, WordPiece, static
  artifacts, harness/MCP `--embedder static:<path>`, and RRF instrumentation
  landed. The D1 result was negative: static token-vector tables top out below
  BM25 on LongMemEval full-pool multi-session, so static is an instrument, not
  the served default.
- **1C-B pure-Rust encoder — closed/trend-closed.** `SKENC001`,
  `skinki-encoder`, converter, goldens, query/passage prefix support, and
  `encoder-embed` landed. bge-small proved the machinery but not the quality
  bar, so the sidecar fallback was rejected as a product shape and the next
  move became 1D.
- **1D multilingual-e5-small — closed negative.** e5 was parity-green and beat
  bge-small on the 41q/201k trend row, but the full 594k/121q D2 row failed:
  semantic-real recall@10 0.152 and `rrf(bm25+real)` 0.160 vs the 0.30 bar.
  Fable's cold-indexing work makes these measurements practical and remains
  useful for future models, but e5-small is not the served default and should
  not receive the SDOT/int8 acceleration ticket.
- **1E base-class encoder — specced, ready to build.** The 1D diagnosis is a
  model *weight-class* problem (e5-small delivers a consistent ~1.15× over BM25;
  the bar needs ~2.2×), not a machinery problem (parity 1.0000000, prefixes
  applied, fusion beats both parents). **Stage 1E** pushes the existing block
  one weight class up — a 768-dim MIT/Apache base-class encoder
  (multilingual-e5-base primary) through the *unchanged* forward pass, artifact,
  and harness — before inventing (Gemma bridge/port, int8). The 1D lesson is
  encoded as an invariant: the trend row is a cheap **abort**, never a **pass**
  (it was a misleading pass signal for e5-small). See
  [`STAGE_1E`](specs/STAGE_1E_BASE_CLASS_ENCODER.md).

## Stage 2 — Storage substrate (pure-Rust, gate passed) — Done

- **Hypothesis:** a pure-Rust, mmap-backed append-only log + content-addressed
  unit store can hold years of capture within budget without pulling in a heavy
  embedded DB.
- **Built:** `skinki-store` crate — compact binary encoding (lossless
  `created_utc_secs`, `text_len` derived from framing), segmented append-only
  files, 128-bit FNV1a dedup, zero-copy mmap reads, `derive_units` sentence
  splitter. Benchmarked via `skinki store-bench`.
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
  units (budget < 1 s). See [`specs/STAGE_2B.md`](specs/STAGE_2B.md).

## Stage 3 — Graph & retrieval quality

- **Hypothesis:** a local incremental GraphRAG (LightRAG extraction + HippoRAG 2
  PPR) on **Gemma 4B** gives multi-hop quality close to big-model RAG within the
  cost budget.
- **Pre-design constraint (do the math first):** full-LLM extraction of a 5M-unit
  backfill is infeasible by ~100x on the M1 Air — extraction is **two-tier by
  construction** (deterministic tier over everything; LLM tier only for a
  selected ≤~5% on backfill), and all LLM outputs go to an append-only artifact
  log (replayable; AGENTS.md rule 3). Arithmetic and ticket implications:
  [`specs/STAGE_3_BUDGET.md`](specs/STAGE_3_BUDGET.md).
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
- **Result (real text — closed, honest).** The synthetic gate is green; the
  real-text close-out measured the graph against two real benchmarks and a real
  embedder. **LoCoMo:** BM25 multi-hop recall@10 = 0.784 — *no gap to close.*
  **LongMemEval** (pooled, `multi-session`, n=20, 9.7k entries, recall@10):
  bm25 0.193, co-mention graph 0.168, typed-fact graph 0.168, **EmbeddingGemma
  0.291** (+51% over BM25). The synthetic 2.5–3× win — driven by templated
  intro/rec/venue patterns — **did not transfer** to free-form dialogue.
  **Decisions:** (1) the **default retriever is EmbeddingGemma**, not BM25, on
  real text; (2) the **graph is retained as a structural substrate** (provenance,
  derivation ledger, staleness) for Stage 5, **not** as a retrieval ranker;
  (3) the **multi-hop gap stays open** (EmbeddingGemma still misses ~71% of
  evidence turns) and is *not* pursued via the LLM-entity-graph — the live
  candidates are **query-focused summarization** and **iterative/multi-step
  retrieval** (see [`STAGE_3.md`](specs/STAGE_3.md) round 4 and Stage 5 below).
  Live-LLM extraction and 3B (communities/RAPTOR/PPR-at-scale) are dropped for
  retrieval — the close-out removes their justification.

## Stage 4 — "Sleep" engine (consolidation)

- **Hypothesis:** all expensive work (extraction, communities, summaries, PPR,
  dedup) runs as **interruptible incremental background jobs only on idle + on
  power**, with no realtime or battery regression.
- **Tests:** scheduler on macOS power/thermal/idle signals; incremental Leiden;
  resumable queue; latency during/after; drain over a simulated week.
- **Gate:** zero perceptible realtime impact + within the battery budget.
- **Result (policy, simulated):** `skinki-sleep` ships the scheduler — priority
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
- **Result (structural-bridge detection earned + gated).** `crates/skinki-insight`
  ships the keystone: a fairness boundary (`InsightInput` — the answer key is
  *unreachable*, not just "don't read it"), a Benjamini–Hochberg FDR validation
  core, structural cite-or-silence (0 uncited by construction), the reference
  detector + the naive co-mention contrast, all behind `insight-eval
  --assert-gate` (green in CI on two seeds). Round 1 measured an honest blocker
  (the planted insight was statistically undetectable — bridge entities weren't
  rare, so the naive detector found all 5 bridges *and* all 5 apophenia hubs while
  the validated detector that rejected the hubs found nothing). Round 2 fixed it
  with **D0** — rare/unique bridge names, scoped to V2 and RNG-neutral so the V1
  byte-frozen golden held — and the reference engine now clears the full keystone:
  **recall 1.000, precision 1.000, false-insight 0.000, apophenia 0/5, 0 uncited,
  deterministic**, beating the naive contrast (precision 0.19). The FDR/surprise
  validation — not the corpus — does the work. Remaining for "done": temporal +
  contradiction detectors and the replayed-LLM narrator (delegatable behind the
  frozen interface). See [`specs/STAGE_5.md`](specs/STAGE_5.md).

## Stage 6 — Portable engine (primary artifact, "FFmpeg")

- **Hypothesis:** everything above packages into a Rust crate `skinki` with a
  stable C-ABI/FFI + CLI, headless and embeddable anywhere; `no_std`-friendly in
  hot paths where possible.
- **Tests:** Swift bindings (for skinki) and Python bindings (for CI/eval);
  third-party embedding; reproducible benchmarks; an **MCP server** over the
  query/insight surface — the cheapest distribution channel ("memory for
  agents": Claude Code, Cursor, any MCP host) and a forcing function for a
  clean headless API. (The Stage 1 `RaBitQ::save/load` index format is the
  seed of what `sk_open` consumes.)
- **Gate:** documented, benchmarked, embeddable by a third party.

## Stage 7 — skinki product integration (secondary)

- **Hypothesis:** the engine delivers a consumer-ready experience on an M1 Air
  end-to-end within all budgets.
- **Tests:** capture (multilingual voice STT + text), agent interface,
  calibrated pop-up insights, mascot reactions; onboarding/privacy; `.dmg`
  (codesign + notarize).
- **Gate:** a `.dmg` that meets every budget on a live M1 Air. The parked
  is a planned future product; no app ships in this repo.

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
  shape, not a literal blockchain). `crates/skinki-ledger` + `ledger-bench
  --assert-gate` reach **invalidation-recall 1.000 at 0 over-invalidation** on
  the corpus's planted `Contradiction` ground truth, vs a provenance-free
  baseline's **0.000**. Cross-cuts L0 provenance, Stage 3 graph, and the Stage 5
  keystone. Remaining: durable persistence (on `skinki-store`) and Stage-3
  integration. See
  [`specs/DERIVATION_LEDGER.md`](specs/DERIVATION_LEDGER.md).
- External validation of the architecture's core pattern: DeepSeek V4's
  CSA/HCA attention (query-dependent selection over compressed entries +
  heavily-compressed global context) is the same "compress + select" shape we
  build *outside* the model — RAPTOR summaries ≈ the compressed global view,
  two-stage retrieval + context assembler ≈ the query-dependent selection.
  Inspiration for the assembler's structure; not a component to adopt.
