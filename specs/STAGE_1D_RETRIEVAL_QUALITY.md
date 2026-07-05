# Stage 1D — Retrieval quality: mid-size multilingual encoder, int8, space bridge (SPEC, DRAFT)

> **Context.** `STAGE_1C_B` D2 (2026-07-05, trend-close) proved the
> pure-Rust encoder machinery works end-to-end (teacher parity
> 1.0000000, byte-determinism, `RRF(BM25+encoder)` beating both parents
> +21%/+14% on the 41q/201k trend row) and that **bge-small's 33M params
> are the ceiling**, not the code: the inherited bars need 1.64×/2.24×
> over BM25, the model delivers 1.06×/1.21×. The human decision framing
> this stage: the memory engine stays a **self-contained binary** (the
> sidecar is rejected as a product shape for the embedder; the LLM stays
> an external, swappable interface), the product is international
> (English-first, multilingual mandatory — a US/EU-only architecture is
> disqualified at design time), and models are expected to improve every
> few months, so everything here must stay **model-agnostic behind
> `SKENC001` + converter** — no model is ever load-bearing.

> Read [`../AGENTS.md`](AGENTS.md). Everything from `STAGE_1C` §6 and
> `STAGE_1C_B` still holds: artifact logs, replay-only gates, 0 network,
> deterministic forward. All quality iteration runs on the cheap **trend
> row** (41q/201,233-entry verified prefix of the D1 pool); only the
> final frozen gate replays the full D1 row.

## 1. Hypothesis

Reference-class retrieval quality (semantic-real+c2f = 0.438 on the D1
row) is reachable **inside** the zero-dep engine by three independently
killable moves, ordered by predictability:

- **M1 — better weights, same machinery.** A cleanly-licensed (MIT/
  Apache) mid-size encoder served through the existing `skinki-encoder`
  forward. First candidate: **multilingual-e5-small** — 12 layers ×
  hidden 384 = *the same per-token compute as bge-small* (the extra
  ~90M params are the 250k-vocab embedding table, which is a lookup,
  not FLOPs — mmap-class cost), multilingual for free, "query: " /
  "passage: " prefix discipline (both sides — the 1C-B prefix lesson,
  generalized). Escalation candidates for quality: e5-base / gte-base /
  arctic-embed-m class (~110M non-embedding-heavy — needs M2 or int8
  to fit the latency budget).
- **M2 — closed-form space bridge for query latency.** Embed queries
  with the *small* in-engine encoder, map them into the *big* model's
  space via a linear map fitted offline by least squares on ~100k
  locally-computed embedding pairs. No GPU, no training loop, no new
  deps: the fit is dev tooling (1B `distill` shape), the map is a
  384×D f32 matrix inside the artifact, the apply is one GEMV —
  deterministic by construction. Kill-switch: if the bridged query
  loses > 10% of the big tower's own-query recall on the trend row,
  the bridge is dead and queries pay the big-tower latency instead.
- **M3 — int8 escalation (design ticket, human-approved 2026-07-05).**
  Quantized weights + i8→i32 blocked GEMM in safe Rust (autovectorized;
  **no SDOT `unsafe` without a separate human D-ticket**). Integer
  accumulation is *more* deterministic than f32, not less. Only needed
  if an M1-escalation model busts latency/backfill budgets at f32.

Falsifiable per move, and jointly: if no combination clears the §2 bars
on the D1 row, the honest record is that engine-internal encoding caps
below reference class, and the served default remains
`RRF(BM25 + best-available)` with the measured number — not a fudged bar.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Pooled multi-session recall@10, `rrf(bm25+encoder)` | **≥ 0.30** on the D1 row (inherited second bar; stretch 0.35; reference 0.438) | `longmemeval-eval --pooled` replay, `rrf(bm25+real)` column |
| Pooled multi-session recall@10, encoder solo | reported, no bar (solo is not the served config; fusion is) | same run |
| Query embed latency (≤ 32 tokens, warm) | p95 ≤ 50 ms via M2 bridge or small tower; ≤ 150 ms hard cap via big tower (global retrieval budget) | telemetry over the eval query set |
| Backfill throughput (sleep-time, M1-class) | 5M ≤ 10 days aggregate, interruptible (Stage 4) | `encoder-bench` + measured dump rates |
| Multilingual sanity | tokenizer + e2e parity vs HF reference on a RU/DE/ES/EN golden set (retrieval-quality eval per language = recorded gap, no public LongMemEval analog) | converter golden dumps + `#[ignore]` parity test |
| Artifact size | ≤ 500 MB on disk, embedding table mmap-resident only under load | loader + telemetry |
| Bit-determinism | byte-identical across runs / thread counts / arch (inherited CI property tests) | CI |
| Deps / unsafe / network | **unchanged: none / quarantine only / 0** | review + CI |

## 3. Task decomposition (draft — tickets firm up after K0)

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| **K0 kill-switch: SentencePiece Unigram tokenizer** in pure Rust (~500 lines; 1C-B T5's port, promoted) | impl | frontier-reviewed | byte-parity vs HF tokenizer on a 1k-string multilingual golden corpus; deterministic |
| T1 multilingual-e5-small: converter path (mean pooling flag, query/passage prefixes in `EmbedderSpec`), artifact + layer/e2e goldens | impl | cheaper | parity ≥ 0.999 cosine on goldens; loads under §2 budgets |
| T2 trend-row eval of T1 (prefixed both sides) + fusion | impl | cheaper | trend-row table recorded here; GO/NO-GO vs bge-small numbers |
| **D1 bridge design**: pair-corpus recipe, ridge/Procrustes choice, artifact layout for the 384×D map | design | **frontier** | written before T3; kill criterion frozen (≤ 10% recall loss vs big-tower queries) |
| T3 bridge fit (dev tooling, offline, no GPU) + `SKENC001` map section + GEMV apply | impl | cheaper | trend-row bridged-query recall within kill criterion; determinism tests |
| T4 (conditional) escalation model at f32; **T5 (conditional) int8 GEMM bench + quantized forward** — T0-style sustained bench *before* the forward is written | impl | frontier (T5 core) | bench table here; §2 latency/backfill green |
| T6 (exploratory, delegatable) sleep-time doc2query spike: local-LLM generated per-session questions → artifact log → indexed alongside entries; measures BM25-only lift first | impl | cheaper | trend-row lift recorded; replay-only; zero query-time cost |
| **D2 verdict**: served-default decision on §2 bars (full D1-row replay) | design | **frontier + human** | numbers + decision recorded here |

## 4. Out of scope

- The LLM serving stack (external interface by human decision — not this
  stage, not this engine).
- SDOT/AMX/Metal `unsafe` (separate D-ticket + human approval if int8
  autovectorization proves insufficient).
- Fine-tuning / training beyond the closed-form bridge fit.
- Push-policy / sleep-scheduling productization (Stage 4B/7 territory).
