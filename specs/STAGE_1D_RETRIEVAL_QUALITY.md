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
- **M3 — int8, default-on (human decision 2026-07-05: "не сжать —
  расточительство").** Quantized weights + i8→i32 blocked GEMM in safe
  Rust (autovectorized; **no SDOT `unsafe` without a separate human
  D-ticket**). Integer accumulation is *more* deterministic than f32,
  not less. Sequencing is Law-2-mandatory: the f32 baseline (T1/T2)
  lands first *because it is the ruler* — int8 ships only if its
  trend-row `rrf` recall drop vs own-f32 is ≤ 0.01 and embeddings stay
  byte-deterministic; otherwise f32 stays and the delta is recorded.
  Wins either way: 4× artifact/RSS (472 MB → ~120 MB for e5-small),
  2–3× GEMM as measured, and headroom for the base-class escalation.

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
| ✅ **K0 kill-switch: SentencePiece Unigram tokenizer** in pure Rust (1C-B T5's port, promoted). Design constraints: Rust never parses the SP protobuf — the converter (Python dev tooling) extracts vocab+scores **and the `precompiled_charsmap`** into our own table format; Rust normalization = trie longest-match over that shipped table + SP whitespace/`▁` handling (NOT a hand-rolled NFKC); Viterbi over log-probs with fixed tie-break | impl | frontier-reviewed | **Done (2026-07-05, reviewed + merged).** `skinki_vector::unigram` + `SKUNI001` table (`scripts/dump_unigram_fixtures.py`; darts-clone charsmap decoded from the model file — 224,711 rules; public `nmt_nfkc.tsv` verified to DRIFT from compiled models, never use it). Parity 1305/1305 byte-exact vs `AutoTokenizer` (EN/RU/DE/ES/CJK/emoji/edge cases), `golden_parity` + `real_artifact_loads` `#[ignore]` tests, toy fixture committed. Tie-break: exact score tie → longer piece (derived from `Lattice::Viterbi` candidate order, unit-tested). **Known corner (disclosed):** HF's fast-tokenizers backend diverges from reference `sentencepiece` C++ on adversarial stacked-combining-mark inputs; our Rust matches reference `sentencepiece` (verified bit-for-bit) |
| ✅ **T1 multilingual-e5-small: converter path (mean pooling, query/passage prefixes), artifact + layer/e2e goldens** | impl | cheaper | **Done (2026-07-05).** `SKENC001` bumped to **v2** (header gains `tok_kind` + `query_prefix_len`/`passage_prefix_len` + the prefix bytes; v1 rejected loudly — silent rehash guard). Two tokenizer kinds: `WordPiece` (vocab inline, bge path unchanged byte-for-byte under the new header) and `Unigram` (`<u32 sku_size><SKUNI001 bytes>` inlined — one file = one model, no FS path coupling). New `ArtifactTokenizer` enum abstracts `encode_content` + `bos_id`/`eos_id`; `RustEncoder` forward **unchanged** (already did mean pooling + post-LN; XLM-R is BERT-shaped). Prefix asymmetry fixed at the trait seam: `Embedder::embed_query()` with a default that delegates to `embed()` (no existing impl touched); `SemanticRetriever::search` → `embed_query`, `::index` → `embed`. **Prefixes are a model contract stored in the artifact** (e5: `"query: "` / `"passage: "`), never hardcoded in Rust. `EmbedderSpec::Encoder { path }` + `parse("encoder:<path>")` lights up `locomo-eval` / `longmemeval-eval` / `skinki-mcp` automatically. **K0 landmine honored:** converter's golden dump uses a Rust-convention mirror (`wordpiece_*` for BERT, full `unigram_normalize`+`unigram_segment`+id-offset for XLM-R) and feeds ids straight to `model(input_ids=...)`; `AutoTokenizer.encode` is never called inside the dump. e2e goldens are dumped over *passage-prefixed* text (matches `RustEncoder::embed`). **Measured parity vs the torch teacher: bge-small unchanged (min cosine 1.0000000 over 32 goldens; per-layer max abs ≤ 4.1e-6 across 13 states). multilingual-e5-small: min cosine 1.0000000 over 32 goldens, layer-parity green** — the §2 parity bar (≥ 0.999) cleared with margin on both models. e5 artifact 480 MB (budget ≤ 500). `real_artifact_loads`/`real_e5_artifact_loads` + 4 `#[ignore]` parity tests green. `cargo test`/clippy/fmt clean; deps unchanged (`skinki-baseline → skinki-encoder`, both internal). |
| ✅ **T2 trend-row eval of multilingual-e5-small (solo + RRF fusion)** | impl | cheaper | **Done (2026-07-07).** Trend-row table recorded below; this is the GO/NO-GO input for D2/human, no verdict here. **Step-0 prerequisite fix:** `run_encoder_embed` in `skinki-harness` had been routing `queries.json` through `RustEncoder::embed_batch` → `embed()` (the *passage* prefix, e5's `"passage: "`), which silently zeroed the asymmetric e5 query-prefix gain that T1 baked into the artifact. The encoder crate owns the query/passage split (`embed_query` vs `embed` since 1D-T1); the harness dump path just wasn't wired to it. Fixed in this PR: `dump_embeddings` takes `as_queries: bool`; entries go through `embed_batch` (passage prefix), queries go through `embed_query` (query prefix). Texts in `entries.json`/`queries.json` stay raw — prefixes live in the artifact (`SKENC001` v2), so hand-prepending would double-prefix them. Threaded query path mirrors `embed_batch`'s band partition (sequences only, never per-row arithmetic → byte-identical across `threads`, rule 2). New regression `run_encoder_embed_routes_queries_through_query_prefix` on a prefixed toy WordPiece artifact (Mean pooling — CLS pooling collapses the toy's token-count signal) asserts `entries.f32 == embed()` and `queries.f32 == embed_query()` (and ≠embed), plus 1-vs-4-thread invariance for the query branch. `toy_dup_matches_committed_fixture` guards the deliberate harness-side dup of the toy builder against drift from `skinki-encoder::format::toy`. **Measured trend row (41q / 201,233-entry pool, `multilingual-e5-small`, Mean pooling, prefixed both sides):** recall@10 — bm25 **0.341** / semantic-real **0.394** / `rrf(bm25+real)` **0.423**. Same row's answer@10 / ndcg@10: bm25 0.244 / 0.240; semantic-real 0.317 / 0.261; rrf 0.268 / 0.283. **Vs the inherited `STAGE_1C_B` D2 bge-small row** (same pool, same 41 queries, same BM25): solo recall 0.362 → 0.394 (**+8.8%**, +0.032 absolute); `rrf` recall 0.411 → 0.423 (**+2.9%**, +0.012 absolute); bm25 unchanged at 0.341 (lexical baseline, no encoder in the path — confirms the pool parity). e5 clears bge on both encoder columns with the same per-token compute budget (12×384 forward, FLOPs-for-FLOPs); the prompt did not ask for a GO/NO-GO and this ticket records the number, not the call. **Exact verified commands:** `cargo run --release -p skinki-harness -- longmemeval-eval --path …/longmemeval_m_cleaned.json --pooled --question-type multi-session --limit 41 --dump-texts <WORKDIR>/t2_dump` (Step 1 — verified 201,233 entries / 41 queries, abort bar); `cargo run --release -p skinki-harness -- encoder-embed --artifact fixtures/encoder_e5_small.skenc --texts <WORKDIR>/t2_dump --threads 8` (Step 2 — embeds both sides, query prefix applied automatically; ~8–10 h wall, same FLOPs as bge); `cargo run --release -p skinki-harness -- longmemeval-eval --path …/longmemeval_m_cleaned.json --pooled --question-type multi-session --limit 41 --embeddings-file <WORKDIR>/t2_dump/entries.f32 --query-embeddings-file <WORKDIR>/t2_dump/queries.f32` (Step 3 — columns of interest: bm25, semantic-real, rrf(bm25+real)). Gates: `cargo build && cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --check` green; no gate weakened; no new deps; only `run_encoder_embed` + its tests touched in the harness |
| **D1 bridge design**: pair-corpus recipe, ridge/Procrustes choice, artifact layout for the 384×D map | design | **frontier** | written before T3; kill criterion frozen (≤ 10% recall loss vs big-tower queries) |
| T3 bridge fit (dev tooling, offline, no GPU) + `SKENC001` map section + GEMV apply | impl | cheaper | trend-row bridged-query recall within kill criterion; determinism tests |
| **T5 int8 (default-on, after the T2 f32 baseline)**: quantization recipe in the converter + i8→i32 blocked GEMM bench (T0-style sustained, *before* the forward is written) + quantized forward | impl | frontier (GEMM core + recipe review) | bench table here; trend-row `rrf` drop vs own-f32 ≤ 0.01; byte-determinism preserved; §2 latency/backfill green |
| T4 (conditional) base-class escalation model (e5-base / gte-base / arctic-m) on the int8 path | impl | cheaper | trend-row table; §2 budgets hold |
| T6 (exploratory, delegatable) sleep-time doc2query spike: local-LLM generated per-session questions → artifact log → indexed alongside entries; measures BM25-only lift first | impl | cheaper | **Code landed (2026-07-07, feat/1d-t6-doc2query).** `tools/doc2query-longmemeval.py` (llama-server + ThreadPoolExecutor + incremental per-instance `doc2query.artifacts.jsonl`, resumable — copy of the `extract-graph-llm-longmemeval.py` skeleton). Harness column `bm25+doc2query` via `longmemeval-eval --pooled --doc2query-artifacts <dir>`: BM25 index where each entry's indexed text = entry text + "\n" + its generated questions; entry ids unchanged (retrieval returns the source entry), answer@10 checked against the SOURCE entry text (apples-to-apples vs the `bm25` column). Determinism: replay parses JSONL into a `BTreeMap` keyed by `entry_index`, last-write-wins on dup keys, corrupt/torn final lines skipped — `rebuild(log)` is byte-identical regardless of line order; no LLM in the eval path. Unit tests (5, no LLM): per-instance index→global-id mapping, no-logs-errors-loud, corrupt-line tolerance, expanded-text preserves ids/ground-truth, end-to-end BM25 lift on a synthetic log. **Preliminary numbers (9/41 instances, 44,899 entries, 9 queries, Qwen2.5-0.5B-Instruct-Q4_K_M):** generation 92.3% covered (41,440/44,899), 2.73 q/covered entry. Lift: `recall@10` 0.593→0.500 (−0.093 ⬇), `answer@10` 0.333→0.222 (−0.111 ⬇), `ndcg@10` 0.394→0.411 (+0.017 ⬆). **Negative signal** at 0.5B — generated questions appear to dilute BM25's top ranks (noise > signal at this model size). Consistent with the T-ticket's "if a 0.5B model produces junk questions, that's a recorded finding, not a failure." Full trend row (41q) pending — verdict tentative at 9q (high variance). Replay-only; zero query-time cost |
| **D2 verdict**: served-default decision on §2 bars (full D1-row replay) | design | **frontier + human** | numbers + decision recorded here |

## 4. Out of scope

- The LLM serving stack (external interface by human decision — not this
  stage, not this engine).
- SDOT/AMX/Metal `unsafe` (separate D-ticket + human approval if int8
  autovectorization proves insufficient).
- Fine-tuning / training beyond the closed-form bridge fit.
- Push-policy / sleep-scheduling productization (Stage 4B/7 territory).
