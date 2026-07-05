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
| T2 trend-row eval of T1 (prefixed both sides) + fusion | impl | cheaper | trend-row table recorded here; GO/NO-GO vs bge-small numbers |
| **D1 bridge design**: pair-corpus recipe, ridge/Procrustes choice, artifact layout for the 384×D map | design | **frontier** | written before T3; kill criterion frozen (≤ 10% recall loss vs big-tower queries) |
| T3 bridge fit (dev tooling, offline, no GPU) + `SKENC001` map section + GEMV apply | impl | cheaper | trend-row bridged-query recall within kill criterion; determinism tests |
| **T5 int8 (default-on, after the T2 f32 baseline)**: quantization recipe in the converter + i8→i32 blocked GEMM bench (T0-style sustained, *before* the forward is written) + quantized forward | impl | frontier (GEMM core + recipe review) | bench table here; trend-row `rrf` drop vs own-f32 ≤ 0.01; byte-determinism preserved; §2 latency/backfill green |
| T4 (conditional) base-class escalation model (e5-base / gte-base / arctic-m) on the int8 path | impl | cheaper | trend-row table; §2 budgets hold |
| T6 (exploratory, delegatable) sleep-time doc2query spike: local-LLM generated per-session questions → artifact log → indexed alongside entries; measures BM25-only lift first | impl | cheaper | trend-row lift recorded; replay-only; zero query-time cost |
| **D2 verdict**: served-default decision on §2 bars (full D1-row replay) | design | **frontier + human** | numbers + decision recorded here |

## 4. Out of scope

- The LLM serving stack (external interface by human decision — not this
  stage, not this engine).
- SDOT/AMX/Metal `unsafe` (separate D-ticket + human approval if int8
  autovectorization proves insufficient).
- Fine-tuning / training beyond the closed-form bridge fit.
- Push-policy / sleep-scheduling productization (Stage 4B/7 territory).
