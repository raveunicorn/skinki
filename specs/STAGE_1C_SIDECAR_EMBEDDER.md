# Stage 1C — Sidecar sentence encoder: the real semantic base (SPEC)

> Successor to Stage 1B, whose D1 verdict + addendum falsified *static*
> embedding in all forms (input-table 0.090, canonical Model2Vec 0.116,
> retrieval-tuned 130 MB 0.086 — vs BM25 0.134, bar ≥ 0.22, live-encoder
> reference 0.438). The failure is architectural: context-free token vectors
> lose word order/negation/speaker at encoding time and their NN margin
> collapses at ~600k entries. The fix is a real sentence encoder (attention)
> run **locally** at ingestion + query time — the honest fallback §1 of
> STAGE_1B promised to record, not hide. T8 (RRF fusion, merged) is the
> serving pattern waiting for this discriminative base: the expected
> end-state is `RRF(BM25 + sidecar)`.

- **Status:** draft — T0 (feasibility bench) is runnable now; **D1 (form +
  model choice) blocks everything after it and needs the human** (it may add
  a dependency boundary and a model license to the repo).
- **Owner of the design (frontier/human):** frontier drafts; human approves
  D1 (deps/license are law-level).
- **Delegatable to (cheaper model):** yes — T0, T2–T6 are mechanical behind
  the replay gate. D1/D2 are frontier+human.

> Read [`../AGENTS.md`](../AGENTS.md). Rule-3 shape throughout: the encoder
> runs **outside every gate**; each embedding that feeds the engine goes to
> an **append-only artifact log**; every downstream structure must be
> `rebuild(log)`-deterministic; CI replays logs and never runs inference.
> Network stays 0 (model file is local; downloading it is dev tooling, like
> the LongMemEval dataset).

## 1. Hypothesis

A locally-run sentence encoder (33–120M params), invoked out-of-process at
ingestion and query time and replayed from artifact logs everywhere else,
lifts LongMemEval multi-session pooled recall@10 from BM25's 0.134 to
**≥ 0.22 single-shot** (the inherited 1B bar) and **≥ 0.30 fused as
`RRF(BM25 + sidecar)`** (the T8 pattern), while leaving the engine's idle
RAM, cold start, and minimal-deps law untouched (the encoder lives behind a
process boundary and a log, not in the dependency tree). Falsifiable: if
replayed sidecar embeddings cannot clear 0.22 on the D1 row, the model
choice (not the architecture) is wrong — escalate model size once, then
record failure honestly. The 0.438 semantic-real reference says the
headroom exists.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Pooled multi-session recall@10 (single-shot, replayed sidecar embeddings) | **≥ 0.22** | `longmemeval-eval --pooled` + embeddings/query-embeddings files (existing replay path) |
| Pooled multi-session recall@10, `RRF(BM25 + sidecar)` | **≥ 0.30** | same run, `hybrid-rrf` column (T8, already ships) |
| Query embed latency (warm sidecar, ≤ 32-token query) | p95 ≤ 30 ms | telemetry (fits inside the 150 ms retrieval p95) |
| Ingestion throughput (batched) | ≥ 200 turns/s (5M-unit backfill ≤ ~7 h, sleep-time) | T0 bench, then telemetry |
| Sidecar RSS while loaded | ≤ 400 MB transient; **0 when idle** (TTL-killed) | `ps`/getrusage in bench |
| Engine idle RAM / cold start | unchanged (< 250 MB / < 1 s; engine never waits for the sidecar to answer BM25) | existing telemetry |
| Sidecar cold spawn → ready | ≤ 700 ms (mmap'd weights) | T0 bench |
| Engine workspace deps | unchanged (sidecar binary is external tooling, not a workspace dep) | `Cargo.toml` review |
| Network in engine + CI | 0 | ci.yml review; gates replay logs only |
| Crash behavior | sidecar death → loud log + graceful BM25(+hash) degradation, never a hang | T5 fault-injection test |

## 3. Public interface

```rust
// skinki-baseline (extends the Stage 1B seam; no new crate needed for the client)

/// Grows one variant. `Sidecar` speaks the §4 protocol over stdio to a child
/// process; `cmd` is the launcher line (e.g. "python3 tools/emb-sidecar.py
/// --model <dir>"). parse() syntax: `sidecar:<cmd>`.
pub enum EmbedderSpec { Hash, Static { path: PathBuf }, Sidecar { cmd: String } }

/// The sidecar client implements the existing `Embedder` trait (per-text,
/// query path) plus batching for ingestion:
pub trait BatchEmbedder: Embedder {
    /// One round-trip for many texts (ingestion). Order-preserving.
    fn embed_batch(&self, texts: &[&str]) -> io::Result<Vec<Vec<f32>>>;
}

// skinki-store or skinki-vector (T2 decides the crate; log format is fixed):
/// Append-only embedding artifact log, one JSON object per line:
/// { "v":1, "unit_id":…, "input_sha256":"…", "model":{"id":"…","revision":"…",
///   "dim":384,"quant":"int8"}, "vec_f32le_b64":"…" }
/// `rebuild(log)` → vector index, byte-deterministic given the same log.
pub struct EmbArtifactLog { /* writer/reader, replay iterator */ }
```

## 4. Sidecar protocol (frozen by T2)

JSON-RPC 2.0 over stdio, same hand-rolled shape as `skinki-mcp` (no new
deps). Methods:

- `model_info() → { id, revision, dim, quant, max_tokens }` — stamped into
  every log record; a revision change forces a new log segment (staleness
  propagates through the derivation ledger exactly like any method-version
  bump).
- `embed_batch(texts: [String]) → { vecs: [base64 f32 LE] }` — base64 of the
  raw little-endian f32 vector; floats never round-trip through JSON number
  formatting (replay must be byte-exact).

Process policy: spawn on first need; keep warm; kill after 60 s idle
(engine idle RAM unaffected); on crash, one respawn attempt then loud
degradation to BM25(+hash). The engine never blocks its lexical path on the
sidecar.

## 5. Model candidates (D1 picks; licenses checked at D1)

| Model | Params | Dim | Langs | Tokenizer | License | Note |
| --- | --- | --- | --- | --- | --- | --- |
| bge-small-en-v1.5 | 33M | 384 | en | WordPiece | MIT | benchmark comparability (1B teacher); English-only — not the product model |
| multilingual-e5-small | 118M | 384 | 100+ | SentencePiece (XLM-R) | MIT | product candidate — the corpus is Russian-heavy; tokenizer lives in the sidecar, so no Rust SentencePiece port (the D0 problem stays dead) |
| EmbeddingGemma-300M | 308M | 768 | 100+ | SentencePiece | Gemma terms | the 0.438 reference; license + RSS need D1 scrutiny |

T0 benches all three; D1 may pick two (en for the LongMemEval verdict,
multilingual for the product) — the protocol is model-agnostic by design.

## 6. Invariants

- **Rule 3:** every embedding that feeds engine state is in the log; gates
  replay logs; CI runs zero inference; `rebuild(log)` is byte-deterministic.
- Query-time embeddings are ephemeral (they feed no durable structure), but
  every *verdict* uses logged query embeddings (existing
  `--query-embeddings-file` path) so numbers are reproducible.
- Engine crates keep `#![forbid(unsafe_code)]` status quo; no new workspace
  deps; the sidecar binary is dev/product tooling outside the workspace.
- Mixed-model logs are illegal: one log segment = one `model_info` stamp.

## 7. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T0 (measure-first) feasibility bench: onnxruntime (dev tooling) throughput/latency/RSS for the §5 candidates on M1 (batch + single-stream + cold spawn), plus a pure-Rust blocked-GEMM microbench to ground variant-B numbers | impl | cheaper | numbers table recorded in this spec |
| **D1** form + model decision: out-of-process sidecar (recommended below) vs pure-Rust in-crate forward; model pick + license check; approve the tooling dependency boundary | design | **frontier + human** | decision + rationale recorded here; §5 table updated |
| T2 `EmbArtifactLog` (§3): writer/reader/replay + `rebuild(log)` determinism tests + ledger method-stamp wiring | impl | cheaper | unit + property tests green |
| T3 sidecar reference implementation (dev tooling, e.g. `tools/emb-sidecar.py` over onnxruntime) + §4 protocol client (`BatchEmbedder`) + process supervisor (spawn/warm/TTL/crash) | impl | cheaper | protocol round-trip + fault-injection tests green |
| T4 wire into `SemanticRetriever`/IVF + `hybrid-rrf`; produce the D1-row logs (one-time inference, offline) | impl | cheaper | replayed eval runs end-to-end |
| **D2** quality verdict: run §2 rows from replayed logs, record margins, freeze bars (never lower), take the served-default decision (incl. T8 flip) | design | **frontier** | numbers + decision recorded here |
| T5 MCP/serving integration behind `--embedder sidecar:<cmd>` + degradation tests | impl | cheaper | MCP tests green; BM25 path never blocks |
| T6 (product, after D2) backfill runbook: 5M-unit sleep-time embedding job through the Stage-4 scheduler | impl | cheaper | sim + runbook recorded |

**Frontier recommendation for D1 (to be confirmed by the human):** variant
**A (out-of-process)**. Grounding: variant B (pure-Rust in-crate forward) is
~1–2k lines of GEMM/attention and would be the only bit-deterministic
option, but (1) a naive-to-decent Rust f32 GEMM on M1 sustains ~10–40
GFLOP/s vs onnxruntime INT8 at ~50–100×, turning the 5M backfill from ~7 h
into weeks; (2) the multilingual product model forces a SentencePiece
Unigram port — the exact ~500-line swamp the 1B D0 amendment was written to
avoid; (3) rule 3 already gives us replay-determinism, so B's
bit-determinism buys nothing the log doesn't. B stays recorded as a
portability option (Stage 6F WASM might revive it for tiny models).

## 8. Definition of done

- [ ] D1 + D2 decisions recorded in this spec (form, model, license, bars).
- [ ] §2 gate rows green from replayed logs in CI; zero inference in CI.
- [ ] `cargo test`, `clippy -D warnings`, `fmt --check` clean; engine deps unchanged.
- [ ] README honest-status row + HANDOFF updated with measured margins.
- [ ] Decision recorded: served default (BM25 / hybrid / sidecar) and why.

## 9. Out of scope

- Stage-7 packaging of the sidecar into a macOS app bundle (Swift supervision).
- Fine-tuning or distilling custom models (use published checkpoints as-is).
- Pure-Rust encoder port (variant B) unless D1 selects it.
- Cross-encoder rerankers / late-interaction (T6 of 1B stays parked until a
  discriminative base lands and D2 says the +lift is worth it).
