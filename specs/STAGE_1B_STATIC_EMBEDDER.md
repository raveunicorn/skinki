# Stage 1B — Static-distilled semantic embedder in pure Rust + the production retrieval path (SPEC)

> Batch 2 of the 2026-07 review (`REVIEW_FRONTIER_2026_07.md` §2). Closes
> README open-problem #4 ("Port EmbeddingGemma to Rust or as a sidecar") the
> Law-1 way: the *model runs once, offline*; the engine ships a **static
> token→vector table** distilled from it (Model2Vec recipe — the L2a plan in
> `ARCHITECTURE.md` that was never built). Then the production path finally
> serves what the benchmarks measured: a real semantic embedder + the Stage-1
> IVF index + coarse-to-fine.

- **Status:** ready to build
- **Owner of the design (frontier/human):** frontier — the artifact format, the
  tokenizer contract, and the parity-gate design are locked below. The
  distillation recipe is standard (Model2Vec).
- **Delegatable to (cheaper model):** **yes** — T1–T6 are mechanical behind the
  parity gate. D1 (quality verdict + bar freezing) is frontier.

> Read [`../AGENTS.md`](../AGENTS.md). The distillation script runs **offline,
> once**, outside any gate (rule-3 shape: the artifact is the replay). Gates
> consume the checked-in artifact + goldens only; 0 network in CI. No new Rust
> dependencies (the Python script may use `sentence-transformers`; it is
> dev-tooling, not a runtime dep).

## 1. Hypothesis

A static embedder distilled from EmbeddingGemma (token embeddings → PCA →
Zipf reweighting, per Model2Vec) and executed in ~200 lines of pure Rust
(BPE encode → table lookup → weighted mean → L2 norm) retrieves **markedly
better than both the hash embedder (LongMemEval multi-session recall@10
0.068) and BM25 (0.193)**, at deterministic, sub-millisecond query embedding
within the RAM budget — making a *real* semantic retriever the engine's
default without violating the minimal-deps law. Falsifiable: if the distilled
table cannot beat BM25 on LongMemEval multi-session, static distillation is
insufficient and the honest fallback is an out-of-process sidecar (recorded,
not hidden).

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| **Cross-impl parity** | exact | Rust embeddings byte-equal the Python reference on the golden strings (f32-exact: the reference dumps f32 LE; Rust reproduces bit-for-bit) |
| **LongMemEval multi-session recall@10 (single-shot)** | **≥ 0.22** (must beat BM25 0.193; hash is 0.068) | `longmemeval-eval --pooled --question-type multi-session --retriever semantic-static-v2` |
| **LongMemEval multi-session recall@10 (coarse-to-fine)** | **≥ 0.30** first bar (semantic-real+c2f was 0.438 with the live model) | same harness, `--strategy coarse-to-fine` |
| No single-session regression | ≥ hash embedder per type | per-type table |
| Artifact size | ≤ 48 MB on disk | file size |
| Resident RAM | ≤ 64 MB (mmap the table) | telemetry |
| Query embed latency | p95 ≤ 1 ms | telemetry over the eval query set |
| Corpus embed throughput | ≥ 10k entries/s | `--scale-report` |
| Determinism | byte-identical embeddings across runs/platforms | golden hash |
| IVF serving parity | IVF-served top-k ⊇ ≥ 95% of brute-force top-k on the eval set | new integration test |

> The two LongMemEval rows cannot run in CI (dataset not redistributable);
> they are gated **locally** via the runbook (T6) and their measured numbers
> are recorded in this spec. CI asserts parity, goldens, size, latency.

## 3. Public interface

Artifact format `SKEMB001` (little-endian throughout):

```
magic "SKEMB001" | version u32 | dim u32 | vocab u32 | flags u32
tokenizer section: vocab strings (len-prefixed UTF-8, id = order) |
                   merges count u32 | merges (pairs of token ids, BPE order)
table: vocab × dim f32   (unit-norm rows)
weights: vocab f32       (Zipf/SIF down-weighting, pre-multiplied out of table
                          is NOT allowed — weights apply at pooling time)
```

```rust
// skinki-vector/src/static_embed.rs  (#![forbid(unsafe_code)] respected;
// mmap via the existing quarantined store module only)
pub struct StaticEmbedder { /* mmap'd table + tokenizer */ }

impl StaticEmbedder {
    pub fn load(path: &Path) -> io::Result<Self>;
    pub fn dim(&self) -> usize;
}
impl Embedder for StaticEmbedder {
    /// BPE-encode `text` (lowercase NFC), look up each token's row, sum
    /// weighted by `weights[token]`, divide by the weight sum, L2-normalize.
    /// Empty/OOV-only text -> the zero vector.
    fn embed(&self, text: &str) -> Vec<f32>;
}
```

Distillation (dev tooling, checked in but never run by CI):

```
scripts/distill_static_embedder.py
  --teacher google/embeddinggemma-300m --dim 256 --out model.skemb \
  --golden-out golden_embeddings.f32   # the parity fixture
```

Serving path: `SemanticRetriever` (now in `skinki-baseline`, per Stage 5C T7)
gains an index-backed mode — above `IVF_THRESHOLD = 50_000` entries it builds
the Stage-1 two-stage/IVF index (`skinki-vector::ivf`) instead of the
brute-force scan; below it, brute-force (exact) stays. `skinki-mcp` and the
harness take `--embedder static:<path>|hash` with `static` the default when a
model file is present.

## 4. Invariants (must always hold)

- The gate never runs the teacher model; the artifact + goldens are the replay.
- Rust embedding is bit-deterministic (fixed summation order: token order as
  produced by the tokenizer; f32 accumulation left-to-right; no fast-math).
- Tokenization is defined by the artifact (vocab+merges), not by any external
  tokenizer crate — the Python script dumps them; Rust reimplements BPE
  greedy-merge exactly (lowercase, NFC, byte-fallback per the dumped config).
- The artifact is versioned and hash-pinned; loading stamps a
  `MethodStamp{ id: M_EMBEDDER, version }` so a future ledger integration can
  flag embeddings as stale when the model artifact changes.
- No new Rust deps; mmap stays inside the existing quarantine.
- **License check (human):** distilled Gemma vectors inherit Gemma's terms of
  use — a human confirms redistribution terms before the artifact is committed
  or published. If redistribution is disallowed, ship the *script* + a
  download-and-distill step instead of the artifact (gate then keys on a
  locally-produced artifact hash).

## 5. Test plan

- **Unit:** BPE encoder against 20 dumped (string → token-ids) fixtures incl.
  Cyrillic, emoji, empty, whitespace-only; pooling math on a 3-token toy table
  computed by hand; zero-vector on OOV-only input.
- **Golden (parity):** 32 fixed strings (English + Russian + mixed) → Rust
  output bytes == `golden_embeddings.f32` from the Python reference.
- **Property:** `embed` is pure (same input twice → identical bytes);
  cosine(embed(s), embed(s)) == 1 for non-empty s.
- **Integration:** IVF-served retriever vs brute-force on a 60k synthetic
  corpus: top-10 overlap ≥ 95% per query, p95 latency reported.
- **Metric (local runbook):** the two LongMemEval rows in §2.
- **Gate command:** `cargo test -p skinki-vector static_embed` +
  `cargo run --release -p skinki-harness -- longmemeval-eval --pooled
  --question-type multi-session --retriever semantic-static-v2 --assert-gate`
  (the flag is added by D1 after the first measured margin).

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `distill_static_embedder.py` (Model2Vec recipe: token-embed the vocab through the teacher, PCA→dim, Zipf weights) + artifact writer + golden dump | impl (dev tooling) | cheaper | artifact + goldens produced; format matches §3 byte layout |
| T2 `StaticEmbedder` in Rust: format reader (mmap), BPE encoder, pooling | impl | cheaper | §5 unit + parity goldens green |
| T3 wire `--embedder` into `skinki-mcp` + harness; static becomes default when the model file exists | impl | cheaper | MCP unit tests green; hash path still available |
| T4 IVF-backed `SemanticRetriever` mode (threshold switch, index build/load) | impl | cheaper | §5 integration test green; latency reported |
| T5 coarse-to-fine productionization: per-session/instance pooling as the documented strategy flag; runbook for the LongMemEval dataset (download, layout, exact commands) — this also discharges HANDOFF 3B's "gate the 0.438" TODO | impl | cheaper | `--strategy coarse-to-fine` reproduces the measured number ±0.01 with the live-model embeddings; with static embeddings hits the §2 bar |
| **D1** quality verdict: run §2's LongMemEval rows, record the margins in this spec, freeze the `--assert-gate` bars (never lower later) | design | **frontier** | numbers recorded; bars frozen; fallback decision (sidecar) taken only if the hypothesis failed |
| T6 (optional, measure-first) late-interaction rerank: keep per-token vectors for the top-50 coarse candidates, MaxSim rerank; measure on multi-session | impl | cheaper | measured lift recorded (adopt only if > +0.02 recall@10) |

## 7. Definition of done

- [ ] Parity + goldens + IVF integration green in CI; LongMemEval margins
      measured, recorded here, and locally gated.
- [ ] `skinki-mcp` default retriever is the static embedder (hash relegated to
      `--embedder hash`); README honest-status row updated with the measured
      number ("the served retriever" vs "the benchmarked retriever" gap closed).
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [ ] Decision recorded: static distillation sufficient (by what margin) or
      sidecar fallback (why).

## 8. Out of scope

- Running any transformer at engine runtime (that is the sidecar fallback,
  a separate decision).
- Fine-tuning / training custom embedders.
- The end-to-end QA gate (Stage 5D) — it consumes this stage's retriever.
- Multilingual quality *measurement* (no labeled Russian benchmark yet; the
  tokenizer/parity tests cover Cyrillic correctness, not quality).
