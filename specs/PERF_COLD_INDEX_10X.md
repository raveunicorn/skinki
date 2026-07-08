# PERF — cold indexing path ≥ 10× (profiling record + staged plan)

**Goal.** Cut the cold indexing path (N text entries → searchable
BM25 + embedding + store + graph index, N = 201k…5M) by ≥ 10× with **zero**
retrieval-quality loss and **zero** loss of bit-determinism.

**Status.** Tranche 1 (this PR) is landed and measured: **bit-identical
outputs** (byte-compared embeddings for bge + e5, byte-compared corpus,
toy-golden regression, teacher parity) at **~2.1× encoder wall / 2.0× CPU**
and **149× corpus generation**. Tranches 2–3 (the road to ≥ 10×) change
*which* bits are produced — still fully deterministic, but they require
re-blessing goldens and re-running the quality row, i.e. a human decision.
They are specified below with measured physics, not vibes.

---

## 1. Where the time actually goes (measured, M-series, 2026-07-08)

Cold path at ~200k entries, release build, per stage:

| Stage | Cost @200k | Cost @5M (extrapolated) | Verdict |
| --- | --- | --- | --- |
| corpus `generate` (write) | 16.4 s → **0.11 s** after fix | ~7 min → ~3 s | was 94% syscalls (unbuffered `serde_json::to_writer`); fixed in this PR |
| BM25 `index()` | 0.39 s (incl. load + eval) | ~10–60 s | not a bottleneck; parallel-shard plan noted §4 |
| L0/unit store ingest | 2.0 M units/s | ~2.5 s | not a bottleneck |
| graph build (`graph-eval`) | 0.12 s @11.5k units | ~1 min | not a bottleneck |
| **encoder embeddings** | **0.33 s/entry wall (t=4) → ~18.5 h @201k** | **weeks** | **the cold path ≈ the encoder, 3 orders of magnitude over everything else** |

`sample` profile of `encoder-embed` (bge-small, 4 threads), top-of-stack:

- before: `gemm_serial` 49%, `__ulock_wait` 21% (static-band imbalance),
  `forward_states` 16%, `softmax_row` 10%, `erf64` 4%
- after tranche 1: `gemm_serial` 51% (now at ~50 GF/s/thread), `forward_states`
  26%, `exp64x4` 16% (f64-divide-port bound), softmax/gelu remainder ~5%

## 2. Tranche 1 — bit-identical rewrites (THIS PR, landed)

Every item below produces **byte-identical output** to `main` (verified: 32
lme_pooled entries × {bge, e5} → `cmp` equal; `toy_golden_regression` green
without re-blessing; layer/e2e teacher parity green; 200k corpus `cmp` equal).

1. **GEMM register-tiled microkernel** (`skinki-encoder/src/gemm.rs`).
   The old kernel streamed every C row through cache once per `p` (2 flops
   per C-load+C-store+B-load ⇒ memory-bound at ~26 GF/s/thread). The new
   MR×NR=4×8 microkernel keeps the C tile in registers across the whole K
   range. Per `(i, j)` the f32 additions run in the *identical* ascending-`p`
   order (a register accumulator vs a store/reload round-trip is exact), so
   `kernel_matches_documented_order_bit_exact` passes unchanged. Now sustains
   ~50 GF/s/thread — **the no-FMA f32 peak of one M1 core is ~51 GF/s**, i.e.
   this kernel is done; further f32 gains require FMA (tranche 2).
2. **Attention QK^T 4-way ILP** (`encoder.rs`): four independent `q·k_j`
   accumulator chains interleaved; each dot still sums its own `hd` elements
   strictly left-to-right → bit-exact, but the latency-bound single chain
   becomes 4 parallel chains.
3. **Attention AV register accumulation** (`attn_weighted_sum::<32|64>`):
   const-generic monomorphized paths keep the context row in registers;
   ascending-`j` order per lane unchanged; generic fallback for other `hd`.
4. **`exp64x4` / `erf64x4` / `gelu_slice`** (`math.rs`): four interleaved
   lanes, per lane the identical straight-line op sequence (asserted
   bit-exact by dense-sweep tests). Modest win only — the Horner `r/k`
   divides bound on the single f64 divide port (see §3).
5. **`embed_batch` dynamic scheduling**: workers pull texts off an atomic
   counter instead of owning static bands (text length varies ⇒ seq² work
   varies ⇒ the old join wasted ~21% of wall on small batches). Output slot
   `i` is a pure function of text `i` — scheduling cannot touch determinism.
   Also makes `--threads 8` scale honestly (E-cores join late-arriving work).
6. **Harness**: corpus write buffered (16.4 s → 0.11 s at 200k); `encoder-embed`
   streams in 512-text blocks with progress + ETA on stderr (telemetry only;
   block boundaries don't touch the numbers).

**Measured tranche-1 results** (32 lme_pooled entries, incl. ~1 s model load):

| Config | before | after | wall speedup |
| --- | --- | --- | --- |
| bge-small, t=4 | 10.56 s | 4.92 s | **2.15×** (CPU 2.0×) |
| e5-small, t=4 | 12.22 s | 6.04 s | **2.02×** |
| bge-small, t=8 | — | 2.87 s | 3.7× vs old t=4 |
| 201k-entry row, bge, t=4 (projected) | ~18.5 h | ~8.5 h | 2.2× |

## 3. Why bit-identical stops at ~2×: the physics

- f32 mul+add as *separate* ops (rustc never contracts to FMA) peaks at
  16 flops/cycle/core ≈ 51 GF/s on M1. The GEMM is there now. **2× more GEMM
  requires FMA**, which changes rounding (fused) ⇒ different (still
  deterministic) bits.
- `exp64`'s Horner uses `r/k` per term: 12 sequential f64 divides through a
  single divide port. ILP across lanes doesn't widen the port. Killing the
  divides (precomputed reciprocal coefficients) changes bits.
- Everything else (LayerNorm, pooling, residuals) is already vectorized or
  negligible.

So: bit-**identical** ceiling ≈ 2×. Bit-**deterministic** (but re-blessed)
tiers below get to ≥ 10×.

## 4. Tranche 2 — deterministic, bits change, quality provably unchanged

Requires: regenerate `fixtures/encoder_toy_golden.f32` (one `--ignored` test),
re-run teacher parity (bar: layer ≤ 5e-3, e2e cosine ≥ 0.999 — the actual
quality contract), re-run the 41q/201k trend row (`rrf(bm25+real)` must not
drop). All outputs stay byte-identical across runs/threads/platforms.

1. **FMA GEMM** (`f32::mul_add`): fused rounding is IEEE-exact and identical
   on arm64/x86_64 (hardware FMA; softfloat `fma` is also correctly rounded).
   GEMM 51 → ~100 GF/s/thread. Expected e2e ≈ 1.6–1.8×.
2. **Divide-free `exp64`** (precomputed-coefficient Horner) + f32 softmax/GELU
   in the style PyTorch itself uses (the teacher runs f32 kernels — f64
   intermediate is *extra* precision the parity bar never asked for).
   Expected e2e ≈ 1.2–1.4× on top.
3. **QK^T / AV via the FMA microkernel** (batch heads as small GEMMs).

Combined tranche 1+2 estimate: **~4–5× wall**, still pure fp.

## 5. Tranche 3 — int8 inference (the actual 10×; aligns with STAGE_1D M3)

Stage 1D already carries a human-approved int8 escalation (M3, safe Rust
only). Extending it from storage to **inference** is the only lever with
another 3–4× on this hardware:

- i8×i8→i32 dot products autovectorize to NEON `smull/sadalp` (or `sdot`)
  from safe Rust; **integer accumulation has no rounding order at all** —
  determinism is *stronger* than fp, byte-identical across platforms by
  construction.
- Per-channel weight quant + per-token activation quant on the GEMM inputs,
  f32 residual stream (LayerNorm/softmax/GELU stay fp) — the standard
  BERT-int8 recipe that holds cosine ≈ 0.999+ vs f32.
- Quality gate: same teacher-parity harness + trend-row `rrf` bar; kill the
  move if recall drops at all (1D discipline: killable move, cheap row first).
- Peak: ~4× the f32-FMA rate on M1 (`sdot` 128 int-ops/cycle/core).

Combined estimate vs today's `main`: **GEMM-bound path ~8× · scheduling/
elementwise wins ⇒ ≥ 10× wall** on the 201k row (18.5 h → < 2 h), more with
`--threads 8`.

## 6. Non-encoder follow-ups (cheap, independent, do when touched)

- BM25 `index()`: parallel shard-and-merge (deterministic merge by term then
  entry id) + `to_lowercase` only when a token contains uppercase — ~5–10×
  at 5M, matters once embeddings stop dominating.
- `skinki-store` ingest and graph build are already within budget at 5M.
- IVF build (k-means) at 5M: profile when Stage-3 co-design lands; today it
  runs on synthetic scale-bench only.

## 7. Verification protocol used (and to reuse for tranches 2–3)

1. `cargo test` workspace + clippy + fmt + the CI `--assert-gate` set.
2. Byte-compare: `encoder-embed` on 32 lme_pooled entries × {bge, e5} vs the
   previous binary (`cmp entries.f32`). Tranche 1: **equal**. Tranches 2–3:
   expected to differ → then run parity + trend row instead.
3. `toy_golden_regression` — byte regression vs committed golden. Tranche 1:
   green unchanged. Tranches 2–3: re-bless via `gen_toy_golden -- --ignored`
   in the same PR as the parity/trend evidence.
4. Thread-count invariance: `embed_batch` t=1 vs t=4 vs t=8 byte-equal
   (existing test + measured on real artifacts).
