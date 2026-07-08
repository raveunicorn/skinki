# PERF — cold indexing path ≥ 10× (profiling record + results)

**Goal.** Cut the cold indexing path (N text entries → searchable
BM25 + embedding + store + graph index, N = 201k…5M) by ≥ 10× with **zero**
retrieval-quality loss and **zero** loss of bit-determinism.

**Result (this branch, measured 2026-07-08, 8P+2E M-series dev box).**

| Metric (bge-small, 32 lme_pooled entries ~1.2k chars) | main | branch | speedup |
| --- | --- | --- | --- |
| wall, `--threads 4` (the config all D1/D2/T2 runs used) | 10.56 s | 2.44 s | 4.3× |
| wall, `--threads 8` (now scales: dynamic scheduling) | ~10.5 s¹ | **1.19 s** | **8.9×** |
| CPU per entry | 0.99 s | 0.225 s | 4.4× |
| **201k-row throughput (load amortized), old t=4 → new t=8** | 3.3 e/s | ~40 e/s | **~12×** |
| e5-small wall t=4 → t=8 | 12.22 s | 1.47 s | 8.3× |
| corpus `generate` 200k | 16.4 s | 0.11 s | 149× |

¹ old static-band `embed_batch` gains nothing from E-cores at small batches
(join waits on the slowest band); measured old-t4 is its realistic best here.

The 201k encoder row drops **~18.5 h → ~1.4 h** on this box. On the M1 Air
target (4P+4E) the projected wall win is ~6–7×; the remaining gap to 10×
there is the quarantined-`unsafe` sdot escalation in §5.

**Quality / determinism evidence** (the project's own bars):

- teacher parity: layer bar ≤ 5e-3 green, e2e **min cosine 1.0000000** on
  both artifacts (bge, e5) — unchanged from the D2-recorded value;
- **min cosine 1.000000000 between `main`-binary and branch-binary
  embeddings of 32 real lme_pooled entries, both models** — retrieval
  behavior is provably unchanged;
- byte-determinism: run-to-run and `--threads 1/4/8` outputs `cmp`-equal;
  `fma`/`max` are IEEE correctly-rounded/order-independent on every target,
  so cross-platform byte-identity is preserved by construction;
- 302 workspace tests, clippy `-D warnings`, fmt, and all 8 CI gates PASS;
- `fixtures/encoder_toy_golden.f32` re-blessed **once** (numerics contract
  change, see §3) via the sanctioned `gen_toy_golden -- --ignored` path.

---

## 1. Where the time went (profiling record)

Cold path at ~200k entries: the encoder **is** the cold path — 0.33 s/entry
(⇒ 18.5 h @201k) vs BM25 0.39 s *total*, store ingest 2 M units/s, graph
0.12 s. `sample` top-of-stack, bge t=4, evolution across the branch:

| symbol | main | after tranche 1 | after tranche 2 |
| --- | --- | --- | --- |
| `gemm_serial` | 49% (26 GF/s/thr) | 51% (50 GF/s/thr) | ~68% (79 GF/s/thr) |
| static-band join idle | 21% | — (dynamic) | — |
| `forward_states` (attn loops) | 16% | 26% | 3% (now GEMMs) |
| `softmax_row` + exp | 10% | 24% (f64) | ~15% (f32) |
| `erf64`/GELU | 4% | 10% | ~2% (f32) |

## 2. Tranche 1 — bit-identical to `main` (commit 1)

Byte-compared equal to `main` outputs; toy golden green **without**
re-blessing. Register-tiled GEMM (C tile in registers, exact ascending-`p`
order — 26→50 GF/s/thread), attention QK ILP + register AV, `exp64x4`
lane-interleave, dynamic `embed_batch` (atomic counter; output slot `i` is a
pure function of text `i`, scheduling can't touch determinism), BufWriter
corpus write, `encoder-embed` block streaming with progress/ETA.

## 3. Tranche 2 — deterministic contract change, one golden re-bless (commits 2–3)

The old contract (separate mul+add, f64 elementwise) capped at physics:
no-FMA f32 peak is 51 GF/s/core, and `exp64`'s `r/k` Horner serialized 12
f64 divides on the one divide port. The new contract is equally
deterministic and byte-stable across platforms:

- **fused `mul_add` everywhere hot** (gemm, attention, exp/erf Horner) —
  IEEE correctly-rounded on every target;
- **gemm NR 8→16**: the per-(i,j) ascending-`p` chain is latency-bound
  (~4 cyc), so saturating 4 FMA pipes needs latency×pipes = 16 independent
  vector accumulators → 48→79 GF/s/thread measured;
- **attention as per-head GEMMs**: scores = Qh·Khᵀ, ctx = P·Vh on packed
  panels — gemm's fused ascending-reduction *is* the order the scalar loops
  used (bit-identical step, no extra re-bless), at microkernel throughput;
- **f32 softmax + GELU** (`exp32xn`, `erf32xn`): the torch teacher's own
  kernels are f32 — the f64 intermediate was extra precision the parity bar
  never asked for, at double SIMD width and Horner depth. Divide-free
  precomputed-1/k! Horner; softmax max-pass vectorized (max is
  order-independent — identical value); sums stay strictly left-to-right.

## 4. Falsified: int8 inference in safe Rust (measured, do not re-try)

Prototype (scratchpad `i8bench`): best safe-Rust i8×i8→i32 GEMM shape
autovectorizes to ~47 GMAC/s = **93 GOP/s ≈ 1.2×** the 79 GF/s f32 kernel —
LLVM does not select NEON `sdot` from scalar reduction loops
(`-C target-cpu=native` doesn't change it). The 4× int8 story requires
`std::arch` intrinsics ⇒ `unsafe` ⇒ AGENTS rule-4 human approval. Dynamic
activation quant also risks the ≥ 0.999 parity bar (BERT activation
outliers). **Skip int8 unless §5 is approved.**

## 5. Remaining headroom (for a future ticket, needs human sign-off)

1. **Quarantined `unsafe` sdot microkernel** (like the mmap quarantine):
   `vdotq_s32` int8 GEMM ≈ 3–4× the f32 kernel on the 68% GEMM share ⇒
   ~1.8–2.2× e2e; integer accumulation is order-free, determinism is
   *stronger* than fp. Quality gate: e2e cosine ≥ 0.999 + trend-row `rrf`
   hold. This is what closes 10× on the M1 Air.
2. Weight panel pre-packing at load (NR-blocked B layout, bit-exact,
   ~+10–15% GEMM).
3. QKV fused single GEMM (bit-exact, ~5%).
4. BM25 parallel shard-and-merge at 5M (deterministic merge; matters only
   once embeddings stop dominating).

## 6. Verification protocol (reusable)

1. `cargo test` workspace + clippy + fmt + all 8 CI `--assert-gate`s.
2. Byte-compare embeddings vs previous binary (32 lme_pooled × {bge, e5});
   where bits legitimately change, min-cosine vs previous binary instead
   (this branch: 1.000000000) + teacher parity (layer ≤ 5e-3, e2e ≥ 0.999).
3. Thread-count invariance: t=1/4/8 outputs `cmp`-equal.
4. Toy golden re-bless only together with the full evidence above in the
   same PR (`gen_toy_golden -- --ignored`).
