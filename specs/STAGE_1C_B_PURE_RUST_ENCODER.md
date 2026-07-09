# Stage 1C-B — Pure-Rust sentence encoder: the self-contained variant (SPEC)

> Variant **B** of `STAGE_1C_SIDECAR_EMBEDDER.md`, promoted to its own spec
> by the human decision of 2026-07-04: **try B first, fall back to A** — the
> engine's soul is a self-contained Rust binary ("FFmpeg for personal
> knowledge"), and B is the only variant that is zero-dep, zero-process,
> WASM-portable (Stage 6F), and **bit-deterministic end to end** (rule 2 at
> full strength, not just rule-3 replay). The price is a real one — we write
> a BERT forward pass by hand and its performance is unproven — so this spec
> is built around a **cheap kill-switch measured before the investment**:
> T0 is a ~1-day GEMM microbench; if pure safe Rust cannot sustain the §2
> floor on M1, we stop, record the numbers, and execute variant A unchanged
> (same seam, same logs, same bars — A remains a drop-in fallback by
> construction).

- **Status:** closed / trend-closed. T0, D1, T1, T2, the embedding dump path,
  `EmbedderSpec::Encoder`, and the bge-small D2 trend verdict are done. The
  pure-Rust encoder machinery is kept; bge-small is not the served model. The
  active continuation is [`STAGE_1D_RETRIEVAL_QUALITY.md`](STAGE_1D_RETRIEVAL_QUALITY.md).
- **Owner of the design (frontier/human):** frontier; the B-first strategy
  itself was decided by the human (2026-07-04). D1 go/no-go is frontier +
  human.
- **Delegatable to (cheaper model):** partially — T0 (bench harness), T1
  (format/converter), T3 (batch driver), T4 (wiring) behind golden/parity
  gates. **T2 (the forward pass + in-crate transcendentals) is frontier** —
  it is exactly the kind of subtle numerical core the tier split exists for.

> Read [`../AGENTS.md`](../AGENTS.md). Everything from `STAGE_1C` §6 still
> holds (artifact logs, gates replay, 0 network). B strengthens it: the
> forward pass itself is bit-deterministic, so logs become a cache and an
> audit trail rather than the only source of reproducibility.

## 1. Hypothesis

A hand-written, dependency-free, `forbid(unsafe)` Rust forward pass for a
BERT-class encoder (bge-small-en-v1.5: 12 layers, hidden 384, 21.3M
non-embedding params; WordPiece tokenizer **already shipped by 1B T2**)
can sustain enough throughput on an M1 Air that (a) a query embeds in
p95 ≤ 50 ms, (b) the one-time 5M-unit backfill completes in ≤ 10 days of
interruptible sleep-time (Stage 4 exists for exactly this shape), and
(c) the D1-row quality equals the same model served any other way —
clearing the inherited bars (recall@10 ≥ 0.22 single-shot, ≥ 0.30 as
`RRF(BM25+encoder)`). Falsifiable twice: **T0** — if blocked safe-Rust
f32 GEMM cannot sustain **≥ 40 GFLOP/s aggregate (4 threads, 384-class
shapes, 10-min sustained on passive cooling)**, variant B is dead on
arithmetic and A proceeds; **D2** — if quality or latency bars fail on the
replayed D1 row, same outcome. Arithmetic grounding: one 128-token turn ≈
5.5 GFLOP, so 40 GFLOP/s → ~7 turns/s → 5M in ~8.6 days sleep-time; a
32-token query ≈ 1.4 GFLOP → ~35 ms single-core at 40 GFLOP/s/core-class
rates. M1 P-core peak is ~100 GFLOP/s f32; the bar asks for ~40% of one
core's peak across four throttled cores — aggressive but not heroic.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| **T0 kill-switch: sustained f32 GEMM, 4 threads, 384-class shapes** | **≥ 40 GFLOP/s** (10-min sustained; below → variant A) | `encoder-bench --gemm --full-run --threads 4` (10-min sustained per shape, 4-thread sweep only). **M1 Max, 2026-07-04:** 4-thread min p5 = **46.94 GF/s** (worst case over M∈{32,128}, N∈{384,1536}, K=384); per-shape p5: M32/N384 = 46.94, M32/N1536 = 56.54, M128/N384 = 57.29, M128/N1536 = 61.16. **GATE: PASS** (≥ 40 by ~17%). Note: this is an M1 *Max* (8 P-cores), not the M1 *Air* (4 P-cores) the §1 budget was sized for — the Air will land lower; the §1 arithmetic (~40% of one P-core peak across four throttled cores) should be re-checked against an actual Air before locking D1. |
| Query embed latency (≤ 32 tokens, warm) | p95 ≤ 50 ms | telemetry over the eval query set |
| Backfill throughput (aggregate, sleep-time) | ≥ 6 turns/s sustained (5M ≤ 10 days, interruptible/resumable via Stage 4) | `encoder-bench --backfill-sim` + telemetry |
| Incremental daily ingestion (~300 entries) | ≤ 60 s | derived from throughput bench |
| Pooled multi-session recall@10 (single-shot, replayed) | ≥ 0.22 | `longmemeval-eval --pooled` replay (same as 1C) |
| Pooled multi-session recall@10, `RRF(BM25+encoder)` | ≥ 0.30 | same run, `hybrid-rrf` column |
| Teacher parity | cosine ≥ 0.999 per vector on the 32-string golden set vs the Python f32 reference | conversion-time dump + `#[ignore]` parity test |
| **Bit-determinism** | byte-identical embeddings across runs **and across thread counts (1 vs 4) and platforms (arm64/x86_64 CI)** | property tests in CI |
| Encode-time RSS (weights mmap + activations) | ≤ 300 MB transient; engine idle unchanged (< 250 MB) | telemetry |
| Engine deps | **unchanged — no new crates, `#![forbid(unsafe_code)]`** | `Cargo.toml` + crate attr review |
| Network in engine + CI | 0 | gates replay logs / run goldens only |

## 3. Public interface

```rust
// New crate: skinki-encoder  (#![forbid(unsafe_code)], deps: skinki-vector only)

/// A BERT-class encoder loaded from a `SKENC001` artifact (mmap'd weights).
/// Reuses skinki-vector's WordPiece tokenizer (1B T2) — same vocab section.
pub struct RustEncoder { /* mmap view, header, layer offsets */ }

impl RustEncoder {
    pub fn load(path: &Path) -> io::Result<Self>;
    pub fn dim(&self) -> usize;
    pub fn method_stamp(&self) -> (u32, u64);   // ledger staleness wiring
}
impl Embedder for RustEncoder { /* single text — query path */ }
impl BatchEmbedder for RustEncoder { /* per-sequence forward, threads across
                                        sequences, output order fixed */ }

// skinki-baseline: EmbedderSpec grows `Encoder { path }`, syntax `encoder:<path>`.
```

## 4. `SKENC001` artifact + numerics contract (frozen by T1/T2)

- **Format:** magic `SKENC001` | version | arch tag (`bert`) | dims (layers,
  hidden, ffn, heads, vocab, max_seq) | pooling flag (`cls` — bge — or
  `mean` — e5) | WordPiece vocab section (identical layout to `SKEMB001`) |
  f32 LE tensors at fixed documented offsets (embeddings, per-layer QKV/O,
  FFN, LayerNorm γ/β, final pooling). Converted offline from safetensors by
  `scripts/convert_encoder_to_skenc.py` (dev tooling, like 1B); artifact is
  model weights → gitignored, regenerable; conversion also dumps **per-layer
  golden activations** for one fixed input (the layer-by-layer debugging
  gate) + the 32-string golden embeddings.
- **Numerics:** f32 everywhere; every reduction in a fixed, documented
  order (left-to-right, no pairwise/tree reordering); **no `libm` in the
  forward path** — `exp` (softmax) and `tanh`/`erf` (GELU) are in-crate
  fixed polynomial/rational approximations with recorded max-error bounds,
  so outputs are byte-identical across OS/libc/arch. Threads partition
  *sequences*, never one sequence's arithmetic → thread-count invariance by
  construction.
- **INT8 escalation (recorded, not scheduled):** if T0 passes but backfill
  margins are uncomfortable, NEON `SDOT` via `std::arch` is the one
  sanctioned `unsafe` quarantine candidate (~4× GEMM), behind its own
  D-ticket + human approval. Accelerate/AMX is rejected outright: platform
  dep + no summation-order guarantee = determinism loss.

## 5. Test plan

- **Unit:** header/offset parsing rejects truncation and overflow loudly
  (1B loader lessons applied from day one); LayerNorm/GELU/softmax against
  tiny hand-computed fixtures; approximation error bounds asserted.
- **Golden (layer):** per-layer activations byte-equal to the committed
  conversion-time dump for the fixed probe input — localizes any regression
  to a single layer.
- **Golden (end-to-end):** 32-string embeddings byte-equal to committed
  goldens (regression); cosine ≥ 0.999 vs the Python teacher reference
  (parity) — both `#[ignore]` (need the regenerable artifact), run after
  every regeneration; a **toy** SKENC001 (2 layers, dim 16, seeded) is
  committed for CI-time unit/property tests, mirroring the 1B toy pattern.
- **Property:** determinism (two runs byte-equal; 1-thread vs 4-thread
  byte-equal); purity through `Box<dyn Embedder>`; unit-norm output.
- **Bench gates:** `encoder-bench --gemm --assert-gate` (T0 bar, local
  only — CI machines are not M1 Airs; CI runs it assert-free for smoke).
- **Gate command:** existing replay evals (§2 rows) + `cargo test -p
  skinki-encoder`.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| **T0 kill-switch bench**: blocked/tiled safe-Rust f32 GEMM (the exact loop structure T2 will use) + `encoder-bench --gemm`; sustained 10-min run on the M1 Air at 384-class shapes (K=384, N∈{384,1536}), 1/2/4 threads | impl (bench frontier-reviewed) | cheaper | GFLOP/s table recorded in §2; ≥ 40 sustained → D1 go. **Status:** bench implemented (`crates/skinki-encoder/{gemm,bench}.rs` + `encoder-bench` CLI); **M1 Max 10-min sustained: 4-thread min p5 = 46.94 GF/s, GATE PASS.** Note the M1 Max has 8 P-cores vs the Air's 4; the Air number is still TBD but §1's 40%-of-one-P-core projection was set on the Air's hardware envelope. |
| ✅ **D1 go/no-go** on T0 numbers (+ backfill/latency projections recomputed from measured rates) | design | **frontier + human** | **GO — decided by the human 2026-07-04, on M1 Max data.** T0's question was "is a hand-written encoder worth building": worst-case sustained p5 = 46.94 GF/s with means ≈ 57–61 across shapes answers yes with margin. The M1 Air of §1's original sizing is **not available and not expected** — recorded as a measurement gap, not hidden: the dev/serving machine *is* the Max; Air-class passive-cooling hardware would land ~15–30% lower (possibly at/under the bar) and must be re-benched (one `encoder-bench --gemm --full-run --threads 4 --assert-gate` run) before any Stage-7 claim on such hardware. T3's backfill/latency projections use the measured Max rates **with an explicit ~25% derating column** for Air-class targets. T1/T2 unblocked. |
| ✅ **T1** `SKENC001` format + `convert_encoder_to_skenc.py` + toy artifact + golden dumps (layer + e2e) | impl | cheaper | loader tests + toy goldens green — **done.** `format.rs`: bounds-checked reader (truncation/overflow/trailing-garbage/missing-specials all `InvalidData`, never a panic — 1B loader lessons applied from day one), tensors decoded once into a contiguous f32 arena (safe Rust cannot view mmap bytes as `&[f32]`; ~130 MB arena is inside the §2 RSS budget), W stored `[in][out]` so the forward runs through `gemm` with no transposes, `pos_type_emb` pre-summed with the token-type-0 row. WordPiece tokenizer extracted from `static_embed` into shared `skinki_vector::wordpiece` (1B behavior unchanged — 58 tests green). Converter dumps: 13-state layer goldens for a fixed probe (bug localization per layer) + 32-string e2e goldens with Rust-convention tokenization. Real bge-small artifact converted (133,168,692 bytes) and loads (`real_artifact_loads`, `#[ignore]`); toy fixture (22.6 KB) committed + byte-reproducible |
| ✅ **T2 encoder core**: embeddings→12×(MHA+FFN+LN)→pooling, in-crate transcendentals, fixed-order reductions | impl | **frontier** | **done.** `math.rs`: `exp64` (range reduction + Horner Taylor, rel err < 1e-13, asserted) + `erf64` (A&S 7.1.26, abs err < 1.5e-7) + exact GELU + f64-statistics LayerNorm/softmax — zero `libm` on the forward path. `encoder.rs`: full BERT forward over the T1 arena, every projection via the T0 `gemm` with the pre-filled-bias trick, per-head attention with fixed j-order accumulation, threads partition sequences only. **Measured parity vs the torch teacher (real bge-small artifact): per-layer max abs diff ≤ 4.1e-6 across all 13 states (tolerance 5e-3 — three orders of margin); e2e min cosine over the 32 goldens = 1.0000000 (bar ≥ 0.999).** Determinism: byte-equal across runs, thread counts (1 vs 4), trait objects; Rust-self toy goldens committed for CI regression. **Honest latency note for D2:** warm single query (22 ids) p95 ≈ 67 ms vs the 50 ms bar (small-M GEMM + gelu are the known levers); 128-token turn ≈ 335 ms single-thread → ~12 turns/s at 4 threads (backfill bar ≥ 6 clears ×2) |
| ✅ **T3 batch/dump path**: per-sequence deterministic fan-out + `encoder-embed` replay dump path | impl | cheaper | done; output is byte-identical across thread counts and feeds the existing pooled eval replay path |
| ✅ **T4 wiring**: `EmbedderSpec::Encoder`, query/passage prefix route, `rrf(bm25+real)` replay column | impl | cheaper | done/absorbed into 1C-B and 1D T1/T2; the D2 trend row runs end-to-end |
| ✅ **D2 quality/latency verdict** on bge-small bars; served-default decision | design | **frontier** | done below: bge-small fails by trend, hash remains default, 1D is the continuation |
| ✅ **T5 promoted to 1D**: multilingual escalation via Unigram + multilingual-e5-small artifact | impl | frontier-reviewed | done as 1D K0/T1/T2; see `STAGE_1D_RETRIEVAL_QUALITY.md` |

### D2 verdict (2026-07-05, human decision: trend-close)

**The §2 quality bars are falsified-by-trend for bge-small; the encoder
itself and all its infrastructure are keepers.** Measured on the pooled
**trend row** — 41 multi-session questions over the first 201,233 entries of
the D1 corpus (verified byte-exact prefix of the full 594,708-entry pool;
`--limit` pools are prefixes of larger ones, so partial backfills are
evaluable early):

| column | recall@10 | answer@10 | ndcg@10 |
| --- | --- | --- | --- |
| bm25 | 0.341 | 0.244 | 0.240 |
| semantic-real (bge-small, prefixed queries) | **0.362** | 0.268 | 0.248 |
| `rrf(bm25+real)` (depth = k) | **0.411** | **0.317** | **0.262** |

Findings, in order of importance:

1. **BGE query prefix is mandatory.** Embedding queries raw scored 0.289
   (below BM25); prepending the model's documented instruction
   `"Represent this sentence for searching relevant passages: "` to
   *queries only* (documents stay raw) lifted recall +25% to 0.362. Any
   serving path (T4 `EmbedderSpec::Encoder`) must apply this prefix at
   query time. Entry embeddings are unaffected — backfills stay valid.
2. **Fusion beats both parents on every metric** (+21% recall vs BM25,
   +14% vs encoder solo) — real complementarity, unlike 1B's static
   fusion (+8%, "instrument, not default"). The `rrf(bm25+real)` pooled
   column (PrecomputedSemantic adapter + T8 `RrfFusion`) is the new
   second-bar instrument, landed with this verdict.
3. **The absolute §2 bars are out of reach for this model.** On the full
   row (BM25 = 0.134) the bars demand encoder/BM25 ratios of 1.64×
   (solo ≥ 0.22) and 2.24× (RRF ≥ 0.30); the trend row measures 1.06×
   and 1.21×. No plausible pool-size effect closes a 2× ratio gap —
   33M params is a different weight class from the 300M reference
   (semantic-real+c2f 0.438). Coarse-to-fine *hurts* with this model
   (0.362 → 0.122): 384-dim instance-mean vectors are too weak for the
   coarse stage, confirming the base-model ceiling.

**Decisions (human, 2026-07-05):** the full-row rerun is **waived** — no
decision depends on the exact full-pool number (bge-small will not be the
served default either way), and the opportunistic shard backfill
(resumable runner, 37/115 shards at close) may append the archival number
later. The A-fallback (sidecar) is **rejected as a product shape** for the
embedder: the memory engine stays a self-contained binary; the LLM stays
an external, swappable interface — that boundary is the product. The
earned path forward is **`STAGE_1D`**: mid-size cleanly-licensed
multilingual encoder (T5's e5-small first) + int8 escalation (§9's
recorded escalation, now human-approved for design) + a closed-form
space-bridge for query-time latency. Hash remains the served default
until a 1D candidate clears its bars. Deliverables kept: `skinki-encoder`
(parity 1.0000000, byte-deterministic), `SKENC001` + converter + goldens,
`encoder-embed` dump CLI, the prefix finding, the `rrf(bm25+real)`
instrument, and the prefix-evaluability of pooled backfills.

**Perf update (2026-07-08):** Fable's cold-indexing work keeps the same
quality/determinism contract but makes the 1D measurements practical. The
profiling record in [`PERF_COLD_INDEX_10X.md`](PERF_COLD_INDEX_10X.md) reports
the 201k encoder row dropping from ~18.5h to ~1.4h on the dev box and confirms
teacher parity / thread-count determinism remain green. This does not reopen
bge-small as a served model; it unblocks the full e5 D2 run in 1D.

## 7. Relationship to `STAGE_1C` (variant A)

Everything A defined stays true and shared: the `EmbedderSpec` seam, the
artifact-log format, the replay-only gates, the §2 quality bars, the T8
fusion pattern. B replaces only *where the forward pass runs*. The fallback
is therefore mechanical: if T0 or D2 kills B, variant A's T2–T5 proceed
against the identical seam with the identical bars, losing nothing but the
calendar time B consumed — which is capped by design (T0 first, core last).
`STAGE_1C`'s D1 row records the human decision: **B-first (this spec),
A on kill-switch.**

## 8. Definition of done

- [x] **T0 bench implemented** — `crates/skinki-encoder` (GEMM + bench) +
      `encoder-bench --gemm` CLI.
- [x] **T0 10-min sustained on M1 Max (2026-07-04)** — 4-thread min p5 =
      **46.94 GF/s**, GATE PASS (≥ 40 by ~17%). Numbers in §2.
- [x] **D1 go/no-go recorded (human, 2026-07-04): GO on M1 Max data.** The
      M1 Air of §1's original sizing is unavailable and not expected; the
      gap is recorded in the D1 row (§6) together with the ~25% derating
      rule for Air-class projections and the exact re-bench command should
      such hardware ever matter (Stage 7).
- [x] **D2 verdict recorded (human, 2026-07-05): §2 quality bars
      falsified-by-trend for bge-small; trend-close, full-row rerun
      waived.** Numbers, the query-prefix finding, and the kept
      deliverables are in the §6 verdict block. Goldens/parity/determinism
      green; `cargo test`/clippy/fmt clean; deps unchanged.
- [x] Served-default decision recorded in D2: hash stays the served
      default; the sidecar fallback is rejected as a product shape;
      `STAGE_1D` is the earned continuation.

## 9. Out of scope

- INT8/SDOT quarantine (recorded escalation, own D-ticket + human approval).
- Accelerate/AMX/GPU/Metal (dep + determinism loss — rejected).
- Fine-tuning; models beyond the §5 candidates of `STAGE_1C`.
- Stage-7 packaging (B makes it trivial — one binary — but that is 7's job).
