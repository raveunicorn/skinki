# Kortex — portable local memory + insight engine

Kortex is the headless engine at the heart of Skinki: a portable Rust core that
ingests a lifetime of raw thoughts (voice/text) and turns them into structured,
linked, retrievable memory — and, eventually, non-obvious, *cited* insights.

This directory currently contains **Stage 0: the measuring stick**. Before we
build (or invent) anything — compression, GraphRAG, the "sleep" consolidator,
the Insight Engine — we need an honest, reproducible way to know whether each
step actually helps, and whether it fits the hard budget of an **M1 Air, 8 GB**.

> Philosophy: we earn the right to invent (custom quantizers, a `.kx` codec,
> new algorithms) only when a measured baseline proves existing SOTA can't meet
> the budget. Stage 0 builds the instrument that produces those measurements.

## What Stage 0 delivers

1. A **deterministic synthetic corpus generator** ("years of thoughts") that
   plants machine-checkable ground truth of five kinds.
2. An **evaluation harness** with retrieval / QA / insight metrics.
3. **Telemetry**: query latency (p50/p95) and peak RAM, with a battery hook
   stubbed for Stage 4.
4. A **BM25 lexical baseline** — the yardstick every future engine must beat.

### Why synthetic?

Real journals don't come labeled. To measure recall, multi-hop reasoning, and
insight discovery we must *know* the answers in advance. The generator plants:

| Phenomenon       | What it tests                                  | Ground truth |
|------------------|------------------------------------------------|--------------|
| Recall facts     | Can we retrieve a single stated fact?          | `RecallQuery` |
| Multi-hop chains | Can we join two distant entries to answer?     | `MultiHopQuery` |
| Temporal patterns| Does entity A lead event B by a fixed lag?     | `TemporalPattern` |
| Contradictions   | A belief stated, then reversed over time       | `Contradiction` |
| Insight bridges  | A rare entity secretly linking two clusters    | `InsightBridge` |

Generation uses a hand-rolled SplitMix64 PRNG, so the same `--seed` yields a
**byte-identical** corpus on any machine (no `rand` drift, CI-safe).

## Crate map

```
kortex/
  crates/
    kortex-corpus/      deterministic generator + planted ground truth
    kortex-eval/        RetrievalSystem trait + metrics + Report
    kortex-telemetry/   latency percentiles + peak RSS (unsafe only here + vector mmap)
    kortex-baseline/    BM25 lexical retriever (the yardstick)
    kortex-vector/      Stage 1: embeddings, quantizers (int8/PQ/RaBitQ), two-stage, mmap
    kortex-harness/     `kortex` CLI: generate / eval / demo / compress-bench
```

## Stage 1 — memory compression (the first "impossible task")

Stage 1 asks whether ~5M memory vectors can be searched within the M1 Air RAM
and latency budget while preserving full-precision nearest neighbors. The gate:
**recall@10 >= 95% vs exact float32**, with idle RAM < 250 MB at 5M vectors and
p95 < 150 ms.

We implement the candidate codecs from scratch in `kortex-vector` and bench them
against an exact float32 baseline:

- **int8 scalar** (4x), **Product Quantization** (per-subspace k-means; 16-64x),
- **RaBitQ** — a fast random rotation (sign-flip + Walsh-Hadamard) then **1-bit**
  sign codes (~28x) or **multi-bit** uniform codes, with RaBitQ's unbiased
  inner-product estimator,
- a **two-stage** pipeline: a 1-bit coarse scan shortlists candidates, then a
  precise stage re-ranks only those,
- an **mmap-backed** code store so the bulk of the index lives on disk
  (demand-paged), not in RAM.

```bash
cargo run --release -p kortex-harness -- compress-bench --source corpus --years 5 --entries-per-day 6
cargo run --release -p kortex-harness -- compress-bench --source synthetic --dim 256 --vectors 4000
```

### Result: GATE PASSED

The winning configuration is **Matryoshka-truncation to 256 dims + two-stage
(1-bit RaBitQ coarse scan -> float rerank of candidates fetched from mmap)**:

| config (dim 256) | bytes/vec | recall@10 | p95 | resident RAM @5M |
| --- | --- | --- | --- | --- |
| float32 (reference) | 1024 | 1.000 | 2.3 ms | 4883 MB |
| int8 scalar | 256 | 0.995 | 3.1 ms | 1221 MB |
| RaBitQ 1-bit | 36 | 0.78 | 2.9 ms | 172 MB |
| RaBitQ 7-bit | 228 | 0.98 | 13.7 ms | 1087 MB |
| **two-stage 1-bit -> float** | **36 resident** | **1.000** | **2.5 ms** | **172 MB** |

172 MB resident at 5M vectors is **under the 250 MB budget**, with recall 1.000
and p95 well under 150 ms. The same pipeline at dim 768 hits recall 1.000 but
629 MB resident (over budget) — so **dimensionality is the decisive lever**, as
hypothesized. A harder synthetic set (overlapping clusters) reproduces the
verdict (recall 0.999 at 172 MB), confirming it isn't an artifact of easy data.

Because existing building blocks (RaBitQ + Matryoshka + two-stage) clear the
budget, **we did not need to invent a custom quantizer** — the "beat-or-invent"
gate resolves to *beat*. The mmap store is verified to serve byte-identical codes
from disk, the foundation for the cold-start and idle-RAM budgets.

> Caveat: vectors here come from a deterministic static "Model2Vec-lite" hash
> embedder over the corpus (no model download). The harness is built to swap in
> real EmbeddingGemma vectors later; the compression-fidelity conclusion (codec
> preserves geometry) is independent of embedding quality.

## Usage

```bash
# One-shot: generate a corpus in memory and score the BM25 baseline.
cargo run --release -p kortex-harness -- demo --seed 42 --years 3 --k 10

# Stress the scale knob (~1M entries) to exercise latency/RAM telemetry.
cargo run --release -p kortex-harness -- demo --years 10 --entries-per-day 270

# Or persist a corpus and evaluate it (writing a JSON report).
cargo run --release -p kortex-harness -- generate --seed 42 --years 5 --out corpus.json
cargo run --release -p kortex-harness -- eval --corpus corpus.json --k 10 --report-out report.json

cargo test   # determinism, ground-truth integrity, metric correctness
```

## Example report (~1M entries, BM25 baseline)

```
=== Kortex Stage 0 — Eval Report ===
system           : bm25-lexical
corpus entries   : 984052
recall@10        : R=1.000  P=0.100  nDCG=0.373  answer-in-topk=1.000  (n=80)
multi-hop@10     : R=0.000  P=0.000  nDCG=0.000  answer-in-topk=0.475  (n=40)
insight          : recall=0.000  precision=n/a  false-rate=n/a  (planted=10, surfaced=0, matched=0)
query latency    : p50=0.839ms  p95=24.982ms  mean=8.701ms  max=25.476ms  (n=120)
peak RSS         : 526.8 MB
```

### How to read it (this is the whole point)

- **Recall stays perfect** even at ~1M entries — single-entry facts are lexically
  findable. A good engine must not regress here.
- **Multi-hop collapses to ~0** at scale — BM25 can't join two distant entries.
  (`answer-in-topk` is non-zero only because the answer word coincidentally
  appears in unrelated entries — a reminder that lexical overlap is a weak proxy
  for reasoning.) This is the target for **Stage 3 (GraphRAG)**.
- **Zero insights surfaced** — BM25 has no notion of bridges. This is the target
  for **Stage 5 (Insight Engine)**, where the keystone metric is a low
  *false-insight rate* (no apophenia) alongside high insight recall.
- **~527 MB RSS for ~1M raw entries** is the naive, uncompressed cost. This is
  the number **Stage 1 (RaBitQ + Model2Vec compression)** must crush so memory
  fits comfortably on 8 GB *next to* a local LLM.

The fact that the harness cleanly separates recall (solved) from multi-hop
(unsolved) from insight (unsolved) is the evidence that it measures the right
things. These gaps are the explicit, quantified targets for Stages 1-5.

## License

MIT OR Apache-2.0
