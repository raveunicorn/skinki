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
    kortex-telemetry/   latency percentiles + peak RSS (only crate with `unsafe`)
    kortex-baseline/    BM25 lexical retriever (the yardstick)
    kortex-harness/     `kortex` CLI: generate / eval / demo
```

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
