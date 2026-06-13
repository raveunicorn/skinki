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
| Apophenia traps  | Does the system *avoid* false insights?        | `NegativeBridge` (V2) |

Generation uses a hand-rolled SplitMix64 PRNG, so the same `--seed` yields a
**byte-identical** corpus on any machine (no `rand` drift, CI-safe).

### Difficulty: V2 (default) vs V1 (legacy)

The original corpus (`--difficulty v1`) rendered every phenomenon through a
single template — lexically so easy that BM25 scored recall 1.000 and a regex
could find every planted insight. A measuring stick a regex can max out cannot
measure Stages 3-5, so the default is the hardened **V2**:

- **Paraphrase banks** — each phenomenon has 6-10 surface forms; entries share
  only partial vocabulary with the questions.
- **Coreference** — ~40% of multi-hop second hops drop the person's name
  ("That new acquaintance from the climbing gym told me to read..."); the join
  must go through the venue anchor, not string matching.
- **Negative bridges** — hub entities casually spanning 4+ clusters. A naive
  co-occurrence detector fires on every pair; each such hit is a *certified*
  false insight (`neg-hits` in the report).
- **Distractors** — entries lexically near the needles (same person, different
  book, no recommendation semantics) that pull lexical rankers off target.
- **Topic drift** — cluster weights mutate per year, so the background
  distribution is non-stationary.

V1 remains available behind `--difficulty v1` and is pinned by a golden-hash
test (`v1_entries_match_legacy_golden`) so the legacy numbers stay reproducible.

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
# Codec fidelity matrix (small-N, fast):
cargo run --release -p kortex-harness -- compress-bench --source corpus --years 5 --entries-per-day 6
cargo run --release -p kortex-harness -- compress-bench --source synthetic --dim 256 --vectors 4000

# At-scale validation (1M: ~1 GB disk, ~1 min; 5M: ~5 GB disk, ~3 min):
cargo run --release -p kortex-harness -- scale-bench --scale 1m --assert-gate
cargo run --release -p kortex-harness -- scale-bench --scale 5m --clusters 1024

# Real-embedding validation (dev-only script; see tools/export-embeddings.py):
#   kortex generate --years 5 --entries-per-day 6 --out /tmp/corpus.json
#   python3 tools/export-embeddings.py --corpus /tmp/corpus.json --out /tmp/real.f32 --dim 256
cargo run --release -p kortex-harness -- compress-bench --vectors-file /tmp/real.f32 --dim 256
```

### Result, part 1 — codec fidelity (small-N matrix, dim 256)

The winning family is **Matryoshka-truncation to 256 dims + two-stage
(1-bit RaBitQ popcount scan -> float rerank of candidates fetched from mmap)**:

| config (dim 256) | bytes/vec | recall@10 | resident RAM @5M |
| --- | --- | --- | --- |
| float32 (reference) | 1024 | 1.000 | 4883 MB |
| int8 scalar | 256 | 0.988 | 1221 MB |
| RaBitQ 1-bit | 40 | 0.80 | 191 MB |
| RaBitQ 7-bit | 228 | 0.98 | 1087 MB |
| **two-stage 1-bit -> float** | **40 resident** | **1.000** | **190.7 MB** |

(40 B/vec = 32 B of sign codes + 4 B ranking factor + 4 B precomputed popcount
for the fast scan.) At dim 768 the same pipeline also reaches recall 1.000 but
breaks the RAM budget — **dimensionality is the decisive lever**, as
hypothesized.

### Result, part 2 — at scale, measured instead of projected

The original gate was declared on 4k-20k vectors with RAM *projected* to 5M and
latency not projected at all. `kortex scale-bench` now builds real 1M-5M-vector
indexes (streamed to disk, never resident) and measures end to end. On an
Apple-silicon dev machine (not yet the M1 Air target):

| n | geometry | config | recall@10 | p95 | resident | cold open + 1st query |
| --- | --- | --- | --- | --- | --- | --- |
| 1M | 64 clusters | refine=16384 | 1.000 | 32.9 ms | 38.1 MB | 55 ms |
| 5M | 1024 clusters (mild) | refine=4096 | 1.000 | 119.1 ms | 190.7 MB | 281 ms |
| 5M | 64 clusters (adversarial) | refine=65536 | 1.000 | **157.7 ms** | 190.7 MB | 290 ms |

**Verdict: PASS at 1M; PASS at 5M on mild geometry; FAIL by ~5% at 5M on
adversarial geometry** (recall 1.000 only at p95 157.7 ms vs the 150 ms budget;
within-budget latency tops out at recall 0.898). What the honest measurement
taught us — none of which the projection could have shown:

1. **The original "p95 ~2.5 ms" hid a real bug.** At 1M vectors p95 was 103 ms
   — dominated not by the scan but by a full `sort` of all n candidates per
   query inside `select_top_k`. An O(n) selection plus a popcount fast scan
   (the query quantized into 4-bit planes; per-vector work = 4 AND+popcounts
   per u64 word) brought the 1M scan to ~25 ms.
2. **`refine` does not transfer across scale.** 160 candidates suffice at 20k
   vectors; 5M needs 4k-64k, growing with cluster size — 1-bit codes built
   against a *global* centroid cannot rank neighbors inside a dense cluster,
   so the shortlist must cover the cluster.
3. **The earned next move is IVF-style partitioning** (per-list centroids):
   it restores within-cluster discrimination — the standard deployment of
   RaBitQ — and cuts the scan cost by 10-50x at the same time, which would
   retire both the recall and the latency risk with margin. Tactical
   alternatives (SIMD popcount, a threaded scan) buy 2-8x latency but don't
   fix the geometry problem. To be co-designed with the Stage 3 index work.

Caveats, stated plainly: numbers are from a dev machine, not the 8 GB M1 Air;
rerank and cold-open ran against a warm page cache (lower bounds — a true cold
start pays SSD reads for the touched candidate pages). The codec-fidelity
conclusion (the codec preserves float-search geometry) is independent of
embedding quality; the *required refine* is not — which is why part 3 exists.

### Result, part 3 — real-embedding validation

Same pipeline, real vectors: V2 corpus texts embedded with
**nomic-ai/nomic-embed-text-v1.5** (a non-gated, Matryoshka-trained text
embedder; see `tools/export-embeddings.py`), 200 held-out queries.

Codec matrix at 21.5k vectors:

| config | recall@10 | resident @5M | verdict |
| --- | --- | --- | --- |
| two-stage 1-bit -> float, dim 768 (native) | 0.998 | 648.5 MB | over RAM budget |
| **two-stage 1-bit -> float, dim 256 (Matryoshka)** | **0.987** | **190.7 MB** | **PASS** |

Refine sweep at 101.6k vectors (dim 256, `scale-bench --vectors-file`):

| refine | recall@10 | p95 |
| --- | --- | --- |
| 160 | 0.928 | 2.5 ms |
| **1024** | **1.000** | **2.9 ms** |
| 4096 | 1.000 | 4.2 ms |

What this settles:

1. **Matryoshka-256 is confirmed on a real MRL model** — truncation to 256
   dims keeps the two-stage fidelity gate green (0.987-1.000), and native 768
   stays over the RAM budget, exactly as the synthetic runs predicted.
2. **Real geometry is far milder than the adversarial synthetic**: recall
   1.000 needs `refine ≈ 1k at 100k vectors (~1%)`, vs 16-65k (16%+) on the
   78k-point synthetic clusters. The adversarial set remains the stress
   ceiling, not the expectation.
3. **IVF is still the earned next move**, but for a different reason than
   recall: the flat coarse scan is O(n) and eats ~120 ms of the 150 ms budget
   at 5M *regardless of geometry* — partitioning is about scan cost first,
   adversarial robustness second.

Caveat: nomic-embed is a stand-in for EmbeddingGemma (the product target,
gated on Hugging Face) — same model class (multilingual-capable MRL text
embedder), so the geometry conclusion should transfer; re-confirm when the
product embedder is wired in.

### Result, part 4 — IVF closes the scan-cost gap (the earned next move, gated)

Parts 2-3 named the next move: the flat coarse scan is O(n) and eats ~120 ms of
the 150 ms budget at 5M *regardless of geometry*. `kortex-vector/src/ivf.rs`
implements it — an inverted-file index that clusters vectors into `nlist` lists
with their own centroids and quantizes each vector's residual against its
**list** centroid (not the global one), so the 1-bit codes discriminate
*within* a cluster. Search ranks the `nlist` centroids exactly, then scans only
the `nprobe` best lists. Run it with `scale-bench --index ivf --nprobe ...`.

Measured at 1M, dim 256, mild geometry (the realistic regime — part 3 showed
real embeddings are far milder than the adversarial synthetic):

| index | nprobe | recall@10 | p95 | resident @5M |
| --- | --- | --- | --- | --- |
| flat (part 2) | — | 1.000 | 32.9 ms | 190.7 MB |
| **ivf** | 16 | **1.000** | **2.6 ms** | 229.4 MB |
| ivf | 64 | 1.000 | 3.4 ms | 229.4 MB |

**IVF cuts p95 ~12x (32.9 → 2.6 ms) at recall 1.000**, well inside the latency
budget, for ~8 extra B/vec (the per-slot original-id array) — 229 MB projected
at 5M, still under the 250 MB RAM budget. The O(n) scan is gone.

Two honest caveats, both stated rather than hidden:

1. **The 5M RAM projection is now split, not linear.** IVF's per-slot arrays
   scale with `n` but the centroid table scales with `nlist ≈ sqrt(n)`, so the
   old `total × 5M/n` projection over-counted at small N. `resident_bytes_at`
   projects the two terms separately, making the verdict N-independent — which
   is what lets the CI gate run on a tiny mild-geometry index yet report the
   true ~5M RAM.
2. **IVF does *not* rescue the adversarial extreme.** On a few enormous blobs
   (e.g. 8 clusters / ~60k points each), k-means can't separate the sub-lists
   inside a blob and the 1-bit residual shortlist needs a large `refine`, so
   recall 1.0 still costs a high `nprobe` — IVF is no better than flat there.
   Part 3 already classed that synthetic case as the stress *ceiling*, not the
   expectation; the win above is on the realistic geometry. Pushing the extreme
   further (multi-bit residual codes, OPQ rotation) is a Stage-3 co-design item,
   not a Stage-1 blocker.

**Verdict: Stage 1 closed.** Flat clears the budget on mild geometry and maps
its own limit; IVF removes the scan-cost ceiling on the realistic regime at
recall 1.000 within budget, and both are pinned by `--assert-gate` in CI.

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

## Example report (~1M entries, V2 corpus, BM25 baseline)

```
=== Kortex Stage 0 — Eval Report ===
system           : bm25-lexical
corpus entries   : 987380
recall@10        : R=0.138  P=0.014  nDCG=0.080  answer-in-topk=0.588  (n=80)
multi-hop@10     : R=0.075  P=0.015  nDCG=0.048  answer-in-topk=0.300  (n=40)
insight          : recall=0.000  precision=n/a  false-rate=n/a  neg-hits=0  (planted=10, surfaced=0, matched=0)
query latency    : p50=0.758ms  p95=24.804ms  mean=8.610ms  max=25.798ms  (n=120)
peak RSS         : 502.6 MB
```

For contrast, the same run on the legacy corpus (`--difficulty v1`) gives
recall@10 R=1.000 — the V1 facts are lexically findable by construction.

### How to read it (this is the whole point)

- **Recall no longer saturates** on V2 (R=0.138 at ~1M vs 1.000 on V1):
  paraphrases and distractors break pure lexical overlap. The facts are still
  findable *in principle* — every entry names the person and the book — so this
  gap is exactly the headroom that semantic retrieval (Stage 1 embeddings and
  beyond) must capture. A good engine must push this back toward 1.0 without
  the V1 crutch of shared template words.
- **Multi-hop collapses at scale** (R=0.075) — BM25 can't join two distant
  entries, and on V2 ~40% of chains additionally require coreference through a
  venue anchor. This is the target for **Stage 3 (GraphRAG)**.
- **Zero insights surfaced** — BM25 has no notion of bridges. This is the target
  for **Stage 5 (Insight Engine)**. On V2 the report also tracks `neg-hits`:
  surfaced "insights" that fall into planted apophenia traps (hub entities
  spanning many clusters). The keystone metric is high insight recall with a
  low false-insight rate **and zero trap hits** — a naive co-occurrence
  detector maxes recall but lights up `neg-hits`, and the harness will catch it.
- **~503 MB RSS for ~1M raw entries** is the naive, uncompressed cost. This is
  the number **Stage 1 (RaBitQ + Model2Vec compression)** must crush so memory
  fits comfortably on 8 GB *next to* a local LLM.

The harness now separates four regimes: lexically solved (V1 recall), semantic
retrieval (V2 recall — open), relational reasoning (multi-hop — open), and
grounded discovery (insight vs apophenia — open). These gaps are the explicit,
quantified targets for Stages 1-5.

## License

MIT OR Apache-2.0
