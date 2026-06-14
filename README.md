<div align="center">

# skinki

**A portable, local-first memory engine — built to test one idea:
that intelligence can live in the memory, not the model.**

100% on-device · 0 bytes of network · embeddable in any language via a stable C-ABI

</div>

---

## What this is

`skinki` is a headless Rust engine that ingests text, compresses it to fit
on a laptop, links it into a knowledge graph, and serves grounded, **cited**
retrieval — with every fact traceable to its source and a ledger that knows
when a fact has gone **stale**. Think "FFmpeg for memory": not an app, a
primitive you embed.

The bet, stated so it can be proven false:

> **Intelligence lives in the memory substrate, not the model.** On an 8 GB
> laptop the model is small; the leverage is a substrate that does the index,
> graph, consolidation, and context assembly so the model only has to verbalize.

This repo is the honest attempt to test that bet, with a reproducible benchmark
gate behind every claim. **It is not finished, and it does not all work** — see
the next section, which is the whole point.

## Honest status (read this first)

Everything here is measured, not asserted. The good and the bad:

**Validated, gated in CI:**
- **Compression** — 5M vectors searchable at recall 1.000, p95 ~2.6 ms, <250 MB
  RAM (Matryoshka-256 + 1-bit RaBitQ → float rerank, IVF). Measured, not projected.
- **Storage** — pure-Rust append-only log + content-addressed units, mmap,
  crash recovery, within hard byte budgets.
- **Graph (synthetic)** — a typed-relation multi-hop retriever beats BM25 by
  **2.5–3×** on the synthetic corpus, ledger-wired, RAM-budgeted.
- **Derivation Ledger** — store the *reasoning chain*, hash-pin its premises;
  a changed premise breaks the link and every dependent conclusion is flagged.
  On planted contradictions: **catches 100% of stale conclusions vs 0% for a
  provenance-free baseline**, 0 false flags.
- **Portability** — a stable **C-ABI** (`skinki.h`), a pure-`ctypes` Python
  binding, and an **MCP server** (memory for agents) — cross-language search
  parity gated in CI.

**Measured and *failed* on real data (the honest part):**
- On the real **LoCoMo** conversation benchmark, a good **semantic embedder
  (EmbeddingGemma)** beats BM25 by ~39% — but that win is the *embedder*, which
  any RAG can use.
- Our **graph**, given a real LLM extractor (Qwen-2.5-3B), **does not beat
  BM25** on real dialogue — worse in every category, including multi-hop. The
  synthetic graph win did **not** transfer. The hand-crafted structure the
  synthetic benchmark rewarded doesn't generalize. Recorded, not hidden.

So: the substrate (compression, storage, provenance/staleness, portability) is
real and useful today; the claim that our *graph* beats baselines is **true on
synthetic, false on real data so far**. The unique bets — staleness on evolving
real data, and an insight engine — are **not yet proven**. That's the open work.

## Quickstart

```bash
git clone https://github.com/raveunicorn/skinki && cd skinki
cargo build --release

# Generate a synthetic corpus and score the BM25 baseline:
./target/release/skinki demo --seed 42 --years 3 --k 10

# Run any of the gates that guard a claim above:
./target/release/skinki compress-bench --source synthetic --dim 256 \
    --vectors 4000 --queries 100 --assert-gate          # compression
./target/release/skinki ledger-bench --assert-gate       # staleness propagation
./target/release/skinki graph-eval  --assert-gate        # graph multi-hop (synthetic)
```

Every gate is deterministic (seeded SplitMix64; same seed → byte-identical
output) and runs with no network access.

## Use it

**As memory for an AI agent (MCP)** — register the server with any MCP host
(Claude Code, Cursor, …):

```json
{ "mcpServers": { "skinki-memory": {
    "command": "skinki-mcp", "args": ["--corpus", "/path/to/corpus.json"] } } }
```

It exposes `search` (graph multi-hop retrieval) and `assemble_context` (a
budgeted, cited, dated context package — feed *that* to the model instead of raw
chunks). The agent reasons; skinki remembers.

**Embedded in any language (C-ABI)** — the thing cloud memory APIs can't do.
Drop in `libskinki_ffi` + `crates/skinki-ffi/include/skinki.h`:

```c
sk_engine* e; sk_open("./index", &e);
uint32_t ids[10]; size_t n;
sk_search(e, query, dim, 10, ids, &n);
sk_free_engine(e);
```

Pure-`ctypes` Python binding in [`bindings/python/skinki.py`](bindings/python/skinki.py)
— no PyO3, no build step. Cross-language results are byte-identical to the Rust
path (gated by `scripts/ffi-gate.sh`).

## What's inside

| crate | role |
| --- | --- |
| `skinki-corpus` | deterministic synthetic corpus + planted ground truth |
| `skinki-eval` | `RetrievalSystem` trait + retrieval/QA/insight metrics |
| `skinki-baseline` | BM25 — the yardstick every layer must beat |
| `skinki-vector` | compression: quantizers, two-stage search, mmap, IVF |
| `skinki-store` | append-only L0 log + content-addressed unit store |
| `skinki-graph` | typed-relation GraphRAG + budgeted context assembler |
| `skinki-ledger` | hash-linked derivation DAG + staleness propagation |
| `skinki-sleep` | interruptible/resumable offline consolidation scheduler |
| `skinki-ffi` | the stable C-ABI (the only `unsafe`, quarantined + reviewed) |
| `skinki-mcp` | MCP server — search + context assembly over stdio |
| `skinki-harness` | the `skinki` CLI: generate / eval / all the `*-bench` gates |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the layered design and
[`ROADMAP.md`](ROADMAP.md) for the staged, hypothesis-driven plan.

## Open problems (come help)

This is where the crowd matters more than one person:

1. **Make the graph earn its place on real text.** Entity co-mention loses to
   lexical/semantic retrieval. Does fact-precise matching + coreference
   resolution + true path-finding beat a *semantic* baseline on a recognized
   multi-hop benchmark? Unknown.
2. **Validate staleness on evolving real data.** The ledger catches planted
   contradictions; show it catches real ones a retriever misses.
3. **The insight engine** — proactive, *statistically validated*, cited
   discovery of non-obvious links, with **zero** uncited claims. The hardest and
   most valuable piece; unbuilt.

If you can break a gate, beat a baseline, or kill/confirm one of these — open an
issue or PR. The benchmark decides, not opinion.

## Principles

- **Local & private.** 0 bytes of network. Every claim traceable to source bytes.
- **Determinism is law.** Seeded RNG only; same seed → identical output. (LLM
  outputs are *replayable* via an append-only artifact log, not bit-reproducible.)
- **Earn invention with a benchmark.** Push the best existing building block
  against a hard budget first; invent only where it *measurably* breaks.
- **Minimal dependencies.** `serde`, `serde_json`, `clap`, `libc`, `anyhow` — and
  that's nearly all of it.

## License

MIT OR Apache-2.0.
