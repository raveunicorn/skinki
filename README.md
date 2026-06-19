<img width="2190" height="551" alt="skinki-header" src="https://github.com/user-attachments/assets/8e3eacf8-8ffb-4895-a41e-bbcdfad15d48" />


# skinki — a portable, local-first memory engine
## built to test one idea: that intelligence can live in the memory, not the model.

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

</div>

---

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
- **Insight Engine (Stage 5)** — on synthetic, the keystone is **earned across
  three detector families**: deterministic discovery + Benjamini–Hochberg FDR +
  "cite-or-silence" surfaces planted **insight bridges** (recall/precision 1.000),
  **temporal lead/lag** (recall 0.800), and **contradictions** (recall 1.000),
  every one at **false-insight 0.000, 0 apophenia hits, 0 uncited claims** on two
  seeds — all asserted by `insight-eval --assert-gate` in CI. A naive co-mention
  baseline, by contrast, floods every apophenia trap (precision 0.19), proving
  the validation does the work. The engine is wired into the MCP server as a
  `discover_insights` tool; narration is replayed from a checked-in artifact log.
- **Coarse-to-fine retrieval** — on LongMemEval `multi-session` (the multi-hop
  regime), instance-level coarse pooling + turn-level fine search lifts
  recall@10 from semantic-real's **0.291 to 0.438 (+46%)** — the first retrieval
  strategy that measurably closes the multi-hop gap, without the LLM-entity graph.
- **Production ingest pipeline** — `skinki ingest` writes to the L0 store;
  `skinki-mcp --store` serves from live ingested data (no static corpus.json
  bottleneck).

**Measured and *failed* on real data (the honest part):**
- On two real conversation benchmarks (**LoCoMo**, **LongMemEval**), a good
  **semantic embedder (EmbeddingGemma)** is the best retriever — on LongMemEval
  `multi-session` (the multi-hop regime) it scores recall@10 **0.291 vs BM25
  0.193 (+51%)**. But that win is the *embedder*, which any RAG can use.
- Our **graph**, given a real LLM extractor (Qwen-2.5-3B), **does not beat
  BM25** on real dialogue. On LoCoMo there's no multi-hop gap to close (BM25
  cat-2 recall@10 = 0.784); on LongMemEval where the gap *is* real, both the
  co-mention and typed-fact graphs land **−0.025 below BM25**. The synthetic
  2.5–3× graph win — driven by templated intro/rec/venue patterns — did **not**
  transfer to free-form dialogue. Recorded, not hidden.

So: the substrate (compression, storage, provenance/staleness, portability) is
real and useful today. **Stage 3 closes honestly:** on real text the graph is a
*structural substrate* (provenance, ledger, staleness — the inputs to the
insight engine), **not** a retrieval ranker; the default retriever is the
semantic embedder. The **multi-hop gap remains open** — even EmbeddingGemma
misses ~71% of evidence turns — and the next attempt is *not* the LLM-entity
graph but query-focused summarization / iterative retrieval.

**Stage 5 (the insight engine) has its keystone gated on synthetic — across all
three detectors.** A deterministic discovery + Benjamini–Hochberg FDR +
"cite-or-silence" pipeline surfaces insight bridges (recall/precision 1.000),
temporal lead/lag (recall 0.800), and contradictions (recall 1.000), each at
**false-insight 0.000, zero apophenia, zero uncited claims** on two seeds — all
asserted by `insight-eval --assert-gate`. Getting the temporal and contradiction
detectors there took real precision work (a density-corrected temporal null +
word-boundary matching; name-anchored stance attribution for contradictions) —
the naive versions hallucinated 60–78% false insights. The next frontier:
validate on **real** data (the synthetic win must transfer — Stage 3 just taught
us it need not) and complete the narration with a live (replayed) LLM narrator.

**The multi-hop gap is partially closed — by retrieval strategy, not graph.**
Coarse-to-fine (instance-level embedding + targeted fine search) lifts
LongMemEval multi-session recall@10 from semantic-real 0.291 to **0.438
(+46%)**. The iterative query expansion approach attempted first gave *zero*
lift — a clean negative. The remaining gap (~0.56) is likely an embedder
ceiling; a larger model or learned query decomposition is the next lever.

**Ingest → search → insights: the full loop is wired.**
`skinki ingest` → L0 store → `skinki-mcp --store` serves `search`,
`assemble_context`, and `discover_insights` over stdio. The server accepts a
`--retriever` flag (graph or semantic); the default retriever is the semantic
hash embedder. Coarse-to-fine with EmbeddingGemma is the next production
upgrade (requires the model as a sidecar or port — not in this repo yet).


## Quickstart

<img width="2190" height="551" alt="skinki-quickstart" src="https://github.com/user-attachments/assets/6c9151b5-21de-46d2-93b6-0505dfce7214" />

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

</div>

---

## Where this fits

<img width="2190" height="551" alt="skinki-for-devs" src="https://github.com/user-attachments/assets/5f37a561-506b-44ad-8b0c-de270cd0b2f7" />


This is about the engine's *shape*, not proven deployments — it's early. But
"local, embeddable, 0-network, stable C-ABI" is a shape cloud memory APIs can't
take, which is exactly where it's interesting:

- **Memory for AI agents** — persistent, private recall across sessions over MCP.
- **Embedded / edge / games** — one library + header into a C/C++/Rust/legacy
  codebase, no backend, deterministic (NPC memory, on-device assistants, sensors).
- **On-prem / regulated** — air-gapped knowledge with provenance, where data
  legally can't leave the box (the staleness/ledger angle is built for this).
- **Big-data pipelines** — a fast, deterministic, dependency-light memory
  primitive you embed rather than a service you call.

If none of these is you but the benchmark discipline is, the repo is also just an
honest case study in building a memory engine from scratch.

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

1. **Validate the insight engine on real data.** Stage 5's keystone (structural
   bridges, FDR, cite-or-silence) clears every budget on synthetic — but so did
   Stage 3's graph. Measured on real conversations, does the engine surface
   *genuine* non-obvious links without hallucinating citations? Unknown.
2. **Validate staleness on evolving real data.** The ledger catches planted
   contradictions; show it catches real ones a retriever misses — and that
   the staleness flag actually changes agent behaviour.
3. **Close the remaining multi-hop gap.** Coarse-to-fine lifted recall from
   0.291 to 0.438; the remaining ~0.56 is likely an embedder ceiling. A larger
   model (EmbeddingGemma → full dim, or a bigger backbone) or learned query
   decomposition is the next lever.
4. **Port EmbeddingGemma to Rust or as a sidecar.** The engine is 100% Rust;
   the current semantic retriever is a fast hash-of-tokens (deterministic, no
   model). EmbeddingGemma (the SOTA retriever) lives in Python and must be
   called out-of-process for embedding. Porting it would make coarse-to-fine
   the production default.
5. **Ingest real data continuously.** `skinki ingest` works; the missing piece
   is a daemon or filesystem watcher that feeds new text into the store
   automatically (e.g. voice transcripts, chat logs, notes).

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

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at
your option. Unless you explicitly state otherwise, any contribution you submit
for inclusion shall be dual-licensed as above, with no additional terms.
