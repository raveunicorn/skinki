<div align="center">

# Skinki — Exocortex

**A portable, local-first memory & insight engine — your second brain that thinks in a dimension you can't.**

Capture a lifetime of raw thoughts. Get back structured, linked memory and
non-obvious, *cited* insights. 100% on-device. No cloud. No subscription.

</div>

---

## The pivot

Skinki began as a local macOS AI assistant. It has since been refocused around
its hardest, most valuable core: **the memory**. The thesis is simple —

> **Intelligence lives in the memory substrate, not in the model.**

On an 8 GB M1 Air the "brain" is only a ~4B model. So the leverage is in the
memory: a substrate that ingests years of voice/text, compresses it, links it
into a knowledge graph, consolidates it offline ("sleep"), and surfaces grounded
insights — making a small model punch far above its weight.

The primary product is therefore a **headless, embeddable Rust engine**
(`kortex`) — think "FFmpeg for personal knowledge." The macOS app is a secondary
consumer wrapper.

## Two laws of the architecture

1. **Intelligence lives in the memory, not the model.** All the heavy lifting —
   index, graph, consolidation, context assembly — happens in the substrate.
2. **You earn the right to invent with a benchmark.** We push the best existing
   building blocks (RaBitQ, Model2Vec, Lance/Cozo, LightRAG/HippoRAG 2) against
   hard budgets first, and invent a new format/algorithm only where they
   objectively break. Maximum with minimum means.

## Repository layout (monorepo)

```
Skinki/
  kortex/              # PRIMARY: headless Rust memory + insight engine
    crates/
      kortex-corpus/      deterministic synthetic corpus + planted ground truth
      kortex-eval/        RetrievalSystem trait + metrics + Report
      kortex-telemetry/   latency p50/p95 + peak RSS (M1 Air budgets)
      kortex-baseline/    BM25 lexical baseline (the yardstick)
      kortex-harness/     `kortex` CLI: generate / eval / demo
  apps/
    skinki-macos/      # SECONDARY: parked SwiftUI/Tuist consumer wrapper (Stage 7)
  ARCHITECTURE.md      # Exocortex layered architecture (this repo)
  ROADMAP.md           # staged, hypothesis-driven plan (Stage 0 → 7)
```

- Engine: [`kortex/`](kortex/) — see [`kortex/README.md`](kortex/README.md).
- App: [`apps/skinki-macos/`](apps/skinki-macos/) — see its [README](apps/skinki-macos/README.md).

## Hard budgets (worst-case ~10 years, ~5M memory units)

| Budget | Target on M1 Air 8 GB |
| --- | --- |
| Idle engine RAM (model unloaded) | < 250 MB (mmap, not all in RAM) |
| Retrieval latency (vector + 1-2 graph hops) | p50 < 50 ms, p95 < 150 ms |
| Cold start to first result | < 1 s (mmap) |
| Recall after compression | >= 95% vs full-precision |
| False-insight rate / uncited claims | < 5% / **0** |
| Network | 0 bytes |

## Status

**Stage 0 — the measuring stick (done).** A reproducible eval harness with a
deterministic synthetic corpus, retrieval/QA/insight metrics, latency+RAM
telemetry, and a BM25 baseline that cleanly separates recall (solved) from
multi-hop and insight discovery (the targets for Stages 1-5).

**Stage 1 — memory compression (done, gate passed).** From-scratch codecs
(int8/PQ/RaBitQ) benchmarked against exact float32. The winning config —
Matryoshka-256 + two-stage (1-bit RaBitQ scan -> float rerank from mmap) —
hits **recall@10 = 1.000 at ~172 MB resident per 5M vectors** (budget 250 MB),
p95 ~2.5 ms. Existing building blocks clear the budget, so no custom quantizer
was needed yet. See [`kortex/README.md`](kortex/README.md#stage-1--memory-compression-the-first-impossible-task).

```bash
cd kortex
cargo run --release -p kortex-harness -- demo --years 10 --entries-per-day 270   # Stage 0
cargo run --release -p kortex-harness -- compress-bench --source corpus          # Stage 1
cargo test
```

See [`ROADMAP.md`](ROADMAP.md) for all stages and their decision gates.

## License

MIT OR Apache-2.0.
