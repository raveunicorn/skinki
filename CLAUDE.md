# CLAUDE.md — working notes for Claude in this repo

This file orients an AI agent (or human) quickly. The **binding contract** is
[`AGENTS.md`](AGENTS.md) — read it first; it is law. This file adds the
practical "how to actually work here" layer and the current state of play.

## What this project is (one paragraph)

skinki is an **exocortex**: a portable, local-first **memory + insight engine**.
The bet is "**intelligence lives in the memory substrate, not the model**" — on
an 8 GB M1 Air the LLM is only ~4B, so the leverage is a substrate that ingests
years of voice/text, compresses it, links it into a knowledge graph,
consolidates it offline ("sleep"), and surfaces grounded, *cited* insights. The
repo **is** the headless Rust engine — "FFmpeg for personal knowledge"; the
cargo workspace lives at the repo root. (A macOS consumer product is a planned
Stage 7; no app ships in this repo.)

## Two laws (everything else follows from these)

1. **Intelligence in the memory, not the model.** Index, graph, consolidation,
   context assembly do the heavy lifting; the small model only verbalizes.
2. **Earn the right to invent with a benchmark.** Push the best existing
   building block against a hard budget first; invent a new format/algorithm
   only where it *measurably* breaks. "Maximum with minimum means."

## Repo map

```
skinki/                      PRIMARY — the engine (all real work)
  crates/
    skinki-corpus/    deterministic synthetic corpus + planted ground truth
    skinki-eval/      RetrievalSystem trait + retrieval/QA/insight metrics
    skinki-telemetry/ latency p50/p95 + peak RSS         (unsafe: getrusage)
    skinki-baseline/  BM25 lexical baseline (the yardstick)
    skinki-vector/    Stage 1: embeddings, quantizers, two-stage, mmap, IVF (unsafe: mmap)
    skinki-store/     Stage 2/2B: append-only L0 + unit store, rotation, dedup (unsafe: mmap)
    skinki-sleep/     Stage 4: interruptible/resumable consolidation scheduler + macOS signals
    skinki-ledger/    Derivation Ledger: hash-linked reasoning DAG + deterministic staleness propagation
    skinki-graph/     Stage 3: deterministic GraphRAG (typed IntroEdge/RecEdge walk + 3C context assembler), ledger-backed
    skinki-insight/   Stage 5: Insight Engine — deterministic discovery + BH-FDR validation + cite-or-silence (frontier-owned keystone; infra gated, detection blocked on D0)
    skinki-ffi/       Stage 6: C-ABI (cdylib/staticlib) over Stage-1 search (unsafe: all in ffi.rs, R1-reviewed)
    skinki-mcp/       Stage 6: MCP server over stdio — search + assemble_context for agents (safe, hand-rolled JSON-RPC)
    skinki-harness/   `skinki` CLI: generate/eval/demo/compress-bench/scale-bench/store-bench/sleep-sim/ledger-bench/graph-eval/insight-eval
  bindings/python/    pure-ctypes binding over the C-ABI (no PyO3)
  specs/              per-stage delegation contracts (STAGE_<n>.md from TEMPLATE.md)
ARCHITECTURE.md  ROADMAP.md  AGENTS.md   the vision, the staged plan, the rules
```

The layered architecture (L0 capture → L1 units → L2 vector/graph index →
L3 sleep consolidation → L4 insight engine → L5 agent/query) is in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Current state (read before planning work)

| Stage | Focus | Status |
| --- | --- | --- |
| 0 | Eval harness + synthetic corpus (V2) | **Done** |
| 1 | Memory compression (Matryoshka-256 + two-stage 1-bit RaBitQ → float rerank; IVF) | **Done**; IVF closes the scan-cost gap (1M mild: recall 1.000 @ p95 2.6 ms), gated |
| 2 / 2B | Storage substrate + durability (pure Rust, mmap, append-only) | **Done** |
| 3 | Incremental local GraphRAG (two-tier; see `STAGE_3.md`) | **Deterministic tier done + gated** (multi-hop 2.5–3× BM25, ledger-wired, 3C assembler); LLM tier measured = not earned |
| 4 | "Sleep" consolidation scheduler | **Done** (policy proven in sim; real jobs plug in at Stage 3/5) |
| 5 | Insight Engine (anti-hallucination keystone) | **Infrastructure built + gated** (`skinki-insight`: frozen interface, BH-FDR `validate`, cite-or-silence, apophenia-safe reference + naive contrast, `insight-eval --assert-gate`). Detection (recall) **blocked on D0** — measured: V2 insight ground-truth not yet detectable (bridge entities not rare). See `specs/STAGE_5.md` |
| 6 | Portable `skinki` (C-ABI/FFI + Python binding; MCP server) | **Done** — C-ABI + Python parity gated; `skinki-mcp` ships to agents (Swift → Stage 7) |
| 7 | skinki macOS product | Parked |

**IVF (Stage 1 close-out, done):** `skinki-vector/src/ivf.rs` adds an **IVF
index** with per-list 1-bit RaBitQ residual codes (`ivf_two_stage_search`,
`--index ivf --nprobe ...` in `scale-bench`). Measured win on realistic (mild)
geometry at 1M: recall@10 1.000 at p95 2.6 ms vs flat's 32.9 ms (~12x scan-cost
cut), 229 MB projected at 5M. Guarded in CI by a small mild-geometry
`scale-bench --index ivf --assert-gate` run; the 5M RAM projection is **split**
(per-vec linear + sqrt(n) centroids via `resident_bytes_at`) so the gate is
N-independent and fast. IVF does not rescue the synthetic adversarial extreme
(huge blobs) — that's the documented stress ceiling, deferred to Stage-3
co-design (multi-bit residuals / OPQ), not a Stage-1 blocker.

## Hard budgets (worst case ~10 years, ~5M units, M1 Air 8 GB)

| Budget | Target |
| --- | --- |
| Idle engine RAM (model unloaded) | < 250 MB (mmap, not all resident) |
| Retrieval latency (vector + 1–2 graph hops) | p50 < 50 ms, p95 < 150 ms |
| Cold start to first result | < 1 s |
| Recall after compression | ≥ 95% vs full-precision |
| False-insight rate / uncited claims | < 5% / **0** |
| Network | **0 bytes** |

## How to build, test, and check the gate

All commands run from the repo root:

```bash
cd skinki
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check        # CI enforces; `cargo fmt` to fix

# Active stage gates (CI runs exactly these):
cargo run --release -p skinki-harness -- compress-bench \
    --source synthetic --dim 256 --vectors 4000 --queries 100 --assert-gate   # Stage 1
cargo run --release -p skinki-harness -- store-bench --years 5 --assert-gate   # Stage 2
cargo run --release -p skinki-harness -- sleep-sim --assert-gate               # Stage 4
cargo run --release -p skinki-harness -- ledger-bench --assert-gate            # Derivation Ledger
cargo run --release -p skinki-harness -- graph-eval --assert-gate              # Stage 3 GraphRAG + 3C
cargo run --release -p skinki-harness -- insight-eval --assert-gate            # Stage 5 Insight Engine infra
bash scripts/ffi-gate.sh                                                        # Stage 6 C-ABI/Python parity
```

The Stage-1 IVF gate (`scale-bench --index ivf ... --assert-gate`) and the
`ffi` job run in CI too — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
for the exact set.

A change is correct **iff** build + test + clippy + fmt + the relevant
`--assert-gate` all pass. Never weaken a gate to make it pass; if a budget is
genuinely wrong, raise it with the human first.

> ⚠️ **Known flake:** the `skinki-store` test suite is **non-deterministically
> flaky under parallel execution** — a fresh `cargo test` can fail ~7 tests
> (fsync/rename timing in `temp_dir`-based fixtures), then pass on re-run, and
> always passes with `--test-threads=1`. This violates AGENTS.md Rule 2
> (determinism is law) and can randomly redden CI. If you touch `skinki-store`,
> consider hardening fixture isolation (unique per-run temp dirs incl. PID/nonce,
> not just per-test names) — but confirm scope with the human first.

## Non-negotiables you will trip over (full list in AGENTS.md)

- **Determinism is law.** Seeded `Rng` (SplitMix64) only — never `rand`,
  wall-clock, thread timing, or `HashMap` iteration order in anything that
  affects results. Same seed → byte-identical output. Timing is for *telemetry
  only*.
- **LLM outputs (Stage 3+) are replayable, not bit-deterministic.** Every LLM
  output that feeds the engine goes to an **append-only artifact log**; every
  downstream structure must be `rebuild(log)`-deterministic. Gates evaluate
  replayed artifacts; never run inference inside a gate.
- **No `unsafe`** outside the already-quarantined spots (telemetry getrusage;
  `skinki-vector::store` + `skinki-store` mmap). Safe crates keep
  `#![forbid(unsafe_code)]`.
- **Minimal deps.** Only `serde`, `serde_json`, `clap`, `libc`, `anyhow`, and
  internal `skinki-*`. Any new third-party dep needs explicit human approval.
- **Interface-first, tests with every change**, comments explain *why* not what.

## Working agreements for this environment

- **Branch:** develop on the assigned feature branch; create it locally if
  missing; never push to `main` without explicit permission. Push with
  `git push -u origin <branch>`.
- **GitHub:** use the `mcp__github__*` tools (no `gh` CLI). Repo scope is
  `raveunicorn/skinki`. **Do not open a PR unless explicitly asked.**
- **Commit messages** are descriptive and scoped (e.g.
  `feat(vector): ...`, `docs(roadmap): ...`); never include the model id.

## How to delegate / pick up a stage

Each `specs/STAGE_<n>.md` is a contract: hypothesis, fixed trait
interface, budgets/invariants, test plan, and a ticket table splitting **design
tickets** (subtle — keep on a frontier model) from **impl tickets** (mechanical
— safe to delegate). The gate decides correctness, not reviewer taste. Stages 2,
4, and 6 were built this way (done — Stage 6's `unsafe` FFI boundary frontier-
reviewed); Stage 3's mechanical tickets were delegated behind a frontier-owned
spec while its algorithm cores stayed frontier. **Stage 5** (Insight Engine) is
the remaining "soul" — frontier-only, heavy review, no hand-off.
