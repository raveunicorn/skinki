# AGENTS.md — rules for any coding agent/model in this repo

This file is the contract for **any** model working here (frontier or cheaper —
Composer, DeepSeek, Sonnet, etc.). Follow it exactly. The guiding idea: the
**eval gate is law**, so a model can move fast on a stage *as long as the gate
stays green*.

## What this repo is

- `kortex/` — the **primary** artifact: a headless, local-first Rust memory +
  insight engine ("FFmpeg for personal knowledge"). All real work happens here.
- `apps/skinki-macos/` — a **parked** SwiftUI/Tuist consumer wrapper (Stage 7).
  Do not touch unless a task explicitly targets it.
- Vision and plan: [`README.md`](README.md), [`ARCHITECTURE.md`](ARCHITECTURE.md),
  [`ROADMAP.md`](ROADMAP.md). Per-stage contracts: [`kortex/specs/`](kortex/specs/).

## Golden rules (non-negotiable)

1. **The gate is law.** Each stage has a fitness function encoded as tests and/or
   a `--assert-gate` CLI check. Your change is correct iff `cargo test`, clippy,
   `cargo fmt --check`, and the relevant gate all pass. Never weaken a gate to
   make it pass; if a budget is truly wrong, raise it with the human first.
2. **Determinism.** No nondeterminism in logic: use the seeded `Rng`
   (SplitMix64), never `rand`, wall-clock, thread timing, or `HashMap` iteration
   order for anything that affects results. The same seed must reproduce
   byte-identical output. (Timing is fine for *telemetry only*, never for logic.)
3. **No `unsafe`** except in the two quarantined modules that already use it
   (`kortex-telemetry`, `kortex-vector::store` for mmap). Safe crates keep
   `#![forbid(unsafe_code)]` (or the `cfg(not(unix))` variant). Do not add
   `unsafe` elsewhere.
4. **Minimal dependencies.** Allowed crates: `serde`, `serde_json`, `clap`,
   `libc`, `anyhow`, and the internal `kortex-*` crates. Adding any new
   third-party dependency requires explicit human approval and a one-line
   justification in the PR.
5. **Interface-first.** Implement against the trait/spec defined for the stage.
   Keep public APIs small; don't reach across crate boundaries.
6. **Tests with every change.** New behavior ships with unit/property/golden
   tests. Prefer property tests for math (invariants) and golden tests for
   anything with a fixed expected output.

## Commands (run from `kortex/`)

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check          # CI enforces formatting; run `cargo fmt` to fix
# Stage 1 gate (fast CI variant):
cargo run --release -p kortex-harness -- compress-bench \
    --source synthetic --dim 256 --vectors 4000 --queries 100 --assert-gate
```

All four (build, test, clippy, fmt) plus the active stage gate must be green
before you open a PR. CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
runs exactly these and will reject a regression.

## Crate map (where things go)

| Crate | Responsibility |
| --- | --- |
| `kortex-corpus` | Deterministic synthetic corpus + planted ground truth. |
| `kortex-eval` | `RetrievalSystem` trait + retrieval/QA/insight metrics. |
| `kortex-telemetry` | Latency percentiles + peak RSS. (`unsafe`: getrusage.) |
| `kortex-baseline` | BM25 lexical baseline (the yardstick). |
| `kortex-vector` | Stage 1: embeddings, quantizers, two-stage, mmap store. (`unsafe`: mmap.) |
| `kortex-harness` | `kortex` CLI: generate / eval / demo / compress-bench. |

## Workflow for a delegated stage

1. Read `kortex/specs/STAGE_<n>.md` (the contract: hypothesis, interface,
   budgets, invariants, test plan, task tickets).
2. Implement the **impl tickets** behind the defined trait. Leave **design
   tickets** (marked "frontier/human") alone unless assigned.
3. Make the stage gate green. Run the full command list above.
4. Keep the diff scoped to the stage's crate(s). Update docs only as the spec
   says.

## Style

- Comments explain *why* (intent, trade-offs, math), never restate the code.
- Match existing formatting (`cargo fmt`); no manual alignment that fmt undoes.
- Prefer small, pure functions; keep hot numeric loops index-based where it
  mirrors the math (the `needless_range_loop` lint is allowed in `kortex-vector`).
