# Specs — the delegation contract per stage

Each stage of the [roadmap](../../ROADMAP.md) gets a `STAGE_<n>.md` written from
[`TEMPLATE.md`](TEMPLATE.md). A spec turns a stage into a thing a **cheaper but
capable model** (Composer, DeepSeek, Sonnet, ...) can build safely, because:

- the **interface** is fixed (a Rust trait) — the implementer can't break the
  rest of the system;
- the **gate** is a number checked by CI (`--assert-gate` / tests) — work is
  correct by construction of our metrics, not by reviewer vibes;
- **determinism + golden tests** make verification automatic.

This is why we built the eval harness first: it is the guardrail that makes
delegation cheap. See [`../../AGENTS.md`](../../AGENTS.md) for the hard rules.

## How to delegate a stage

1. A frontier model / human writes `STAGE_<n>.md`: hypothesis, budgets, the
   trait, invariants, test plan, and a task table splitting **design tickets**
   (subtle, keep on frontier) from **impl tickets** (mechanical, delegate).
2. Hand the impl tickets + the spec + `AGENTS.md` to the cheaper model.
3. Accept the PR only if CI is green (build, test, clippy, fmt, stage gate).

## Which model tier per stage (and why)

Rule of thumb: ~20% of the code is the "soul" (subtle algorithm cores, the
`unsafe` FFI boundary, statistical validity) — keep that on a frontier model
with heavy review. The other ~80% (plumbing, OS integration, bindings, UI
scaffolding) goes to cheaper models *precisely to preserve budget and attention
for the 20% that matters*.

| Stage | Criticality | Subtlety | Build with | Notes |
| --- | --- | --- | --- | --- |
| 0 Harness | High | Low | done (frontier) | The ruler; must be exact. |
| 1 Compression | High | Very high | done (frontier) | Quantization math fails silently. |
| 2 Storage substrate | Med-high | Medium | cheaper + frontier on design | Mostly plumbing (Lance/Cozo, append-only log). Frontier only for the "use vs invent `.kx`" call. |
| 3 GraphRAG | High | High | frontier + human | Extraction quality, incremental updates, PPR. |
| 4 Sleep/scheduler | Medium | Med-low (fiddly) | cheaper impl, frontier design | OS power/thermal APIs are mechanical; "interruptible/resumable" is subtle. |
| 5 Insight Engine | Highest | Highest | frontier only, max review | The soul + anti-hallucination. Never delegate. |
| 6 FFI/bindings | Low-med | Low-med | cheaper, frontier reviews `unsafe` | C-ABI, cbindgen, PyO3, Swift bridge. |
| 7 macOS app | Medium (UX) | Medium | cheaper boilerplate, human UX | SwiftUI scaffolding is delegatable; interaction design is not. |

## Index

**Safe to delegate now** (mechanical impl behind a fixed interface; subtle design
decisions pre-made and isolated as frontier-only tickets):

- [`STAGE_2.md`](STAGE_2.md) — storage substrate: append-only L0 log +
  content-addressed unit store (pure Rust, mmap). The Lance-vs-`.kx` call is a
  deferred design ticket; the delegatable slice produces the data for it.
- [`STAGE_2B.md`](STAGE_2B.md) — **done**: store durability (size-based
  rotation, write-through + fsync, torn-tail recovery, persistent dedup runs,
  O(one-segment) reopen). Documents the locked design and the extended gate.
- [`STAGE_4.md`](STAGE_4.md) — "sleep" scheduler: interruptible, resumable,
  signal-gated background jobs, proven in a deterministic simulator (jobs are
  stubs here; real ones come from Stage 3/5).
- [`STAGE_6.md`](STAGE_6.md) — portable engine: stable C-ABI + Swift/Python
  bindings with cross-language result parity (one frontier-reviewed `unsafe`
  boundary).

**Keep on a frontier model** (algorithm cores / anti-hallucination / UX):

- Stage 3 (GraphRAG extraction + PPR) and Stage 5 (Insight Engine) — the "soul";
  no spec hand-off, built with heavy review.
- Stage 7 (macOS app) — parked; interaction design is human, boilerplate can be
  delegated later.

Stages 0 and 1 are complete; their "specs" are the shipped code, tests, and the
results documented in [`../README.md`](../README.md).
