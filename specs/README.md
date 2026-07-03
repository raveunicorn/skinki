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

## Batch 2026-07 — execution order (from `../REVIEW_FRONTIER_2026_07.md`)

Seven specs, written to be delegated **in this order** (each names its own
gate; later ones consume earlier ones' outputs):

| # | Spec | What it delivers | Depends on |
| --- | --- | --- | --- |
| 1 | [`STAGE_5C_HARDENING.md`](STAGE_5C_HARDENING.md) | bug fixes (candidate-id collision, Unicode word boundaries, per-family FDR, honest temporal null), insight scale gate, store test soak, cleanups | — |
| 2 | [`STAGE_1B_STATIC_EMBEDDER.md`](STAGE_1B_STATIC_EMBEDDER.md) | pure-Rust static-distilled semantic embedder + IVF-backed serving + coarse-to-fine productionized | 5C T7 |
| 3 | [`STAGE_6B_AGENT_MEMORY.md`](STAGE_6B_AGENT_MEMORY.md) | MCP write path (`remember`), staleness-annotated results, `memory_asof`, full insight surface | 5C T1 |
| 4 | [`STAGE_0B_V3_CORPUS.md`](STAGE_0B_V3_CORPUS.md) | V3: LLM-paraphrased corpus, artifact-frozen — measures the template-coupling transfer gap | — (parallel-safe) |
| 5 | [`STAGE_5D_LAW1_EVAL.md`](STAGE_5D_LAW1_EVAL.md) | **the Law-1 experiment**: end-to-end QA, small model ± substrate, on LongMemEval | 1B; 5B's judgment seam |
| 6 | [`STAGE_2C_INTEGRITY.md`](STAGE_2C_INTEGRITY.md) | from-scratch SHA-256 + CRC record frames + scrub-on-sleep job | — (parallel-safe) |
| 7 | [`STAGE_4B_SALIENCE.md`](STAGE_4B_SALIENCE.md) | reinforcement/decay ranking, near-dup consolidation, forgetting-as-demotion | 1B, 6B |

[`STAGE_5B_REAL_INSIGHT.md`](STAGE_5B_REAL_INSIGHT.md) (already specced)
slots between 4 and 5: build its T2 judgment seam early (5D reuses it), run
its detectors after V3's transfer verdict (0B D2) says what they must survive.

**On external benchmark comparisons** (memory products: Mem0, Zep, Letta,
etc.): do **not** chase their marketing tables. The credible sequence is
(a) the Law-1 experiment (5D) with reproducible fixtures, (b) LongMemEval /
LoCoMo numbers anyone can re-run with one command, and only then (c) a
same-harness comparison where a competitor is run through *our* gate, not
their blog's. A comparison we can't make deterministic and reproducible is
marketing, and this repo's whole identity is that it never publishes those.
The (c) protocol is specced in [`STAGE_5F_COMPARISON.md`](STAGE_5F_COMPARISON.md),
deliberately blocked on 5D's verdict.

## Batch 2026-07-B — product & production backlog (pick up any time)

Independent of batch A's order; each is a complete delegation contract. Only
listed dependencies constrain when to start:

| Spec | What it delivers | Depends on |
| --- | --- | --- |
| [`STAGE_6C_DISTRIBUTION.md`](STAGE_6C_DISTRIBUTION.md) | release binaries + one-line install + install smoke gate + semver/CHANGELOG + PRIVACY.md | — |
| [`STAGE_6D_PORTABLE_MEMORY.md`](STAGE_6D_PORTABLE_MEMORY.md) | `skinki export/import` — your memory as one verified file (`SKPKG001`) | 2C strengthens hashes |
| [`STAGE_6E_CONNECTORS.md`](STAGE_6E_CONNECTORS.md) | `skinki-connect`: Markdown/Obsidian, Telegram, WhatsApp, jsonl, txt + the `skinki watch` poller (README open-problem #5) | — |
| [`STAGE_6F_WASM.md`](STAGE_6F_WASM.md) | `wasm32` target behind a `no-mmap` feature + native↔wasm parity gate + in-browser demo page | — |
| [`STAGE_2D_ROBUSTNESS.md`](STAGE_2D_ROBUSTNESS.md) | fuzzing every decoder (total-decoder contract) + `skinki doctor` + `FORMATS.md` registry & migration policy | 2C for the CRC row |
| [`STAGE_5E_DOGFOOD.md`](STAGE_5E_DOGFOOD.md) | the owner-corpus measurement protocol: personal QA, owner-judged false-insight, "genuine & new" counter | 6E; best after 1B |
| [`STAGE_5F_COMPARISON.md`](STAGE_5F_COMPARISON.md) | same-harness competitor comparison, fixtures-replayed, honest table | **blocked on 5D verdict** |

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
- [`STAGE_3.md`](STAGE_3.md) — incremental local GraphRAG. The **design** (graph
  schema, retrieval core, tier-0/tier-1 split, replay + ledger wiring) is
  frontier-owned and reviewed heavily, but the **impl tickets** (gazetteer NER,
  pattern relations, graph/CSR plumbing, CLI/gate, golden tests) are delegatable
  behind the locked interface. Two design tickets (D1 ranking core, D2 selection
  policy) stay frontier.

**Keep on a frontier model** (algorithm cores / anti-hallucination / UX):

- Stage 5 (Insight Engine) — the "soul"; no spec hand-off, built with heavy
  review. Stage 3 (GraphRAG) now has a frontier-owned spec
  ([`STAGE_3.md`](STAGE_3.md)) whose mechanical impl tickets are delegatable; its
  algorithm cores (D1/D2) stay frontier. The compute-budget arithmetic that
  constrains the Stage 3 design is in
  [`STAGE_3_BUDGET.md`](STAGE_3_BUDGET.md) (extraction must be two-tier; LLM
  outputs replayable per AGENTS.md rule 3).
- Stage 7 (macOS app) — parked; interaction design is human, boilerplate can be
  delegated later.

Stages 0 and 1 are complete; their "specs" are the shipped code, tests, and the
results documented in [`../README.md`](../README.md).

**Design notes (pre-code, "do the math before the code"):**

- [`STAGE_3_BUDGET.md`](STAGE_3_BUDGET.md) — extraction compute arithmetic that
  forces a two-tier design.
- [`DERIVATION_LEDGER.md`](DERIVATION_LEDGER.md) — **proposal**: staleness-aware
  memory via a content-addressed Merkle DAG of derivations (hash-linked reasoning
  chains). Cross-cuts Stages 2/3/5; awaiting go/no-go.
