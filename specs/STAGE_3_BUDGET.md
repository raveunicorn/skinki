# Stage 3 — Extraction compute budget (design note, do the math before the code)

- **Status:** pre-design arithmetic, written before any Stage 3 code exists.
- **Purpose:** Stage 3 plans "LightRAG-style entity/relation extraction on
  Gemma 4B". This note checks whether that is *physically affordable* on the
  target machine — because if it isn't, the extraction interface must be
  designed tiered from day one, not retrofitted.

## 1. The workload

Design target (worst case, from [`../../ROADMAP.md`](../../ROADMAP.md)):
**~10 years, ~5M memory units.** Two distinct regimes:

| Regime | Volume | Deadline |
| --- | --- | --- |
| Steady state | ~5M / 3650 ≈ **1,400 new units/day** | within the nightly "sleep" window |
| Backfill (first import of a life archive) | **5M units at once** | weeks, not years |

## 2. The machine

M1 Air 8 GB, passively cooled. A 4B-class model at Q4 occupies ~2.5-3 GB plus
KV cache; it can run only when the engine's own working set is small (which
the substrate guarantees) and only on idle+power (Stage 4 policy). Realistic
sustained rates for a 4B Q4 on M1 (memory-bandwidth-bound, with thermal
throttling on a fanless chassis):

- prefill: ~200-350 tok/s (instruction prefix amortized via prompt caching)
- decode: ~20-30 tok/s

Per-unit extraction (entities + relations as compact JSON): ~30-40 prefix-new
tokens to prefill + ~60-120 tokens to decode →

> **t_unit ≈ 3-4 s per unit** (call it 3.5 s). Decode dominates; shaving the
> output schema matters more than shaving the prompt.

## 3. The arithmetic

**Steady state:** 1,400 units/day × 3.5 s ≈ **82 min/day** of inference.
A nightly plugged-in sleep window (4-8 h) covers this with margin even after
thermal derating — *full-LLM extraction of daily inflow is affordable*.

**Backfill:** 5M units × 3.5 s ≈ 4,860 hours ≈ **202 days of continuous
inference**. At a generous 8 h of sleep window per night that is ~600 calendar
days. **Infeasible by ~2 orders of magnitude.** No prompt tweak closes a 100x
gap.

General decision rule for the LLM share `s` of a backfill of `N` units within
a sleep-time budget of `T` hours:

```
s_max = T * 3600 / (N * t_unit)
e.g.  T = 240 h (8 h/night for a month), N = 5M, t_unit = 3.5 s
      s_max ≈ 4.9%
```

## 4. The conclusion the interface must encode

Extraction is **two-tier by construction**, not as an optimization:

- **Tier 0 — deterministic, ~10^5+ units/s:** gazetteer/dictionary NER
  (people, books, tools from prior tier-1 output and user data), pattern
  relations, embedding-cluster topic assignment, rule-based coreference
  candidates. Runs over *everything*; minutes for 5M units. Fully under
  AGENTS.md rule 2 (bit-deterministic).
- **Tier 1 — LLM, ~0.3 units/s:** only units selected by a **deterministic**
  salience/uncertainty policy (novel entity candidates, retrieval-hot units,
  ambiguous coreference, high topic entropy). Backfill share ≤ ~5% (formula
  above); steady-state share may be up to 100% of daily inflow.
- All tier-1 outputs go to the **append-only artifact log** per AGENTS.md
  rule 3 (replayable): graph builds are byte-deterministic replays of the log;
  gates never re-run inference.

Ticket implications for the Stage 3 spec (when written):

| Ticket | Consequence of this note |
| --- | --- |
| Extractor trait | must accept tiered providers; tier is data, not code |
| Selection policy | pure function of unit features; seeded, testable |
| Artifact log | reuse `skinki-store` append/rotation/recovery machinery |
| Gate | extraction quality measured on replayed artifacts at both tiers |
| Bench | `extract-bench` must report units/s per tier + projected backfill days |

## 5. Sensitivity

- A 2x faster model (smaller, speculative decoding, better quantization)
  moves s_max from ~5% to ~10% — useful, not regime-changing.
- An M-series Pro/Max/Studio (3-5x decode) makes backfill share ~15-25% —
  still tiered. The two-tier design is not Air-specific.
- If EmbeddingGemma-quality embeddings make tier-0 topic/entity clustering
  strong enough, the tier-1 share can drop further; measure, don't assume.
