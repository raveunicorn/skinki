# Stage 3B — Closing the multi-hop retrieval gap (SPEC)

> A retrieval follow-up to Stage 3's real-text close-out. **Not** the
> LLM-entity-graph (that was measured and rejected — see [`STAGE_3.md`](STAGE_3.md)
> round 4). This is a separate, measurable bet, ready to hand to an
> implementer behind the gate.

- **Status:** **draft — ready to build.** The measurement instrument already
  exists (`longmemeval-eval`, merged); this spec adds the retrieval strategy on
  top of it.
- **Owner of the design (frontier/human):** frontier — the strategy choice
  (iterative vs coarse-to-fine) and the determinism/replay seam.
- **Delegatable to (cheaper model):** **yes** — this is a measurable, mechanical
  retrieval loop over an existing harness; ideal for DeepSeek with the gate as the
  arbiter.

## 1. Hypothesis

LongMemEval `multi-session` questions need evidence from **several distant
sessions**; a single dense query vector cannot be close to all of them at once
(each evidence turn matches only one facet), so single-shot top-k structurally
under-recalls. We believe a **multi-step retrieval** loop — retrieve, read,
form a follow-up query for what's still missing, retrieve again (2–3 rounds) —
and/or **coarse-to-fine session pooling** (retrieve session summaries, then drill
to turns) **lifts recall@10 on `multi-session` above the semantic-real baseline
(0.291)** by a measured margin, **without** the LLM-entity graph.

Falsifiable: if neither strategy beats 0.291 on `multi-session` (pooled), the gap
is an *embedder ceiling*, not a *strategy* gap — a clean negative that redirects
effort to the embedder (dim, model) instead of retrieval orchestration.

## 0. Pre-work (do the cheap ablation first)

Before building anything, **separate the embedder ceiling from the strategy gap**:
re-run `semantic-real` at full embedding dim (no Matryoshka-256 truncation) on
`multi-session`. If recall jumps materially, part of the 0.291 is compression,
not strategy — record it and re-baseline. One command, no new code.

## 2. Budgets / fitness function (the gate)

Measured by `longmemeval-eval --pooled` on `multi-session` (the multi-hop
regime), recall@10. Baselines already measured: bm25 0.193, **semantic-real
(EmbeddingGemma) 0.291**.

| Metric | Budget | How measured |
| --- | --- | --- |
| **multi-session recall@10** | **> 0.291** (first target: ≥ 0.34, ~+15%) | `longmemeval-eval --pooled --question-type multi-session` |
| No single-session regression | ≥ semantic-real per-type | the other 5 question types must not drop |
| Cost | ≤ ~3 retrieval rounds; on-device, 0 network | round count + latency in the report |
| Determinism / replay | byte-identical | the loop is deterministic; any LLM query-rewrite is replayed from an artifact log (rule 3), never inferred in a gate |

> First target ≥ 0.34 is the honest "beat the baseline by a real margin" bar; raise
> after the first measurement, never lower without sign-off.

## 3. Public interface

Reuse `skinki_eval::RetrievalSystem`. Add an iterative wrapper over an existing
single-shot retriever (semantic-real or BM25):

```rust
/// Wraps a base retriever in a multi-step loop. Deterministic given the base
/// retriever and the (replayed) query-expansion source.
pub struct IterativeRetriever<R: RetrievalSystem> {
    base: R,
    rounds: usize,
    expander: QueryExpander, // deterministic (extractive) OR replayed-LLM
}

/// Produces the next query from the original question + the texts retrieved so
/// far. v0 is EXTRACTIVE/deterministic (keyword salience over retrieved-but-
/// unconfirmed facets). An LLM expander implements the same seam and is replayed.
pub enum QueryExpander { Extractive, ReplayedLlm { log: PathBuf } }
```

Coarse-to-fine variant (second ticket): build per-session summaries offline (a
Stage-4 job), retrieve summaries to pick candidate sessions, then turn-level
dense within them.

## 4. Invariants

- **Determinism / replay:** the loop is deterministic; an LLM query-rewrite, if
  used, writes to an append-only artifact log and is replayed in the gate (rule
  3) — never inferred in CI.
- **0 bytes network**, on-device only.
- **No regression** on single-session types (the loop must not hurt the easy
  cases).
- **No `unsafe`** outside the existing quarantine.

## 5. Test plan

- **Unit:** the expander is deterministic; the loop terminates at `rounds`;
  round-2 query differs from round-1 when round-1 leaves facets uncovered.
- **Metric:** `longmemeval-eval --pooled --question-type multi-session` recall@10
  beats 0.291 by the §2 margin; per-type table shows no single-session regression.
- **Gate command:** `longmemeval-eval --pooled --question-type multi-session`
  (a `--assert-gate` flag is added once the first margin is measured).

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T0: full-dim `semantic-real` ablation (no code) | measure | DeepSeek/human | recorded; re-baselined if it moves |
| T1: extractive `IterativeRetriever` (deterministic, 2–3 rounds) over semantic-real | impl | DeepSeek | beats 0.291 on multi-session, no single-session regression |
| T2: per-session summary pooling (coarse-to-fine), summaries as a Stage-4 job | impl | DeepSeek | measured lift vs T1; summaries don't drop the evidence turn |
| T3 (optional): replayed-LLM query expander + artifact log + replay golden | impl | DeepSeek (frontier reviews) | `rebuild(log)` byte-identical; gate replays |
| D1: pick the production strategy from T1/T2/T3 measurements; add `--assert-gate` | design | frontier | gate green at the chosen margin |

## 7. Definition of done

- [ ] `longmemeval-eval --pooled --question-type multi-session` recall@10 beats
      semantic-real by the §2 margin, no single-session regression, deterministic/
      replayed.
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [ ] Decision recorded: which strategy won, by what margin — or the honest
      negative (it's an embedder ceiling, not a strategy gap).

## 8. Out of scope

- The LLM-entity-relation graph (measured + rejected in Stage 3 round 4).
- Insight discovery (Stage 5) — different problem, different gate.
- A new embedder / fine-tuning — if T0 shows an embedder ceiling, that's a
  separate track, not this spec.
