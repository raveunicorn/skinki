# Stage 5B — Validating the Insight Engine on real data (SPEC)

> The real-data follow-up to Stage 5. Stage 5's keystone (structural, temporal,
> contradiction detection + apophenia rejection + cite-or-silence) is **earned on
> synthetic only**, and the detectors are visibly coupled to the synthetic
> generator (topic lexicon, exact template cues, planted day-series). Stage 3
> just taught us a synthetic win need not transfer. This stage builds the
> instrument that measures whether it does — and, if not, the real-signal
> detectors that make it.

- **Status:** **draft — ready to build.** The corpus loaders (`longmemeval`,
  `locomo`) and the Stage-3 LLM-extraction artifacts already exist; this stage
  adds the real-insight eval mode + the oracle-judge replay seam on top.
- **Owner of the design (frontier/human):** **frontier** — the oracle-judge
  contract, the false-insight measurement without planted ground truth, and the
  real-signal detector cores are the anti-hallucination keystone; heavy review.
- **Delegatable to (cheaper model):** the instrument plumbing (dump/replay,
  label remapping, CLI) — yes. The real-signal detector algorithms (T3/T4) —
  no, frontier-owned, like the synthetic keystone.

> Read [`STAGE_5.md`](STAGE_5.md) (what's earned on synthetic) and
> [`STAGE_3.md`](STAGE_3.md) round 4 (the precedent: build the real instrument,
> expect an honest negative first). Determinism is law (rule 2); the **LLM
> oracle judge is replayed** from an append-only artifact log (rule 3) — never
> inferred in CI; **0 bytes network** in any gate.

## 0. The core difficulty (why this is two layers, not "just run it")

1. **The detectors don't speak real text.** `StructuralBridgeDetector` keys on the
   synthetic *topic lexicon*; `ContradictionDetector` on *exact templates* ("was
   a mistake"); both find entities from `ground_truth.entities`. On real
   dialogue none of that exists. So a naive "run the synthetic detectors on a
   real corpus" surfaces ~nothing — a real run needs **real-signal detectors**
   fed by real entity/relation extraction.
2. **Real data has no planted insight labels.** Recall can be measured only where
   a real benchmark already carries the signal (LongMemEval `knowledge-update` ≈
   contradictions; `temporal-reasoning` ≈ leads). **Precision / false-insight,
   the anti-hallucination keystone, needs an oracle judge** — a human or a strong
   LLM ruling each surfaced insight genuine-vs-apophenia. That judge is the
   replayed-LLM seam.

The instrument (the measuring stick) comes **first**; it will likely show the
synthetic detectors don't transfer — which is the point (it localizes the gap
before we trust the synthetic numbers). The real-signal detectors follow,
measured by the same instrument.

## 1. Hypothesis

The synthetic-validated insight keystone **transfers to real dialogue**:
real-signal detectors fed by the Stage-3 extraction substrate surface genuine
non-obvious links / reversals / leads at **false-insight < 5% on real text**
(oracle-judged) and recall above baseline where real labels exist. Falsifiable:
if false-insight stays high on real data (apophenia returns), or recall collapses,
the keystone is synthetic-only — a clean Law-2 negative that says *what* breaks
(extraction noise, stance ambiguity, weak community structure) and redirects.

## 2. Budgets / fitness function (the gate)

Measured by `insight-eval --real` over a real corpus, with the LLM oracle judge
**replayed from a checked-in fixture** — never inferred in CI.

| Metric | Budget | How measured |
| --- | --- | --- |
| **False-insight rate (real, oracle-judged)** | **< 0.05** (hard, `CLAUDE.md`) | fraction of surfaced insights an oracle rules *spurious* (no planted GT needed) |
| **Uncited claims** | **= 0** (hard) | every surfaced insight cites ≥1 real entry (structural, reused) |
| Contradiction recall (real) | **≥ 0.50** (first target) | vs LongMemEval `knowledge-update` remapped to `Contradiction` GT |
| Temporal recall (real) | report (no bar yet) | vs `temporal-reasoning` where a lead→event pair is recoverable |
| Oracle-replay determinism | byte-identical | `rebuild(judgment log)` twice → identical scores (rule 3) |
| Network in gate | **0 bytes** | gate consumes the replayed fixture only |

> The first target bars are deliberately modest (real data is hard, per Stage 3).
> The `< 0.05` false-insight and `= 0` uncited budgets are fixed by `CLAUDE.md`
> and non-negotiable — they are the whole reason this stage exists. A measured
> negative (false-insight ≫ 0.05 on real) is a valid, recorded outcome.

## 3. Public interface

Reuse `skinki_eval::{DiscoveredInsight, InsightScores}`, the existing
`longmemeval` / `locomo` loaders, and the Stage-3 extraction artifact format.

```rust
// --- Real entity/relation source: reuse the Stage-3 LLM extraction log -------
// The graph/extraction substrate Stage 3 built (entities, facts, timestamps per
// entry) is exactly the input the real-signal insight detectors need. No new
// extraction — replay the existing artifacts (rule 3).
pub struct RealInsightInput<'a> {
    pub entries: &'a [Entry],                 // real turns (from longmemeval/locomo)
    pub extraction: &'a [LlmExtraction],      // replayed Stage-3 artifacts (entities/facts/day)
}

// --- Oracle judge (replayed; rule 3) -----------------------------------------
/// One oracle ruling on a surfaced insight. Written offline by a human or a
/// strong LLM over a dumped manifest; replayed here. `verdict` is the
/// anti-hallucination label.
pub struct InsightJudgment {
    pub insight_id: u64,
    pub verdict: Verdict,        // Genuine | Spurious | Unsure(→counts as spurious)
    pub judge: String,           // model id or "human", for provenance
}
pub enum Verdict { Genuine, Spurious, Unsure }

pub struct JudgmentLog;          // JSON-lines, append-only; rebuild() is deterministic
impl JudgmentLog {
    pub fn dump_manifest(insights: &[DiscoveredInsight], corpus: &Corpus, dir: &Path) -> Result<()>;
    pub fn replay(path: &Path) -> Result<Vec<InsightJudgment>>;
}

/// Score surfaced insights against replayed oracle judgments: false-insight =
/// (#Spurious + #Unsure) / #surfaced. No planted ground truth required.
pub fn score_real_false_insight(
    insights: &[DiscoveredInsight],
    judgments: &[InsightJudgment],
) -> InsightScores;

/// Remap a labelled benchmark into insight ground truth (recall side).
/// LongMemEval `knowledge-update` → `Contradiction { entry_before, entry_after }`.
pub fn knowledge_update_as_contradictions(insts: &[LongMemEvalInstance]) -> Vec<Contradiction>;
```

## 4. Invariants (must always hold)

- **Oracle judge is replayable, not inferred** (rule 3): live judging writes
  `InsightJudgment` records to an append-only log; `rebuild(log)` is
  deterministic; the gate replays a **checked-in fixture**, never calls a model.
- **No planted-answer leakage:** the real detectors see only `RealInsightInput`
  (entries + replayed extraction) — never any benchmark's answer field. Same
  type-level fairness boundary as synthetic `InsightInput`.
- **Cite-or-silence preserved:** every surfaced real insight cites ≥1 real entry;
  the oracle judges the *cited* evidence, so an uncited or mis-cited claim is
  judged Spurious by construction.
- **Determinism (rule 2):** detection + scoring deterministic; only the offline
  oracle step is non-deterministic, and it is quarantined behind the replay log.
- **0 bytes network** in any gate.
- **No `unsafe`.**

## 5. Test plan

- **Unit:** `knowledge_update_as_contradictions` maps the right turns;
  `score_real_false_insight` computes the rate (Unsure counts as spurious);
  `JudgmentLog` round-trips; dump manifest lists every surfaced insight + its
  cited text.
- **Golden:** a tiny checked-in (corpus excerpt + extraction + judgment-log)
  fixture → locked false-insight + recall scores; `rebuild` twice identical.
- **Metric:** `insight-eval --real` reproduces false-insight (oracle) and
  contradiction recall on a real corpus; prints the synthetic-detector vs
  real-signal-detector columns side by side (the transfer contrast).
- **Gate command:**
  `cargo run --release -p skinki-harness -- insight-eval --real --judgments <fixture>`
  (a `--assert-gate` is added once the first real margin is measured).

## 6. Task decomposition

Instrument first (measure the transfer), then the real-signal detectors the
measurement demands. Build order is Law 2 inside the stage.

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| **T0**: `insight-eval --real` skeleton — load a real corpus + replayed Stage-3 extraction → `RealInsightInput`; run the (current synthetic) detectors; report what they surface | impl | DeepSeek | runs end to end; honestly shows the synthetic detectors surface ~nothing on real text |
| **T1**: `knowledge_update_as_contradictions` remap + contradiction recall on LongMemEval | impl | DeepSeek | recall measured against remapped GT; unit-tested |
| **T2**: oracle-judge seam — `dump_manifest` (insight + cited text) + `JudgmentLog` replay + `score_real_false_insight` + a small fixture | impl | DeepSeek (frontier reviews the contract) | `rebuild(log)` byte-identical; false-insight computed from replay |
| **T3 (design)**: real-signal detectors — structural via **embedding community detection** (not the topic lexicon), contradiction via **embedding stance / LLM-extracted endorse→regret** (not templates); temporal reuses the day-series cross-correlation over extracted entities | **design** | **frontier** | each real detector clears false-insight < 0.05 (oracle) ∧ its recall bar on the instrument |
| **T4**: wire real detectors behind the existing `Detector` trait; feed from the extraction substrate | impl | DeepSeek (frontier reviews cores) | gate green on the real fixture; determinism golden |
| **D1 (verdict)**: record the transfer result — does the synthetic keystone hold on real text, by how much, or is it a Law-2 negative naming what broke | design | frontier | honest measurement logged in this spec |

## 7. Definition of done

- [ ] `insight-eval --real` runs on a real corpus with a replayed oracle-judgment
      fixture; false-insight (oracle) and contradiction recall reported, both
      seeds / both benchmarks where available.
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean; the
      replay-determinism golden is in CI (no inference, no network).
- [ ] Decision recorded: the keystone **transfers** (real false-insight < 0.05 at
      useful recall) — or the honest negative, naming the failure mode
      (extraction noise / stance ambiguity / weak real community structure).
- [ ] Docs: README honest-status row + ROADMAP Stage 5 updated with the real
      verdict; this spec Status → done.

## 8. Out of scope (deferred)

- **Live on-device LLM inference** (for extraction or judging) — both are
  *replayed* from artifact logs here; the live model harness is Stage 6/7.
- **A bespoke human-labelled insight benchmark at scale** — start with oracle
  judging of surfaced insights (cheap, ground-truth-free) + remapped
  `knowledge-update`; a large curated set is a later investment if the cheap
  signal is promising.
- **End-to-end product loop** on a user's multi-year corpus — related but a
  separate validation (Stage 7); this stage validates the *engine's quality*,
  not the capture/UX.
- **Re-opening the Stage-3 retrieval gap** — orthogonal; tracked in
  [`STAGE_3B_MULTIHOP.md`](STAGE_3B_MULTIHOP.md).
