# Stage 5D — The Law-1 experiment: end-to-end QA with a small model (SPEC)

> Batch 5 of the 2026-07 review (`REVIEW_FRONTIER_2026_07.md` §1). The
> project's founding bet — *"intelligence lives in the memory substrate, not
> the model"* — is stated as falsifiable in the README, but no gate tests it:
> every existing gate measures a component. This stage builds the experiment
> that tests the **composition**: the same small model, with and without the
> substrate, scored on end-to-end QA accuracy on a public benchmark. This is
> the number the outside world can actually compare.

- **Status:** ready to build (consumes Stage 1B's retriever; reuses Stage 5B's
  T2 judgment-log seam — build that ticket first if 5B hasn't started)
- **Owner of the design (frontier/human):** frontier — the three-condition
  design, the token-budget accounting, and the judge contract are locked.
- **Delegatable to (cheaper model):** **yes** — the harness (T1–T4) is
  mechanical replay plumbing. Producing the answer/judgment artifacts (P1) is
  an offline human-supervised run. The verdict (D1) is frontier.

> Read [`../AGENTS.md`](../AGENTS.md). Rule 3 everywhere: answers and
> judgments are produced offline once, artifact-logged, and the gate replays
> fixtures — **no inference, 0 network in CI**. Determinism is law for
> condition construction (context assembly, chunk selection, token counting).

## 1. Hypothesis

At an equal prompt-token budget, a small model answering from **skinki's
assembled context** (3C: budgeted, cited, dated, staleness-flagged package
over the Stage-1B retriever with coarse-to-fine) scores **materially higher
end-to-end QA accuracy** on LongMemEval than the same model over **naive RAG**
(top-k raw chunks, same retriever) — and recovers a large fraction of the
long-context ceiling at a small fraction of its tokens. Falsifiable: if
(assembled − naive) is not clearly positive at equal tokens, the substrate's
assembly layer adds nothing beyond retrieval and Law 1 is unsupported at this
stage's scope — a recorded, headline negative.

## 2. Budgets / fitness function (the gate)

Conditions, all over the **same pooled LongMemEval corpus** and the same
question set, all answers produced offline by the **same pinned small model**
(4B-class; the Stage-3 close-out used Qwen-2.5-3B — reuse it for continuity,
record the exact quant):

| Condition | Prompt construction (deterministic) |
| --- | --- |
| A `naive-rag` | top-k raw entries (Stage-1B retriever), concatenated, truncated to the token budget |
| B `skinki` | `assemble_context` package (same retriever, same token budget) incl. dates + staleness flags |
| C `oracle-ctx` | the benchmark's gold evidence turns only (the retrieval-free ceiling for this model) |

| Metric | Budget | How measured |
| --- | --- | --- |
| QA accuracy, per question type + overall | report (D1 freezes the bar; the *claim* needs **B − A ≥ +0.05 absolute overall** at equal budget) | replayed judge over replayed answers |
| Token honesty | A and B within ±2% prompt tokens per question | deterministic token estimator, printed per condition |
| Ceiling fraction | B / C reported | same |
| Replay determinism | byte-identical scores across runs | `rebuild(fixtures)` twice |
| Coverage | ≥ 200 questions, all 6 LongMemEval types represented | manifest count |
| Network in gate | 0 | fixtures only |

> If B − A lands positive but < 0.05, that is a *finding*, not a pass — record
> it and investigate which package elements (dates? pre-joins? staleness?)
> carry weight via the T5 ablation before claiming anything publicly.

## 3. Public interface

```rust
// skinki-harness (new module law1.rs) — everything below is deterministic.

/// One prompt the offline runner must answer. Dumped as a manifest so the
/// answer-production step is a dumb loop over records (any runner, any host).
#[derive(Serialize, Deserialize)]
pub struct QaTask { pub question_id: String, pub condition: String, // "naive-rag" | "skinki" | "oracle-ctx"
                    pub prompt: String, pub est_prompt_tokens: usize,
                    pub gold_answer: String }

/// Offline-produced answer, one JSON line per QaTask (rule 3).
#[derive(Serialize, Deserialize)]
pub struct AnswerRecord { pub question_id: String, pub condition: String,
                          pub answer: String, pub model: String, pub v: u64 }

/// Judge ruling, replayed (reuses Stage 5B's JudgmentLog shape/seam).
#[derive(Serialize, Deserialize)]
pub struct QaJudgment { pub question_id: String, pub condition: String,
                        pub correct: bool, pub judge: String }

// CLI:
//   law1-eval dump    --lme <dataset> --k 10 --token-budget 1500 --out <dir>
//   law1-eval score   --answers <jsonl> --judgments <jsonl> [--assert-gate]
```

Prompt templates are fixed string constants in the module (versioned `V=1` in
the manifest): a terse instruction + the condition's context + the question.
Condition B's context is the `ContextPackage` rendered as dated, cited lines
plus a `[STALE: superseded by …]` marker on flagged facts (Stage 6B's
semantics; if 6B hasn't landed, staleness rendering is behind a flag and the
first run proceeds without it — record which).

Judging: LongMemEval ships gold answers; the judge prompt is the benchmark's
standard equivalence check ("does the answer contain/entail the gold
answer?"). Produced offline by a strong model **or** a human over the dumped
(answer, gold) pairs; `Unsure → incorrect`. The gate replays the log.

## 4. Invariants (must always hold)

- Same retriever, same k, same token budget, same prompt skeleton across A/B —
  the *only* varying factor is context construction. C varies retrieval-free.
- Dump → answer → judge → score is four separate steps; the gate runs only
  `score` on checked-in fixtures.
- Token estimator is the repo's deterministic `est_tokens` (shared with 3C) —
  not a model tokenizer — applied identically to A and B.
- Every number printed carries its n (questions per type).
- No inference, no network, no `unsafe` in any gate path.

## 5. Test plan

- **Unit:** manifest construction (A and B within token tolerance on a toy
  corpus; C uses exactly the gold evidence ids); scorer arithmetic (correct /
  total per type); `Unsure → incorrect`.
- **Golden:** a tiny checked-in fixture (6 questions × 3 conditions with
  hand-written answers/judgments) → locked score table; `score` twice →
  byte-identical.
- **Metric:** the full run per §2 from the real fixtures (checked in, ~200
  questions × 3 conditions ≈ small JSONL files).
- **Gate command:** `cargo run --release -p skinki-harness -- law1-eval score
  --answers fixtures/law1-answers.jsonl --judgments fixtures/law1-judgments.jsonl
  --assert-gate` (bar frozen by D1 after the first honest run).

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `law1-eval dump`: three-condition manifest over the pooled LongMemEval corpus (reuses `build_pooled_corpus`, Stage-1B retriever, `assemble_context`) | impl | cheaper | unit tests; token-tolerance holds on the real dump |
| T2 `AnswerRecord`/`QaJudgment` logs on `skinki-eval::jsonl` + `law1-eval score` + golden | impl | cheaper | golden green; deterministic |
| P1 produce the artifacts: run the pinned model over the manifest (offline, any host), then the judge; commit fixtures | production run | human + any model | fixtures committed with model ids + quant recorded |
| T4 runbook: exact dataset layout, model, quant, commands — so anyone reproduces P1 | impl | cheaper | a third party can re-produce (documented) |
| T5 ablation dump variants (B minus dates / minus staleness / minus pre-joins) | impl | cheaper | per-variant scores in the report |
| **D1** the verdict: record A/B/C per type, freeze the `--assert-gate` bar, write the honest headline (positive or negative) into README/ROADMAP | design | **frontier** | numbers + decision recorded; bar frozen |

## 7. Definition of done

- [ ] `law1-eval score --assert-gate` green in CI on checked-in fixtures.
- [ ] `cargo test`, clippy, fmt clean.
- [ ] README gains the experiment as the headline table (three conditions ×
      accuracy × tokens); ROADMAP records the verdict either way.
- [ ] Decision recorded: is Law 1 supported at this scope, by what margin, and
      which package elements carry it (T5).

## 8. Out of scope

- Live on-device inference in any gate (Stage 6/7 harness work).
- LoCoMo end-to-end (add as a second corpus *after* the LongMemEval pipeline
  is proven; same machinery).
- Prompt-engineering optimization — templates are fixed v1; improving them is
  a later, separately-measured change.
- Comparisons against *other memory products* — see the benchmarking note in
  `specs/README.md` (external comparisons are marketing until this internal
  experiment is solid).
