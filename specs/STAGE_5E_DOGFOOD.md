# Stage 5E — Dogfood eval: the engine on its owner's real memory (SPEC)

> Product backlog, batch 2026-07-B. LongMemEval/LoCoMo measure retrieval on
> *other people's* synthetic-ish dialogues. The product claim — "surfaces
> genuine insights from YOUR years of notes, cited, without hallucinating" —
> can only be measured on a real personal corpus, and there is exactly one
> ethically available: the owner's own. This stage turns dogfooding from
> "poke around and vibe" into a **repeatable measurement protocol** with the
> same replay discipline as every other gate. N=1 and unpublishable in raw
> form — but it is the only eval where a discovered insight can be checked
> against the one oracle who was actually there.

- **Status:** ready to build (needs 6E connectors to get the data in; judge
  seam shared with 5B/5D; most valuable after 1B's embedder)
- **Owner of the design (frontier/human):** frontier — the protocol and the
  privacy boundary are locked. The measured numbers stay private; the
  *aggregates* are publishable.
- **Delegatable to (cheaper model):** **yes** for the harness (T1–T3); the
  actual runs and judgments are the human owner's (that's the point).

> Read [`../AGENTS.md`](../AGENTS.md). Hard privacy boundary: **no personal
> data, fixtures, or derived text ever enters the repo.** CI exercises the
> code path on a synthetic stand-in; the real run happens locally and only
> aggregate numbers (counts, rates) may be quoted in docs.

## 1. Hypothesis

On the owner's real multi-year corpus, the pipeline (ingest → retrieval →
insights → staleness) produces: personal-QA answer-in-top-k **≥ 0.7 at k=10**
on owner-written questions with known answers; oracle-judged (by the owner)
false-insight **< 0.05** among surfaced insights; and **≥ 1 insight per run
the owner marks "genuine and I hadn't connected it"** — the first measured
instance of the product actually doing its job. Falsifiable on all three; a
failure names the layer (retrieval miss / detector silence / apophenia) via
the per-layer report.

## 2. Budgets / fitness function (the gate)

CI gate (synthetic stand-in, proves the instrument):

| Metric | Budget | How measured |
| --- | --- | --- |
| Protocol round-trip | question file → scores; judgment file → rates; deterministic | golden on the stand-in fixture |
| Privacy guard | `dogfood` refuses to write any report outside `--out`; repo-path detection test | unit test |

Local protocol (the real measurement, recorded privately):

| Metric | Target | How measured |
| --- | --- | --- |
| Personal QA answer-in-top-10 | ≥ 0.7 | owner-written `questions.yaml`, ≥ 30 questions with `expect:` substrings |
| False-insight rate (owner-judged) | < 0.05 | judgment log over the insight manifest (5B seam) |
| "Genuine & new" insights | ≥ 1 per full run | judgment label `genuine_new` |
| Staleness spot-check | 10 owner-known reversals: flagged ≥ 8 | `memory_asof`/staleness rows in the report |
| End-to-end wall time | full pipeline over the corpus ≤ one sleep window (8 h) | timed report |

## 3. Public interface

```
skinki dogfood --store <dir> --questions questions.yaml --out <private-dir> \
               [--judgments <private-dir>/judgments.jsonl]
```

```yaml
# questions.yaml — owner-authored; NEVER committed.
- q: "какую книгу мне советовал N в 2023?"
  expect: ["название книги"]        # substring(s) that must appear in a hit
- q: "..."
  expect: ["..."]
```

Outputs into `--out` (all private):

```
report.md          # per-layer scores + the three headline numbers
insights.jsonl     # the manifest: every surfaced insight + its cited entries'
                   # text — exactly 5B's dump_manifest shape (reused)
judgments.jsonl    # owner fills verdicts: genuine_new | genuine_known |
                   # spurious | unsure  (unsure counts as spurious)
```

```rust
// skinki-harness (module dogfood.rs). Reuses: connectors (6E) for ingest,
// the production retriever (1B), InsightEngine::full (5C), staleness (6B),
// JudgmentLog (5B seam — extend Verdict with GenuineNew | GenuineKnown,
// scored as genuine; the split exists to count the headline metric).
pub fn run_dogfood(store: &Path, questions: &Path, out: &Path,
                   judgments: Option<&Path>) -> anyhow::Result<DogfoodReport>;
```

The runbook (`docs/DOGFOOD.md`, committed): how to import your data (6E),
write good questions (answerable-from-corpus, dated, mixed easy/hard), judge
insights honestly (the labels' definitions), and what aggregate numbers are
safe to quote publicly (counts and rates only, never text).

## 4. Invariants (must always hold)

- Nothing personal in the repo: CI fixtures are synthetic; the harness
  refuses `--out` inside the repo tree (tested).
- Determinism: same store + same questions → identical report (modulo the
  judgment file, which is human input replayed like any artifact).
- The insight manifest cites entry text so the owner judges *evidence*, not
  vibes — same contract as 5B's oracle.
- Scoring code is shared with 5B (one `score_real_false_insight`), not
  forked.
- 0 network; no new deps.

## 5. Test plan

- **Unit:** questions.yaml parsing (incl. Cyrillic); `expect` matching is
  substring, case-insensitive, Unicode-correct; verdict counting
  (`unsure → spurious`, `genuine_new` tallied).
- **Golden (CI):** synthetic stand-in store (generated from V2/V3) + a
  stand-in questions file + a stand-in judgment log → locked report.
- **Privacy:** `--out` inside the repo → hard error.
- **Gate command:** `cargo test -p skinki-harness dogfood` (CI); the real
  protocol is the runbook, executed by the owner per release.
- **Cadence:** the release checklist (Stage 6C) gains a "dogfood run
  performed, aggregates recorded" checkbox.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `dogfood` subcommand: pipeline orchestration + report + privacy guard | impl | cheaper | CI golden + privacy tests green |
| T2 questions.yaml format + QA scoring (answer-in-top-k over `expect`) | impl | cheaper | unit tests incl. Cyrillic |
| T3 judgment extension (`GenuineNew`/`GenuineKnown`) shared with the 5B seam + rate computation | impl | cheaper (frontier reviews the shared-seam change) | 5B tests still green; counts correct |
| T4 `docs/DOGFOOD.md` runbook | impl | cheaper (human reviews) | a stranger could run it |
| **P1** the first real run: import own corpus (6E), author ≥ 30 questions, judge every surfaced insight, record aggregates | human | **owner** | three headline numbers exist; per-layer failures filed as issues |

## 7. Definition of done

- [ ] CI golden + privacy tests green; `cargo test`, clippy, fmt clean.
- [ ] First real run performed; aggregates (rates/counts only) recorded in
      this spec's measurement log and, if they hold, in README honest-status
      ("on a real 5-year personal corpus: QA X, false-insight Y, Z insights
      judged genuine-and-new by the owner").
- [ ] Decision recorded: which layer is the weakest on real personal data —
      that verdict feeds 5B's detector priorities directly.

## 8. Out of scope

- Publishing any personal data or per-item results — aggregates only, ever.
- Multi-user studies / recruiting testers (a later, consent-designed effort;
  this spec is deliberately N=1).
- Automated LLM judging of personal insights (the owner *is* the oracle here;
  an LLM judge adds privacy surface for no validity gain at N=1).
