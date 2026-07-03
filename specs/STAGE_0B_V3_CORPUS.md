# Stage 0B — V3 corpus: LLM-paraphrased, artifact-frozen (SPEC)

> Batch 4 of the 2026-07 review (`REVIEW_FRONTIER_2026_07.md` §3). V2's text is
> template-generated, so every Stage-5 detector could (and partly does) win by
> matching the generator's own surface forms — the exact failure mode Stage 3
> round 4 exposed on real text. V3 keeps V2's **planted, machine-checkable
> ground truth** (ids, days, tags never move) but replaces every entry's text
> with an **offline LLM paraphrase, byte-frozen in an artifact log** — free-form
> language, deterministic forever, no network in any gate. This is the missing
> rung between V2 and real benchmarks, and the dress rehearsal for Stage 5B.

- **Status:** ready to build
- **Owner of the design (frontier/human):** frontier — the paraphrase contract
  (what must survive rewording) and the validator rules are the fairness
  boundary; locked below.
- **Delegatable to (cheaper model):** **yes** for the pipeline, validator,
  loader, goldens (T1–T4). The paraphrase *prompt* + acceptance review (D1) and
  the transfer verdict (D2) are frontier.

> Read [`../AGENTS.md`](../AGENTS.md). Rule 3 applies to the corpus itself
> here: `produce(paraphrase log)` runs a model once, offline; `rebuild(log)` —
> i.e. V3 generation — is byte-deterministic. **Never touch V1/V2 generation**
> (their goldens are law). V3 is additive: a new `Difficulty::V3` arm only.

## 1. Hypothesis

Planted ground truth survives paraphrase: with entity names, titles, venues,
dates and stance *meaning* preserved but all templates destroyed, the corpus
still admits deterministic scoring against the same `GroundTruth` — while
detector/retriever scores **drop toward their real-text values**, exposing
template coupling as a measured number (the "transfer gap") instead of a
suspicion. Falsifiable both ways: if scores do NOT drop, V2 wasn't
template-coupled after all (record it); if GT integrity cannot survive
paraphrase (validator rejects most entries), the paraphrase contract is too
strict or the phenomena genuinely need surface forms — either is a recorded
Law-2 verdict.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| V3 determinism | byte-identical corpus from (V2 skeleton, checked-in artifact) | golden hash test, both seeds |
| V1/V2 untouched | existing golden hashes unchanged | existing tests |
| GT integrity | 100% of *kept* paraphrases pass the validator; ≥ 90% of entries paraphrased (≤ 10% fallback-to-original) | validator report in the gate |
| Lexical difficulty | BM25 recall@10 on V3 ≤ 0.8 × its V2 score (paraphrase must hurt lexical matching, else it was too tame) | `eval --difficulty v3` |
| Paraphrase leakage | 0 entries where the validator's required surface forms are absent | validator (hard) |
| Transfer report | Stage-5 detector table (recall/precision/false-insight per detector) on V3 vs V2, both seeds, printed side by side | `insight-eval --difficulty v3` (informational first; D2 freezes bars) |

> The detectors are **expected to degrade** on V3 — that is the measurement,
> not a failure. Bars for V3 detector performance are frozen by D2 *after* the
> first honest run, per repo convention (never lowered later).

## 3. Public interface

```rust
// skinki-corpus
pub enum Difficulty { V1, V2, V3 }

/// One paraphrase record. `entry` indexes the V2 corpus for (seed, years,
/// entries_per_day) recorded in the log header line. Append-only JSON-lines.
#[derive(Serialize, Deserialize)]
pub struct Paraphrase { pub entry: EntryId, pub text: String,
                        pub model: String, pub v: u64 }

/// V3 = the V2 corpus with entry texts replaced from the artifact log.
/// Ids, days, dates, kinds, tags, GroundTruth: UNCHANGED. Entries missing
/// from the log keep their V2 text (the recorded fallback). Deterministic.
pub fn generate_v3(config: &GenConfig, paraphrase_log: &Path) -> io::Result<Corpus>;

/// The fairness keeper: which surface forms a paraphrase of a tagged entry
/// MUST retain for the ground truth to stay scoreable. Deterministic; used by
/// the offline pipeline to accept/reject and by a gate audit.
pub fn required_forms(corpus_v2: &Corpus, entry: EntryId) -> Vec<String>;
```

`required_forms`, locked (per tag of the V2 entry):

| Tag | Must appear verbatim (case-insensitive) |
| --- | --- |
| Recall | person name, book title |
| MultiHopA | both person names, venue phrase |
| MultiHopB (named) | person name, book title |
| MultiHopB (coref) | venue phrase, book title — and **no** PEOPLE name |
| TemporalLead/Trail | the lead/trail entity name |
| ContradictionBefore/After | tool X's name (After also: tool Y's name) |
| Insight / NegativeBridge | the bridge/hub entity name |
| routine (untagged) | nothing — free rewrite |

Everything else — sentence shape, cue phrases ("was a mistake", "recommended
the book", topic phrasing) — **must go**. The paraphraser is explicitly
instructed to vary stance wording, add hedges, split/merge sentences, use
pronouns and colloquialisms; the *meaning* (who, what, when, which stance)
must be preserved. Topic-cluster wording may drift freely: V3 detectors get
cluster signal from embeddings (Stage 1B) or co-occurrence, **not** from
`topic_lexicon()` — which is precisely the coupling this corpus exists to
break.

Offline pipeline (dev tooling; any strong LLM; never in CI):

```
scripts/paraphrase_corpus.py --seed 42 --years 5 --entries-per-day 6 \
    --out fixtures/v3-paraphrase-s42.jsonl
# dumps V2 entries + required_forms per entry, prompts the model, validates,
# retries rejected entries up to 3x, falls back to original text, writes log.
```

Checked-in fixtures: `fixtures/v3-paraphrase-s42.jsonl`, `...-s7.jsonl`
(~11.5k lines each; gzip if > 5 MB, loader handles `.gz`).

## 4. Invariants (must always hold)

- V1/V2 RNG streams and goldens byte-identical (V3 consumes **no** RNG — it is
  a pure text substitution over the V2 skeleton).
- `GroundTruth` ids/entries are shared by V2 and V3 verbatim.
- The validator is the *only* authority on acceptance; a paraphrase it rejects
  never enters the log (the pipeline enforces it; a gate audit re-checks the
  checked-in log against `required_forms` — defense in depth).
- The gate never calls a model; fixtures are the replay (rule 3).
- Coref MultiHopB entries stay coref (no person name reintroduced) — the
  validator's one *negative* requirement.

## 5. Test plan

- **Unit:** `required_forms` per tag on a hand-built mini corpus; loader
  handles missing entries (fallback), duplicate entries (last wins is an
  ERROR — reject), `.gz`.
- **Golden:** V3 corpus hash for seeds 42 & 7 locked; V1/V2 goldens untouched.
- **Audit (in gate):** re-validate every log record against `required_forms`;
  0 violations.
- **Metric:** `eval --difficulty v3` (BM25 drop row) and `insight-eval
  --difficulty v3` (transfer table) both run in CI from fixtures.
- **Gate command:** `cargo run --release -p skinki-harness -- insight-eval
  --difficulty v3 --assert-gate` (asserting determinism + GT-integrity +
  BM25-drop; detector bars added by D2).

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `required_forms` + `generate_v3` + `Difficulty::V3` arm + loader | impl | cheaper | unit + golden tests green; V1/V2 untouched |
| **D1** paraphrase prompt + pipeline acceptance criteria; run the pipeline once for seeds 42 & 7; eyeball 100 random paraphrases | design | **frontier/human** | fixtures committed; ≥ 90% paraphrased; audit 0 violations |
| T2 `paraphrase_corpus.py` (dump → prompt → validate → retry → log) | impl (dev tooling) | cheaper | produces D1's fixtures; validator-clean |
| T3 harness: `--difficulty v3` in `eval` / `insight-eval` + the BM25-drop and GT-integrity assertions | impl | cheaper | gate runs in CI from fixtures |
| **D2** transfer verdict: record the V2→V3 detector table in `STAGE_5.md` (round 6); freeze V3 detector bars; decide which detectors need the 5B real-signal rework *now* | design | **frontier** | honest numbers recorded; bars frozen |

## 7. Definition of done

- [ ] V3 gate green in CI (fixtures only, 0 network).
- [ ] `cargo test`, clippy, fmt clean; V1/V2 goldens untouched.
- [ ] Transfer gap recorded in `STAGE_5.md` + README honest-status (expected
      headline: "on paraphrased text the detectors score X vs 1.000 on
      templates — here is what that taught us").
- [ ] Decision recorded (D2): which detector cores survive paraphrase, which
      were template artifacts, and what 5B must therefore build.

## 8. Out of scope

- Real-data validation (Stage 5B) — V3 is synthetic with free-form *text*, not
  real *phenomena*.
- Multilingual V3 (a Russian V3 is a cheap later variant of the same pipeline
  — note it, don't build it yet).
- Regenerating V2 or changing planted phenomena counts/shapes.
- Detector rework to pass V3 — that is 5B's T3 design work; this stage only
  *measures* the gap.
