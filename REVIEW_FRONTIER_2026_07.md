# skinki — frontier deep review (2026-07)

A full-repo review: every spec, every crate core, the gates, the metrics, the
production path. Written to be handed to implementers — each finding names the
file, the failure, and the fix. Ordered by leverage, not by severity of tone.

**One-line verdict:** the *engineering culture* (gates, determinism, replay,
honest negatives) is already world-class and rarer than any algorithm here.
The three things standing between this repo and an objectively-defensible
"best memory engine" claim are: (1) the **missing end-to-end Law-1
experiment**, (2) a **production retrieval path that serves the weakest
retriever ever measured in this repo**, and (3) a **synthetic keystone whose
detectors are coupled to the generator's own templates**. All three are
fixable with means the repo already owns.

---

## 0. What is genuinely excellent — do not touch

- **The gate discipline.** `--assert-gate` as law, "never weaken a gate", the
  measurement logs inside specs recording *failures* (Stage 3 round 4, Stage 5
  round 1) — this is the moat. No competitor publishes their negatives.
- **The replay contract** (rule 3). `produce(log)` nondeterministic,
  `rebuild(log)` deterministic, gates never infer — this is the correct
  architecture for LLM-in-the-loop systems and almost nobody else has it.
- **Cite-or-silence as a structural property** (`InsightEngine::discover`
  drops uncited narrations; `InsightInput` makes the answer key unreachable by
  type). Correct instinct, correctly implemented — §3 below only *extends* it.
- **The derivation ledger concept.** "Memory that knows when it went stale" is
  the single most differentiated idea in the repo (see §5 — it is currently
  invisible to users, which is the tragedy to fix).
- **RaBitQ/IVF implementation quality** (`skinki-vector`): the streaming
  builder byte-identical to batch, the popcount fast path with a rerank-parity
  test instead of a per-score tolerance, the split N-independent RAM gate —
  museum-grade already.
- The honest Stage-3 close-out. Killing your own 2.5–3× synthetic win on real
  data and writing it in the README is exactly the credibility that will make
  the eventual positive claim believable.

---

## 1. The keystone gap: the Law-1 experiment does not exist yet

**The bet** — "intelligence lives in the memory substrate, not the model" —
is stated in the README as falsifiable. But nothing in the repo actually
tests it. Every gate measures a *component* (recall@k of a retriever,
false-insight of a detector, bytes of a store). The bet is about the
*composition*: a small model + this substrate vs. the same model without it.

**The missing gate (call it `law1-eval`):**

> On LongMemEval (and LoCoMo), fix one small model (Gemma-4B class, replayed
> per rule 3). Measure end-to-end **QA accuracy** under three conditions:
> (a) model + top-k raw chunks (naive RAG — the strawman everyone ships),
> (b) model + skinki's assembled context (3C: budgeted, cited, dated,
> pre-joined, staleness-flagged),
> (c) model + full/long context where it fits (the "no memory system" ceiling).
> The bet is TRUE iff (b) > (a) by a real margin at equal token budget, and
> (b) approaches (c) at a fraction of the tokens.

This is *the* headline experiment. Everything else in the repo is a lemma.
It is also the only claim format the outside world compares on (LongMemEval
QA accuracy is what Zep/Mem0/Letta-class systems publish). Retrieval
recall@10 = 0.438 is a good internal number; nobody outside can rank it.

Mechanics: answers judged by a replayed LLM judge (same seam as Stage 5B's
oracle — build it once, use it twice). All inference offline, artifact-logged,
gate replays. This slots into the existing harness with no new laws.

**Priority: this is the #1 strategic item.** Until it exists, "world's best"
is unfalsifiable — by the repo's own Law 2, that means it isn't earned.

---

## 2. The production path serves the worst retriever in the repo

Measured on LongMemEval multi-session (recall@10): hash embedder **0.068**,
BM25 0.193, EmbeddingGemma 0.291, coarse-to-fine 0.438. What does
`skinki-mcp` — the actual front door agents talk to — serve by default?
The **hash embedder** (`StaticHashEmbedder`, `skinki-mcp/src/lib.rs:111`).
The engine's shop window runs the 0.068 system while the README's headline
numbers come from a Python-side model and a strategy that never shipped.

Meanwhile the crown jewel of Stage 1 — the IVF/RaBitQ index (recall 1.000,
p95 2.6 ms at 1M) — is **shelfware in the product path**: `SemanticRetriever`
in both the harness and MCP does a brute-force `dot()` over `Vec<Vec<f32>>`.
The beautiful index is only exercised by `scale-bench` and the FFI demo.

**The fix is already in the architecture diagram and was never built.**
`ARCHITECTURE.md` L2a says "Model2Vec first-pass". A Model2Vec-class static
embedder is *literally a token→vector lookup table + mean pooling* — no
transformer, no runtime, no new dependency:

1. **Offline (Python, one-time, artifact):** distill EmbeddingGemma into a
   static table (Model2Vec/potion recipe: embed the tokenizer vocabulary,
   PCA, Zipf-weight). Ship the table as a versioned binary artifact
   (hash-pinned; a `MethodStamp` for the ledger — the machinery exists).
2. **In Rust (~200 lines):** tokenizer + table lookup + weighted mean + L2
   norm. Fully deterministic, `forbid(unsafe)`, zero deps. This is the
   *replayed-model* pattern applied to an embedder: the model ran once,
   offline; the engine replays its weights forever.
3. **Wire it as the MCP/harness default**, replace brute-force scan with the
   Stage-1 two-stage/IVF index, and put **coarse-to-fine on top** (it is
   already the measured winner; it just needs the runbook + `--assert-gate`
   per HANDOFF).

Expected: static-distilled embedders retain a large fraction of their
teacher's retrieval quality; even landing midway between hash (0.068) and
Gemma (0.291) transforms the shipped product, and closes README open-problem
#4 without violating the deps law. Measure, don't assume — but this is the
highest ratio of measured-upside to effort in the entire repo.

Bonus within the same artifact: token-level vectors make a **late-interaction
(MaxSim/ColBERT-style) rerank** nearly free — the token embeddings *are* the
model. That is the most plausible lever for the remaining multi-hop gap
(0.438 → ?) that doesn't require a bigger model. One ticket, one measurement.

---

## 3. The synthetic keystone measures the corpus, not the intelligence

Stage 5's numbers (recall/precision 1.000, false-insight 0.000) are real but
weaker than they look, for three reasons the spec half-admits (§8 "treat as
earned on synthetic") — and one it doesn't:

1. **Template coupling is answer-key leakage through the side door.**
   `ContradictionDetector`'s cue lists (`"was a mistake"`, `"regret picking"`,
   `"is the best"`, `"wins"`, `"coming back to"` — `skinki-insight/src/lib.rs:724-733`)
   are verbatim the generator's own templates
   (`skinki-corpus/src/lib.rs:769-778`). `profile_entities` reads
   `skinki_corpus::topic_lexicon()` — the generator's private vocabulary.
   The `InsightInput` type guarantee protects the *ground-truth ids*, but the
   detector constants smuggle the *generator's distribution* in. Recall 1.000
   is closer to a tautology than a finding.
2. **The naive contrast is a strawman.** `CoMentionDetector` sets
   `p=0, surprise=1` for everything — of course it floods. "The FDR does the
   work" needs an *ablation* contrast, not a scarecrow: (a) reference minus
   FDR (floors only), (b) reference minus surprise floor (FDR only),
   (c) full reference. If (c) ≫ (a),(b) the statistics are earned; today
   that claim is asserted, not measured.
3. **Two-seed overfit resistance is weak because both seeds were used during
   tuning.** Every constant (MIN_MENTIONS=5, MIN_COUNT=4, min_ratio=0.35, the
   1e-6 pre-filter, fdr_q=0.01) was calibrated while seeds 42 and 7 were the
   assertion set. Adopt a **sealed-holdout protocol**: develop on {42, 7},
   gate *additionally* on seeds drawn after freeze (e.g. from the release-tag
   hash) that no tuning ever saw. One-line change to the gate, real epistemic
   upgrade.
4. **The unstated one — V2's text is trivially separable.** "Lots of caffeine
   again today." / "Woke up with a migraine. Rough." A regex distinguishes
   planted from routine. Stage 3 already demonstrated exactly this failure
   mode (templated wins don't transfer).

**The structural fix — V3: LLM-paraphrased corpus with frozen artifacts.**
The missing rung between V2 and real benchmarks: take the V2 generator's
output, paraphrase every entry with an LLM **once, offline**, store the
paraphrases in a byte-frozen artifact log (rule 3), and pin the whole V3
corpus with a golden hash. Result: planted, machine-checkable ground truth
(the ids never move) + free-form, coref-laden, template-free surface text —
deterministic forever, no network in any gate. Every Stage-5 detector must
then survive without the topic lexicon and without verbatim cues, which is
precisely what Stage 5B's real-signal detectors (T3) need anyway. Build V3
*before* 5B's oracle-judge instrument and 5B gets a labeled dress rehearsal.

---

## 4. Concrete defects found in the code (fix regardless of strategy)

**4.1 `CandidateId` collision across detectors — latent bug in `full_produce`.**
`StructuralBridgeDetector` uses the *entity id* as candidate id
(`lib.rs:407`); `TemporalLeadDetector` and `ContradictionDetector` each use
their own counter starting at **0**. `InsightEngine::discover` pools all
candidates into `BTreeMap<CandidateId, &InsightCandidate>` (`lib.rs:1093`) —
colliding ids silently shadow candidates, and `validate`'s id-tiebroken sorts
become ambiguous. The measured gates hide this by running each detector in an
isolated engine, but `full_produce` (used for narration fixtures) is live and
wrong. Fix: namespace the id (`kind` tag in the high bits, or key by
`(kind, id)`), and add a test that `full_produce` surfaces the union of the
isolated engines.

**4.2 `contains_word` is ASCII-only — broken for the product's own target
language.** `lib.rs:500-517` checks word boundaries with
`is_ascii_alphanumeric()`. Every Cyrillic byte is non-ASCII ⇒ treated as a
boundary ⇒ any substring inside a Russian word "matches at a word boundary".
The 1763-phantom-mentions bug this function was built to kill comes straight
back the day Russian text (the roadmap's own STT target) enters the store.
Note `skinki-baseline`'s tokenizer already does this right
(`char::is_alphanumeric` — Unicode-aware). Fix: char-based boundary check;
add a Cyrillic regression test. Audit the same pattern anywhere `as_bytes()`
meets natural language.

**4.3 Insight discovery does not scale to its own design target — and no gate
notices.** `profile_entity_days` lowercases **every entry for every entity**
(`lib.rs:519-538`): O(V·N) string allocations; `TemporalLeadDetector` is
O(V²·91 lags) with a linear scan of `b_days` inside `count_at_lag`. Fine at
11.5k entries / ~60 entities; hopeless at 5M entries with a real extracted
vocabulary (thousands of entities) inside a sleep window. The repo's own
discipline demands an **insight-at-scale budget in the gate** (discovery
minutes per 1M entries), like Stage 1's `scale-bench`. Cheap wins first:
lowercase entries once (as `profile_entities` already does), binary-search
`b_days` (they're sorted), prune the pair loop by co-occurrence before the
lag scan.

**4.4 The temporal detector's "FDR" isn't FDR.** A hard `p_value > 1e-6`
pre-filter (`lib.rs:643`) runs *before* BH; BH's `m` then counts only
survivors. Selection before BH voids the FDR guarantee — the procedure is
effectively fixed-threshold testing with a Bonferroni×91 correction (which is
defensible!). Either drop the pre-filter and let BH see the full candidate
set, or rename the guarantee honestly. Also: `InsightEngine::discover` pools
*all* detector families into one `validate` call — contradiction candidates
with `p=0` occupy the top BH ranks and raise the cutoff for everyone else.
Validate **per family**, then merge.

**4.5 The temporal null assumes uniform days.** `p_null ≈ n_b(2tol+1)/max_day`
models B's mentions as uniform over the calendar. Real diaries are bursty
(weekday cycles, vacations, the V2 generator's own year-drift). On real data
this null will hallucinate leads out of shared seasonality — the exact
apophenia class the keystone exists to kill. The standard deterministic fix:
a **circular-shift permutation null** (shift B's day-series by k seeded
offsets, take the empirical tail) — preserves B's autocorrelation, stays
bit-deterministic, ~20 lines.

**4.6 The contradiction detector bypasses statistics entirely** (`p=0,
surprise=1, support=2` hardcoded — `lib.rs:800-804`). Sound *on this corpus*
(exact anchored match, no search), unsound as a shape: on real text stance
attribution is noisy and there is no statistical floor to absorb that noise.
Stage 5B's T3 design should give contradictions a real statistic (stance
confidence × temporal separation × source count), so `validate` has teeth
when exactness disappears.

**4.7 The ledger's hash is not up to its job.** `ContentHash` = two
*correlated* FNV-1a-64 passes (`skinki-ledger/src/lib.rs:70-76`). FNV is
non-cryptographic and near-linear; dual-seeding does not make 128 honest
bits. The ledger's **entire guarantee** — "changed premise ⇒ changed hash" —
rides on collision resistance; a silent collision is precisely the silent
staleness the system exists to abolish. This is the one place in the repo
where the guarantee itself *earns* a real hash (Law 2). Within the deps law:
implement **SHA-256 from scratch** (~120 lines, test vectors from FIPS 180-4,
`forbid(unsafe)`) — a museum-quality module, and the FFI/store dedup can
migrate later at leisure. Perf is irrelevant here (hashing happens at derive
time, not query time).

**4.8 No per-record integrity in the store.** Torn-tail recovery validates
*framing* only (`skinki-store`: length-prefix + tail truncation). Bit-rot
inside a committed record — the actual multi-year failure mode on consumer
SSDs — is undetectable: `decode_event` will happily return corrupted text,
and every downstream hash (ledger! dedup!) silently diverges. For a system
whose pitch includes "fault-tolerant" and "provenance to source bytes":
add a **per-record checksum** in the frame (CRC32 or a truncation of 4.7's
SHA-256) + a **scrub job** for the sleep scheduler (finally a real Stage-4
`Job`, and it's on-theme: memory that checks its own integrity while it
sleeps). Fix the known `skinki-store` test flake in the same PR (unique
per-run temp dirs) — a determinism-law repo with flaky CI undermines its own
sermon.

**4.9 Small but real:**
- `BitWriter::finish` (`quant.rs:69-76`): the padding loop increments a
  counter and does nothing — dead code; delete or pad bytes for real.
- `SemanticRetriever` is duplicated verbatim in `skinki-mcp/src/lib.rs` and
  `skinki-harness/src/main.rs` — move one copy next to `Embedder` in
  `skinki-vector`.
- Three hand-rolled JSON-lines logs (`ArtifactLog`, `NarrationLog`, 5B's
  planned `JudgmentLog`) — extract one generic `JsonlLog<T>` (~30 lines);
  three museum plaques become one.
- Doc drift: BM25 multi-hop recall is quoted as 0.075 (STAGE_3 §1) and 0.325
  (round-1 log) without flagging the different corpus scale; unify.
- `record_insight_derivations` hashes a `Debug`-formatted `Vec` into the
  output id — deterministic today, fragile across rustc versions; format
  explicitly.

---

## 5. The ledger is the killer feature — and it is invisible

Nothing in the query surface exposes staleness. `skinki-mcp` serves `search`,
`assemble_context`, `discover_insights` — none of them carries a staleness
flag; the MCP server doesn't even hold a `Ledger`. The one capability **no
cloud memory API has** (Mem0/Zep/Letta store facts; none can answer *"this
answer depends on a belief you reversed on March 3rd"*) is gated, measured,
recall-1.000 — and unreachable by any user or agent.

Three shippable moves, in order:

1. **Staleness in every result.** Each search hit / context fact carries
   `fresh | stale{why: broken_premise(hash), superseded_by(entry)}`. The 3C
   assembler already flags contradictions; wire `stale_closure` into it.
2. **`remember` — the write path.** The MCP server is read-only. Agent memory
   that agents cannot write to is a demo, not memory. `skinki ingest` already
   exists; expose it as an MCP tool (`remember(text) → unit ids + content
   hashes`), plus derivations when the agent records a *conclusion* with its
   premises. This single tool turns skinki from "a corpus browser" into the
   thing the README promises ("the agent reasons; skinki remembers").
3. **Belief time-travel.** L0 is append-only and the ledger is a DAG with
   logical time — so *"what did my memory believe on day X, and why?"* is a
   pure function you already have the data for. As a demo it is unanswerable
   by any competitor; as a debugging tool for agent memory it is genuinely
   novel. Cheap: filter derivations by `produced ≤ T`, replay staleness.

Also: the MCP server computes only `InsightEngine::structural()` at startup
(`lib.rs:117`) — the temporal and contradiction detectors, both gated, never
reach users. Serve `full` (after fixing 4.1).

---

## 6. Missing memory dynamics (the "alive" part of a memory)

The substrate stores, compresses, links, and invalidates. It does not yet
**prioritize** or **forget** — and both were in Stage 3's test list
("salience/reconsolidation: use counts + recency feed ranking;
reinforced-on-use links") but never built. For a 10-year corpus this is not
polish; frequency/recency/reinforcement is the difference between a memory
and an archive. The skinki-shaped (deterministic, gate-able) version:

- **Salience ledger:** retrieval hits append `(unit, logical_time)` events
  (append-only, same machinery as everything else). Ranking becomes
  `semantic_score × f(recency, use_count)` with a *pinned, versioned* `f`
  (a `MethodStamp`, so changing the formula flags re-ranking as stale —
  the ledger eats its own dogfood).
- **Gate:** plant a "habit" ground truth in V3 (an entity queried often must
  outrank an equally-similar never-queried one; a decayed entity must not).
- **Forgetting as compaction, not deletion:** L0 is sacred; but the *index*
  can demote (units that never earn reinforcement fall out of the hot IVF
  lists into a cold tier). RAM budget improves as a side effect.
- **Semantic near-dup consolidation** as a sleep job (embedding-threshold
  merge with provenance union) — the corpus of a real decade is 30% repeats.

---

## 7. What "the best in the world, objectively" means — the claim ladder

To be *measurably* the best, pick claims with public comparators and a gate
per rung. The honest ladder, bottom to top:

1. **Component records (have today):** 5M vectors, recall 1.000, p95 2.6 ms,
   <250 MB, 0 network, deterministic — publishable now as "the best
   *embeddable* vector memory at this budget"; nobody competes in the
   0-dependency C-ABI niche.
2. **Retrieval on public benchmarks (near):** coarse-to-fine 0.438 on
   LongMemEval multi-session, reproducible by `cargo run` — once §2 lands and
   the runbook + gate exist. State it as "recall@10 X vs BM25 Y vs dense Z,
   same hardware, deterministic" — a category where honest numbers are rare.
3. **End-to-end QA with a small model (§1 — the bet itself):** small model +
   skinki ≥ naive RAG by a wide margin, → long-context quality at ~10× fewer
   tokens. This is the claim that makes the project *great* rather than good,
   and it is currently untested.
4. **The unique capabilities (no comparator exists):** staleness-aware
   answers, belief time-travel, cited insights at false-insight < 5% on real
   data (Stage 5B). These aren't "better X"; they are "only skinki does X" —
   provided §3/§5 land so the claims are earned and *visible*.

Anti-goal, stated once: do not chase leaderboard SOTA on retrieval-only with
bigger embedders — that race is won by whoever spends more GPU, contradicts
Law 1, and abandons the only defensible moat (local, deterministic, cited,
staleness-aware, embeddable).

---

## 8. Prioritized plan (what I would do, in order)

| # | Item | Why first | Tier |
| --- | --- | --- | --- |
| 1 | **Bug batch 4.1 + 4.2 + flake + 4.9** | correctness debt is cheap now, expensive after 5B builds on it | delegatable, frontier reviews 4.1 |
| 2 | **Static-distilled embedder in Rust + IVF wired into MCP/harness (§2)** | biggest shipped-quality jump; unblocks coarse-to-fine as default; closes README problem #4 | distillation script + Rust embedder delegatable behind a parity gate |
| 3 | **`remember` tool + staleness flags in MCP (§5.1–5.2)** | turns the demo into a product; the unique feature becomes visible | delegatable |
| 4 | **V3 corpus: LLM-paraphrased, artifact-frozen (§3)** | the instrument every later claim needs; de-risks 5B | paraphrase pipeline delegatable; corpus contract frontier |
| 5 | **Law-1 end-to-end gate (§1)** | the thesis experiment; reuses 5B's judge seam | frontier design, delegatable harness |
| 6 | **Stage 5B on V3 + real benchmarks** (with 4.4/4.5/4.6 statistics fixes) | the keystone's real-data verdict | frontier cores (as specced) |
| 7 | **SHA-256 + per-record CRC + scrub job (4.7/4.8)** | the fault-tolerance story made real | delegatable, test vectors decide |
| 8 | **Salience/forgetting (§6), late-interaction rerank (§2-bonus), belief time-travel (§5.3)** | the "alive memory" layer, each behind its own gate | mixed |

Items 1–3 are a couple of focused weeks of delegated work behind existing
gates. Items 4–6 are the campaign that decides whether the bet is true. If it
is — you will have the measurements to prove it, which is the only kind of
"great" that survives contact with an audience.
