# Stage 1E — Base-class encoder escalation: 768-dim through the proven machinery (SPEC)

> Successor to Stage 1D, whose D2 verdict (2026-07-09) closed
> **multilingual-e5-small negative** on the full D1 row
> (`rrf(bm25+e5)` recall@10 **0.160** vs the **0.30** bar). The machinery is
> not the problem — it is parity-green (teacher min cosine 1.0000000,
> byte-deterministic across runs/thread-counts/arch, query/passage prefix
> discipline applied, fusion beats both parents). The problem is the **model
> weight class**: e5-small is 118M/384-dim, the reference (EmbeddingGemma) is
> 300M/768-dim, and §1's root-cause analysis shows the ~2× ratio gap the bar
> demands is unreachable from a base that delivers only ~1.15× over BM25.
> Stage 1E is the smallest, most predictable step that attacks the diagnosed
> cause: **one weight class up — a base-class encoder (~12 layers × hidden
> 768, 768-dim) through the unchanged forward pass and `SKENC001` v2 artifact.**
>
> Read [`../AGENTS.md`](../AGENTS.md). Everything from `STAGE_1C` §6,
> `STAGE_1C_B`, and `STAGE_1D` still holds: artifact logs, replay-only gates,
> 0 network, deterministic forward. **The single most important methodological
> lesson from 1D is encoded as an invariant here: the cheap trend row is an
> *abort* signal, never a *pass* signal** (see §1 verdict and §4 invariant
> "trend row is a cheap abort, never a pass").

- **Status:** draft — ready to build. No new crate; this reuses `skinki-encoder`
  + the `SKENC001` v2 seam + the `encoder-embed` / `longmemeval-eval` paths that
  1D already wired. The only net-new code is a checkpoint through the existing
  converter and the measured verdict.
- **Owner of the design (frontier/human):** frontier — the candidate ranking,
  the trend-vs-full methodological rule, and the kill criteria. D2 (served-default
  decision) is frontier + human; license check at D1 is human (law-level per
  `STAGE_1C`).
- **Delegatable to (cheaper model):** **yes** — T1 (converter extension) and
  T2 (trend-row eval) are mechanical behind the parity + replay gates. D1 (model
  pick + license) and D2 (full-row verdict + served decision) are frontier +
  human. The forward pass and the harness paths are unchanged from 1D.

## 1. Hypothesis + root-cause verdict on 1D

### 1.1 Why e5-small failed to transfer (trend → full) — recorded verdict

The trend row was a **verified byte-exact prefix** of the full pool (41 queries
over the first 201,233 of 594,708 entries). Comparing the two:

| row | pool entries | queries | bm25 r@10 | semantic-real(e5) r@10 | `rrf(bm25+e5)` r@10 | encoder/BM25 (solo) | rrf/BM25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trend (prefix) | 201,233 | 41 | 0.341 | 0.394 | 0.423 | 1.16× | 1.24× |
| full (D1) | 594,708 | 121 | 0.134 | 0.152 | 0.160 | 1.13× | 1.19× |
| reference (EmbeddingGemma, full, c2f) | 594,708 | 121 | — | 0.291 | — | — | — (0.438 w/ c2f) |

Three findings, in order of importance:

1. **The haystack grew 3× and the base signal is pool-size-fragile.** BM25
   *itself* collapsed 0.341 → 0.134 (−60%) from the prefix to the full pool —
   the trend queries were localized in the prefix. The encoder's *absolute*
   recall fell with it. **Use the BM25 column as the haystack-difficulty
   barometer**: a trend row whose BM25 is far above the full-row BM25 is a
   localized, optimistic sample, not a quality predictor.
2. **The encoder's lift *ratio* over BM25 held across both rows** (1.16×→1.13×
   solo, 1.24×→1.19× rrf). e5-small is consistently ~1.15–1.2× better than BM25.
   The bar needs ~2.2× (`rrf` ≥ 0.30 vs BM25 0.134). **No pool-size effect
   closes a ~2× ratio gap** — that gap is the model's weight class, not noise.
3. **It is a weight-class problem, not a machinery problem.** The pure-Rust
   encoder is parity-green (min cosine 1.0000000 vs the torch teacher across 32
   goldens, per-layer ≤ 4.1e-6); the query prefix is applied (1C-B D2 measured
   raw queries at 0.289 vs +25% with the e5 `"query: "` prefix); fusion beats
   both parents (+19% recall over BM25, +5% over encoder solo on the full row).
   The machinery did its job; 118M/384-dim is a different weight class from the
   300M/768-dim reference. Coarse-to-fine also collapses with e5-small on the
   full row (0.017) — confirming the base-embedder ceiling caps c2f too (c2f
   amplifies whatever the base does).

**The trend row was a misleading pass signal.** On the trend row e5-small's
`rrf` (0.423) looked like a comfortable pass of the 0.30 bar; on the full row it
landed at 0.160. **Lesson, encoded as invariant §4:** the trend row is a cheap
*abort* (if a candidate can't beat the prior model's trend number, stop), never
a *pass* — only the full D1 row decides the gate.

### 1.2 Hypothesis

A cleanly-licensed (MIT/Apache) **base-class** encoder — ~12 layers × hidden
768, 768-dim, ~110M *non-embedding* params (the next weight class up from
e5-small's 384-dim, halfway to the 300M reference) — served through the
**unchanged** `skinki-encoder` forward pass and `SKENC001` v2 artifact, lifts
the **full D1 row** `rrf(bm25+encoder)` recall@10 to **≥ 0.30** (stretch 0.35;
reference 0.438). Per-token compute is ~2× e5-small (12×768 vs 12×384 — the extra
vocab mass of multilingual models is a lookup, not FLOPs), so the 1D cold-index
path keeps the full-row dump practical (~2× the 1D e5-small wall time). The
diagnosed 1D failure is a model-weight-class problem; 768-dim + the larger tower
is the smallest step that can plausibly close a ~2× ratio gap from a base that
delivers only ~1.15×.

Falsifiable twice: **T2 abort** — if base-class cannot beat e5-small's *trend*
number, the step is not helping and the full row is not run; **D2 verdict** — if
base-class on the full row still misses 0.30, the gap is *not* weight class
alone and the honest record is that engine-internal encoding caps below
reference class, redirecting to the EmbeddingGemma bridge/port (Stage 1F, to be
specced) or the quarantined-int8 ticket on the winning model.

## 2. Budgets / fitness function (the gate)

Inherited from `STAGE_1D` §2 unless noted; bars frozen here are never lowered.

| Metric | Budget | How measured |
| --- | --- | --- |
| **Pooled multi-session recall@10, `rrf(bm25+encoder)`** | **≥ 0.30** on the full D1 row (inherited second bar; stretch 0.35; reference 0.438) | `longmemeval-eval --pooled` replay, `rrf(bm25+real)` column, **no `--limit`** |
| Pooled multi-session recall@10, encoder solo | reported, no bar (fusion is the served config); sanity: **> BM25 0.134 and > e5-small solo 0.152** | same run, `semantic-real` column |
| **T2 abort bar (trend row, 41q/201k)** | base-class `rrf(bm25+encoder)` **> e5-small trend `rrf` 0.423** *and* solo > 0.394 | `longmemeval-eval --pooled --limit 41` replay (cheap abort; **not** a pass signal) |
| Query embed latency (≤ 32 tokens, warm) | p95 ≤ 150 ms hard cap (global retrieval budget); **target ≤ 100 ms** via base tower, else flag M2 bridge (T3) | telemetry over the eval query set |
| Backfill throughput (sleep-time, M1-class) | full D1 row (594k) dump ≤ ~10 h on the dev box; 5M projection ≤ 10 days, interruptible (Stage 4) | `encoder-embed` measured rates (base ≈ 2× e5-small wall) |
| Teacher parity (conversion) | min cosine ≥ 0.999 over the 32-string golden set vs the Python f32 reference; per-layer max abs ≤ 5e-3 | converter golden dumps + `#[ignore]` parity test |
| Multilingual sanity | tokenizer + e2e parity vs HF reference on a RU/DE/ES/EN golden set (retrieval-quality per language = recorded gap, no public LongMemEval analog) | converter golden dumps + `#[ignore]` parity test |
| Artifact size | ≤ 1.1 GB on disk, embedding table mmap-resident only under load | loader + telemetry |
| Bit-determinism | byte-identical across runs / thread counts / arch (inherited CI property tests) | CI |
| Deps / unsafe / network | **unchanged: none added / quarantine only (none needed here) / 0** | `Cargo.toml` + crate attr review; CI |

> The full D1 row cannot run in CI (LongMemEval is not redistributable); it is
> gated **locally** via the T2/D2 runbook and its measured numbers are recorded
> in this spec. CI asserts parity, goldens, size, determinism — the replay-only
> shape from 1B/1C-B/1D.

## 3. Public interface

**No new public interface.** This stage reuses, unchanged:

- `skinki_encoder::RustEncoder` (the parity-green, byte-deterministic forward
  pass from 1C-B T2 / PERF tranches 1–2).
- `SKENC001` **v2** (1D T1) — the header already carries every field a
  base-class model needs: `arch = 1` (BERT post-LN), `pooling` (CLS or mean),
  `tok_kind` (WordPiece or Unigram), `query_prefix`/`passage_prefix`. **A
  base-class checkpoint is the same architecture as e5-small, one weight class
  larger — no format bump, no forward-pass change.**
- `EmbedderSpec::Encoder { path }` + `parse("encoder:<path>")` (1D T1) — lights
  up `loco-eval` / `longmemeval-eval` / `skinki-mcp` automatically.
- `embed_query` vs `embed` prefix asymmetry (1D T1) — the model contract
  (prefixes) lives in the artifact, never hardcoded in Rust.
- `PrecomputedSemantic` + `RrfFusion` (1C-B D2 / 1B T8) — the `rrf(bm25+real)`
  column this stage's gate reads.
- `encoder-embed` (1C-B T3, PERF streaming) — the resumable replay dump path.

The only new artifact is the converted `SKENC001` checkpoint itself (model
weights → gitignored, regenerable, like all `SKENC001`/`SKEMB001` artifacts).

## 4. Invariants (must always hold)

- **Determinism (rules 2/3):** the forward pass is byte-deterministic (fixed
  left-to-right reduction order, in-crate transcendentals, threads partition
  sequences not arithmetic — inherited and unchanged from 1C-B/PERF). The gate
  replays dumped embeddings; CI runs zero inference.
- **Replay (rule 3):** every embedding that feeds a verdict is dumped once,
  offline, through `encoder-embed`; the gate consumes `entries.f32` /
  `queries.f32`. `rebuild(log)` is byte-identical regardless of dump order.
- **Trend row is a cheap abort, never a pass (NEW, the 1D lesson).** A trend-row
  number ≥ 0.30 does **not** clear the gate; only the full D1 row does. The
  trend row's only sanctioned use is the §2 abort bar (stop early if the
  candidate can't beat the prior model's trend number). The BM25 column is the
  haystack-difficulty barometer — record it alongside every trend number.
- **No `unsafe` added** (rule 4). This stage needs no quarantined intrinsics;
  f32 is the ruler (PERF §4 measured safe-Rust int8 at only ~1.2× f32 and
  forbids retrying it as scalar safe Rust). Int8/SDOT stays parked (§5 / out of
  scope).
- **Minimal deps (rule 5):** no new workspace crates, no new runtime deps. The
  converter is Python dev tooling (offline, outside every gate) — same shape as
  the 1B/1C-B/1D converters.
- **License (law-level, human at D1):** the served model must be MIT/Apache so
  the self-contained, redistributable binary soul is preserved. Gemma-terms
  models are **out of scope for 1E** (they are the Stage 1F bridge/port
  question). A human confirms redistribution terms before any artifact is
  committed/published; if redistribution is disallowed, ship the script + a
  download-and-convert step (gate keys on a locally-produced artifact hash).
- **Do not weaken gates (rule 1):** the ≥ 0.30 `rrf` bar is inherited and frozen;
  if a budget is genuinely wrong, raise it with the human first.

## 5. Candidate strategies — ranked by expected value

This is the decision record for *why base-class is the next move*, evaluated
against the §1 diagnosis (a pool-size-fragile base that is a weight class too
small), not against vibes.

| Rank | Strategy | Expected value | Cost / risk | Verdict for 1E |
| --- | --- | --- | --- | --- |
| **1** | **Base-class encoder (e5-base / gte-base / bge-base / arctic-embed-m) through `SKENC001`** | **High.** Directly attacks the diagnosed cause (weight class). MIT/Apache → clean for the redistributable binary. Reuses *all* proven machinery (parity-green forward, v2 artifact, prefixes, fusion, cold-index path). ~2× e5-small compute — within budget, no R&D. Most predictable: the smallest step that can close a ~2× ratio gap. | Low. New checkpoint through an existing converter; no new crate, no `unsafe`, no deps. Wall time ~2× 1D. | **TOP — this stage.** |
| 2 | EmbeddingGemma-class bridge / port (300M/768-dim, the 0.438 reference) | High ceiling (it *is* the reference) but deferred. Gemma-terms license is law-level and may force a download-and-convert step (breaks "one redistributable binary" simplicity). 300M through pure-Rust f32 is ~4–5× e5-small backfill → needs the M2 space-bridge or quarantined int8 to hit query latency (its own kill-switched R&D). | High. License + a bridge/int8 R&D dependency. | **Defer to Stage 1F**, gated on 1E's verdict. If 1E's clean MIT base clears 0.30, Gemma is never needed. If 1E lands in [0.16, 0.30), Gemma (or int8-on-base) is the earned escalation. |
| 3 | Late-interaction / MaxSim (ColBERT-style multi-vector) | Medium, and an *amplifier* not a fix. Per-token vectors blow up the index (~128× storage/RAM) and likely break the < 250 MB idle budget at 594k entries; a rerank-only variant (top-50 candidates, MaxSim rerank — 1B T6's parked idea) is feasible. Amplifies whatever the base does — on a base a weight class too small it won't clear 0.30 alone. | Medium-high (index/store rework, or a rerank-only ticket). | **Defer.** Reopen as a rerank-only ticket *after* a discriminative base (1E or 1F) lands. |
| 4 | Learned / query-decomposition or reranking | Lowest now, partly falsified. 1D T6 doc2query measured **negative** at 0.5B (BM25 lift −0.093). `STAGE_3B` iterative retrieval gave **zero lift** per the README. A cross-encoder rerank is another model + dep. These are strategy moves; the diagnosis is a *base* ceiling, so they can't rescue a base a weight class too small. | Low–medium, but low expected lift given prior negatives. | **Defer / out of scope.** |

**Why base-class and not straight to Gemma** (the repo's own principle, AGENTS
"earn invention with a benchmark" / README "push the best existing building
block against a hard budget first; invent only where it measurably breaks"):
base-class is pushing the *existing* block (proven encoder + a bigger clean
checkpoint) before inventing (bridge / port / int8). If it clears 0.30, the
served-retriever problem is solved with the cheapest possible move and no
license/R&D overhead. If it fails, we've sharpened the Gemma decision: the gap is
not weight class alone, it is something about the reference specifically
(training data, Matryoshka, instruction tuning) — which is exactly what a bridge
or port would need to justify.

## 6. Candidate models (D1 picks; all MIT/Apache; all convertible through the existing converter)

| Model | Layers × hidden | ~Non-emb params | Dim | Langs | Tokenizer | License | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **multilingual-e5-base** | 12 × 768 | ~110M (+ 250k XLM-R vocab) | 768 | 100+ | Unigram (XLM-R) | MIT | **Primary candidate.** The natural e5-small→e5-base step; same Unigram tokenizer already shipped (1D K0), same `"query:"`/`"passage:"` prefixes already in v2, multilingual (product requirement). The extra mass is mostly the embedding table — mmap-class cost, not FLOPs. |
| bge-base-en-v1.5 | 12 × 768 | ~109M | 768 | en | WordPiece | MIT | Clean benchmark comparability with bge-small (1B/1C-B); WordPiece already shipped. English-only — not the product model, but the cleanest single-variable ablation vs e5-small. |
| gte-multilingual-base / gte-base-en | 8–12 × 768 | ~110M | 768 | multi/en | Unigram/WordPiece | MIT/Apache | Lighter (8L for gte-multilingual) if e5-base's wall time is uncomfortable; retrieval-tuned. |
| arctic-embed-m / arctic-embed-l-v2.0 | 12 × 768 | ~110M | 768 | en/multi | WordPiece/Unigram | Apache | Retrieval-tuned; strong MTEB but check LongMemEval transfer specifically. |

D1 picks **at most two**: multilingual-e5-base (product/multilingual) and, if the
machine budget allows a same-day ablation, bge-base-en (the cleanest
single-variable "does 768-dim alone help, holding the e5 vs bge recipe constant"
data point). One is enough for the verdict.

## 7. Test plan

- **Unit / format:** the converted base-class artifact loads via the unchanged
  `RustEncoder::load` (header fields all already supported in v2); a new
  `real_<model>_artifact_loads` `#[ignore]` test mirrors the 1D e5/bge pattern.
- **Golden (layer):** per-layer activations byte-equal to the converter's
  committed layer-golden dump for the fixed probe input (localizes any
  regression to one layer) — `#[ignore]`, run after every regeneration.
- **Golden (end-to-end parity):** 32-string embeddings — Rust-self golden
  (regression) + cosine ≥ 0.999 vs the Python torch teacher reference (parity),
  both `#[ignore]` (need the regenerable artifact).
- **Property:** determinism (two runs byte-equal; 1-thread vs 8-thread
  byte-equal — inherited from PERF §6); purity through `Box<dyn Embedder>`;
  unit-norm output.
- **Metric (local runbook, the gate):** the full D1 pooled multi-session row
  (§2); the trend row is run first only as the §2 abort bar.
- **Gate command (local):**
  ```
  cargo run --release -p skinki-harness -- longmemeval-eval \
      --path <…/longmemeval_m_cleaned.json> --pooled \
      --question-type multi-session \
      --embeddings-file <dump>/entries.f32 \
      --query-embeddings-file <dump>/queries.f32
  ```
  (the `rrf(bm25+real)` column is the gate; a `--assert-gate` flag is added by
  D2 after the first measured margin, never before). CI runs parity/goldens/
  size/determinism only.

## 8. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| **D1 model pick + license check**: choose ≤2 base-class candidates from §6; confirm MIT/Apache redistribution terms (human, law-level) | design | **frontier + human** | model(s) + license recorded here; converter target fixed |
| **T1 converter extension + artifact + goldens**: extend `scripts/convert_encoder_to_skenc.py` (already does e5-small/bge-small) for the D1 base-class checkpoint(s); dump layer + e2e goldens; produce the `SKENC001` v2 artifact. **No format bump, no forward-pass change** — base-class is the same architecture, one weight class up | impl | cheaper | `real_<model>_artifact_loads` green; layer-parity ≤ 5e-3; e2e min cosine ≥ 0.999 over 32 goldens; artifact size ≤ 1.1 GB; `#[ignore]` parity tests green; `cargo test`/clippy/fmt clean; deps unchanged |
| **T2 trend-row eval (cheap abort, NOT a pass)**: run base-class solo + `rrf(bm25+encoder)` on the 41q/201k trend row using the exact 1D T2 command sequence | impl | cheaper | trend table recorded; **abort bar**: base `rrf` > e5-small trend 0.423 *and* solo > 0.394 → proceed to D2; else stop and record (the step is not helping). BM25 column recorded as the haystack barometer. |
| **D2 full-row verdict + served-default decision**: run base-class over the full D1 pooled row; score solo + `rrf`; measure query latency; decide served default | design / measurement | **frontier + human** | verdict recorded here; gate bar `rrf` ≥ 0.30; latency verdict explicit; served-default decision (flip from hash, or not) recorded |
| **T3 (conditional) M2 space-bridge for query latency**: pair-corpus ridge/Procrustes fit (offline, no GPU, no deps) → 768×D map in the artifact → one GEMV apply at query time | design + impl | **frontier design; cheaper impl** | start only if D2 says quality passes but base-tower query p95 > 150 ms; kill-switch: bridged query loses > 10% of base tower's own-query recall on the trend row → dead, queries pay base-tower latency |
| **T4 (conditional) coarse-to-fine re-test with base-class**: re-run the `coarse2fine(3)` column with 768-dim instance means (e5-small's 384-dim means collapsed to 0.017; 768-dim may finally be discriminative — the 0.438 reference was c2f) | impl | cheaper | start only if D2 clears 0.30 solo/rrf and c2f is the path to the stretch 0.35 / reference 0.438; measured lift recorded |
| **T5 SDOT/int8 (PARKED — do not start)**: quarantined `std::arch` SDOT microkernel (PERF §5) | impl/design | **human approval required** | **Not started in 1E.** Reserved for a model that *clears* the quality bar but then has a latency/throughput problem (see §9 — not e5-small, not a rescue for a quality failure). |

## 9. Kill criteria (explicit, measured)

1. **T2 abort:** base-class trend `rrf` ≤ e5-small trend `rrf` (0.423) **or**
   base-class trend solo ≤ 0.394 → **stop.** The weight-class step is not
   helping on the easy row; the full row will not rescue it (1D proved the trend
   is the optimistic case). Record and do not run the full row.
2. **D2 solo sanity:** base-class full-row solo recall ≤ BM25 (0.134) **or** ≤
   e5-small full-row solo (0.152) → the step is a regression; **abort.**
3. **D2 partial:** base-class full-row `rrf` ∈ [0.16, 0.30) → **partial.**
   Record the honest number; do *not* freeze a below-bar `--assert-gate`. The
   earned escalation is either (a) the quarantined-int8 ticket **on this model**
   (only if latency is the binding constraint — it isn't a quality fix), or (b)
   Stage 1F: the EmbeddingGemma bridge/port (the 0.438 reference, license + cost
   permitting). The decision is frontier + human.
4. **D2 pass:** base-class full-row `rrf` ≥ 0.30 → **keep.** Freeze the
   `--assert-gate` at the measured margin, flip the served default from `hash`
   to `encoder:<base-class artifact>` in the runbook (the MCP/harness wiring
   already exists from 1D), and record the verdict in README honest-status +
   ROADMAP.
5. **Parity/determinism regression at any point:** if teacher parity drops below
   0.999 or byte-determinism breaks → **block** (these are rule-1/rule-2
   invariants, not tuning knobs).

## 10. Definition of done (the served-default decision is the deliverable)

- [ ] D1 model + license recorded (human sign-off).
- [ ] T1 converter + artifact + goldens green (parity ≥ 0.999, layer ≤ 5e-3,
      determinism byte-identical across runs/thread-counts).
- [ ] T2 trend row recorded; abort-bar decision taken.
- [ ] D2 full-row verdict recorded with solo + `rrf` + latency; served-default
      decision taken (flip to base-class, or the honest partial/negative).
- [ ] README honest-status + ROADMAP Stage-1 row updated with the 1E number.
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean; deps
      unchanged; no `unsafe` added; 0 network in CI.

## 11. Out of scope (explicit "what NOT to do next")

- **SDOT/int8 on e5-small (or on any model that has not cleared 0.30).**
  Acceleration makes a weak retriever *faster*; it does not change recall.
  e5-small's `rrf` is 0.160 vs 0.30 — a **0.14 absolute quality gap**. Spending
  the project's most expensive resource — a human-approved `unsafe` quarantine
  (AGENTS rule 4) — on a model that has already failed the quality bar would burn
  the `unsafe` budget on a loser that will never be served. PERF §4 already
  measured safe-Rust int8 at only ~1.2× f32 and forbade retrying it as scalar
  safe Rust; a real int8 path is SDOT/VDOT intrinsics behind a separate D-ticket
  + human approval, reserved for a model that *passes* quality and then has a
  latency problem. Int8-on-e5-small is premature by construction.
- **More e5-small tuning** (dim sweeps, prefix variants, fusion-depth sweeps).
  The diagnosis is the model's weight class; tuning within the class will not
  close a ~2× ratio gap (1D's trend→full transfer proves the absolute number is
  pool-fragile, not parameter-tunable).
- **The EmbeddingGemma bridge/port.** High ceiling but license (Gemma terms) +
  cost (~4–5× backfill, needs bridge or int8) gate it behind 1E. Stage 1F (to
  specced) is the earned escalation *only if* 1E lands partial (kill criterion
  #3). If a clean MIT base clears 0.30, Gemma is never needed.
- **Late-interaction / MaxSim as a primary.** An amplifier, not a fix; it blows
  up the index at scale and cannot rescue a base a weight class too small.
  Rerank-only variant reopens after a discriminative base lands.
- **Query-decomposition / cross-encoder reranking.** Partly falsified already
  (1D T6 doc2query negative at 0.5B; `STAGE_3B` iterative retrieval zero lift).
  Strategy moves can't fix a base ceiling.
- **A new crate, a new runtime dep, or any `unsafe`.** None is needed; the seam
  is complete from 1D.
