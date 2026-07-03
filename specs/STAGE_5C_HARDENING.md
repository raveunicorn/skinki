# Stage 5C — Insight & substrate hardening (SPEC)

> Batch 1 of the 2026-07 frontier review (`REVIEW_FRONTIER_2026_07.md` §4).
> Pure correctness debt: latent bugs, Unicode readiness, statistical honesty,
> scale headroom, and the store test flake. **No new features.** Every existing
> gate must stay green with unchanged budgets; this stage only adds tests and
> fixes defects those tests expose.

- **Status:** ready to build
- **Owner of the design (frontier/human):** frontier — every fix below is fully
  designed here (exact algorithms given); nothing is left to implementer taste.
- **Delegatable to (cheaper model):** **yes, all tickets** — the subtle parts
  (T3 statistics, T4 null model) are frozen as pseudocode in this spec; frontier
  reviews the diffs for T3/T4 only.

> Read [`../AGENTS.md`](../AGENTS.md) first. Determinism is law (rule 2). Do
> not touch `skinki-corpus` generation (golden hashes). Never weaken a gate.

## 1. Hypothesis

The Stage-5 keystone and the store contain five concrete, test-provable defects
(candidate-id collision, ASCII-only word boundaries, pooled multi-family FDR,
a mislabeled FDR procedure with a uniform-days null, and a non-deterministic
test fixture pattern) plus scale headroom no gate measures. Fixing them keeps
every existing gate green **and** makes the following new assertions pass —
falsifiable per ticket by its named test.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| All existing gates | unchanged, green | `insight-eval --assert-gate`, `graph-eval`, `store-bench`, `ledger-bench`, CI suite |
| Full-engine union (T1) | `full` surfaces = union of isolated engines | new unit test `full_engine_surfaces_union_of_isolated_engines` |
| Unicode word boundary (T2) | Cyrillic substrings never match | new unit tests (see §5) |
| Per-family validation (T3) | contradiction `p=0` does not change the structural acceptance set | new unit test |
| Temporal null honesty (T4) | gate green on both seeds with the permutation null; procedure documented as what it is | `insight-eval --assert-gate` |
| **Insight throughput** (T5) | ≥ 2,000 entries/s (all 3 detectors combined) at ≥ 100k entries; time scales ≲ 1.5× linear from 10k→100k | new `insight-eval --scale-report` (asserted with 4× CI headroom: hard floor 500 entries/s) |
| Store test determinism (T6) | `cargo test -p skinki-store` green 20/20 consecutive runs, default threads | `scripts/store-test-soak.sh` |

## 3. Public interface

No public interface changes outside `skinki-insight`; the one type change:

```rust
/// Candidate ids are namespaced by detector family so pooling candidates from
/// several detectors can never collide (the top byte is the kind tag).
pub type CandidateId = u64;

#[inline]
pub fn candidate_id(kind: InsightKind, seq: u64) -> CandidateId {
    debug_assert!(seq < (1 << 56));
    ((kind_tag(kind) as u64) << 56) | seq
}
fn kind_tag(k: InsightKind) -> u8 {
    match k { InsightKind::StructuralBridge => 1,
              InsightKind::TemporalLead => 2,
              InsightKind::Contradiction => 3 }
}
```

Every detector constructs its ids via `candidate_id(kind, seq)`
(`StructuralBridgeDetector` keeps `seq = entity id`; the others keep their
running counters). `InsightEngine::discover` keys candidates by the (now
collision-free) id, **validates per `InsightKind`** (one `validate` call per
family present in the candidate set), and merges accepted sets in
`(kind_tag, effect desc, id asc)` order.

New shared helper (one home, both callers):

```rust
// skinki-eval::jsonl — the one JSON-lines append/replay used by ArtifactLog,
// NarrationLog, and Stage 5B's JudgmentLog. Append-only; replay in file order.
pub fn jsonl_append<T: Serialize>(path: &Path, rec: &T) -> io::Result<()>;
pub fn jsonl_replay<T: DeserializeOwned>(path: &Path) -> io::Result<Vec<T>>;
```

## 4. Invariants (must always hold)

- Determinism (rule 2) everywhere, including the new permutation null (seeded
  SplitMix64 only).
- All existing golden hashes (V1 corpus, graph, sleep trace) unchanged.
- `#![forbid(unsafe_code)]` stays on every crate that has it.
- Natural-language matching is **char-based, Unicode-aware** — no
  `as_bytes()` / `is_ascii_*` on user text anywhere in `skinki-insight`.
- No new dependencies.

## 5. Test plan

- **T1 unit:** build `InsightEngine::full_produce`-shaped engine (all three
  detectors) on V2 seeds 42 & 7; assert its surfaced set equals the union of
  `structural()`, `temporal()`, `contradiction()` outputs (same descriptions,
  same citation sets).
- **T2 unit:** `contains_word("сплошной контраст", "раст") == false`;
  `contains_word("посадил раст в саду", "раст") == true`;
  `contains_word("distrust", "rust") == false`; `contains_word("rust!", "rust")
  == true`. Same cases through `profile_entity_days` end to end.
- **T3 unit:** a candidate set mixing structural candidates (p ∈ {0.001…0.2})
  with contradiction candidates (p=0): the structural acceptance set must be
  identical with and without the contradictions present.
- **T4:** existing two-seed temporal assertions stay green; new property test —
  on a null corpus (no planted temporal pattern; generate V2 and strip by
  scoring against empty GT is NOT valid — instead run the detector on a
  seed-shuffled day permutation of the entries) the detector surfaces 0
  candidates on both seeds.
- **T5:** `insight-eval --scale-report` prints entries/s at ~10k and ~100k
  entries (V2, `entries_per_day` scaled) and the 10k→100k time ratio; asserts
  the §2 floors.
- **T6:** `scripts/store-test-soak.sh` runs `cargo test -p skinki-store` 20×,
  exits non-zero on any failure.
- **Gate command:** `cargo run --release -p skinki-harness -- insight-eval
  --assert-gate` (must include the new `--scale-report` assertions when
  `--assert-gate` is passed) + full CI suite.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| ✅ **T1** candidate-id namespacing + per-family `validate` in `discover` | impl | done (frontier, PR #8) | union test green; existing gate green; no golden moved |
| ✅ **T2** Unicode `contains_word` + audit every `as_bytes`/`is_ascii_*` touch of entry text in `skinki-insight` | impl | done (frontier, PR #8) | §5 T2 tests green on Cyrillic + ASCII |
| ✅ **T3** per-family FDR (`validate_per_kind`) + isolation test | impl | done (frontier, PR #8) | §5 T3 test green; pooled-vs-per-family distortion demonstrated on a fixed vector |
| ✅ **T4** temporal null rework — **done (frontier, PR #8)**, with a stronger design than first drafted: (a) the `1e-6` pre-filter and the analytic binomial+Bonferroni p are deleted; (b) **split-half selection** — the lag δ* is chosen on A's odd-indexed mention days and tested at that *fixed* lag on the held-out even half, so search optimism never reaches the p-value and no multiple-testing correction is needed; (c) the p-value is an **exact circular-shift enumeration**: `p = #{δ in 0..D : c_test(δ) ≥ c_test(δ*)} / D` with period `D = max_day+1` and B's days smeared ±tol — deterministic, **no RNG at all** (strictly better than the sampled-shift variant this ticket originally specified), resolution 1/D, and the null inherits B's real day distribution so shared burstiness is not mistaken for signal. Effect-size floors (MIN_COUNT=4, MIN_RATIO=0.35 on full data; MIN_TEST_COUNT=2 on the held-out half) stay as guards. Per-family BH at `q=0.01` gates the survivors. | impl (design frozen above) | done (frontier) | **Measured:** temporal recall 0.800, precision 1.000, false-insight 0.000 on seeds 42 & 7; day-shuffled null-corpus test silent on 3 fixed shuffles; gate PASS |
| **T5** perf: ~~lowercase every entry **once** per `propose`~~ and ~~binary-search `b_days` in `count_at_lag`~~ (both done in PR #8); remaining: the same lowercase-once fix in `profile_entities`, and add `--scale-report` to `insight-eval` with the §2 throughput floors | impl | cheaper | §2 throughput floors; identical surfaced sets before/after (golden) |
| **T6** store fixture isolation: every `skinki-store` test uses `temp_dir/skinki_<testname>_<pid>_<COUNTER.fetch_add>` dirs; add `scripts/store-test-soak.sh` | impl | cheaper | 20/20 soak green |
| **T7** cleanups: extract `skinki-eval::jsonl` and port `ArtifactLog`/`NarrationLog` to it; move the duplicated `SemanticRetriever` to `skinki-baseline` (add its `skinki-vector` dep) and reuse from mcp+harness; delete `BitWriter::finish`'s no-op padding loop; make `record_insight_derivations` hash an explicitly-formatted string (no `{:?}`); fix the 0.075-vs-0.325 BM25 doc drift in `STAGE_3.md` | impl | cheaper | CI green; behavior-identical goldens |

## 7. Definition of done

- [ ] All §2 budgets asserted and green in CI (insight gate extended, soak
      script wired as a CI step or documented as pre-merge for store PRs).
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [ ] `specs/STAGE_5.md` measurement log gains a "round 5 (hardening)" entry
      recording the T4 procedure change and re-measured numbers.
- [ ] Decision recorded: did the permutation null change any surfaced insight
      vs the analytic null, and in which direction.

## 8. Out of scope

- Real-data detectors and the oracle judge (Stage 5B).
- The V3 corpus (Stage 0B) — T4's null-corpus test uses day-shuffling, not V3.
- Ledger/store hash upgrades (Stage 2C).
- Any retrieval work (Stage 1B).
