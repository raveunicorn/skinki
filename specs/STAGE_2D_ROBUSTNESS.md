# Stage 2D — Robustness: fuzzing, `skinki doctor`, the format registry (SPEC)

> Product backlog, batch 2026-07-B. The repo is full of hand-rolled binary
> and JSON-lines decoders (`skinki-store` framing, `rabitq.idx`, the artifact
> logs, soon `SKEMB001` and `SKPKG001`) — exactly the code class where
> hostile or bit-rotted bytes cause panics and silent misreads. Production
> readiness = every decoder provably total (error, never panic), one command
> that tells a user their store is healthy, and one document that says what
> every on-disk byte means and how formats evolve.

- **Status:** ready to build (doctor's CRC row lights up after Stage 2C;
  everything else independent)
- **Owner of the design (frontier/human):** frontier — the "total decoder"
  contract and the migration policy are locked; targets and plumbing are
  mechanical.
- **Delegatable to (cheaper model):** **yes, all tickets.**

> Read [`../AGENTS.md`](../AGENTS.md). `cargo-fuzz` is dev/CI tooling (not a
> runtime dep). Fuzz targets live under `fuzz/` and never ship.

## 1. Hypothesis

Every decoder in the workspace satisfies the **total-decoder contract** —
arbitrary input bytes produce `Ok` or a typed `Err`, never a panic, OOM, or
silent wrong value that framing could have caught — provable by fuzzing with
a checked-in regression corpus; and a `skinki doctor` command surfaces store
health (formats, integrity, sizes, index freshness) so "is my memory okay?"
has a one-command answer. Falsifiable per decoder: the fuzzer either finds a
counterexample (fix it, keep the case) or runs clean.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Fuzz targets exist | every decoder listed in §3 has a target | `fuzz/` inventory test |
| **No panics** | 0 crashes on: the full checked-in regression corpus (CI, every PR) + 60 s/target smoke (CI, nightly) | `cargo fuzz` runs |
| Deep runs | ≥ 4 CPU-hours/target once per release, findings triaged | release checklist item |
| Memory bound | decoders reject length fields implying > 1 GiB allocations *before* allocating | targeted unit tests + fuzz |
| `doctor` correctness | healthy fixture → exit 0 + golden report; each injected fault class → exit 1 + the right diagnostic line | fault-matrix test |
| `doctor` speed | ≤ 5 s on a 1 GB store (without `--scrub`) | bench report |
| FORMATS.md coverage | every magic string in the workspace appears in the registry (grep-driven test) | doc-coverage test |

## 3. Public interface

Fuzz targets (`fuzz/fuzz_targets/`), one per decoder:

```
store_decode_event, store_decode_unit, store_open_segment (arbitrary segment
bytes in a tmp dir), rabitq_load, ivf_load, jsonl_replay (each record type),
ledger_load, corpus_json_load, skemb_load (after 1B), skpkg_import (after 6D),
connector_parse_* (after 6E, one per connector)
```

Every target is the same shape: `fuzz_target!(|data: &[u8]| { let _ =
decode(data); });` — the *contract* is "returns, never panics"; sanitizers
catch the rest. Found cases land in `fuzz/corpus/<target>/` (checked in) and
CI replays them as plain tests via a tiny `#[test]` harness (so the
no-regression guarantee doesn't require nightly/cargo-fuzz on every PR).

```
skinki doctor --store <dir> [--scrub] [--json]
```

Report rows (each: OK / WARN / FAIL + one action line):

| Row | Checks |
| --- | --- |
| formats | every file's magic+version known (FORMATS.md registry compiled in) and readable by this binary |
| framing | segment tails valid; torn-tail state |
| integrity | (with `--scrub`, post-2C) CRC pass over all v2 records |
| dedup | runs sorted, non-overlapping, binary-searchable |
| ledger | loads; stale-closure self-check on a probe |
| index | index files present, dim/count consistent with the store, embedder artifact hash matches (post-1B) |
| sizes | bytes by category; growth since last `doctor` (state file) |

`FORMATS.md` (repo root): the registry — one section per magic
(`KXRABQ01`, `SKSEG01/02`, `SKEMB001`, `SKPKG001`, dedup runs, jsonl record
types), each with: layout, version history, and the **migration policy,
locked**: *read every version ever shipped; write only the newest; never
rewrite committed bytes; a format a released binary cannot read = major
version bump (per Stage 6C).* Reserved bits/flags documented (e.g. 6D's
encryption flag).

## 4. Invariants (must always hold)

- Total-decoder contract on every listed decoder (panic found ⇒ bug, fixed,
  case kept forever).
- Length-prefix sanity *before* allocation, everywhere.
- `doctor` is read-only (even `--scrub` — report, never repair; repair is a
  human decision, per Stage 2C).
- `doctor --json` output is stable (machine-readable; a future menu-bar app
  will poll it).
- Fuzz corpus is deterministic CI input; nightly smoke may *add* cases but
  PRs replay a pinned set.
- No new runtime deps.

## 5. Test plan

- **Unit:** per-decoder hostile-input cases (truncations, huge lengths, wrong
  magics, future versions) — the fuzzer's greatest hits, hand-promoted.
- **Regression:** the checked-in fuzz corpus replayed as `#[test]`s on every
  PR (no nightly needed).
- **Fault matrix (`doctor`):** healthy fixture golden; then per-row injected
  faults (unknown magic, torn tail, unsorted dedup run, dim mismatch, stale
  scan-state) → the right FAIL row, exit 1.
- **Doc coverage:** grep all `b"..."` magics in the workspace ⊆ FORMATS.md.
- **Gate command:** `cargo test -p skinki-harness doctor` + the corpus-replay
  test suite + (nightly CI) `cargo fuzz run <each> -- -max_total_time=60`.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 fuzz scaffolding + first 6 targets (store, rabitq, jsonl, ledger, corpus) + corpus-replay harness | impl | cheaper | targets run; replay suite in CI |
| T2 fix everything T1 finds; promote cases to unit tests; length-sanity audit across all decoders | impl | cheaper | 60 s smoke clean per target |
| T3 `skinki doctor` + fault-matrix tests + `--json` | impl | cheaper | matrix green; golden report |
| T4 `FORMATS.md` + doc-coverage test + migration policy text | impl | cheaper (human reviews policy wording) | coverage test green |
| T5 CI wiring: PR replay job, nightly smoke job, release deep-run checklist | impl | cheaper | jobs green; checklist in CONTRIBUTING |
| T6 (rolling) add targets as new formats land (1B skemb, 6D skpkg, 6E parsers) | impl | cheaper | inventory test enforces it |

## 7. Definition of done

- [ ] All targets clean on smoke; replay suite green in CI; `doctor` shipped.
- [ ] `cargo test`, clippy, fmt clean.
- [ ] `FORMATS.md` complete with the migration policy; README links `doctor`
      under a "trust but verify" section.
- [ ] Decision recorded: findings count by class (panics / OOMs / silent
      misreads) — the honest tally of what fuzzing caught.

## 8. Out of scope

- Automatic repair of any fault (report-only by design).
- Fuzzing the MCP JSON-RPC dispatch (serde_json is the parser there; a thin
  target is welcome under T6 but not a gate).
- Property-based testing frameworks (proptest etc. — a dep; hand-rolled
  seeded properties remain the house style).
- Differential fuzzing against other implementations (nothing to differ
  against; formats are ours).
