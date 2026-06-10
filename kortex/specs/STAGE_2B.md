# Stage 2B — Store durability: rotation, crash recovery, persistent dedup (SPEC)

- **Status:** done (gate passed)
- **Owner of the design (frontier/human):** implemented (frontier); this spec
  documents the locked design as the contract for future changes.
- **Why it exists:** the Stage 2 substrate passed its gate but had three
  durability holes invisible to that gate: (1) `open()` rebuilt the full dedup
  `HashMap` by scanning *all* segments — cold start grew with history and the
  map itself grew RAM linearly (~250+ MB at 5M events); (2) every `sync()`
  created a new segment, so frequent durable syncs meant thousands of tiny
  files and a linear segment lookup; (3) appends lived only in RAM until
  `sync()`, with no recovery story for a torn final write.

> Read [`../../AGENTS.md`](../../AGENTS.md). Gate is law; determinism is
> mandatory; no new dependencies (pure Rust + allowed crates). `unsafe` stays
> quarantined in the mmap module.

## 1. Hypothesis

Size-based segment rotation + write-through appends + a persisted, sorted-run
dedup index make the store durable and give O(one segment) reopen — without a
heavy embedded DB and without regressing any Stage 2 budget.

## 2. Budgets / fitness function (the gate)

Measured by `kortex store-bench` (now also at `--entries-per-day 270`, ~500k
entries / ~894k units). All previous budgets retained, two added:

| Metric | Budget | Actual (5y, 270/d) |
| --- | --- | --- |
| Content overhead | ≤ 1.25x | **1.211x** |
| Index bytes per unit | ≤ 24.0 | **20.0** |
| Random-access `get_unit` p95 | < 1 ms | **0.3 us** |
| Ingest, buffered (sync once at end) | ≥ 50,000 units/s | **2.5M** |
| **Ingest, durable (fsync per event)** | **≥ 100 events/s** | **~240** |
| **Cold reopen incl. one random read** | **< 1000 ms** | **80 ms @ 894k units** |
| Torn-tail recovery | committed records intact | ✅ (tests) |
| Determinism | byte-identical files (segments + runs) | ✅ |

`store-bench --assert-gate` exits non-zero if any metric misses.

## 3. Design, locked

- **Segment rotation by size** (`StoreOptions::segment_target_bytes`, default
  64 MiB), not by `sync()`. Ids stay `(segment << 32) | byte_offset`. Segment
  lookup is a binary search over the finished-segment list.
- **Write-through appends.** Records go to the current segment file as they
  arrive (buffered writer); `sync()` = flush + fsync. The current segment is
  readable via an mmap of its validated-at-open prefix plus an in-RAM tail
  mirror of bytes appended since open; records never straddle that boundary.
- **Torn-tail recovery.** `open()` framing-validates the last segment of each
  stream (`validated_prefix_len`) and physically truncates garbage past the
  last complete record. Committed bytes are never rewritten (append-only in
  spirit; the truncation only removes a record that was never acknowledged).
- **Persistent dedup runs.** At event-segment rotation, the RAM delta
  (hash → id since the last run) is drained into an immutable sorted run
  `dedup-NNNN.run` (magic + count + covered-watermark + sorted
  `(u128 hash, u64 id)` entries), written atomically (tmp + fsync + rename +
  dir fsync). Lookups: RAM delta, then binary search over each mmap'd run.
  More than 8 runs → compacted into one. Corrupt/missing runs are discarded
  wholesale and the index rebuilt by scanning — a slower open, never wrong.
  Key invariant making rebuild trivial: segments contain only *unique* events
  (duplicates are never appended), so a suffix scan can't collide with runs.
- **Unit-count watermark.** `counts.meta` (JSON, atomic rename) is written at
  unit-segment rotation; `open()` adds a scan of only the units past the
  watermark. Missing/corrupt meta → full-scan fallback.
- **Reopen cost model:** mmap finished segments + mmap runs + framing-scan at
  most one segment tail per stream. Independent of total history size.

## 4. Invariants

- Provenance: `unit_text(id) == &event_text(u.event)[u.byte_start..u.byte_end]`.
- Ids stable across reopen; reopen continues the same segment until target.
- `unit_count() == units().count()` even before `sync()`.
- After `sync()` returns, all appended records survive crash/power loss; a
  torn unsynced tail never corrupts committed records.
- Same input → byte-identical segment *and run* files.
- No `unsafe` outside the mmap module; no new dependencies.

## 5. Test plan (implemented)

- Rotation: multi-segment ids readable live + across reopen; reopen continues
  the same segment (no per-open segment churn).
- Crash: torn event/unit tails recovered; append-after-recovery lands cleanly.
- Dedup: hit via RAM delta, via persisted run after rotation, via runs after
  reopen, via full-rebuild fallback after deleting all runs; compaction bounds
  run count and stays correct.
- Counts: `unit_count` correct across reopen with meta, and via full-scan
  fallback without it.
- Determinism: byte-identical files including runs.
- Gate: `cargo run --release -p kortex-harness -- store-bench --years 5 --assert-gate`

## 6. Out of scope

- Encryption-at-rest, multi-device sync — later.
- A `.kx` single-file container — only if a future budget breaks.
- Background compaction scheduling — Stage 4 ("sleep") owns job scheduling;
  compaction here is inline and bounded.
