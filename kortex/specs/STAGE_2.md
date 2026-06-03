# Stage 2 — Storage substrate: append-only L0 log + content-addressed unit store (SPEC)

- **Status:** done (gate passed)
- **Owner of the design (frontier/human):** implemented; D1 decision recorded below.
- **Delegatable to (cheaper model):** was yes — all impl tickets (T1–T6) completed.

> Read [`../../AGENTS.md`](../../AGENTS.md) first. The gate is law; determinism is
> mandatory; no new third-party dependencies (pure Rust + the already-allowed
> `serde`, `serde_json`, `libc`, `clap`, `anyhow`).

## 1. Hypothesis

A pure-Rust, mmap-backed **append-only L0 log** plus a **content-addressed unit
store** can hold years of raw capture with low overhead, O(1) random access
(p95 < 1 ms), and lossless provenance back to the exact source bytes — *without*
pulling in a heavy embedded DB. If true, we avoid a big dependency; if a later
budget breaks, we have the numbers to justify a custom `.kx` codec.

## 2. Budgets / fitness function (the gate)

Measured by `kortex store-bench` over the Stage 0 synthetic corpus. **Two metrics
replace the original single "overhead" metric per review decision.**

| Metric | Budget | Actual (5y, 270/d) |
| --- | --- | --- |
| Content overhead (event store / raw text) | ≤ 1.25x | **1.21x** |
| Index bytes per unit (unit store / unit count) | ≤ 24.0 | **20.0** |
| Random-access `get_unit` p95 (shuffled) | < 1 ms | **0.3 us** |
| Ingest throughput | ≥ 50,000 units/sec | **2.5M** |
| Provenance round-trip | 100% | ✅ |
| Dedup | identical events stored once | ✅ |
| Determinism | byte-identical segment files | ✅ |

`store-bench --assert-gate` exits non-zero if **any** gate metric misses.

## 3. Public interface

New crate `kortex-store`. Implement exactly these (names/signatures are the
contract; internals are your freedom):

```rust
pub type EventId = u64; // offset-derived, stable within a log
pub type UnitId = u64;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RawEvent {
    pub source: Source,      // Voice | Text | Import
    pub created_utc_secs: i64,
    pub text: String,        // raw captured text (UTF-8)
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Source { Voice, Text, Import }

/// A memory unit with lossless provenance back to L0 bytes.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Unit {
    pub event: EventId,
    pub byte_start: u32,     // span within RawEvent.text (UTF-8 byte offsets)
    pub byte_end: u32,
}

pub struct Store { /* mmap segments + index */ }

impl Store {
    pub fn open(dir: &std::path::Path) -> anyhow::Result<Store>;
    pub fn append_event(&mut self, ev: &RawEvent) -> anyhow::Result<EventId>;
    pub fn append_unit(&mut self, u: &Unit) -> anyhow::Result<UnitId>;
    pub fn event_text(&self, id: EventId) -> anyhow::Result<&str>;
    pub fn unit_text(&self, id: UnitId) -> anyhow::Result<&str>;
    pub fn event_count(&self) -> usize;
    pub fn unit_count(&self) -> usize;
    pub fn units(&self) -> impl Iterator<Item = (UnitId, Unit)> + '_;
    pub fn sync(&mut self) -> anyhow::Result<()>;
}

pub fn derive_units(event: EventId, text: &str) -> Vec<Unit>;
```

### Encoding (actual, lossless)

Event record: `created_utc_secs (i64 LE) | source (u8) | text bytes (UTF-8)`.
`text_len` is derived from the outer length prefix — no cap, no `assert!`.

Unit record: `event_id (u64 LE) | byte_start (u32 LE) | byte_end (u32 LE)`.

### Storage layout (the design, locked)

- **Segmented append-only files.** `events-000.seg`, `units-000.idx`. Each record:
  little-endian `len: u32` + payload. `EventId`/`UnitId` = byte offset within
  segment.
- **mmap reads.** Pattern reused from `kortex-vector::store`. `unsafe` only in
  this module; everything else `#![forbid(unsafe_code)]`.
- **Content addressing / dedup.** 128-bit hash from two FNV1a-64 passes with
  different seeds. `HashMap<u128, EventId>` in RAM during ingest.

## 4. Invariants (must always hold)

- **Provenance:** `store.unit_text(id) == &store.event_text(u.event)[u.byte_start..u.byte_end]`.
- **Append-only:** existing bytes never rewritten; ids stable across reopen.
- **Determinism:** same input → byte-identical segment files.
- **No `unsafe`** outside the mmap module. **No new dependencies.**
- `unit_count() == units().count()` even before `sync()`.

## 5. Test plan

- **Unit:** append/read round-trip; reopen-and-continue; ids stable.
- **Property:** provenance round-trip = 100%; `unit_count` matches `units()` before sync.
- **Golden:** byte-identical segment files (not just size); locked overhead ratio.
- **Dedup:** same event → same `EventId`.
- **Gate:** `cargo run --release -p kortex-harness -- store-bench --years 5 --assert-gate`

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| D1: Lance/Cozo vs pure-Rust decision | design | frontier/human | **Resolved: kept pure Rust.** All budgets met. |
| D2: `.kx` codec | design | frontier/human | Deferred — no budget broke. |
| T1: crate skeleton + types + encoding | impl | cheaper | ✅ |
| T2: segmented append-only log + mmap | impl | cheaper | ✅ |
| T3: 128-bit content hash + dedup | impl | cheaper | ✅ |
| T4: `derive_units` + provenance | impl | cheaper | ✅ |
| T5: `store-bench` CLI + `--assert-gate` | impl | cheaper | ✅ |
| T6: full test suite | impl | cheaper | ✅ |

## 7. Definition of done

- [x] `store-bench --assert-gate` green; content overhead 1.21x, index 20.0 B/unit, p95 0.3 us, ingest 2.5M/s.
- [x] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [ ] CI: add a `store gate` step mirroring Stage 1's gate job.
- [x] ROADMAP Stage 2 row → done; this spec Status → done.
- [x] D1 decision: kept pure Rust (Lance/Cozo not needed).

## 8. Out of scope

- Entity/relation extraction, embeddings, graph — **Stage 3**.
- Any `.kx` compression codec — only if a future budget breaks.
- Encryption-at-rest, multi-device sync — later.