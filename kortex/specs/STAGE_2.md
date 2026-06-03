# Stage 2 — Storage substrate: append-only L0 log + content-addressed unit store (SPEC)

- **Status:** ready-to-build
- **Owner of the design (frontier/human):** done below — the "use Lance/Cozo vs
  invent `.kx`" call is deferred (it needs *this stage's* benchmark data), so the
  delegatable slice is a pure-Rust substrate that produces that data.
- **Delegatable to (cheaper model):** **yes** — all impl tickets (T1–T6). The two
  design tickets (D1, D2) stay on frontier and are *not* part of this hand-off.

> Read [`../../AGENTS.md`](../../AGENTS.md) first. The gate is law; determinism is
> mandatory; no new third-party dependencies (pure Rust + the already-allowed
> `serde`, `serde_json`, `libc`, `clap`, `anyhow`).

## 1. Hypothesis

A pure-Rust, mmap-backed **append-only L0 log** plus a **content-addressed unit
store** can hold years of raw capture with low overhead (<= 1.25× raw bytes),
O(1) random access (p95 < 1 ms), and lossless provenance back to the exact source
bytes — *without* pulling in a heavy embedded DB. If true, we avoid a big
dependency; if a later budget breaks, we have the numbers to justify a custom
`.kx` codec.

## 2. Budgets / fitness function (the gate)

Measured by `kortex store-bench` over the Stage 0 synthetic corpus.

| Metric | Budget | How measured |
| --- | --- | --- |
| Storage overhead | <= 1.25× raw UTF-8 text bytes | total store size / sum(len(text)) |
| Random-access `get_unit` p95 | < 1 ms (warm mmap) | telemetry over N random ids |
| Ingest throughput | >= 50,000 units/sec | wall-clock over full corpus |
| Provenance round-trip | 100% | property test (see §5) |
| Dedup | identical events stored once | golden test |
| Determinism | byte-identical files for same corpus+seed | golden hash of segment files |

`store-bench --assert-gate` exits non-zero if overhead, p95, or throughput miss.

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
    /// Open or create a store rooted at `dir` (creates segment + index files).
    pub fn open(dir: &std::path::Path) -> anyhow::Result<Store>;

    /// Append a raw capture event; deduplicates by content hash (§ T3).
    pub fn append_event(&mut self, ev: &RawEvent) -> anyhow::Result<EventId>;

    /// Append a derived unit (must reference an existing event + valid span).
    pub fn append_unit(&mut self, u: &Unit) -> anyhow::Result<UnitId>;

    /// Zero-copy read of the raw event text (borrows the mmap).
    pub fn event_text(&self, id: EventId) -> anyhow::Result<&str>;

    /// The exact source slice a unit points at (the provenance contract).
    pub fn unit_text(&self, id: UnitId) -> anyhow::Result<&str>;

    pub fn event_count(&self) -> usize;
    pub fn unit_count(&self) -> usize;
    pub fn units(&self) -> impl Iterator<Item = (UnitId, Unit)> + '_;

    /// Flush buffers + index to disk (crash-safe append boundary).
    pub fn sync(&mut self) -> anyhow::Result<()>;
}

/// Deterministic L0 -> L1 derivation for this stage (real enrichment is Stage 3).
/// Splits an event's text into units on sentence boundaries, preserving spans.
pub fn derive_units(event: EventId, text: &str) -> Vec<Unit>;
```

### Storage layout (the design, locked)

- **Segmented append-only files.** `events-000.seg`, `units-000.idx`, etc. Each
  record: little-endian `len: u32` + payload, payload = bincode-free manual
  encoding or `serde_json` (prefer compact manual: `created_utc_secs` i64,
  `source` u8, `text_len` u32, text bytes — no JSON overhead, to hit the 1.25×
  budget). `EventId`/`UnitId` = byte offset of the record within its segment
  (stable, gives O(1) seek).
- **mmap reads.** Reuse the mmap pattern from
  [`kortex-vector::store`](../crates/kortex-vector/src/store.rs) (factor a shared
  `MmapBytes` if convenient, or duplicate the minimal unix `mmap`/`munmap`). On
  non-unix, fall back to reading into RAM. `unsafe` is allowed **only** in this
  mmap module, like the existing one; everything else stays
  `#![forbid(unsafe_code)]`.
- **Content addressing / dedup (T3).** Hash each event payload with a 128-bit
  hash built from two `fnv1a` passes over the bytes with different seeds
  (concatenate to 128 bits — pure Rust, no deps; collision risk negligible at
  ~5M). Keep a `HashMap<u128, EventId>` in RAM during ingest; on duplicate,
  return the existing `EventId` and store nothing.

## 4. Invariants (must always hold)

- **Provenance:** for every unit, `store.unit_text(id) ==
  &store.event_text(u.event)[u.byte_start..u.byte_end]`. Byte offsets are valid
  UTF-8 boundaries.
- **Append-only:** existing bytes are never rewritten; ids are stable across
  reopen.
- **Determinism:** same `(corpus, seed)` → byte-identical segment files (so a
  golden hash test is possible).
- **No `unsafe`** outside the mmap module. **No new dependencies.**
- Reopening a store and continuing to append must keep all prior ids valid.

## 5. Test plan

- **Unit:** append/read round-trip for events and units; reopen-and-continue; ids
  stable after `sync` + reopen.
- **Property:** for random corpora, `unit_text` equals the source span for every
  unit produced by `derive_units` (provenance round-trip = 100%).
- **Golden:** ingest a fixed seed corpus, assert a locked hash of the segment
  files (determinism) and a locked storage-overhead ratio.
- **Dedup:** appending the same event twice yields one stored record and equal
  `EventId`s.
- **Gate command:**
  `cargo run --release -p kortex-harness -- store-bench --years 5 --assert-gate`

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| D1: Lance/Cozo vs pure-Rust decision | design | frontier/human | uses store-bench numbers; **not in this hand-off** |
| D2: `.kx` semantic delta/dedup codec | design | frontier/human | only if a budget breaks; deferred |
| T1: `kortex-store` crate skeleton + types + manual record encoding | impl | cheaper | builds; types serialize round-trip |
| T2: segmented append-only log + mmap reader | impl | cheaper | reopen keeps ids; mmap zero-copy reads |
| T3: 128-bit content hash + ingest dedup | impl | cheaper | dedup golden test passes |
| T4: `derive_units` deterministic sentence split + provenance | impl | cheaper | provenance property test = 100% |
| T5: `store-bench` CLI subcommand + report + `--assert-gate` | impl | cheaper | gate exits non-zero when budgets miss |
| T6: full test suite (unit/property/golden/dedup) | impl | cheaper | `cargo test` green |

## 7. Definition of done

- [ ] `store-bench --assert-gate` green; overhead <= 1.25×, p95 < 1 ms, ingest >= 50k/s.
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [ ] CI: add a `store gate` step mirroring Stage 1's gate job.
- [ ] ROADMAP Stage 2 row → done; this spec Status → done; record the D1 decision
      (kept pure-Rust, or adopted Lance/Cozo, with the bench numbers).

## 8. Out of scope

- Entity/relation extraction, embeddings, graph — **Stage 3**.
- Any `.kx` compression codec — only if D1 shows the budget breaks.
- Encryption-at-rest, multi-device sync — later.
