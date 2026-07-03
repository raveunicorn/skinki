# Stage 2C — Integrity: a real hash, per-record checksums, scrub-on-sleep (SPEC)

> Batch 6 of the 2026-07 review (`REVIEW_FRONTIER_2026_07.md` §4.7–4.8). Two
> defects in the trust chain: (1) the ledger's `ContentHash` is two correlated
> FNV-1a-64 passes — not collision-resistant, yet the ledger's **entire
> guarantee** ("changed premise ⇒ changed hash") rides on it; (2) the store
> validates *framing* only — bit-rot inside a committed record is silently
> served and silently poisons every downstream hash. For an engine whose pitch
> includes "fault-tolerant" and "provenance to source bytes", both must go.

- **Status:** ready to build
- **Owner of the design (frontier/human):** frontier — the hash choice
  (SHA-256, from scratch — the one place the guarantee itself earns it, Law 2),
  the frame layout, and the migration rules are locked below.
- **Delegatable to (cheaper model):** **yes, everything** — SHA-256 and CRC32
  are specification-defined algorithms with official test vectors; the gate is
  the vectors, not taste.

> Read [`../AGENTS.md`](../AGENTS.md). No new deps (both algorithms are
> implemented from scratch — that is the point). `#![forbid(unsafe_code)]` in
> the new crate. Never weaken `store-bench` budgets.

## 1. Hypothesis

A from-scratch SHA-256 (≈120 lines) and CRC32 (≈30 lines), both pinned by
official test vectors, plus a 4-byte checksum in the record frame and a
resumable scrub `Job`, close both trust holes **within the existing
performance budgets**: durable ingest stays ≥ 100 events/s, buffered ingest
≥ 50k units/s, content overhead ≤ 1.25×, reopen < 1 s. Falsifiable by the
existing `store-bench --assert-gate` and the new corruption-injection tests.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| SHA-256 correctness | all FIPS 180-4 + NIST CAVP short/long vectors pass | unit tests (vectors checked in) |
| CRC32 correctness | IEEE 802.3 test vectors (`"123456789"` → `0xCBF43926`, etc.) | unit tests |
| Corruption detection | 100% of single-byte flips in committed payloads detected on read & by scrub; 0 false alarms on clean stores | injection tests |
| `store-bench` budgets | unchanged: overhead ≤ 1.25×, index ≤ 24 B/unit, durable ingest ≥ 100/s, buffered ≥ 50k/s, reopen < 1 s, p95 random access < 1 ms | `store-bench --assert-gate` (unchanged thresholds) |
| Ledger gate | unchanged: invalidation-recall 1.000, over-invalidation 0 | `ledger-bench --assert-gate` |
| Scrub throughput | ≥ 100 MB/s; interruptible/resumable (Stage-4 `Job` contract) | `store-bench --scrub-report` + sleep-sim style checkpoint test |
| Hash throughput | SHA-256 ≥ 150 MB/s single-thread (scalar is ~2–4× that; generous CI floor) | bench report (informational) |

## 3. Public interface

New micro-crate **`skinki-hash`** (`#![forbid(unsafe_code)]`, zero deps —
a museum piece):

```rust
/// FIPS 180-4 SHA-256. Streaming + one-shot.
pub struct Sha256 { /* 8×u32 state + block buffer */ }
impl Sha256 {
    pub fn new() -> Self;
    pub fn update(&mut self, bytes: &[u8]);
    pub fn finish(self) -> [u8; 32];
    pub fn digest(bytes: &[u8]) -> [u8; 32];
}

/// CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320), table-driven.
pub fn crc32(bytes: &[u8]) -> u32;
```

Ledger migration (`skinki-ledger`):

```rust
impl ContentHash {
    /// SHA-256 truncated to 128 bits (the type stays u128; truncation of a
    /// cryptographic hash keeps collision resistance at the 2^64 birthday
    /// bound — honest bits this time).
    pub fn of(bytes: &[u8]) -> Self;   // same signature, new function
}
```

Store frame v2 (`skinki-store`): segment header magic bumps
`SKSEG01 → SKSEG02`; record frame becomes `len: u32 | crc32(payload): u32 |
payload`. `open()` accepts both versions (v1 segments have no CRC — read
as before); **new segments are always v2**. Reads verify CRC and return a
typed `CorruptRecord { segment, offset }` error instead of decoded garbage.

```rust
/// Resumable integrity scrub over all segments (a Stage-4 Job): verifies
/// every v2 record's CRC, reports corrupt (segment, offset, id) triples.
/// Checkpoint = (segment index, byte offset).
pub struct ScrubJob { /* store handle + cursor */ }
impl skinki_sleep::Job for ScrubJob { /* step/checkpoint/restore */ }
```

## 4. Invariants (must always hold)

- Committed bytes are never rewritten: v1 segments stay v1 forever; CRC
  arrives only with new segments (append-only law).
- `ContentHash::of` change moves ledger hashes — the ledger has **no durable
  persistence yet** (v0 JSON snapshots only), so this is the last cheap moment
  to migrate; any golden that pins hash *values* is updated, gate *metrics*
  are untouched.
- Dedup (`skinki-store`'s FNV-128) migrates **only if** `store-bench` ingest
  budgets hold with SHA-256 (T5 measures; if not, dedup keeps FNV — a dropped
  duplicate is a bounded, non-poisoning risk, recorded as such).
- Scrub never mutates; it reports. Repair policy (re-derive vs quarantine) is
  a human decision logged as an issue, not automated here.
- Determinism, 0 network, no new deps, no `unsafe`.

## 5. Test plan

- **Unit:** the full FIPS/CAVP SHA-256 vector set (empty, "abc", 448-bit,
  million-'a', plus 10 CAVP long-form vectors); CRC32 vectors; streaming ==
  one-shot property.
- **Property:** `Sha256::digest` equals chunked `update` for random splits
  (seeded); frame round-trip encode/decode.
- **Injection:** build a store, flip one byte at a seeded offset in a
  committed v2 record → read returns `CorruptRecord`; scrub finds exactly it;
  torn-tail recovery still truncates only the tail (unchanged behavior).
- **Migration:** open a v1-format fixture directory (checked in, tiny) →
  reads fine; new appends create v2 segments; reopen mixed store green.
- **Golden:** ledger-bench numbers unchanged; a locked SHA-256-based
  `ContentHash` golden.
- **Gate command:** `cargo run --release -p skinki-harness -- store-bench
  --years 5 --assert-gate` + `ledger-bench --assert-gate` + `cargo test -p
  skinki-hash -p skinki-store -p skinki-ledger`.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `skinki-hash`: SHA-256 + CRC32 + vectors | impl | cheaper | all vectors green; property tests |
| T2 ledger `ContentHash::of` → truncated SHA-256; update hash-value goldens; `skinki-insight`/`skinki-graph` derivation call-sites recompile unchanged (signature stable) | impl | cheaper | `ledger-bench --assert-gate` green |
| T3 store frame v2 + version-aware `open()` + `CorruptRecord` error + v1 fixture | impl | cheaper | migration + injection tests green; `store-bench` budgets hold |
| T4 `ScrubJob` (Stage-4 `Job`, checkpointable) + `--scrub-report` in `store-bench` | impl | cheaper | resumability test (sleep-sim pattern); throughput floor |
| T5 measure dedup-on-SHA-256 ingest cost; migrate or record the keep-FNV decision | impl + decision | cheaper measures, frontier decides | numbers in this spec; budgets green either way |

## 7. Definition of done

- [ ] All §5 tests + both gates green in CI.
- [ ] `cargo test`, clippy, fmt clean.
- [ ] README fault-tolerance claims updated to name the mechanisms (CRC
      frames, scrub job, cryptographic content addressing) — claims now
      backed by injection tests.
- [ ] Decision recorded: dedup hash migrated or kept, with the measured
      ingest numbers.

## 8. Out of scope

- Durable ledger persistence on `skinki-store` segments (still the
  DERIVATION_LEDGER follow-up; this stage only fixes what the hashes *are*).
- Automatic repair/re-derivation of corrupt records.
- Merkle-tree segment digests / snapshot signing (nice later; not earned yet).
- SIMD/hardware-accelerated hashing.
