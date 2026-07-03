# Stage 6E — Connectors & the watcher: real data in, continuously (SPEC)

> Product backlog, batch 2026-07-B. Algorithms don't make a memory feel
> personal — **having your data inside it in 10 minutes** does. This stage
> ships deterministic import adapters for the formats people actually have
> (Obsidian/Markdown vaults, Telegram/WhatsApp exports, plain transcripts)
> plus the polling watcher that keeps a store fed continuously — closing
> README open-problem #5 ("a daemon or filesystem watcher").

- **Status:** ready to build (works today over `skinki ingest`; pairs
  naturally with Stage 6B's write path)
- **Owner of the design (frontier/human):** frontier — the `Connector`
  contract and idempotence rules are locked; each parser is mechanical.
- **Delegatable to (cheaper model):** **yes, every ticket** — parsers with
  golden fixtures are the ideal delegation shape.

> Read [`../AGENTS.md`](../AGENTS.md). Rule-2 nuance, stated once: **capture
> timestamps are data, not logic.** A connector records each item's
> *source-asserted* time (front-matter date, chat message timestamp, file
> mtime as last resort) into `RawEvent::created_utc_secs`; nothing downstream
> branches on wall clock. Parsing itself is a pure function of the input
> bytes. No new deps — every format below is parseable with std + serde_json.

## 1. Hypothesis

A `Connector` trait with per-format parsers, feeding the existing
content-hash-dedup'd store, makes imports **idempotent by construction**
(re-running any import or watcher scan appends zero duplicates) and correct
per format (golden fixtures) — so "point skinki at my vault / my chat export"
is one command, safe to re-run forever, and the watcher is just that command
in a loop.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Idempotence (hard) | second run of any import/scan → 0 new events | test per connector + watcher |
| Parse correctness | each connector's golden fixtures → locked `RawEvent` sequences (text, time, source) | golden tests |
| Robustness | malformed input → skip-with-report, never panic; partial files never half-ingested | injection tests |
| Throughput | ≥ 5 MB/s parse+ingest on the fixture corpus | bench report |
| Watcher latency | new/changed file ingested within one poll interval (default 30 s) | integration test with a tmp dir |
| Watcher cost | idle scan of a 10k-file tree ≤ 200 ms, no re-reads of unchanged files (size+mtime gate) | bench report |
| Ordering determinism | same input tree → same event order (sorted paths, then in-file order) | golden |

## 3. Public interface

```rust
// New crate `skinki-connect` (#![forbid(unsafe_code)]; deps: serde,
// serde_json, anyhow, internal skinki-store).

/// One imported item, pre-store: the connector's whole job is turning source
/// bytes into these, deterministically.
pub struct SourceItem {
    pub text: String,
    pub created_utc_secs: i64,   // source-asserted time (data, not logic)
    pub source: skinki_store::Source,
    /// Stable provenance ref, e.g. "vault/notes/2024-03-01.md#2" — recorded
    /// into the event text? NO — kept as a prefix line? NO — stored in the
    /// item and written as a structured header line `[src: <ref>]` prepended
    /// to text (v0; a first-class provenance field is a store-format change
    /// deferred to FORMATS.md).
    pub src_ref: String,
}

pub trait Connector {
    fn name(&self) -> &'static str;
    /// Pure: bytes (+ path for context) -> items, in deterministic order.
    fn parse(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<SourceItem>>;
    /// Which files this connector claims (extension/name predicate).
    fn matches(&self, path: &Path) -> bool;
}

pub fn ingest_items(store: &mut Store, items: &[SourceItem]) -> IngestReport;
pub struct IngestReport { pub appended: usize, pub deduped: usize, pub skipped: Vec<(String, String)> }
```

Connectors, v1 (each = one ticket, one fixture dir under
`crates/skinki-connect/fixtures/<name>/`):

| Connector | Input | Item granularity | Time source |
| --- | --- | --- | --- |
| `markdown` (incl. Obsidian) | `.md` tree | one item per top-level `#`/`##` section; whole file if none | front-matter `date:` → filename `YYYY-MM-DD` → mtime |
| `telegram` | Telegram Desktop `result.json` | one item per text message (join `text` fragments) | message `date` |
| `whatsapp` | exported `_chat.txt` | one item per message line (multi-line continuation folded) | line timestamp |
| `jsonl` | generic `{"text","ts"?,"source"?}` lines | one per line | `ts` or mtime |
| `txt` | plain `.txt` tree | one per file | mtime |

Watcher:

```
skinki watch --store <dir> --source <path> --connector markdown \
             [--interval-secs 30] [--once]
```

Poll loop (no filesystem-notification dep): scan the tree, gate on
(size, mtime) vs a persisted scan-state file in the store dir, parse changed
files, `ingest_items`, `sync()`. `--once` does a single pass (this is also
what tests and cron users call). Content-hash dedup makes even a lost
scan-state harmless — the worst case is a re-parse, never a duplicate.

## 4. Invariants (must always hold)

- Idempotence via the store's content-hash dedup — connectors do **not**
  implement their own dedup; they must produce byte-stable text for unchanged
  input (no timestamps injected into text, `src_ref` header is stable).
- Parsing never panics on arbitrary bytes (fuzz targets in Stage 2D cover
  every `parse`).
- The watcher writes through the same `Store` API as `ingest`/`remember` —
  one write path, one durability story.
- Ordering: files sorted by path, items in source order — same tree, same
  event sequence.
- No new deps; no network; `#![forbid(unsafe_code)]`.

## 5. Test plan

- **Golden (per connector):** fixture inputs (incl. Cyrillic content,
  emoji, edge timestamps, empty files) → locked item sequences.
- **Idempotence (per connector + watcher):** run twice, assert
  `appended == 0` on the second run; touch one file, assert only its items
  appended.
- **Injection:** truncated JSON, invalid UTF-8, absurd sizes → skip-with-
  report, store unchanged for the bad file, good files ingested.
- **Watcher integration:** tmp tree, `--once`, add a file, `--once` again;
  scan-state deleted → third run appends 0 (dedup safety net).
- **Gate command:** `cargo test -p skinki-connect` — in CI.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 crate + `Connector`/`SourceItem`/`ingest_items` + `txt` + `jsonl` | impl | cheaper | goldens + idempotence green |
| T2 `markdown`/Obsidian (front-matter, sections) | impl | cheaper | goldens incl. a mini-vault fixture |
| T3 `telegram` + `whatsapp` | impl | cheaper | goldens from anonymized fixtures |
| T4 `skinki watch` (poll loop, scan-state, `--once`) | impl | cheaper | watcher tests green; cost budgets |
| T5 docs: "get your data in" page with per-source recipes | impl | cheaper | commands verified by the tests |

## 7. Definition of done

- [ ] All §5 tests green in CI; `cargo test`, clippy, fmt clean.
- [ ] README quickstart gains "import your notes/chats" as step 2.
- [ ] Decision recorded: which requested connector is next (issue template
      asks for a sample file + expected items — a new connector should be a
      one-afternoon community PR against the trait).

## 8. Out of scope

- Voice/STT (Stage 7 — capture hardware/OS territory).
- Apple Notes (no stable export format without OS automation — Stage 7).
- A first-class provenance field in the store record (FORMATS.md reserved;
  `[src:]` header line is v0).
- OS-native file watching (notify APIs) — polling is deterministic, portable,
  and cheap enough per the budgets; revisit only with measured pain.
