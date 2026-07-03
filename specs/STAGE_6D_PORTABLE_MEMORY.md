# Stage 6D — Portable memory: export / import as a single file (SPEC)

> Product backlog, batch 2026-07-B. The architecture already implies the
> killer property — a memory is a directory of deterministic, self-describing
> files — but nobody can *hold* it yet. This stage makes it literal: **your
> memory is one file** you can back up, move between machines, hand to a new
> agent, or archive for a decade. "A portable file, not a vendor account" is
> the product one-liner this repo was built to earn.

- **Status:** ready to build (integrity checks strengthen after Stage 2C, but
  the container works without it)
- **Owner of the design (frontier/human):** frontier — the container format
  and determinism rules are locked below.
- **Delegatable to (cheaper model):** **yes, all tickets.**

> Read [`../AGENTS.md`](../AGENTS.md). No new deps: the container is a ~100-
> line custom format (deterministic tar would drag in a crate and
> non-determinism; ours is simpler and byte-stable by construction).

## 1. Hypothesis

A single-file container with a hash-verified manifest round-trips a store
**byte-identically** (export → import → export produces the same bytes), and
an imported memory is immediately fully functional (search, staleness,
insights) — so backup, migration, and sharing become one command each, with
corruption detected at import time rather than discovered months later.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Round-trip | export → import → export: archives byte-identical; store dirs file-identical | golden test |
| Functional import | imported store serves search / `remember` / staleness identically to the source (same responses on a scripted session) | integration test |
| Corruption rejection | any single flipped byte in the archive → import fails with the offending member named; nothing partially written | injection test |
| Determinism | same store → byte-identical archive (independent of filesystem iteration order, mtimes, platform) | golden hash |
| Overhead | archive ≤ 1.02× the sum of member sizes + manifest | size check |
| Big-store sanity | export+import of a ~1 GB synthetic store streams (RAM ≤ 256 MB peak) | bench report |

## 3. Public interface

Container format `SKPKG001` (little-endian):

```
magic "SKPKG001" | format_version u32
manifest_len u32 | manifest JSON:
  { "skinki_version": "...", "created_note": "...",   // data, not logic
    "members": [ { "path": "events/seg-0000.log",
                   "len": u64, "sha256": "hex" }, ... ] }  // sorted by path
members: raw bytes, concatenated in manifest order
trailer: sha256 of everything above
```

```rust
// skinki-store (or a thin new module in the harness — implementer's call,
// but the format code must live in a library crate, not main.rs)
pub fn export_store(store_dir: &Path, out: &Path) -> anyhow::Result<()>;
pub fn import_store(archive: &Path, dest_dir: &Path) -> anyhow::Result<()>;
```

```
skinki export --store <dir> --out memory.skinki
skinki import --archive memory.skinki --store <dir>   # refuses non-empty dest
```

Members included: L0 event/unit segments, dedup runs, the ledger snapshot,
the usage log (Stage 4B, when present), any built index files (`rabitq.idx`
etc.), and a `config.json`. The embedder model artifact is included **by
reference** (its sha256 in the manifest) with `--bundle-model` to inline it —
memories are small, models are 40 MB and shared.

Determinism rules, locked: members sorted by path; no timestamps anywhere in
the container; `created_note` is caller-supplied data (empty by default);
hashes via `skinki-hash` once Stage 2C lands (a local sha256 copy until then
is acceptable but must be replaced — leave a `TODO(2C)`).

## 4. Invariants (must always hold)

- Import is atomic-by-rename: unpack into `dest.tmp-<pid>`, verify **every**
  member hash + the trailer, then a single rename; any failure leaves no
  `dest` at all.
- Export never mutates the store; it snapshots after `sync()`.
- The manifest is the authority: unknown extra bytes → reject; missing
  member → reject; version from the future → reject with a clear message.
- Round-trip byte-identity (the §2 golden) is a permanent regression test.
- No new deps; streaming I/O (no whole-archive buffering).

## 5. Test plan

- **Unit:** manifest encode/decode; member ordering; future-version rejection.
- **Golden:** fixed synthetic store → locked archive hash (platform-stable).
- **Property:** round-trip identity on randomized (seeded) small stores.
- **Injection:** flip byte at seeded offsets (header / manifest / member /
  trailer) → import fails, names the member, dest absent.
- **Integration:** scripted MCP session (Stage 6B's golden) replayed against
  source and imported store → identical responses.
- **Gate command:** `cargo test -p skinki-store export_import` (all of the
  above in-crate) — wired into CI.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 container writer/reader + manifest (streaming) | impl | cheaper | unit + golden + property green |
| T2 atomic import + verification + injection tests | impl | cheaper | injection matrix green |
| T3 CLI `export`/`import` + docs section ("your memory is a file") | impl | cheaper | integration test green; docs match |
| T4 `--bundle-model` + by-reference model check on import (warn if the referenced model is absent locally) | impl | cheaper | both paths tested |
| T5 1 GB streaming bench + RAM ceiling | impl | cheaper | bench report row |

## 7. Definition of done

- [ ] All §5 tests green in CI; `cargo test`, clippy, fmt clean.
- [ ] README gains the export/import quickstart + the one-liner.
- [ ] Decision recorded: none pending (format frozen as SKPKG001; future
      changes go through the Stage 6C versioning policy).

## 8. Out of scope

- Encryption at rest (a real later feature — needs a key-management design,
  not a checkbox; note it in FORMATS.md as reserved flag bits).
- Partial/incremental export (full snapshots first; incremental is a later
  spec once someone actually has a 10 GB store).
- Cloud sync of any kind (violates the 0-network law; users sync the file
  with whatever they already trust).
