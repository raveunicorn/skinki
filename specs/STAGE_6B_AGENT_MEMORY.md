# Stage 6B — Agent memory surface: write path, staleness, time-travel (SPEC)

> Batch 3 of the 2026-07 review (`REVIEW_FRONTIER_2026_07.md` §5). The MCP
> server is currently a read-only corpus browser and the ledger — the engine's
> most differentiated capability — is invisible to every consumer. This stage
> makes skinki *memory*: agents can **write**, every result knows whether it is
> **stale and why**, and the append-only substrate answers **"what did I
> believe on day X?"** — three capabilities no cloud memory API offers.

- **Status:** ready to build (depends on Stage 5C T1 for full-insight serving)
- **Owner of the design (frontier/human):** frontier — the tool contracts and
  the staleness-annotation semantics are locked below.
- **Delegatable to (cheaper model):** **yes, all tickets** — this is JSON-RPC
  plumbing over existing, gated machinery (`Store`, `Ledger`, `stale_closure`,
  `assemble_context`); frontier reviews T2 (staleness semantics) only.

> Read [`../AGENTS.md`](../AGENTS.md) and `crates/skinki-mcp` (hand-rolled
> JSON-RPC; `Server::handle` is pure and unit-tested — keep it that way; all
> new tools must be testable without stdio). 0 network; no new deps.

## 1. Hypothesis

Exposing the existing write (`Store::append_event` + `derive_units`), ledger
(`stale_closure`), and provenance machinery through three MCP tools turns the
server from a demo into an agent memory whose unique behaviors are
**machine-verifiable**: a memory written via `remember` is retrievable in the
same session; a premise contradicted later flags every dependent conclusion in
subsequent `search`/`assemble_context` responses; and `memory_asof` reproduces
the pre-contradiction worldview byte-deterministically. Falsifiable by the
integration goldens in §5.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Write→read loop | a `remember`'d text is retrievable by `search` within the same server session | integration test |
| Durability | `remember` survives server restart (store `sync()` on each write) | integration test: write, drop server, reopen, search |
| **Staleness annotation** | after a contradicting `remember`, every hit whose entry is in the stale closure carries `"stale": {...}`; hits outside it carry none | integration golden (see §5) |
| Over-flagging | 0 false stale flags on the planted-contradiction corpus | reuse `ledger-bench` ground truth through the server path |
| As-of determinism | `memory_asof(T)` twice → byte-identical JSON | golden |
| Full insight surface | `discover_insights` serves structural + temporal + contradiction (post 5C) | unit test lists all three kinds |
| Protocol hygiene | stdout carries only JSON-RPC; all new tools schema-described in `tools/list` | existing test style |
| Latency | `remember` p95 ≤ 50 ms (excl. fsync ≤ 1 write) | telemetry |

## 3. Public interface

Three new MCP tools (JSON schemas in `tools/list`, exact shapes below), plus
annotations on the two existing read tools.

```jsonc
// tools/call "remember"
{ "text": "string (required)",
  "source": "text|voice|import (default text)",
  "date": "YYYY-MM-DD (optional; default = today from the host, recorded as data — never used in logic)",
  // Optional: record a derived conclusion with provenance. When present the
  // server records a skinki_ledger::Derivation: output = hash of `text`,
  // inputs = content hashes of the cited entries, method = MethodStamp{M_AGENT, 1}.
  "premises": [ 123, 456 ] }
// -> { "event_id": u64, "unit_ids": [u64], "content_hash": "hex128" }

// tools/call "memory_asof"
{ "query": "string", "as_of": "YYYY-MM-DD", "k": 10 }
// -> like "search", but only entries with date <= as_of participate, and
//    staleness is evaluated against the ledger restricted to derivations
//    recorded at logical positions whose entries are <= as_of.

// search / assemble_context results gain, per hit/fact:
{ "id": 42, "date": "...", "text": "...",
  "stale": {                       // ABSENT when fresh
     "reason": "broken_premise" | "superseded",
     "via": [ 77 ]                 // entry ids that broke it (best effort)
  } }
```

Server-side contract:

```rust
// skinki-mcp — Server gains a Ledger + (in --store mode) a writable Store.
impl Server {
    /// Staleness index, rebuilt after every `remember`: the stale closure of
    /// the ledger given the CURRENT content hashes of all entries (an entry
    /// edited/superseded upstream has a hash mismatch vs its pinned premise).
    /// Deterministic; O(ledger) per rebuild is acceptable at this stage.
    fn refresh_stale_index(&mut self);
    /// Entry-level staleness view used to annotate results: an entry is
    /// annotated iff it is cited as evidence by a derivation whose output is
    /// in the stale closure, or it IS a superseded contradiction `before`.
    fn stale_info(&self, entry: EntryId) -> Option<StaleInfo>;
}
```

Contradiction wiring (the demo that sells the whole engine): on server start
and after each `remember`, run the (gated) `ContradictionDetector`; each
surfaced reversal `(X, before-entries, after-entries)` records a superseding
derivation, so `before` entries and everything derived from them annotate as
`superseded` with `via = after-entries`.

## 4. Invariants (must always hold)

- `Server::handle` stays pure (no I/O in dispatch besides the injected
  store/ledger handles); every tool unit-testable via JSON values.
- Determinism: same store + same call sequence → byte-identical responses
  (dates come from the caller, never from wall clock in logic — rule 2).
- Provenance: every `remember` returns content hashes; every stale flag is
  traceable (`via`).
- Append-only: `remember` never mutates or deletes; supersession is a new
  record + staleness, exactly the ledger's model.
- 0 network; no new deps; `#![forbid(unsafe_code)]` stays.

## 5. Test plan

- **Unit:** each tool's dispatch (missing params → INVALID_PARAMS; happy path
  shapes); `stale_info` on a hand-built 3-node ledger.
- **Integration golden (the keystone test, scripted JSON-RPC session):**
  1) `remember("Convinced Postgres is the best choice...", premises=[])`;
  2) `remember("decision: use Postgres for billing", premises=[<id of 1>])`;
  3) `search("billing database")` → both hits fresh;
  4) `remember("Changed my mind: Postgres was a mistake. Sqlite is better.")`;
  5) `search("billing database")` → hit 1 `superseded via [4]`, hit 2
     `broken_premise via [4]`, unrelated hits unflagged;
  6) `memory_asof("billing database", as_of = day of step 3)` → the step-3
     worldview, byte-identical to the recorded golden.
- **Durability:** steps 1–2, drop the `Server`, reopen from the same
  `--store` dir, step 3 still returns both.
- **Over-flag:** run the `ledger-bench` planted-contradiction corpus through
  the server; assert 0 flags outside the ground-truth dependent set.
- **Gate command:** `cargo test -p skinki-mcp` (all of the above are in-crate
  tests; no stdio needed) — wired into CI as-is.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `remember` tool: store append + `derive_units` + optional premise `Derivation` + `sync()` | impl | cheaper | write→read + durability tests green |
| **T2** staleness index + result annotation (`stale_info`, contradiction wiring above) | impl | cheaper, **frontier reviews semantics** | integration golden steps 3–5 green; over-flag test 0 |
| T3 `memory_asof` (date-filtered search + as-of ledger restriction) | impl | cheaper | step 6 golden green; deterministic |
| T4 serve the full insight engine (structural+temporal+contradiction — requires 5C T1) + re-run insights after `remember` | impl | cheaper | kinds test green |
| T5 tool schemas in `tools/list` + README/MCP docs update (the "memory for agents" section shows the write path) | impl | cheaper | docs match behavior; existing tests updated |

## 7. Definition of done

- [ ] All §5 tests green in CI; `cargo test`, clippy, fmt clean.
- [ ] README honest-status: "ingest → search → insights" row becomes
      "remember → search (staleness-aware) → insights → as-of", with the
      integration golden named as the proof.
- [ ] Decision recorded: staleness annotation latency at the target corpus
      size (is the O(ledger) rebuild fine, or does 2C need an incremental
      dirty-set sooner than planned).

## 8. Out of scope

- Exposing these over the C-ABI (FFI v1) — a later, mechanical addition once
  the MCP shapes have survived contact with real agents.
- Auth/multi-user/transport beyond stdio — single-user local by design.
- Semantic (embedding-similarity) contradiction detection — Stage 5B T3.
- The salience/usage ledger (Stage 4B) — `remember` does not record reads.
