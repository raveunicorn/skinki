# Stage 6 — Portable engine: stable C-ABI/FFI + Swift/Python bindings (SPEC)

- **Status:** **done** (C-ABI + Python `ctypes` parity gated in CI; `skinki-mcp`
  MCP server ships the graph search + context-assembler surface to agents). The
  `unsafe` boundary (R1) is reviewed and signed off.
- **Owner of the design (frontier/human):** done below — the C-ABI shape, memory
  ownership, and error model are locked. The **only** frontier-review item is the
  `unsafe` boundary in `ffi.rs` (R1).
- **Delegatable to (cheaper model):** **yes** for all wiring (T1–T6). R1 (the
  `unsafe` review) must be signed off by a frontier model/human before merge.

> Read [`../../AGENTS.md`](../../AGENTS.md). Gate is law; determinism mandatory.
> **No new runtime deps:** Python uses `ctypes` (pure C-ABI), Swift uses a module
> map over a hand-written header. `cbindgen` is *optional* and dev-only.

## 1. Hypothesis

The engine packages into a Rust crate exposing a small, stable **C-ABI** so any
host (Swift app, Python eval, third party) can build the index and run searches
and get **byte-identical** results to the pure-Rust path — making `skinki` a
genuinely portable "FFmpeg for personal knowledge."

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Cross-language result parity | **exact** | C, Python, (macOS: Swift) search ids == Rust `two_stage_search` ids |
| ABI stability | header matches symbols | symbol-existence test + `skinki.h` checked in |
| No leaks on the happy path | 0 | open→search×N→free under a leak check (e.g. `cargo test` + valgrind in CI optional) |
| Panic safety | no unwHelpers across FFI | `panic = "abort"` for the cdylib, or `catch_unwind` at every boundary |

The gate here is the **cross-language equality test**, not a numeric budget.

## 3. Public interface (the C-ABI v0)

New crate `skinki-ffi` (`crate-type = ["cdylib", "staticlib"]`). Hand-written
header `skinki/crates/skinki-ffi/include/skinki.h`:

```c
#include <stdint.h>
#include <stddef.h>

typedef struct sk_engine sk_engine;   // opaque handle

// Status codes: 0 = OK, negative = error (see sk_last_error for detail).
int  sk_open(const char* index_dir, sk_engine** out_engine);
int  sk_search(sk_engine* engine,
               const float* query, size_t dim,
               size_t k,
               uint32_t* out_ids, size_t* out_len); // caller allocates out_ids[k]
void sk_free_engine(sk_engine* engine);
const char* sk_last_error(void);                     // thread-local, NUL-terminated
const char* sk_version(void);
```

### Design, locked

- **Opaque handle**: `sk_engine*` wraps the Rust engine; never expose Rust types.
- **Memory ownership**: results are written into a **caller-allocated** buffer
  (`out_ids` of length `k`); the engine writes `*out_len` actually filled. No
  engine-allocated buffers to free → simplest, leak-free contract.
- **Errors**: integer status + a **thread-local** last-error string
  (`sk_last_error`). No panics cross the boundary: set
  `panic = "abort"` in the cdylib profile, or wrap every `extern "C"` body in
  `std::panic::catch_unwind` and return an error code. (Implementer picks one;
  document it.)
- **v0 engine**: back the handle with the existing Stage 1 pipeline
  (`skinki-vector` two-stage search over a loaded index). `sk_open` loads a
  prebuilt index directory; index *building* can stay Rust/CLI-only for v0.
- **Bindings are thin**:
  - **Python** (`bindings/python/skinki.py`): pure `ctypes` over the cdylib —
    a `skinkiEngine` class with `open`/`search`/`close`. No PyO3, no build step.
  - **Swift** (`bindings/swift/`): a module map exposing `skinki.h` + a small
    `skinkiEngine` Swift wrapper. Compiles on macOS only.

## 4. Invariants

- Calling `sk_search` from any language returns ids identical to Rust
  `two_stage_search` on the same seeded index + query.
- The header in `include/skinki.h` exactly matches the exported symbols (a test
  asserts every declared symbol is present in the built library).
- No `unsafe` outside `ffi.rs`; that module is small and reviewed (R1).
- Determinism preserved end-to-end. No new runtime dependencies.

## 5. Test plan

- **Rust unit:** the safe inner functions behind the FFI (handle lifecycle,
  error-slot set/get) tested without crossing the ABI.
- **C harness** (`tests/ffi/`): a tiny `.c` that opens an index, searches, prints
  ids; a script builds the staticlib and runs it.
- **Python parity test:** `ctypes` calls `sk_search`; asserts ids == a JSON dump
  of Rust results for the same seed/query (produced by a harness subcommand).
- **Swift parity test** (macOS CI only): same assertion via the Swift wrapper.
- **Symbol test:** assert `skinki.h` declarations all resolve in the cdylib.
- **Gate command:** a `scripts/ffi-gate.sh` that builds the lib and runs the C +
  Python parity tests; CI invokes it. Non-zero on any mismatch.

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| R1: review the `unsafe` in `ffi.rs` (ptr validity, lifetimes, panic strategy) | review | **frontier/human** | sign-off required before merge |
| T1: `skinki-ffi` crate (cdylib+staticlib) + opaque handle + thread-local error | impl | cheaper | builds both lib types |
| T2: `extern "C"` wrappers for open/search/free/last_error/version | impl | cheaper | symbols exported |
| T3: hand-written `include/skinki.h` + symbol-existence test | impl | cheaper | symbol test passes |
| T4: C harness + build script | impl | cheaper | C test prints matching ids |
| T5: Python `ctypes` binding + parity test | impl | cheaper | Python ids == Rust ids |
| T6: Swift module map + wrapper + (macOS) parity test; `scripts/ffi-gate.sh` + CI | impl | cheaper | gate green |

## 7. Definition of done

- [x] `scripts/ffi-gate.sh` green: Rust C-ABI + Python `ctypes` parity == Rust
      `two_stage_search`. (Swift deferred to Stage 7; the C-harness is replaced
      by the in-process Rust integration test calling the `extern "C"` fns, which
      exercises the identical ABI without requiring a C compiler in CI.)
- [x] R1 `unsafe` review signed off (all `unsafe` in `ffi.rs`; null-checked,
      `catch_unwind` on every boundary, one `into_raw`/`from_raw` pair).
- [x] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [x] CI: `ffi` job builds the lib + runs the Rust + Python parity gate.
- [x] **Plus (beyond v0): `skinki-mcp`** — an MCP server over stdio exposing
      `search` + `assemble_context` (the Stage-3 graph + 3C assembler) to any MCP
      host; hand-rolled JSON-RPC, `forbid(unsafe)`, unit-tested dispatch.
- [x] ROADMAP Stage 6 row → done; this spec Status → done.

## 8. Out of scope

- Exposing graph/insight APIs over FFI — added incrementally as Stages 3/5 land
  (this v0 covers index load + search).
- Packaging/distribution (`.dmg`, wheels) — **Stage 7** / release work.
- Async or streaming FFI — v0 is synchronous.
