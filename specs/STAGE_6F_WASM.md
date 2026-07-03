# Stage 6F — WASM target: skinki in the browser and on the edge (SPEC)

> Product backlog, batch 2026-07-B. The engine is pure Rust, zero heavy deps,
> zero network — which means `wasm32` is *almost free*, and the payoff is a
> whole new consumer class: in-browser private memory (nothing ever leaves
> the tab), edge workers, plugin sandboxes. Also the cheapest possible live
> demo: a static web page where a visitor searches a demo corpus — no
> install, no server, no data leaving the machine. Universality by addition,
> zero change to the essence.

- **Status:** ready to build (independent; a demo page benefits from Stage 1B's
  embedder but works with the hash embedder)
- **Owner of the design (frontier/human):** frontier — the target choice and
  the mmap-fallback seam are locked below.
- **Delegatable to (cheaper model):** **yes, all tickets.**

> Read [`../AGENTS.md`](../AGENTS.md). No new *runtime* deps. `wasmtime` (CI
> smoke runner) and `wasm32` toolchains are dev/CI tooling. The `unsafe`
> quarantine is untouched — WASM builds simply compile it out.

## 1. Hypothesis

The engine's core read path (corpus load → index → search → context assembly
→ insight discovery) compiles to `wasm32-wasip1` behind a `no-mmap` feature
(buffered I/O replacing the two quarantined mmap modules) and produces
**byte-identical results to native** on the golden corpora — proving the
portability claim ("FFmpeg for memory") in the strongest possible way: same
bytes out on a fundamentally different platform.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| **Cross-platform parity (the gate)** | search ids, assembled context, and surfaced insights byte-identical native vs wasmtime on the golden corpus | CI parity job |
| Crates in scope compile | `corpus, eval, baseline, vector (no-mmap), store (no-mmap read), ledger, graph, insight, sleep (no macOS signals), connect (parse only)` on `wasm32-wasip1` | CI build job |
| Binary size | demo `.wasm` ≤ 5 MB (gzipped ≤ 2 MB) | size check |
| Search latency in-browser | p95 ≤ 50 ms at 100k entries (hash or static embedder, brute force acceptable at this n) | demo page bench, recorded |
| No `unsafe` reachable | wasm builds carry `#![forbid(unsafe_code)]` everywhere (mmap modules cfg'd out) | compile-time |
| CI cost | wasm jobs ≤ 5 min | CI timing |

## 3. Public interface

```toml
# skinki-vector / skinki-store gain a default-on feature:
[features]
default = ["mmap"]
mmap = []            # native: the existing quarantined modules
# without "mmap": read-into-Vec / buffered-file fallbacks with the SAME API
```

```rust
// The fallback seam, locked: one internal trait per crate, two impls.
// Public APIs (RaBitQ::load, Store::open, ...) unchanged — callers never know.
trait ByteSource { fn len(&self) -> usize; fn slice(&self, off: usize, len: usize) -> &[u8]; }
// native: Mmap-backed (existing unsafe, quarantined, feature "mmap")
// wasm/no-mmap: Vec<u8>-backed (safe)
```

```rust
// New crate `skinki-wasm` (cdylib for wasm32-unknown-unknown; thin JS glue,
// hand-written — no wasm-bindgen, keeping the deps law; exports mirror the
// C-ABI shape):
//   sk_wasm_open(corpus_bytes_ptr, len) -> handle
//   sk_wasm_search(handle, query_ptr, len, k, out_ptr) -> n
//   sk_wasm_assemble(handle, query_ptr, len, budget, out_json_ptr) -> len
//   sk_wasm_insights(handle, out_json_ptr) -> len
// Memory contract: caller allocates via exported sk_wasm_alloc/free.
```

```
demo/web/            # static page: loads demo .wasm + a bundled demo corpus,
                     # search box, cited results, insights panel. No build
                     # system beyond a plain ES module; deployable on any
                     # static host (GitHub Pages).
```

CI: a `wasm` job — build the in-scope crates for `wasm32-wasip1`, run the
parity test binary under `wasmtime`, compare output hashes against the native
run in the same job.

## 4. Invariants (must always hold)

- Public crate APIs unchanged; the feature only swaps the byte-source impl.
- Determinism across platforms: the parity gate *is* the invariant — any
  float or iteration-order divergence between native and wasm fails CI
  (f32 ops used are IEEE-deterministic; no fast-math anywhere — already law).
- Native builds keep mmap by default; RAM budgets unaffected.
- The demo page makes zero network requests after initial asset load
  (Stage 6C's PRIVACY stance, demonstrable in devtools — say so on the page).
- No new runtime deps; JS glue is hand-written and minimal.

## 5. Test plan

- **Unit:** `ByteSource` fallback equivalence (same bytes via both impls).
- **Parity (the gate):** fixed corpus (seed 42, 2y) → native vs wasmtime:
  identical search-id lists over 50 queries, identical assembled-context
  JSON, identical insight set (hash-compared).
- **Size:** wasm artifact size assertions.
- **Manual-but-scripted:** `demo/web` served locally, a headless check that
  the page loads and one search returns (optional CI, best-effort).
- **Gate command:** the CI `wasm` job (build + wasmtime parity).

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `ByteSource` seam + `no-mmap` fallbacks in `skinki-vector`/`skinki-store` | impl | cheaper | native tests green with both features; equivalence unit test |
| T2 wasm32 compile fixes across in-scope crates (cfg out macOS signals in sleep, etc.) | impl | cheaper | CI build job green |
| T3 parity test + `wasm` CI job (wasmtime) | impl | cheaper | parity gate green |
| T4 `skinki-wasm` cdylib + hand-written JS glue | impl | cheaper | demo callable from JS |
| T5 `demo/web` page + GitHub Pages deploy + size budget | impl | cheaper | live demo; sizes within budget |

## 7. Definition of done

- [ ] `wasm` CI job (build + parity) green and required.
- [ ] Live demo page linked from the README ("try it in your browser — your
      query never leaves the tab").
- [ ] `cargo test`, clippy, fmt clean on native; both feature sets build.
- [ ] Decision recorded: wasm-side write path (store append in OPFS/
      IndexedDB) — wanted or not, based on demo feedback.

## 8. Out of scope

- The write path in wasm (browser persistence is a storage-backend design —
  separate spec if the demo creates demand).
- Running the static embedder's BPE in the demo *if* it blows the size budget
  (hash embedder fallback is acceptable for v1; note which shipped).
- wasm-bindgen / component model / WASI preview 2 — revisit when the
  ecosystem settles; the hand-rolled ABI is enough for the demo.
- Threads/SIMD in wasm (perf work, only after someone measures a need).
