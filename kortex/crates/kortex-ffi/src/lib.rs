//! Stage 6 v0 — a portable C-ABI over the Stage 1 two-stage search engine.
//!
//! This crate builds as both a `cdylib` and `staticlib` (see `Cargo.toml`) so
//! any host (Swift app, Python via `ctypes`, a plain C program) can link
//! against it and get byte-identical results to the pure-Rust
//! `kortex_vector::search::two_stage_search` path on the same seeded index +
//! query. The public C surface is declared in `include/kortex.h` and
//! implemented in [`ffi`].
//!
//! ## Module layout
//! - [`engine`] — safe, pure-Rust inner implementation (handle lifecycle,
//!   index loading, search). Fully unit-tested without crossing the ABI.
//! - [`error`] — thread-local last-error slot used by `kx_last_error`.
//! - [`ffi`] — the `extern "C"` boundary. **The only module in this crate that
//!   may contain `unsafe`** (per `AGENTS.md` rule 4); see its module doc for
//!   the full `unsafe` inventory and panic-safety strategy.

// `forbid` cannot be locally overridden, so we `deny` at the crate level (the
// AGENTS.md default for "safe crates") and `allow` only in `ffi`, the sole
// module permitted to contain `unsafe` (R1 in specs/STAGE_6.md).
#![deny(unsafe_code)]

pub mod engine;
pub mod error;

#[allow(unsafe_code)]
pub mod ffi;
