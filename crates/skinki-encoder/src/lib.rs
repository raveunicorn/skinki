#![forbid(unsafe_code)]
//! Pure-Rust, dependency-free, `forbid(unsafe_code)` BERT-class encoder
//! (Stage 1C-B, variant B of `STAGE_1C`).
//!
//! This crate exists to answer the **T0 kill-switch question** before any
//! real model code is written: can a hand-written, `unsafe`-free, f32 GEMM
//! sustain enough throughput on M1-class hardware to make a self-contained
//! encoder viable? See `specs/STAGE_1C_B_PURE_RUST_ENCODER.md` §1–2.
//!
//! ### Stage of implementation
//!
//! **T0 only.** This revision ships the GEMM microbench (`gemm` + `bench`)
//! so the human can record the §2 numbers and make the D1 go/no-go call.
//! The forward pass (T2), the `SKENC001` artifact format (T1), the batch
//! driver (T3) and the engine wiring (T4) are gated behind that decision
//! and are therefore intentionally absent here.
//!
//! ### Determinism contract (rules 2 / 3)
//!
//! `gemm` sums over `K` strictly left-to-right inside one row of `C`, with
//! no pairwise / tree reordering. Threading partitions **rows of `C`**
//! (the M dimension), never the per-element arithmetic, so 1-thread and
//! N-thread runs are byte-identical by construction. The T2 forward pass
//! will inherit this exact property for free.

pub mod bench;
pub mod format;
pub mod gemm;

pub use gemm::gemm;
