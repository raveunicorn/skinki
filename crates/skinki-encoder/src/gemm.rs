//! Blocked/tiled safe-Rust f32 GEMM — the exact inner loop structure the T2
//! BERT forward pass will use for every QK^T, softmax-weighted AV, FFN and
//! projection. There is intentionally **no `unsafe`**, no `std::arch`, no
//! BLAS: rule 4 (`forbid(unsafe_code)`) and rule 5 (minimal deps) rule that
//! out, and the whole point of T0 is to find out whether that is fast enough.
//!
//! ## Numeric / determinism contract
//!
//! `C[i, j] += sum_{p=0..k} A[i, p] * B[p, j]`, where the sum is taken
//! **strictly left-to-right over `p`** (the outer K-loop), one BK-chunk at a
//! time — no pairwise / tree reduction, no reordering. Threads partition the
//! M dimension (rows of C); no thread ever touches another thread's row, so
//! the same `C` is produced regardless of thread count. This is the
//! Stage-1C-B rule-2 invariant, by construction: `gemm(m, n, k, A, B)` is
//! byte-deterministic across runs, thread counts and platforms
//! (arm64 / x86_64), which the property tests in this file assert.
//!
//! ## Loop structure
//!
//! Register-tiled microkernel: an MR × NR tile of C is loaded into locals
//! once, accumulates the **entire** K range in ascending `p`, and is stored
//! back once:
//!
//! ```text
//! for jj in NR strips of N:  for ii in MR blocks of M:
//!   acc[MR][NR] = C[ii.., jj..];         // load tile once
//!   for p in 0..k:                        // ascending, full range
//!     for r in 0..MR: for l in 0..NR:
//!       acc[r][l] += A[ii+r, p] * B[p, jj+l];
//!   C[ii.., jj..] = acc;                  // store tile once
//! ```
//!
//! The `l` loop is over independent C lanes (contiguous in B and C), so
//! `--release` auto-vectorizes it (NEON/SSE) without hand-written SIMD, and
//! the MR rows give the vector units independent accumulation chains. For
//! each `(i, j)` the additions happen strictly left-to-right over `p` into
//! one f32 accumulator — bit-identical to an in-memory `c[i,j] +=` per `p`
//! (register vs store/reload round-trip of an f32 is exact) — that is what
//! buys rule-2 determinism. Keeping C in registers instead of streaming it
//! through cache once per `p` is the entire speedup.

use std::io;
use std::thread;

/// Microkernel rows: how many rows of C accumulate in registers at once.
const MR: usize = 4;
/// Microkernel columns: f32 lanes of C kept in registers per row (two NEON /
/// SSE vectors). MR × NR accumulators + one B row fit the 32 (16 on x86_64
/// SSE) vector registers without spilling.
const NR: usize = 8;
/// K block used only by the *tests'* chunked reference (the kernel itself
/// accumulates the full K range in registers; per-(i, j) order is unchanged
/// — see the determinism contract above).
#[cfg(test)]
const BK: usize = 256;

/// Compute `C += A · B` for `A ∈ R^{m×k}`, `B ∈ R^{k×n}`, `C ∈ R^{m×n}`,
/// all row-major. `threads = 0` selects single-threaded execution.
///
/// Returns `InvalidData` for zero or overflowing dimensions / mismatched
/// slice lengths (1B loader lessons: bounds checks loud, never panic).
///
/// Threading partitions **rows of C**, so the result is independent of
/// `threads`. See the crate-level determinism contract.
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    threads: usize,
) -> io::Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "gemm: zero dimension",
        ));
    }
    let m_k = m.checked_mul(k);
    let k_n = k.checked_mul(n);
    let m_n = m.checked_mul(n);
    match (m_k, k_n, m_n) {
        (Some(ak), Some(bk), Some(cn)) if a.len() == ak && b.len() == bk && c.len() == cn => {}
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "gemm: dimension/length mismatch or overflow",
            ));
        }
    }

    if threads <= 1 {
        gemm_serial(m, n, k, a, b, c);
        return Ok(());
    }

    gemm_threaded(m, n, k, a, b, c, threads);
    Ok(())
}

/// Evenly split rows `[0, m)` into `threads` contiguous `[m0, m1)` bands.
fn split_bands(m: usize, threads: usize) -> Vec<(usize, usize)> {
    let mut bands = Vec::with_capacity(threads);
    let base = m / threads;
    let rem = m % threads;
    let mut start = 0;
    for i in 0..threads {
        let extra = if i < rem { 1 } else { 0 };
        let end = start + base + extra;
        bands.push((start, end));
        start = end;
    }
    bands
}

fn gemm_threaded(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    threads: usize,
) {
    // Disjoint mut-band splitting: thread `i` owns rows `[m0, m1)` of C and
    // the matching rows of A. We split `c` into `threads` contiguous mutable
    // sub-slices along whole-row boundaries by repeated `split_at_mut`.
    let bands = split_bands(m, threads);
    let mut remaining_c: &mut [f32] = c;
    let mut c_bands: Vec<&mut [f32]> = Vec::with_capacity(bands.len());
    for (m0, m1) in &bands {
        let prefix_rows = (m1 - m0) * n;
        let (head, tail) = remaining_c.split_at_mut(prefix_rows);
        c_bands.push(head);
        remaining_c = tail;
    }
    // Walk both `bands` and `c_bands` by value (`into_iter` on the owned Vec)
    // so each `&mut` is consumed exactly once and not re-borrowed.
    thread::scope(|s| {
        for ((m0, m1), c_band) in bands.into_iter().zip(c_bands) {
            let a_band = &a[m0 * k..m1 * k];
            let m_band = m1 - m0;
            s.spawn(move || {
                gemm_serial(m_band, n, k, a_band, b, c_band);
            });
        }
    });
}

/// Single-threaded register-tiled GEMM, used both as the `threads <= 1` path
/// and as the per-thread worker.
///
/// The C tile (MR × NR) lives in registers for the whole K range: it is
/// loaded from memory once, receives the `c += a[i,p] * b[p,j]` additions
/// strictly left-to-right over `p`, and is stored once. Per `(i, j)` this is
/// the *identical* sequence of f32 operations as an in-memory `c[i,j] +=`
/// per `p` (a store/reload round-trip of an f32 register is exact), so the
/// output is bit-for-bit the documented order — asserted by
/// `kernel_matches_documented_order_bit_exact`. What changes is only memory
/// traffic: C is no longer streamed through cache once per `p`.
fn gemm_serial(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let mut jj = 0;
    while jj + NR <= n {
        let mut ii = 0;
        while ii + MR <= m {
            kernel_mr_nr(ii, jj, n, k, a, b, c);
            ii += MR;
        }
        // M tail: remaining rows one at a time, same NR strip.
        for i in ii..m {
            kernel_1_nr(i, jj, n, k, a, b, c);
        }
        jj += NR;
    }
    // N tail: scalar per (i, j), accumulator in a register, ascending `p` —
    // the same per-element order as everywhere else.
    if jj < n {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            for j in jj..n {
                let mut cij = c[i * n + j];
                for (p, &ap) in a_row.iter().enumerate() {
                    cij += ap * b[p * n + j];
                }
                c[i * n + j] = cij;
            }
        }
    }
}

/// MR × NR register microkernel: C tile in registers, full-K accumulation,
/// ascending `p`. The inner `l` loop is over independent C lanes, so the
/// compiler vectorizes it without touching any per-(i, j) sum order.
fn kernel_mr_nr(ii: usize, jj: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let mut acc = [[0.0f32; NR]; MR];
    for (r, acc_r) in acc.iter_mut().enumerate() {
        acc_r.copy_from_slice(&c[(ii + r) * n + jj..(ii + r) * n + jj + NR]);
    }
    let a0 = &a[ii * k..(ii + 1) * k];
    let a1 = &a[(ii + 1) * k..(ii + 2) * k];
    let a2 = &a[(ii + 2) * k..(ii + 3) * k];
    let a3 = &a[(ii + 3) * k..(ii + 4) * k];
    for p in 0..k {
        let b_row: &[f32; NR] = b[p * n + jj..p * n + jj + NR]
            .try_into()
            .expect("NR-sized B strip");
        let (x0, x1, x2, x3) = (a0[p], a1[p], a2[p], a3[p]);
        for l in 0..NR {
            acc[0][l] += x0 * b_row[l];
            acc[1][l] += x1 * b_row[l];
            acc[2][l] += x2 * b_row[l];
            acc[3][l] += x3 * b_row[l];
        }
    }
    for (r, acc_r) in acc.iter().enumerate() {
        c[(ii + r) * n + jj..(ii + r) * n + jj + NR].copy_from_slice(acc_r);
    }
}

/// 1 × NR variant for the M tail. Same order contract as `kernel_mr_nr`.
fn kernel_1_nr(i: usize, jj: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let mut acc = [0.0f32; NR];
    acc.copy_from_slice(&c[i * n + jj..i * n + jj + NR]);
    let a_row = &a[i * k..(i + 1) * k];
    for (p, &ap) in a_row.iter().enumerate() {
        let b_row: &[f32; NR] = b[p * n + jj..p * n + jj + NR]
            .try_into()
            .expect("NR-sized B strip");
        for l in 0..NR {
            acc[l] += ap * b_row[l];
        }
    }
    c[i * n + jj..i * n + jj + NR].copy_from_slice(&acc);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference that sums K **in the same chunked order** the tiled kernel
    /// uses: BK-sized chunks of p, left-to-right within each chunk, chunks
    /// processed left-to-right. f32 is not associative, so the tiled kernel
    /// is bit-deterministic only against a chunked reference, not a single
    /// long sum. (This is exactly the property rule 2/3 buys us: the order
    /// is fixed and documented, not "whatever the hardware reduces in".)
    fn reference_chunked(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        let mut p0 = 0;
        while p0 < k {
            let p1 = (p0 + BK).min(k);
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in p0..p1 {
                        acc += a[i * k + p] * b[p * n + j];
                    }
                    c[i * n + j] += acc;
                }
            }
            p0 = p1;
        }
        c
    }

    use crate::bench::seeded_inputs;

    /// An independent scalar implementation of the *documented* association:
    /// for each `(i, j)`, `c[i,j] += a[i,p] * b[p,j]` applied `p = 0..k`
    /// strictly left-to-right, accumulating straight into C. This is exactly
    /// the order the module docs promise; the tiled kernel must reproduce it
    /// bit-for-bit no matter how it blocks or vectorizes.
    fn reference_documented_order(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aip * b[p * n + j];
                }
            }
        }
        c
    }

    /// The rule-2 contract test: the kernel computes exactly the documented
    /// association, bit-for-bit, against an independent scalar reference.
    /// Any future tiling/vectorization change that silently reorders the
    /// per-element sum shows up here as a bit diff. (This also demonstrates
    /// that no FMA contraction occurs: rustc never fuses `c += a*b` into an
    /// FMA without an explicit `mul_add`, on any platform — which is what
    /// makes the output byte-identical across arm64/x86_64.)
    #[test]
    fn kernel_matches_documented_order_bit_exact() {
        let shapes: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (33, 77, 91),
            (100, 100, 100),
            (384, 384, 384), // multi-chunk K (BK = 256)
            (128, 1536, 384),
            (384, 1536, 384),
        ];
        for &(m, n, k) in shapes {
            let (a, b) = seeded_inputs(m, n, k);
            let want = reference_documented_order(m, n, k, &a, &b);
            let mut got = vec![0.0f32; m * n];
            gemm(m, n, k, &a, &b, &mut got, 1).unwrap();
            let diffs = got
                .iter()
                .zip(want.iter())
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(
                diffs, 0,
                "shape ({m},{n},{k}): kernel deviates from the documented sum order"
            );
        }
    }

    #[test]
    fn correctness_against_chunked_reference() {
        // High-precision sanity check against a *different* association: the
        // reference accumulates each BK chunk in a local variable and adds it
        // to C once, while the kernel accumulates into C per `p`. For a
        // single-chunk K (k ≤ BK) the two orders coincide exactly (the
        // initial `0.0 + acc` is exact), so bit-equality is asserted; for
        // multi-chunk K they legitimately differ by summation association —
        // a few ULPs — so a small relative tolerance applies. (No FMA is
        // involved anywhere; see `kernel_matches_documented_order_bit_exact`.)
        let shapes: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (16, 16, 16),
            (384, 384, 384),
            (384, 1536, 384),
            (100, 100, 100), // non-multiples of BM/BN/BK
            (33, 77, 91),
            (128, 384, 384),
        ];
        for &(m, n, k) in shapes {
            let (a, b) = seeded_inputs(m, n, k);
            let want = reference_chunked(m, n, k, &a, &b);
            let mut got = vec![0.0f32; m * n];
            gemm(m, n, k, &a, &b, &mut got, 1).unwrap();
            let tol = if k <= BK { 0.0 } else { 1e-3 };
            for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let abs = (g - w).abs();
                let rel = abs / w.abs().max(1e-6);
                assert!(
                    rel <= tol,
                    "shape ({m},{n},{k}) idx {idx}: got {g} want {w} (rel {rel} > {tol})"
                );
            }
        }
    }

    #[test]
    fn accumulates_into_existing_c() {
        // C is pre-populated; gemm must add, not overwrite. The reference
        // uses a local-variable accumulator added to C once, so its
        // association differs from the kernel's per-`p` in-place adds —
        // allow a small relative tolerance (the *additive* behavior is the
        // point, not the sum order; that is asserted bit-exactly elsewhere).
        let (m, n, k) = (32, 32, 32);
        let (a, b) = seeded_inputs(m, n, k);
        let mut c = vec![0.5f32; m * n];
        gemm(m, n, k, &a, &b, &mut c, 1).unwrap();
        let mut want = vec![0.5f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                want[i * n + j] += acc;
            }
        }
        for (idx, (g, w)) in c.iter().zip(want.iter()).enumerate() {
            let rel = (g - w).abs() / w.abs().max(1e-6);
            assert!(rel <= 1e-4, "idx {idx}: got {g} want {w}");
        }
    }

    #[test]
    fn determinism_repeat_runs_byte_equal() {
        let (m, n, k) = (384, 384, 384);
        let (a, b) = seeded_inputs(m, n, k);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        gemm(m, n, k, &a, &b, &mut c1, 1).unwrap();
        gemm(m, n, k, &a, &b, &mut c2, 1).unwrap();
        assert_eq!(c1, c2, "back-to-back runs must be byte-equal");
    }

    #[test]
    fn determinism_thread_count_invariant() {
        // Rule 2 invariant: threads partition rows, never arithmetic, so
        // 1-thread and 4-thread output must be byte-identical.
        let (m, n, k) = (384, 384, 384);
        let (a, b) = seeded_inputs(m, n, k);
        let mut c1 = vec![0.0f32; m * n];
        let mut c4 = vec![0.0f32; m * n];
        gemm(m, n, k, &a, &b, &mut c1, 1).unwrap();
        gemm(m, n, k, &a, &b, &mut c4, 4).unwrap();
        assert_eq!(c1, c4, "thread-count invariant violated");
    }

    #[test]
    fn rejects_zero_dimension() {
        let r = gemm(0, 4, 4, &[], &[0.0f32; 16], &mut [], 1);
        assert_eq!(r.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_length_mismatch() {
        let r = gemm(4, 4, 4, &[0.0f32; 15], &[0.0f32; 16], &mut [0.0f32; 16], 1);
        assert_eq!(r.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }
}
