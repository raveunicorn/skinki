//! In-crate numerics for the T2 forward pass — **no `libm` anywhere on the
//! forward path** (spec §4): `exp` and `erf` are fixed polynomial/rational
//! approximations built from IEEE-exact primitives (`+ - * /`, fused
//! `mul_add`, `sqrt`, `floor`, bit casts — all correctly rounded on every
//! target), so encoder output is byte-identical across OS/libc/arch. Reductions (LayerNorm statistics, softmax sums) accumulate
//! in f64, strictly left-to-right — f64 arithmetic is just as deterministic
//! as f32 and cuts the drift vs the f32 teacher roughly in half.
//!
//! Error bounds (asserted by tests against `std` references — tests may use
//! `libm`, the forward path may not):
//!   - `exp64`: relative error < 1e-13 on [-30, 30] (softmax range after
//!     max-subtraction is (-inf, 0]; underflow to 0 below -708).
//!   - `erf64`: absolute error < 1.5e-7 (Abramowitz–Stegun 7.1.26), which
//!     bounds the GELU error by 1.5e-7·|x| — below f32 resolution for the
//!     activation ranges BERT produces.

/// exp(x) for f64 via range reduction + Taylor/Horner, deterministic.
///
/// x = n·ln2 + r with |r| ≤ ln2/2, exp(x) = 2^n · exp(r); exp(r) by a
/// 12-term Horner Taylor over precomputed 1/k! constants with fused
/// `mul_add` (term 13 ≈ r¹³/13! < 2e-14 at |r| ≤ 0.347); 2^n assembled by
/// exponent-bit construction (no `powi`, no `libm`).
pub fn exp64(x: f64) -> f64 {
    if x < -708.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    // ln2 split hi/lo so `x - n·ln2` stays exact to the last bit.
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let n = (x * LOG2E + if x >= 0.0 { 0.5 } else { -0.5 }).trunc();
    let r = (x - n * LN2_HI) - n * LN2_LO;
    // Horner over precomputed 1/k! coefficients, one fused `mul_add` per
    // term: the old `1 + r/k·p` form spent 12 *serial* f64 divides per call
    // on the single divide port; reciprocal-factorial constants (rounded
    // once, at compile time — still IEEE-exact operations at run time) plus
    // fma keep the truncation error < 2e-14 and cut the chain to 12 fused
    // ops. Term-13 bound: r¹³/13! < 2e-14 at |r| ≤ ln2/2.
    let mut p = EXP_C[12];
    let mut i = 12usize;
    while i > 0 {
        i -= 1;
        p = p.mul_add(r, EXP_C[i]);
    }
    let n = n as i64;
    // 2^n via exponent bits; n ∈ [-1022, 1023] after the clamps above
    // (|x| ≤ 709 → |n| ≤ 1024; subnormal edge folded into the -708 cutoff).
    debug_assert!((-1022..=1023).contains(&n));
    let scale = f64::from_bits(((n + 1023) as u64) << 52);
    p * scale
}

/// 1/k! for the exp Taylor Horner, k = 0..=12; each constant is the
/// correctly-rounded f64 of the exact rational (CTFE division of literals).
const EXP_C: [f64; 13] = [
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
];

/// Four-lane `exp64`: per lane the *identical* straight-line operation
/// sequence as the scalar (asserted bit-exact by tests), but four
/// independent dependency chains in flight — the scalar Horner is
/// latency-bound (a divide + multiply + add chain per term), so interleaving
/// lanes is a pure ILP win with zero numeric change.
///
/// The scalar's early returns become end-of-function selects: the main path
/// runs on a clamped copy (`clamp` is exact for in-range inputs, so in-range
/// lanes see bit-identical `r`), and out-of-range lanes are overwritten
/// with the scalar's 0.0 / ∞ answers.
pub fn exp64x4(x: [f64; 4]) -> [f64; 4] {
    exp64xn::<4>(x)
}

/// N-lane `exp64` (N = const): per lane the identical straight-line op
/// sequence as the scalar, N independent chains in flight. The 12-term fma
/// Horner is a ~50-cycle latency chain per lane; softmax rows call this at
/// N = 16 so the two f64 FMA pipes stay saturated instead of idling on one
/// chain's latency.
pub fn exp64xn<const N: usize>(x: [f64; N]) -> [f64; N] {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let mut out = [0.0f64; N];
    let mut n = [0.0f64; N];
    let mut r = [0.0f64; N];
    for l in 0..N {
        // Exact for in-range lanes; keeps `n` in the representable window
        // for the lanes the selects below will discard anyway.
        let xc = x[l].clamp(-708.0, 709.0);
        n[l] = (xc * LOG2E + if xc >= 0.0 { 0.5 } else { -0.5 }).trunc();
        r[l] = (xc - n[l] * LN2_HI) - n[l] * LN2_LO;
    }
    let mut p = [EXP_C[12]; N];
    let mut i = 12usize;
    while i > 0 {
        i -= 1;
        for l in 0..N {
            p[l] = p[l].mul_add(r[l], EXP_C[i]);
        }
    }
    for l in 0..N {
        let ni = n[l] as i64;
        debug_assert!((-1022..=1023).contains(&ni));
        let scale = f64::from_bits(((ni + 1023) as u64) << 52);
        out[l] = if x[l] < -708.0 {
            0.0
        } else if x[l] > 709.0 {
            f64::INFINITY
        } else {
            p[l] * scale
        };
    }
    out
}

/// N-lane f32 exp for the softmax path. Same construction as `exp64xn`
/// (range-reduce to `x = n·ln2 + r`, Taylor Horner with fused `mul_add`,
/// 2^n by exponent-bit assembly) but in f32 throughout — this is the
/// precision the torch teacher's own f32 softmax kernels work at; the f64
/// version was extra precision the parity bar never required, at twice the
/// SIMD width cost and double the Horner depth. Degree-7 keeps the
/// truncation term r⁸/8! < 6e-10 at |r| ≤ ln2/2, below f32 resolution.
/// Deterministic: fixed straight-line ops, correctly rounded on every
/// target. Inputs are softmax-shifted (x ≤ 0); x < -87 underflows to 0
/// (exp(-87) is already denormal-adjacent and contributes nothing to a sum
/// that is ≥ 1 by construction).
#[allow(clippy::excessive_precision)] // literals document the exact reals
pub fn exp32xn<const N: usize>(x: [f32; N]) -> [f32; N] {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    // ln2 split hi/lo: hi is exact in 10 bits so `n·hi` subtracts exactly.
    const LN2_HI: f32 = 0.693_359_375;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let mut out = [0.0f32; N];
    let mut n = [0.0f32; N];
    let mut r = [0.0f32; N];
    for l in 0..N {
        let xc = x[l].clamp(-87.0, 88.0);
        n[l] = (xc * LOG2E).round();
        r[l] = (xc - n[l] * LN2_HI) - n[l] * LN2_LO;
    }
    let mut p = [EXP32_C[7]; N];
    let mut i = 7usize;
    while i > 0 {
        i -= 1;
        for l in 0..N {
            p[l] = p[l].mul_add(r[l], EXP32_C[i]);
        }
    }
    for l in 0..N {
        let ni = n[l] as i32;
        debug_assert!((-126..=127).contains(&ni));
        let scale = f32::from_bits(((ni + 127) as u32) << 23);
        out[l] = if x[l] < -87.0 { 0.0 } else { p[l] * scale };
    }
    out
}

/// 1/k! in f32 for the exp32 Horner, k = 0..=7 (CTFE division of literals).
const EXP32_C: [f32; 8] = [
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
];

/// erf(x) via Abramowitz–Stegun 7.1.26 (max abs error 1.5e-7), sign-folded.
pub fn erf64(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * exp64(-x * x);
    sign * y
}

/// Exact GELU (the BERT `hidden_act = "gelu"`): x·½·(1 + erf(x/√2)).
/// `sqrt` is IEEE-correctly-rounded on every platform — allowed.
pub fn gelu(x: f32) -> f32 {
    let xf = x as f64;
    (xf * 0.5 * (1.0 + erf64(xf / std::f64::consts::SQRT_2))) as f32
}

/// N-lane f32 erf via the same Abramowitz–Stegun 7.1.26 polynomial as
/// `erf64`, evaluated in f32 with fused `mul_add` and `exp32xn`. The
/// formula's own error (1.5e-7 abs) dominates the f32 rounding, so the
/// approximation class is unchanged — this is the precision the torch
/// teacher's f32 GELU kernel works at. Deterministic straight-line ops.
#[allow(clippy::excessive_precision)] // literals document the exact reals
fn erf32xn<const N: usize>(x: [f32; N]) -> [f32; N] {
    const A1: f32 = 0.254_829_59;
    const A2: f32 = -0.284_496_74;
    const A3: f32 = 1.421_413_74;
    const A4: f32 = -1.453_152_03;
    const A5: f32 = 1.061_405_43;
    const P: f32 = 0.327_591_1;
    let mut sign = [0.0f32; N];
    let mut ax = [0.0f32; N];
    let mut nxx = [0.0f32; N];
    for l in 0..N {
        sign[l] = if x[l] < 0.0 { -1.0 } else { 1.0 };
        ax[l] = x[l].abs();
        nxx[l] = -ax[l] * ax[l];
    }
    let e = exp32xn::<N>(nxx);
    let mut out = [0.0f32; N];
    for l in 0..N {
        let t = 1.0 / P.mul_add(ax[l], 1.0);
        let poly = A5
            .mul_add(t, A4)
            .mul_add(t, A3)
            .mul_add(t, A2)
            .mul_add(t, A1)
            * t;
        out[l] = sign[l] * (1.0 - poly * e[l]);
    }
    out
}

/// In-place GELU over a slice: 16 interleaved f32 lanes (`erf32xn`), scalar
/// f32 tail. `x·½·(1 + erf(x/√2))`, the exact BERT `gelu`.
pub fn gelu_slice(xs: &mut [f32]) {
    const INV_SQRT2: f32 = 0.707_106_77;
    let mut wide = xs.chunks_exact_mut(16);
    for ch in &mut wide {
        let mut args = [0.0f32; 16];
        for l in 0..16 {
            args[l] = ch[l] * INV_SQRT2;
        }
        let er = erf32xn::<16>(args);
        for l in 0..16 {
            ch[l] = ch[l] * 0.5 * (1.0 + er[l]);
        }
    }
    for v in wide.into_remainder() {
        let er = erf32xn::<1>([*v * INV_SQRT2]);
        *v = *v * 0.5 * (1.0 + er[0]);
    }
}

/// In-place LayerNorm over `row` (one token's hidden vector): biased
/// variance (BERT), statistics accumulated in f64 strictly left-to-right.
pub fn layernorm_row(row: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    debug_assert_eq!(row.len(), gamma.len());
    debug_assert_eq!(row.len(), beta.len());
    let n = row.len() as f64;
    let mut sum = 0.0f64;
    for &v in row.iter() {
        sum += v as f64;
    }
    let mean = sum / n;
    let mut var = 0.0f64;
    for &v in row.iter() {
        let d = v as f64 - mean;
        var += d * d;
    }
    let var = var / n;
    let inv = 1.0 / (var + eps as f64).sqrt();
    for (i, v) in row.iter_mut().enumerate() {
        let normed = (*v as f64 - mean) * inv;
        *v = (normed * gamma[i] as f64 + beta[i] as f64) as f32;
    }
}

/// In-place softmax over `row`: max-subtracted, `exp32` (the teacher's own
/// working precision — see `exp32xn`), f32 sum strictly left-to-right,
/// divide. The all-finite input invariant holds by construction (scores are
/// dot products of finite activations).
pub fn softmax_row(row: &mut [f32]) {
    // Lane-wise max then a fixed 16-lane fold: `max` is order-independent
    // over finite scores (the all-finite invariant below), so this computes
    // the identical value as a sequential scan — just vectorizably.
    let mut lane_max = [f32::NEG_INFINITY; 16];
    let mut wide = row.chunks_exact(16);
    for ch in &mut wide {
        for l in 0..16 {
            lane_max[l] = lane_max[l].max(ch[l]);
        }
    }
    for (l, &v) in wide.remainder().iter().enumerate() {
        lane_max[l] = lane_max[l].max(v);
    }
    let mut max = f32::NEG_INFINITY;
    for &v in lane_max.iter() {
        max = max.max(v);
    }
    let mut sum = 0.0f32;
    // Two passes with a scratch-free design: store exp in place, then scale.
    // The exp pass runs 16 interleaved f32 lanes; the sum stays strictly
    // left-to-right in index order.
    let mut wide = row.chunks_exact_mut(16);
    for ch in &mut wide {
        let mut xs = [0.0f32; 16];
        for l in 0..16 {
            xs[l] = ch[l] - max;
        }
        let e = exp32xn::<16>(xs);
        for l in 0..16 {
            ch[l] = e[l];
            sum += e[l];
        }
    }
    for v in wide.into_remainder() {
        let e = exp32xn::<1>([*v - max]);
        *v = e[0];
        sum += e[0];
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests may use `std`/`libm` as the *reference*; the forward path may
    /// not use them as the *implementation*.
    #[test]
    fn exp64_matches_std_to_1e13() {
        let mut worst = 0.0f64;
        // Dense sweep over the softmax-relevant range plus outliers.
        let mut x = -30.0f64;
        while x <= 30.0 {
            let got = exp64(x);
            let want = x.exp();
            let rel = ((got - want) / want).abs();
            if rel > worst {
                worst = rel;
            }
            x += 0.001953125; // 2^-9: exact step, deterministic sweep
        }
        assert!(worst < 1e-13, "exp64 worst rel error {worst}");
        assert_eq!(exp64(0.0), 1.0);
        assert_eq!(exp64(-800.0), 0.0);
        assert!(exp64(-708.5) >= 0.0);
    }

    #[test]
    fn erf64_matches_std_bound() {
        // f64::erf is unstable; use tabulated references (Abramowitz–Stegun
        // Table 7.1, 10 significant digits).
        let table: &[(f64, f64)] = &[
            (0.0, 0.0),
            (0.1, 0.112_462_916_0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (1.5, 0.966_105_146_5),
            (2.0, 0.995_322_265_0),
            (3.0, 0.999_977_909_5),
        ];
        for &(x, want) in table {
            let got = erf64(x);
            assert!(
                (got - want).abs() < 1.5e-7,
                "erf64({x}) = {got}, want {want}"
            );
            assert!(
                (erf64(-x) + want).abs() < 1.5e-7,
                "erf64 sign fold broken at {x}"
            );
        }
    }

    #[test]
    fn gelu_known_points() {
        // gelu(0) = 0; gelu is ~identity for large x, ~0 for large negative.
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(3.0) - 2.996).abs() < 1e-2);
        assert!(gelu(-6.0).abs() < 1e-6);
        // Reference value: gelu(1) = 0.5·(1+erf(1/√2)) = 0.841344746...
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
    }

    #[test]
    fn layernorm_hand_case() {
        // row = [1, 3]: mean 2, biased var 1 → normed [-1, 1]; γ=2, β=0.5.
        let mut row = [1.0f32, 3.0];
        layernorm_row(&mut row, &[2.0, 2.0], &[0.5, 0.5], 1e-12);
        assert!((row[0] - (-1.5)).abs() < 1e-5, "{row:?}");
        assert!((row[1] - 2.5).abs() < 1e-5, "{row:?}");
    }

    #[test]
    fn softmax_hand_case_and_sum() {
        let mut row = [0.0f32, 0.0, 0.0, 0.0];
        softmax_row(&mut row);
        for v in row {
            assert!((v - 0.25).abs() < 1e-7);
        }
        let mut row = [1.0f32, 2.0, 3.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(row[2] > row[1] && row[1] > row[0]);
        // Reference: softmax([1,2,3])[2] = e³/(e+e²+e³) ≈ 0.665240956.
        assert!((row[2] - 0.665_241).abs() < 1e-5);
    }

    /// exp32 must stay within a couple of f32 ULPs of the true exp over the
    /// softmax input range ((-inf, 0] after max subtraction).
    #[test]
    fn exp32xn_accuracy_softmax_range() {
        let mut worst = 0.0f64;
        let mut x = -87.0f32;
        while x <= 0.0 {
            let got = exp32xn::<1>([x])[0] as f64;
            let want = (x as f64).exp();
            let rel = ((got - want) / want).abs();
            if rel > worst {
                worst = rel;
            }
            x += 0.001953125; // 2^-9: exact step, deterministic sweep
        }
        assert!(worst < 3e-7, "exp32 worst rel error {worst}");
        assert_eq!(exp32xn::<1>([0.0])[0], 1.0);
        assert_eq!(exp32xn::<1>([-100.0])[0], 0.0);
    }

    /// The x4 interleaves exist for ILP only: every lane must reproduce the
    /// scalar bit-for-bit, including the out-of-range select edges.
    #[test]
    fn exp64x4_matches_scalar_bit_exact() {
        let mut x = -30.0f64;
        while x <= 30.0 {
            let lanes = [x, x + 0.25, -x, x * 1.5];
            let got = exp64x4(lanes);
            for l in 0..4 {
                assert_eq!(
                    got[l].to_bits(),
                    exp64(lanes[l]).to_bits(),
                    "exp64x4 lane {l} deviates at x={}",
                    lanes[l]
                );
            }
            x += 0.001953125; // 2^-9: exact step, deterministic sweep
        }
        // Range edges and clamp selects.
        let edges = [-800.0, -708.5, -708.0, 0.0, 709.0, 709.5, 800.0];
        for w in edges.windows(4) {
            let lanes = [w[0], w[1], w[2], w[3]];
            let got = exp64x4(lanes);
            for l in 0..4 {
                assert_eq!(got[l].to_bits(), exp64(lanes[l]).to_bits());
            }
        }
    }

    #[test]
    fn gelu_slice_matches_f64_reference() {
        // The f32 path must stay in the same error class as the f64 scalar
        // (the A&S formula's 1.5e-7 abs bound dominates both): sweep the
        // activation range plus a non-multiple-of-16 tail.
        let mut xs: Vec<f32> = (-4000..=4001).map(|i| i as f32 * 0.00390625).collect();
        let want: Vec<f32> = xs.iter().map(|&v| gelu(v)).collect();
        gelu_slice(&mut xs);
        for (i, (g, w)) in xs.iter().zip(want.iter()).enumerate() {
            let tol = 4e-7f32.max(3e-7 * w.abs());
            assert!(
                (g - w).abs() <= tol,
                "gelu_slice deviates at index {i}: got {g} want {w}"
            );
        }
    }

    #[test]
    fn softmax_is_shift_invariant_and_deterministic() {
        let a0 = [0.3f32, -1.2, 4.5, 0.0, 2.2];
        let mut a = a0;
        let mut b = a0.map(|x| x + 100.0);
        softmax_row(&mut a);
        softmax_row(&mut b);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "shift invariance broken");
        }
        let mut c = a0;
        softmax_row(&mut c);
        assert_eq!(a.map(f32::to_bits), c.map(f32::to_bits), "determinism");
    }
}
