//! In-crate numerics for the T2 forward pass — **no `libm` anywhere on the
//! forward path** (spec §4): `exp` and `erf` are fixed polynomial/rational
//! approximations built from IEEE-exact primitives (`+ - * /`, `sqrt`,
//! `floor`, bit casts), so encoder output is byte-identical across
//! OS/libc/arch. Reductions (LayerNorm statistics, softmax sums) accumulate
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
/// 12-term Horner Taylor (term 13 ≈ r¹³/13! < 2e-14 at |r| ≤ 0.347); 2^n
/// assembled by exponent-bit construction (no `powi`, no `libm`).
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
    // Horner: 1 + r(1 + r/2(1 + r/3(... (1 + r/12) ...)))
    let mut p = 1.0f64;
    let mut k = 12.0f64;
    while k >= 1.0 {
        p = 1.0 + r / k * p;
        k -= 1.0;
    }
    let n = n as i64;
    // 2^n via exponent bits; n ∈ [-1022, 1023] after the clamps above
    // (|x| ≤ 709 → |n| ≤ 1024; subnormal edge folded into the -708 cutoff).
    debug_assert!((-1022..=1023).contains(&n));
    let scale = f64::from_bits(((n + 1023) as u64) << 52);
    p * scale
}

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

/// In-place softmax over `row`: max-subtracted, `exp64`, f64 sum
/// left-to-right, divide. The all-finite input invariant holds by
/// construction (scores are dot products of finite activations).
pub fn softmax_row(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    // Two passes with a scratch-free design: store exp in place, then scale.
    for v in row.iter_mut() {
        let e = exp64((*v - max) as f64);
        *v = e as f32;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v = (*v as f64 * inv) as f32;
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
