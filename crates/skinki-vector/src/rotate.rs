//! A fast, deterministic random rotation for RaBitQ.
//!
//! RaBitQ's accuracy comes from quantizing in a *randomly rotated* basis: the
//! rotation spreads each vector's energy across all coordinates ("incoherence"),
//! which makes a 1-bit sign quantization an unbiased, low-variance estimator of
//! inner products. A dense D×D orthonormal matrix would cost O(D^2) per vector;
//! we instead use the standard structured trick — a random sign flip followed by
//! a Walsh-Hadamard transform — giving an O(D log D), norm-preserving,
//! pseudo-random orthogonal transform.
//!
//! Dimensions are padded up to the next power of two (the WHT requires it); the
//! padding is transparent to callers.

use crate::Rng;

/// A reusable random rotation for a fixed input dimension.
#[derive(Clone)]
pub struct Rotator {
    in_dim: usize,
    padded: usize,
    /// Random ±1 per padded coordinate, applied before each WHT pass.
    signs: Vec<Vec<f32>>,
    passes: usize,
}

fn next_pow2(mut n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    n -= 1;
    let mut p = 1;
    while p < n + 1 {
        p <<= 1;
    }
    p
}

/// In-place Fast Walsh-Hadamard Transform; `a.len()` must be a power of two.
fn fwht(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += 2 * h;
        }
        h <<= 1;
    }
}

impl Rotator {
    /// `passes` of (sign-flip + WHT). Two passes give a good random orthogonal
    /// transform; one is often enough. Deterministic given `seed`.
    pub fn new(seed: u64, in_dim: usize, passes: usize) -> Self {
        let padded = next_pow2(in_dim);
        let mut r = Rng::new(seed);
        let passes = passes.max(1);
        let mut signs = Vec::with_capacity(passes);
        for _ in 0..passes {
            let mut s = vec![0.0f32; padded];
            for x in s.iter_mut() {
                *x = if r.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            }
            signs.push(s);
        }
        Rotator {
            in_dim,
            padded,
            signs,
            passes,
        }
    }

    pub fn out_dim(&self) -> usize {
        self.padded
    }

    /// Rotate `v` (length `in_dim`) into a length-`padded` output vector.
    /// The transform is orthonormal, so the L2 norm is preserved.
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        debug_assert_eq!(v.len(), self.in_dim);
        let mut buf = vec![0.0f32; self.padded];
        buf[..self.in_dim].copy_from_slice(v);
        let scale = 1.0 / (self.padded as f32).sqrt();
        for pass in 0..self.passes {
            let s = &self.signs[pass];
            for i in 0..self.padded {
                buf[i] *= s[i];
            }
            fwht(&mut buf);
            for x in buf.iter_mut() {
                *x *= scale;
            }
        }
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dot, l2_norm, Rng};

    #[test]
    fn pow2_rounding() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(256), 256);
        assert_eq!(next_pow2(768), 1024);
        assert_eq!(next_pow2(513), 1024);
    }

    #[test]
    fn rotation_preserves_norm() {
        let rot = Rotator::new(42, 768, 2);
        let mut r = Rng::new(1);
        let v: Vec<f32> = (0..768).map(|_| r.normal()).collect();
        let rotated = rot.apply(&v);
        assert!((l2_norm(&v) - l2_norm(&rotated)).abs() < 1e-3);
    }

    #[test]
    fn rotation_preserves_inner_product() {
        // Orthonormal transforms preserve dot products: <Pa, Pb> == <a, b>.
        let rot = Rotator::new(7, 256, 2);
        let mut r = Rng::new(2);
        let a: Vec<f32> = (0..256).map(|_| r.normal()).collect();
        let b: Vec<f32> = (0..256).map(|_| r.normal()).collect();
        let before = dot(&a, &b);
        let after = dot(&rot.apply(&a), &rot.apply(&b));
        assert!((before - after).abs() < 1e-2, "{before} vs {after}");
    }

    #[test]
    fn rotation_spreads_energy() {
        // A spike vector should become dense after rotation (incoherence).
        let rot = Rotator::new(9, 256, 2);
        let mut v = vec![0.0f32; 256];
        v[0] = 1.0;
        let rotated = rot.apply(&v);
        let max = rotated.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!(max < 0.3, "energy not spread, max coord {max}");
    }
}
