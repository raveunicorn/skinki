//! Stage 1 — memory-compression proof of concept.
//!
//! The Stage 0 harness proved we can *measure* retrieval. Stage 1 asks the first
//! "impossible" question of the Exocortex budget: can ~5M memory vectors be
//! searched within the M1 Air 8GB RAM/latency budget while preserving the
//! geometry of full-precision search?
//!
//! The fitness gate is **recall@k >= 95% versus an exact float32 baseline** at a
//! small per-vector footprint. Crucially, this measures *compression fidelity*
//! (does the codec preserve nearest-neighbor structure?), independent of how
//! good the embeddings themselves are — so we can benchmark honestly without a
//! heavyweight embedding model.
//!
//! We implement the candidate codecs ourselves (int8 scalar, Product
//! Quantization, and RaBitQ at 1 and multiple bits). That is the whole point of
//! the "beat-or-invent" doctrine: only by measuring our own faithful
//! implementations against the budget do we earn the right to invent something
//! new if they fall short.
//!
//! Everything is deterministic (seeded SplitMix64) so results are reproducible.

#![cfg_attr(not(unix), forbid(unsafe_code))]
// These are dense numeric kernels (dot products, quantization, k-means) where
// explicit index loops mirror the underlying math and keep bounds-check elision
// predictable; the iterator-rewrite lint hurts readability here.
#![allow(clippy::needless_range_loop)]

pub mod bench;
pub mod embed;
pub mod exact;
pub mod quant;
pub mod rotate;
pub mod search;
pub mod store;

/// A tiny deterministic SplitMix64 PRNG (matches the corpus generator's), so the
/// whole pipeline — embeddings, rotations, k-means init — is reproducible.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard-normal sample via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.unit() as f64).max(1e-12);
        let u2 = self.unit() as f64;
        (((-2.0 * u1.ln()).sqrt()) * (std::f64::consts::TAU * u2).cos()) as f32
    }

    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }
}

/// Similarity convention: we rank by **inner product** (vectors are L2-normalized
/// on ingest, so inner product equals cosine similarity and is monotone with L2).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

pub fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Normalize in place to unit L2 length (no-op for the zero vector).
pub fn normalize(v: &mut [f32]) {
    let n = l2_norm(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// A flat, row-major set of equal-length vectors — the unit of everything below.
#[derive(Debug, Clone)]
pub struct VectorSet {
    pub dim: usize,
    /// `len() == count * dim`, row-major.
    pub data: Vec<f32>,
}

impl VectorSet {
    pub fn new(dim: usize) -> Self {
        VectorSet {
            dim,
            data: Vec::new(),
        }
    }

    pub fn count(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn get(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        self.data.extend_from_slice(v);
    }

    /// Copy `len` rows starting at `start` into a new set (e.g. to split a
    /// generated set into base + held-out queries from the same distribution).
    pub fn slice_rows(&self, start: usize, len: usize) -> VectorSet {
        let end = (start + len).min(self.count());
        let mut out = VectorSet::new(self.dim);
        if start < end {
            out.data
                .extend_from_slice(&self.data[start * self.dim..end * self.dim]);
        }
        out
    }

    /// Matryoshka-style truncation: keep the first `keep` dimensions and
    /// re-normalize. Used to co-design dimensionality vs RAM (256 vs 768).
    pub fn truncate_dims(&self, keep: usize) -> VectorSet {
        let keep = keep.min(self.dim);
        let mut out = VectorSet::new(keep);
        out.data.reserve(self.count() * keep);
        for i in 0..self.count() {
            let src = self.get(i);
            let mut row: Vec<f32> = src[..keep].to_vec();
            normalize(&mut row);
            out.data.extend_from_slice(&row);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(1);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn normal_is_roughly_standard() {
        let mut r = Rng::new(7);
        let n = 20000;
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for _ in 0..n {
            let x = r.normal() as f64;
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }

    #[test]
    fn truncate_renormalizes() {
        let mut vs = VectorSet::new(4);
        vs.push(&[1.0, 2.0, 3.0, 4.0]);
        let t = vs.truncate_dims(2);
        assert_eq!(t.dim, 2);
        assert!((l2_norm(t.get(0)) - 1.0).abs() < 1e-6);
    }
}
