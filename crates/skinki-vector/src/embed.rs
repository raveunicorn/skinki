//! Vector sources for the compression bench.
//!
//! Two sources, both deterministic:
//!
//! 1. [`StaticHashEmbedder`] — a "Model2Vec-lite" static embedder. Each token is
//!    hashed to a fixed pseudo-random unit-ish vector; a document embedding is
//!    the L2-normalized mean of its token vectors. This is exactly the spirit of
//!    the cheap *first-pass* embedding in the two-stage retrieval plan: no model
//!    download, no network, yet it produces realistic, topically-clustered
//!    vectors directly from the Stage 0 corpus text.
//!
//! 2. [`synthetic_clusters`] — Gaussian clusters on the unit sphere with
//!    controllable separation. Useful to probe a codec's behavior in a clean,
//!    well-understood geometry independent of any text.

use crate::{normalize, Rng, VectorSet};
use skinki_corpus::Corpus;

/// FNV-1a 64-bit hash — small, fast, deterministic across platforms.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn tokenize(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
}

/// A text -> fixed-dim vector embedder. The swappable seam: ship a static
/// embedder by default; a real transformer (EmbeddingGemma/nomic) plugs in
/// behind this same trait via precomputed vectors.
pub trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
    fn dim(&self) -> usize;
}

/// A static (non-contextual) embedder: token -> seeded random vector, averaged.
pub struct StaticHashEmbedder {
    dim: usize,
}

impl StaticHashEmbedder {
    pub fn new(dim: usize) -> Self {
        StaticHashEmbedder { dim }
    }

    /// Deterministic per-token vector, generated on the fly (no table to store).
    fn token_vector(&self, token: &str, out: &mut [f32]) {
        let seed = fnv1a(token.to_lowercase().as_bytes());
        let mut r = Rng::new(seed ^ 0x5125_3a9b_7f01_c3d1);
        for x in out.iter_mut() {
            *x = r.normal();
        }
        normalize(out);
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut acc = vec![0.0f32; self.dim];
        let mut tmp = vec![0.0f32; self.dim];
        let mut n = 0u32;
        for tok in tokenize(text) {
            self.token_vector(tok, &mut tmp);
            for i in 0..self.dim {
                acc[i] += tmp[i];
            }
            n += 1;
        }
        if n == 0 {
            // Empty doc: deterministic but distinct fallback.
            self.token_vector("\u{0}empty", &mut acc);
        }
        normalize(&mut acc);
        acc
    }

    /// Embed every entry of a Stage 0 corpus into a [`VectorSet`].
    pub fn embed_corpus(&self, corpus: &Corpus) -> VectorSet {
        let mut vs = VectorSet::new(self.dim);
        vs.data.reserve(corpus.entries.len() * self.dim);
        for e in &corpus.entries {
            let v = self.embed(&e.text);
            vs.push(&v);
        }
        vs
    }
}

impl Embedder for StaticHashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        StaticHashEmbedder::embed(self, text)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Streaming source of unit vectors drawn from Gaussian blobs on the sphere —
/// the same distribution as [`synthetic_clusters`], but one vector at a time,
/// so multi-million-vector sets can be written to disk without ever living in
/// RAM. Deterministic: a given seed yields the same sequence regardless of
/// how the draws are batched.
pub struct ClusterSampler {
    rng: Rng,
    dim: usize,
    centers: Vec<Vec<f32>>,
    sigma: f32,
}

impl ClusterSampler {
    pub fn new(seed: u64, dim: usize, clusters: usize, noise: f32) -> Self {
        let mut rng = Rng::new(seed);
        // Cluster centers on the sphere.
        let mut centers: Vec<Vec<f32>> = Vec::with_capacity(clusters);
        for _ in 0..clusters {
            let mut c = vec![0.0f32; dim];
            for x in c.iter_mut() {
                *x = rng.normal();
            }
            normalize(&mut c);
            centers.push(c);
        }
        // Per-coordinate noise std so the noise vector's L2 norm is ~`noise`.
        let sigma = noise / (dim as f32).sqrt();
        ClusterSampler {
            rng,
            dim,
            centers,
            sigma,
        }
    }

    /// Fill `out` (length `dim`) with the next vector in the sequence.
    pub fn fill(&mut self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.dim);
        let idx = self.rng.below(self.centers.len());
        // Draw order matches the original synthetic_clusters loop: one normal
        // per coordinate, after the center pick.
        for (i, x) in out.iter_mut().enumerate() {
            *x = self.centers[idx][i] + self.rng.normal() * self.sigma;
        }
        normalize(out);
    }
}

/// Generate `count` unit vectors drawn from `clusters` Gaussian blobs on the
/// sphere. `noise` is the (dimension-independent) relative magnitude of the
/// perturbation around a unit cluster center: smaller `noise` = tighter
/// clusters (easier nearest-neighbor task). The expected cosine of a point to
/// its center is ~1/sqrt(1 + noise^2), independent of `dim`.
pub fn synthetic_clusters(
    seed: u64,
    dim: usize,
    count: usize,
    clusters: usize,
    noise: f32,
) -> VectorSet {
    let mut sampler = ClusterSampler::new(seed, dim, clusters, noise);
    let mut vs = VectorSet::new(dim);
    vs.data.reserve(count * dim);
    let mut v = vec![0.0f32; dim];
    for _ in 0..count {
        sampler.fill(&mut v);
        vs.push(&v);
    }
    vs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dot;
    use skinki_corpus::{generate, GenConfig};

    #[test]
    fn embeddings_are_unit_and_deterministic() {
        let e = StaticHashEmbedder::new(64);
        let a = e.embed("the quick brown fox");
        let b = e.embed("the quick brown fox");
        assert_eq!(a, b);
        assert!((dot(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn similar_text_is_closer_than_unrelated() {
        let e = StaticHashEmbedder::new(128);
        let base = e.embed("distributed systems latency budgets on call");
        let near = e.embed("distributed systems latency budgets review");
        let far = e.embed("jazz harmony band rehearsal new synth");
        assert!(dot(&base, &near) > dot(&base, &far));
    }

    #[test]
    fn embed_corpus_matches_entry_count() {
        let c = generate(&GenConfig {
            seed: 1,
            years: 1,
            ..Default::default()
        });
        let e = StaticHashEmbedder::new(64);
        let vs = e.embed_corpus(&c);
        assert_eq!(vs.count(), c.entries.len());
        assert_eq!(vs.dim, 64);
    }

    #[test]
    fn embedder_trait_object_usable() {
        let e = StaticHashEmbedder::new(32);
        let boxed: &dyn Embedder = &e;
        assert_eq!(boxed.dim(), 32);
        let v = boxed.embed("hello world");
        assert_eq!(v.len(), 32);
        assert!((dot(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn clusters_have_internal_structure() {
        let vs = synthetic_clusters(3, 64, 500, 8, 0.4);
        assert_eq!(vs.count(), 500);
        // A random vector's best match should be quite similar in clustered data.
        let mut best = -2.0f32;
        for j in 1..vs.count() {
            best = best.max(dot(vs.get(0), vs.get(j)));
        }
        assert!(best > 0.5, "expected a close cluster neighbor, got {best}");
    }
}
