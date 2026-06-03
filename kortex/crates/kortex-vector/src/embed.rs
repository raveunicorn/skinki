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
use kortex_corpus::Corpus;

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
    let mut r = Rng::new(seed);
    // Cluster centers on the sphere.
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(clusters);
    for _ in 0..clusters {
        let mut c = vec![0.0f32; dim];
        for x in c.iter_mut() {
            *x = r.normal();
        }
        normalize(&mut c);
        centers.push(c);
    }
    // Per-coordinate noise std so the noise vector's L2 norm is ~`noise`.
    let sigma = noise / (dim as f32).sqrt();
    let mut vs = VectorSet::new(dim);
    vs.data.reserve(count * dim);
    let mut v = vec![0.0f32; dim];
    for _ in 0..count {
        let c = &centers[r.below(clusters)];
        for i in 0..dim {
            v[i] = c[i] + r.normal() * sigma;
        }
        normalize(&mut v);
        vs.push(&v);
    }
    vs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dot;
    use kortex_corpus::{generate, GenConfig};

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
