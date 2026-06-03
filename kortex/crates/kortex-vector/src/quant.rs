//! Candidate compression codecs, implemented from scratch.
//!
//! Each codec implements [`Quantizer`]: it encodes a [`VectorSet`] into a compact
//! byte layout and provides an *asymmetric* estimator of the inner product
//! between a full-precision query and each stored (compressed) vector. We then
//! measure how well the codec's ranking matches exact float32 search.
//!
//! Codecs:
//! - [`FloatStore`] — uncompressed f32 (the 100%-recall, full-RAM reference).
//! - [`ScalarI8`]   — per-dimension int8 scalar quantization (4x).
//! - [`ProductQuantizer`] — PQ with per-subspace k-means codebooks.
//! - [`RaBitQ`]     — rotation + 1-bit sign codes (32x) or B-bit codes, with the
//!   RaBitQ unbiased inner-product estimator.

use crate::exact::select_top_k;
use crate::rotate::Rotator;
use crate::{dot, l2_norm, VectorSet};

/// A compression codec over a fixed vector set.
pub trait Quantizer {
    fn name(&self) -> String;
    fn count(&self) -> usize;
    /// Total compact-storage bytes per vector (the RAM-budget metric).
    fn bytes_per_vector(&self) -> f64;
    /// Estimated inner product against every stored vector (full scan).
    fn scores(&self, query: &[f32]) -> Vec<f32>;
    /// Estimated inner product for a subset of ids (the rerank stage).
    fn scores_subset(&self, query: &[f32], ids: &[u32]) -> Vec<f32>;

    /// Single-stage search: full scan, return top-`k` ids.
    fn search(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = self
            .scores(query)
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i as u32))
            .collect();
        select_top_k(&mut scored, k)
    }
}

// ---------------------------------------------------------------------------
// Bit packing (for multi-bit codes)
// ---------------------------------------------------------------------------

struct BitWriter {
    buf: Vec<u8>,
    bit_len: usize,
}

impl BitWriter {
    fn with_capacity(bytes: usize) -> Self {
        BitWriter {
            buf: Vec::with_capacity(bytes),
            bit_len: 0,
        }
    }
    fn write(&mut self, val: u32, bits: u8) {
        for b in 0..bits as usize {
            let bit = ((val >> b) & 1) as u8;
            let byte_idx = self.bit_len >> 3;
            if byte_idx >= self.buf.len() {
                self.buf.push(0);
            }
            self.buf[byte_idx] |= bit << (self.bit_len & 7);
            self.bit_len += 1;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        while self.bit_len & 7 != 0 {
            self.bit_len += 1;
        }
        self.buf.shrink_to_fit();
        self.buf
    }
}

#[inline]
fn read_bits(buf: &[u8], bit_pos: usize, bits: u8) -> u32 {
    let mut val = 0u32;
    for b in 0..bits as usize {
        let p = bit_pos + b;
        let bit = (buf[p >> 3] >> (p & 7)) & 1;
        val |= (bit as u32) << b;
    }
    val
}

// ---------------------------------------------------------------------------
// FloatStore — uncompressed reference
// ---------------------------------------------------------------------------

pub struct FloatStore {
    data: VectorSet,
}

impl FloatStore {
    pub fn build(vs: &VectorSet) -> Self {
        FloatStore { data: vs.clone() }
    }
}

impl Quantizer for FloatStore {
    fn name(&self) -> String {
        "float32".into()
    }
    fn count(&self) -> usize {
        self.data.count()
    }
    fn bytes_per_vector(&self) -> f64 {
        (self.data.dim * 4) as f64
    }
    fn scores(&self, query: &[f32]) -> Vec<f32> {
        (0..self.data.count())
            .map(|i| dot(self.data.get(i), query))
            .collect()
    }
    fn scores_subset(&self, query: &[f32], ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .map(|&i| dot(self.data.get(i as usize), query))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ScalarI8 — per-dimension int8 scalar quantization
// ---------------------------------------------------------------------------

pub struct ScalarI8 {
    dim: usize,
    count: usize,
    lo: Vec<f32>,
    step: Vec<f32>,
    codes: Vec<u8>, // count * dim, one byte per dim
}

impl ScalarI8 {
    pub fn build(vs: &VectorSet) -> Self {
        let dim = vs.dim;
        let n = vs.count();
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for i in 0..n {
            let v = vs.get(i);
            for d in 0..dim {
                lo[d] = lo[d].min(v[d]);
                hi[d] = hi[d].max(v[d]);
            }
        }
        let step: Vec<f32> = (0..dim)
            .map(|d| {
                let r = hi[d] - lo[d];
                if r > 1e-12 {
                    r / 255.0
                } else {
                    1.0
                }
            })
            .collect();
        let mut codes = vec![0u8; n * dim];
        for i in 0..n {
            let v = vs.get(i);
            for d in 0..dim {
                let q = ((v[d] - lo[d]) / step[d]).round().clamp(0.0, 255.0);
                codes[i * dim + d] = q as u8;
            }
        }
        ScalarI8 {
            dim,
            count: n,
            lo,
            step,
            codes,
        }
    }

    #[inline]
    fn score_with(&self, c0: f32, weighted: &[f32], id: usize) -> f32 {
        let base = id * self.dim;
        let mut s = c0;
        for d in 0..self.dim {
            s += weighted[d] * self.codes[base + d] as f32;
        }
        s
    }

    fn prep(&self, query: &[f32]) -> (f32, Vec<f32>) {
        // dot(query, lo + code*step) = <query,lo> + sum_d query[d]*step[d]*code
        let c0 = dot(query, &self.lo);
        let weighted: Vec<f32> = (0..self.dim).map(|d| query[d] * self.step[d]).collect();
        (c0, weighted)
    }
}

impl Quantizer for ScalarI8 {
    fn name(&self) -> String {
        "int8-scalar".into()
    }
    fn count(&self) -> usize {
        self.count
    }
    fn bytes_per_vector(&self) -> f64 {
        self.dim as f64
    }
    fn scores(&self, query: &[f32]) -> Vec<f32> {
        let (c0, w) = self.prep(query);
        (0..self.count)
            .map(|i| self.score_with(c0, &w, i))
            .collect()
    }
    fn scores_subset(&self, query: &[f32], ids: &[u32]) -> Vec<f32> {
        let (c0, w) = self.prep(query);
        ids.iter()
            .map(|&i| self.score_with(c0, &w, i as usize))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ProductQuantizer — per-subspace k-means
// ---------------------------------------------------------------------------

const PQ_K: usize = 256;
const PQ_ITERS: usize = 12;

pub struct ProductQuantizer {
    count: usize,
    m: usize,   // number of subspaces
    sub: usize, // dim per subspace = dim / m
    /// Codebooks: m * PQ_K * sub, row-major [subspace][centroid][component].
    codebooks: Vec<f32>,
    codes: Vec<u8>, // count * m
}

impl ProductQuantizer {
    /// `m` must divide `vs.dim`.
    pub fn build(vs: &VectorSet, m: usize, seed: u64) -> Self {
        let dim = vs.dim;
        assert!(
            dim.is_multiple_of(m),
            "PQ subspaces ({m}) must divide dim ({dim})"
        );
        let sub = dim / m;
        let n = vs.count();
        let mut codebooks = vec![0.0f32; m * PQ_K * sub];

        // Training sample (deterministic stride, capped for speed).
        let sample_cap = 8192usize.min(n);
        let stride = (n / sample_cap).max(1);
        let sample: Vec<usize> = (0..n).step_by(stride).take(sample_cap).collect();

        for sp in 0..m {
            let off = sp * sub;
            let cb = &mut codebooks[sp * PQ_K * sub..(sp + 1) * PQ_K * sub];
            kmeans_subspace(vs, &sample, off, sub, cb, seed.wrapping_add(sp as u64));
        }

        // Encode all vectors.
        let mut codes = vec![0u8; n * m];
        for i in 0..n {
            let v = vs.get(i);
            for sp in 0..m {
                let off = sp * sub;
                let cb = &codebooks[sp * PQ_K * sub..(sp + 1) * PQ_K * sub];
                codes[i * m + sp] = nearest_centroid(&v[off..off + sub], cb, sub) as u8;
            }
        }
        ProductQuantizer {
            count: n,
            m,
            sub,
            codebooks,
            codes,
        }
    }

    /// Precompute the asymmetric similarity table: table[sp*PQ_K + c] =
    /// dot(query_subspace_sp, centroid_c).
    fn build_table(&self, query: &[f32]) -> Vec<f32> {
        let mut table = vec![0.0f32; self.m * PQ_K];
        for sp in 0..self.m {
            let off = sp * self.sub;
            let q = &query[off..off + self.sub];
            let cb = &self.codebooks[sp * PQ_K * self.sub..(sp + 1) * PQ_K * self.sub];
            for c in 0..PQ_K {
                table[sp * PQ_K + c] = dot(q, &cb[c * self.sub..(c + 1) * self.sub]);
            }
        }
        table
    }

    #[inline]
    fn score_with(&self, table: &[f32], id: usize) -> f32 {
        let base = id * self.m;
        let mut s = 0.0f32;
        for sp in 0..self.m {
            s += table[sp * PQ_K + self.codes[base + sp] as usize];
        }
        s
    }
}

fn nearest_centroid(v: &[f32], codebook: &[f32], sub: usize) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..PQ_K {
        let cen = &codebook[c * sub..(c + 1) * sub];
        let mut d = 0.0f32;
        for j in 0..sub {
            let diff = v[j] - cen[j];
            d += diff * diff;
        }
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

fn kmeans_subspace(
    vs: &VectorSet,
    sample: &[usize],
    off: usize,
    sub: usize,
    codebook: &mut [f32],
    seed: u64,
) {
    use crate::Rng;
    let mut r = Rng::new(seed ^ 0xA17F);
    // Init: distinct random sample points.
    for c in 0..PQ_K {
        let idx = sample[r.below(sample.len())];
        let src = &vs.get(idx)[off..off + sub];
        codebook[c * sub..(c + 1) * sub].copy_from_slice(src);
    }
    let mut counts = vec![0u32; PQ_K];
    let mut sums = vec![0.0f32; PQ_K * sub];
    for _ in 0..PQ_ITERS {
        for x in sums.iter_mut() {
            *x = 0.0;
        }
        for c in counts.iter_mut() {
            *c = 0;
        }
        for &idx in sample {
            let v = &vs.get(idx)[off..off + sub];
            let c = nearest_centroid(v, codebook, sub);
            counts[c] += 1;
            let dst = &mut sums[c * sub..(c + 1) * sub];
            for j in 0..sub {
                dst[j] += v[j];
            }
        }
        for c in 0..PQ_K {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..sub {
                    codebook[c * sub + j] = sums[c * sub + j] * inv;
                }
            } else {
                // Reseed an empty centroid from a random sample point.
                let idx = sample[r.below(sample.len())];
                let src = &vs.get(idx)[off..off + sub];
                codebook[c * sub..(c + 1) * sub].copy_from_slice(src);
            }
        }
    }
}

impl Quantizer for ProductQuantizer {
    fn name(&self) -> String {
        format!("pq-m{}", self.m)
    }
    fn count(&self) -> usize {
        self.count
    }
    fn bytes_per_vector(&self) -> f64 {
        self.m as f64
    }
    fn scores(&self, query: &[f32]) -> Vec<f32> {
        let table = self.build_table(query);
        (0..self.count)
            .map(|i| self.score_with(&table, i))
            .collect()
    }
    fn scores_subset(&self, query: &[f32], ids: &[u32]) -> Vec<f32> {
        let table = self.build_table(query);
        ids.iter()
            .map(|&i| self.score_with(&table, i as usize))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RaBitQ — rotation + sign (1-bit) or uniform (B-bit) codes
// ---------------------------------------------------------------------------

const RABITQ_RANGE_SIGMA: f32 = 3.5;

pub struct RaBitQ {
    count: usize,
    padded: usize,
    bits: u8,
    rotator: Rotator,
    factors: Vec<f32>, // per-vector ranking factor
    codes: Vec<u8>,    // packed: 1-bit sign codes, or B-bit uniform codes
    bytes_per_vec_codes: usize,
    // multi-bit dequant params
    lo: f32,
    step: f32,
}

impl RaBitQ {
    pub fn build(vs: &VectorSet, bits: u8, seed: u64) -> Self {
        assert!(bits >= 1, "bits must be >= 1");
        let dim = vs.dim;
        let n = vs.count();
        let rotator = Rotator::new(seed ^ 0x2B17, dim, 2);
        let padded = rotator.out_dim();

        // Dataset centroid (RaBitQ centers residuals before quantizing).
        let mut centroid = vec![0.0f32; dim];
        for i in 0..n {
            let v = vs.get(i);
            for d in 0..dim {
                centroid[d] += v[d];
            }
        }
        if n > 0 {
            for x in centroid.iter_mut() {
                *x /= n as f32;
            }
        }

        let r = RABITQ_RANGE_SIGMA / (padded as f32).sqrt();
        let levels = 1u32 << bits;
        let step = if bits == 1 {
            0.0
        } else {
            2.0 * r / levels as f32
        };
        let lo = -r;

        let mut factors = vec![0.0f32; n];
        let bytes_per_vec_codes = (padded * bits as usize).div_ceil(8);
        let mut writer = BitWriter::with_capacity(n * bytes_per_vec_codes);

        let mut residual = vec![0.0f32; dim];
        for i in 0..n {
            let v = vs.get(i);
            for d in 0..dim {
                residual[d] = v[d] - centroid[d];
            }
            let norm_r = l2_norm(&residual).max(1e-9);
            for x in residual.iter_mut() {
                *x /= norm_r;
            }
            let x = rotator.apply(&residual); // unit, length padded

            if bits == 1 {
                // Sign codes; factor = norm_r / ||x||_1.
                let mut l1 = 0.0f32;
                for d in 0..padded {
                    let bit = if x[d] >= 0.0 { 1u32 } else { 0u32 };
                    l1 += x[d].abs();
                    writer.write(bit, 1);
                }
                factors[i] = norm_r / l1.max(1e-9);
            } else {
                // Uniform B-bit codes; factor = norm_r / ||x_hat||.
                let mut xhat_norm_sq = 0.0f32;
                for d in 0..padded {
                    let clamped = x[d].clamp(lo, -lo);
                    let mut level = ((clamped - lo) / step).floor() as i64;
                    if level < 0 {
                        level = 0;
                    }
                    if level as u32 >= levels {
                        level = levels as i64 - 1;
                    }
                    let val = lo + (level as f32 + 0.5) * step;
                    xhat_norm_sq += val * val;
                    writer.write(level as u32, bits);
                }
                factors[i] = norm_r / xhat_norm_sq.sqrt().max(1e-9);
            }
        }

        RaBitQ {
            count: n,
            padded,
            bits,
            rotator,
            factors,
            codes: writer.finish(),
            bytes_per_vec_codes,
            lo,
            step,
        }
    }

    /// The packed code bytes (without the per-vector ranking factors). Used by
    /// the bench to exercise the mmap store on the bulk of the index.
    pub fn code_bytes(&self) -> &[u8] {
        &self.codes
    }

    /// Rotate the raw query into the code basis (orthonormal, so inner products
    /// with rotated data residuals are preserved). The constant centroid term is
    /// dropped — it does not affect ranking.
    fn prep(&self, query: &[f32]) -> Vec<f32> {
        // We rotate the query directly; <residual, query> = <R residual, R query>,
        // and the dataset-centroid offset is a per-query constant we can ignore.
        self.rotator.apply(query)
    }

    #[inline]
    fn score_one_bit1(&self, y: &[f32], total_y: f32, id: usize) -> f32 {
        let base = id * self.bytes_per_vec_codes;
        let words = self.padded / 64;
        let mut pos = 0.0f32;
        for w in 0..words {
            let b0 = base + w * 8;
            let word = u64::from_le_bytes(self.codes[b0..b0 + 8].try_into().unwrap());
            let mut bits = word;
            let coord_base = w * 64;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                pos += y[coord_base + j];
                bits &= bits - 1;
            }
        }
        // Signed sum S = sum sign*y = 2*pos - total; score = factor * S.
        self.factors[id] * (2.0 * pos - total_y)
    }

    #[inline]
    fn score_one_multibit(&self, y: &[f32], id: usize) -> f32 {
        let bit_base = id * self.bytes_per_vec_codes * 8;
        let mut s = 0.0f32;
        for d in 0..self.padded {
            let level = read_bits(&self.codes, bit_base + d * self.bits as usize, self.bits);
            let val = self.lo + (level as f32 + 0.5) * self.step;
            s += val * y[d];
        }
        self.factors[id] * s
    }
}

impl Quantizer for RaBitQ {
    fn name(&self) -> String {
        format!("rabitq-{}bit", self.bits)
    }
    fn count(&self) -> usize {
        self.count
    }
    fn bytes_per_vector(&self) -> f64 {
        // codes + a 4-byte ranking factor per vector.
        self.bytes_per_vec_codes as f64 + 4.0
    }
    fn scores(&self, query: &[f32]) -> Vec<f32> {
        let y = self.prep(query);
        if self.bits == 1 {
            let total: f32 = y.iter().sum();
            (0..self.count)
                .map(|i| self.score_one_bit1(&y, total, i))
                .collect()
        } else {
            (0..self.count)
                .map(|i| self.score_one_multibit(&y, i))
                .collect()
        }
    }
    fn scores_subset(&self, query: &[f32], ids: &[u32]) -> Vec<f32> {
        let y = self.prep(query);
        if self.bits == 1 {
            let total: f32 = y.iter().sum();
            ids.iter()
                .map(|&i| self.score_one_bit1(&y, total, i as usize))
                .collect()
        } else {
            ids.iter()
                .map(|&i| self.score_one_multibit(&y, i as usize))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::synthetic_clusters;
    use crate::exact::top_k;

    fn recall_at_k(q: &dyn Quantizer, vs: &VectorSet, queries: &VectorSet, k: usize) -> f64 {
        let mut hit = 0usize;
        let mut total = 0usize;
        for qi in 0..queries.count() {
            let query = queries.get(qi);
            let truth = top_k(vs, query, k);
            let got = q.search(query, k);
            for t in &truth {
                if got.contains(t) {
                    hit += 1;
                }
            }
            total += truth.len();
        }
        hit as f64 / total as f64
    }

    #[test]
    fn float_store_is_exact() {
        let vs = synthetic_clusters(1, 64, 400, 8, 0.4);
        let queries = synthetic_clusters(99, 64, 30, 8, 0.4);
        let fs = FloatStore::build(&vs);
        assert!((recall_at_k(&fs, &vs, &queries, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn int8_high_recall() {
        let vs = synthetic_clusters(2, 128, 800, 10, 0.4);
        let queries = synthetic_clusters(98, 128, 40, 10, 0.4);
        let q = ScalarI8::build(&vs);
        let r = recall_at_k(&q, &vs, &queries, 10);
        assert!(r > 0.9, "int8 recall {r}");
    }

    #[test]
    fn pq_reasonable_recall() {
        let vs = synthetic_clusters(3, 128, 1000, 12, 0.4);
        let queries = synthetic_clusters(97, 128, 40, 12, 0.4);
        let q = ProductQuantizer::build(&vs, 32, 123);
        let r = recall_at_k(&q, &vs, &queries, 10);
        assert!(r > 0.6, "pq recall {r}");
    }

    #[test]
    fn rabitq_1bit_beats_random_and_multibit_improves() {
        // Base + held-out queries from the same distribution (on-manifold).
        let all = synthetic_clusters(4, 256, 1240, 12, 0.4);
        let vs = all.slice_rows(0, 1200);
        let queries = all.slice_rows(1200, 40);
        let r1 = recall_at_k(&RaBitQ::build(&vs, 1, 7), &vs, &queries, 10);
        let r7 = recall_at_k(&RaBitQ::build(&vs, 7, 7), &vs, &queries, 10);
        // 1-bit is a fast but lossy coarse filter (hence the two-stage rerank);
        // adding bits should markedly sharpen recall.
        assert!(r1 > 0.15, "1-bit recall implausibly low: {r1}");
        assert!(r7 > 0.7, "7-bit recall too low: {r7}");
        assert!(r7 > r1 + 0.2, "more bits should clearly help: {r1} -> {r7}");
    }

    #[test]
    fn bytes_per_vector_reports_compression() {
        let vs = synthetic_clusters(5, 256, 200, 8, 0.4);
        let f = FloatStore::build(&vs).bytes_per_vector();
        let one = RaBitQ::build(&vs, 1, 1).bytes_per_vector();
        assert!(one < f / 8.0, "1-bit should be far smaller: {one} vs {f}");
    }
}
