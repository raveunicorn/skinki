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

/// Bit width of the quantized query used by the 1-bit fast scan.
const QUERY_BITS: usize = 4;

/// Magic + version for the on-disk index format (see [`RaBitQ::save`]).
const RABITQ_MAGIC: &[u8; 8] = b"KXRABQ01";

pub struct RaBitQ {
    count: usize,
    in_dim: usize,
    padded: usize,
    bits: u8,
    seed: u64,
    rotator: Rotator,
    factors: Vec<f32>, // per-vector ranking factor
    /// Popcount of each vector's 1-bit code (empty for multi-bit codecs).
    pc: Vec<u32>,
    codes: Vec<u8>, // packed: 1-bit sign codes, or B-bit uniform codes
    bytes_per_vec_codes: usize,
    // multi-bit dequant params
    lo: f32,
    step: f32,
    /// Use the popcount fast path for 1-bit scans (default true).
    fast_scan: bool,
}

/// Incremental RaBitQ encoder: feeds vectors one at a time so an index over
/// millions of vectors can be built from a disk stream without ever holding
/// the float set in RAM. `RaBitQ::build` is a thin wrapper over this.
///
/// The caller supplies the dataset centroid (RaBitQ centers residuals before
/// quantizing), which therefore needs one prior streaming pass.
pub struct RaBitQBuilder {
    bits: u8,
    seed: u64,
    in_dim: usize,
    padded: usize,
    rotator: Rotator,
    centroid: Vec<f32>,
    lo: f32,
    step: f32,
    levels: u32,
    factors: Vec<f32>,
    pc: Vec<u32>,
    writer: BitWriter,
    bytes_per_vec_codes: usize,
    count: usize,
    residual: Vec<f32>,
}

impl RaBitQBuilder {
    pub fn new(in_dim: usize, bits: u8, seed: u64, centroid: Vec<f32>) -> Self {
        assert!(bits >= 1, "bits must be >= 1");
        assert_eq!(centroid.len(), in_dim, "centroid dim mismatch");
        let rotator = Rotator::new(seed ^ 0x2B17, in_dim, 2);
        let padded = rotator.out_dim();
        // The word-wise scan paths assume whole u64 words per vector.
        assert!(padded >= 64, "RaBitQ requires dim > 32 (padded >= 64)");

        let r = RABITQ_RANGE_SIGMA / (padded as f32).sqrt();
        let levels = 1u32 << bits;
        let step = if bits == 1 {
            0.0
        } else {
            2.0 * r / levels as f32
        };
        let lo = -r;
        let bytes_per_vec_codes = (padded * bits as usize).div_ceil(8);

        RaBitQBuilder {
            bits,
            seed,
            in_dim,
            padded,
            rotator,
            centroid,
            lo,
            step,
            levels,
            factors: Vec::new(),
            pc: Vec::new(),
            writer: BitWriter::with_capacity(0),
            bytes_per_vec_codes,
            count: 0,
            residual: vec![0.0f32; in_dim],
        }
    }

    pub fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.in_dim);
        for d in 0..self.in_dim {
            self.residual[d] = v[d] - self.centroid[d];
        }
        let norm_r = l2_norm(&self.residual).max(1e-9);
        for x in self.residual.iter_mut() {
            *x /= norm_r;
        }
        let x = self.rotator.apply(&self.residual); // unit, length padded

        if self.bits == 1 {
            // Sign codes; factor = norm_r / ||x||_1.
            let mut l1 = 0.0f32;
            let mut ones = 0u32;
            for d in 0..self.padded {
                let bit = if x[d] >= 0.0 { 1u32 } else { 0u32 };
                ones += bit;
                l1 += x[d].abs();
                self.writer.write(bit, 1);
            }
            self.factors.push(norm_r / l1.max(1e-9));
            self.pc.push(ones);
        } else {
            // Uniform B-bit codes; factor = norm_r / ||x_hat||.
            let mut xhat_norm_sq = 0.0f32;
            for d in 0..self.padded {
                let clamped = x[d].clamp(self.lo, -self.lo);
                let mut level = ((clamped - self.lo) / self.step).floor() as i64;
                if level < 0 {
                    level = 0;
                }
                if level as u32 >= self.levels {
                    level = self.levels as i64 - 1;
                }
                let val = self.lo + (level as f32 + 0.5) * self.step;
                xhat_norm_sq += val * val;
                self.writer.write(level as u32, self.bits);
            }
            self.factors.push(norm_r / xhat_norm_sq.sqrt().max(1e-9));
        }
        self.count += 1;
    }

    pub fn finish(self) -> RaBitQ {
        RaBitQ {
            count: self.count,
            in_dim: self.in_dim,
            padded: self.padded,
            bits: self.bits,
            seed: self.seed,
            rotator: self.rotator,
            factors: self.factors,
            pc: self.pc,
            codes: self.writer.finish(),
            bytes_per_vec_codes: self.bytes_per_vec_codes,
            lo: self.lo,
            step: self.step,
            fast_scan: true,
        }
    }
}

/// A query pre-quantized to `QUERY_BITS`-bit codes laid out as bit-planes, so
/// a 1-bit data scan reduces to AND + popcount per u64 word.
///
/// Math: data stores sign bits b_d (signed value s_d = 2*b_d - 1) and the
/// reference score is `factor * (2*pos - total)` with `pos = Σ_{b_d=1} y_d`.
/// Quantize y_d ≈ lo + step*q_d (q_d in [0, 2^B-1]); then
/// `pos ≈ lo*popcnt(code) + step * Σ_j 2^j * popcnt(code AND plane_j)`,
/// where plane_j holds bit j of every q_d. popcnt(code) is precomputed per
/// vector at build time, `total` stays exact (f32), so the only work per
/// vector is B AND+popcounts per word.
pub struct QuantizedQuery {
    planes: Vec<u64>, // QUERY_BITS * words, plane-major
    words: usize,
    lo: f32,
    step: f32,
    total: f32,
}

/// Quantize an already-rotated query (length `padded`) into bit-planes for the
/// 1-bit popcount fast scan. Free function so `ivf.rs` can build a
/// [`QuantizedQuery`] for its per-list residual scoring without going through
/// a [`RaBitQ`] instance.
pub(crate) fn quantize_rotated_query(y: &[f32], padded: usize) -> QuantizedQuery {
    debug_assert_eq!(y.len(), padded);
    let words = padded / 64;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut total = 0.0f32;
    for &v in y {
        lo = lo.min(v);
        hi = hi.max(v);
        total += v;
    }
    let levels = (1u32 << QUERY_BITS) - 1; // 15
    let range = hi - lo;
    let (step, inv_step) = if range > 1e-12 {
        let s = range / levels as f32;
        (s, 1.0 / s)
    } else {
        (0.0, 0.0) // degenerate: all coords equal; every q_d = 0
    };
    let mut planes = vec![0u64; QUERY_BITS * words];
    for (d, &v) in y.iter().enumerate() {
        let q = (((v - lo) * inv_step).round() as u32).min(levels);
        let w = d / 64;
        let bit = (d % 64) as u32;
        for (j, plane) in planes.chunks_exact_mut(words).enumerate() {
            if (q >> j) & 1 == 1 {
                plane[w] |= 1u64 << bit;
            }
        }
    }
    QuantizedQuery {
        planes,
        words,
        lo,
        step,
        total,
    }
}

/// Score one 1-bit-coded vector (at `slot`) against a quantized query, using
/// the popcount fast path. `factor` and `pc` are that vector's RaBitQ ranking
/// factor and code popcount. Free function so `ivf.rs` can reuse the kernel
/// for per-list residual codes (see [`QuantizedQuery`] for the math).
#[inline]
pub(crate) fn score_bit1_fast_at(
    codes: &[u8],
    bytes_per_vec_codes: usize,
    slot: usize,
    factor: f32,
    pc: u32,
    qq: &QuantizedQuery,
) -> f32 {
    let words = qq.words;
    let base = slot * bytes_per_vec_codes;
    let mut t = [0u32; QUERY_BITS];
    for w in 0..words {
        let b0 = base + w * 8;
        let code = u64::from_le_bytes(codes[b0..b0 + 8].try_into().unwrap());
        for (j, tj) in t.iter_mut().enumerate() {
            *tj += (code & qq.planes[j * words + w]).count_ones();
        }
    }
    let mut weighted = 0.0f32;
    for (j, &tj) in t.iter().enumerate() {
        weighted += ((1u32 << j) * tj) as f32;
    }
    let pos = qq.lo * pc as f32 + qq.step * weighted;
    factor * (2.0 * pos - qq.total)
}

/// 1-bit RaBitQ residual encoder against a *per-vector* (e.g. per-IVF-list)
/// centroid, sharing the same rotation/sign-code math as
/// [`RaBitQBuilder`]'s 1-bit branch but without baking in a single dataset
/// centroid. Used by `ivf.rs` to encode `(v - list_centroid)` per list.
pub(crate) struct Bit1Encoder {
    rotator: Rotator,
    padded: usize,
    bytes_per_vec_codes: usize,
}

impl Bit1Encoder {
    pub(crate) fn new(dim: usize, seed: u64) -> Self {
        let rotator = Rotator::new(seed ^ 0x2B17, dim, 2);
        let padded = rotator.out_dim();
        // The word-wise scan paths assume whole u64 words per vector.
        assert!(padded >= 64, "Bit1Encoder requires dim > 32 (padded >= 64)");
        Bit1Encoder {
            rotator,
            padded,
            bytes_per_vec_codes: padded / 8,
        }
    }

    pub(crate) fn padded(&self) -> usize {
        self.padded
    }

    pub(crate) fn bytes_per_vec(&self) -> usize {
        self.bytes_per_vec_codes
    }

    pub(crate) fn rotator(&self) -> &Rotator {
        &self.rotator
    }

    /// Encode the residual `(v - c)` into 1-bit sign codes: normalize, rotate,
    /// write sign bits LSB-first per byte into `out` (caller-zeroed, length
    /// `bytes_per_vec()`). Returns `(factor, popcount)` with
    /// `factor = norm_r / ||rotated||_1` — identical math to
    /// `RaBitQBuilder`'s 1-bit branch (same epsilon, same sign convention).
    pub(crate) fn encode_residual_into(&self, v: &[f32], c: &[f32], out: &mut [u8]) -> (f32, u32) {
        debug_assert_eq!(v.len(), c.len());
        debug_assert_eq!(out.len(), self.bytes_per_vec_codes);
        let mut residual: Vec<f32> = v.iter().zip(c).map(|(&vi, &ci)| vi - ci).collect();
        let norm_r = l2_norm(&residual).max(1e-9);
        for x in residual.iter_mut() {
            *x /= norm_r;
        }
        let x = self.rotator.apply(&residual); // unit, length padded
        let mut l1 = 0.0f32;
        let mut ones = 0u32;
        for d in 0..self.padded {
            if x[d] >= 0.0 {
                ones += 1;
                out[d / 8] |= 1u8 << (d % 8);
            }
            l1 += x[d].abs();
        }
        (norm_r / l1.max(1e-9), ones)
    }
}

impl RaBitQ {
    pub fn build(vs: &VectorSet, bits: u8, seed: u64) -> Self {
        let dim = vs.dim;
        let n = vs.count();

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

        let mut b = RaBitQBuilder::new(dim, bits, seed, centroid);
        for i in 0..n {
            b.push(vs.get(i));
        }
        b.finish()
    }

    /// Disable/enable the popcount fast path (reference scoring when off).
    pub fn with_fast_scan(mut self, on: bool) -> Self {
        self.fast_scan = on;
        self
    }

    /// Original (unrotated) vector dimensionality. Persisted in `rabitq.idx`
    /// (see [`RaBitQ::save`]/[`RaBitQ::load`]), so a loaded index is
    /// self-describing — the Stage 6 FFI `kx_open` uses this to open the
    /// matching `base.f32` rerank store without the caller passing `dim`.
    pub fn dim(&self) -> usize {
        self.in_dim
    }

    /// The packed code bytes (without the per-vector ranking factors). Used by
    /// the bench to exercise the mmap store on the bulk of the index.
    pub fn code_bytes(&self) -> &[u8] {
        &self.codes
    }

    /// Bytes that must stay resident in RAM at query time: packed codes plus
    /// the per-vector ranking factor and (1-bit only) precomputed popcount.
    pub fn resident_bytes(&self) -> usize {
        self.codes.len() + self.factors.len() * 4 + self.pc.len() * 4
    }

    /// Persist the index to `dir/rabitq.idx` (little-endian, versioned). The
    /// rotation is reconstructed from (seed, dim) on load, so only codes,
    /// factors and popcounts are stored. This is also the v0 on-disk index
    /// format the Stage 6 FFI `kx_open` will consume.
    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::with_capacity(
            8 + 1
                + 4
                + 8
                + 4
                + 4
                + 4
                + 4
                + self.factors.len() * 4
                + self.pc.len() * 4
                + self.codes.len(),
        );
        buf.extend_from_slice(RABITQ_MAGIC);
        buf.push(self.bits);
        buf.extend_from_slice(&(self.in_dim as u32).to_le_bytes());
        buf.extend_from_slice(&self.seed.to_le_bytes());
        buf.extend_from_slice(&(self.count as u32).to_le_bytes());
        buf.extend_from_slice(&self.lo.to_le_bytes());
        buf.extend_from_slice(&self.step.to_le_bytes());
        buf.extend_from_slice(&(self.pc.len() as u32).to_le_bytes());
        for f in &self.factors {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        for p in &self.pc {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        buf.extend_from_slice(&self.codes);
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("rabitq.idx"), buf)
    }

    pub fn load(dir: &std::path::Path) -> std::io::Result<Self> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let buf = std::fs::read(dir.join("rabitq.idx"))?;
        if buf.len() < 37 || &buf[0..8] != RABITQ_MAGIC {
            return Err(bad("not a kortex rabitq index"));
        }
        let bits = buf[8];
        let rd_u32 = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let in_dim = rd_u32(9) as usize;
        let seed = u64::from_le_bytes(buf[13..21].try_into().unwrap());
        let count = rd_u32(21) as usize;
        let lo = f32::from_le_bytes(buf[25..29].try_into().unwrap());
        let step = f32::from_le_bytes(buf[29..33].try_into().unwrap());
        let pc_len = rd_u32(33) as usize;

        let rotator = Rotator::new(seed ^ 0x2B17, in_dim, 2);
        let padded = rotator.out_dim();
        let bytes_per_vec_codes = (padded * bits as usize).div_ceil(8);

        let mut off = 37;
        let need = off + count * 4 + pc_len * 4 + count * bytes_per_vec_codes;
        if buf.len() != need {
            return Err(bad("rabitq index truncated or corrupt"));
        }
        let mut factors = Vec::with_capacity(count);
        for _ in 0..count {
            factors.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        let mut pc = Vec::with_capacity(pc_len);
        for _ in 0..pc_len {
            pc.push(rd_u32(off));
            off += 4;
        }
        let codes = buf[off..].to_vec();

        Ok(RaBitQ {
            count,
            in_dim,
            padded,
            bits,
            seed,
            rotator,
            factors,
            pc,
            codes,
            bytes_per_vec_codes,
            lo,
            step,
            fast_scan: true,
        })
    }

    /// Rotate the raw query into the code basis (orthonormal, so inner products
    /// with rotated data residuals are preserved). The constant centroid term is
    /// dropped — it does not affect ranking.
    fn prep(&self, query: &[f32]) -> Vec<f32> {
        // We rotate the query directly; <residual, query> = <R residual, R query>,
        // and the dataset-centroid offset is a per-query constant we can ignore.
        self.rotator.apply(query)
    }

    /// Quantize a rotated query into bit-planes for the popcount fast scan.
    pub fn quantize_query(&self, y: &[f32]) -> QuantizedQuery {
        quantize_rotated_query(y, self.padded)
    }

    #[inline]
    fn score_one_bit1_fast(&self, qq: &QuantizedQuery, id: usize) -> f32 {
        score_bit1_fast_at(
            &self.codes,
            self.bytes_per_vec_codes,
            id,
            self.factors[id],
            self.pc[id],
            qq,
        )
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
        // codes + 4-byte factor (+ 4-byte popcount for the 1-bit fast scan).
        let pc = if self.bits == 1 { 4.0 } else { 0.0 };
        self.bytes_per_vec_codes as f64 + 4.0 + pc
    }
    fn scores(&self, query: &[f32]) -> Vec<f32> {
        let y = self.prep(query);
        if self.bits == 1 {
            if self.fast_scan {
                let qq = self.quantize_query(&y);
                (0..self.count)
                    .map(|i| self.score_one_bit1_fast(&qq, i))
                    .collect()
            } else {
                let total: f32 = y.iter().sum();
                (0..self.count)
                    .map(|i| self.score_one_bit1(&y, total, i))
                    .collect()
            }
        } else {
            (0..self.count)
                .map(|i| self.score_one_multibit(&y, i))
                .collect()
        }
    }
    fn scores_subset(&self, query: &[f32], ids: &[u32]) -> Vec<f32> {
        let y = self.prep(query);
        if self.bits == 1 {
            if self.fast_scan {
                let qq = self.quantize_query(&y);
                ids.iter()
                    .map(|&i| self.score_one_bit1_fast(&qq, i as usize))
                    .collect()
            } else {
                let total: f32 = y.iter().sum();
                ids.iter()
                    .map(|&i| self.score_one_bit1(&y, total, i as usize))
                    .collect()
            }
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

    /// The popcount fast scan approximates the f32 reference scan with a
    /// 4-bit-quantized query; after the float rerank of the two-stage pipeline
    /// the end-to-end recall must be indistinguishable from the reference.
    #[test]
    fn fast_scan_matches_reference_after_rerank() {
        use crate::search::mean_recall_two_stage;
        let all = synthetic_clusters(21, 256, 2050, 16, 0.45);
        let vs = all.slice_rows(0, 2000);
        let queries = all.slice_rows(2000, 50);
        let precise = FloatStore::build(&vs);

        let fast = RaBitQ::build(&vs, 1, 9); // fast_scan defaults to true
        let refr = RaBitQ::build(&vs, 1, 9).with_fast_scan(false);

        let r_fast = mean_recall_two_stage(&fast, &precise, &vs, &queries, 10, 100);
        let r_ref = mean_recall_two_stage(&refr, &precise, &vs, &queries, 10, 100);
        assert!(
            (r_fast - r_ref).abs() <= 0.005,
            "fast {r_fast} vs reference {r_ref}: query quantization too lossy"
        );
        assert!(r_fast > 0.95, "two-stage recall regressed: {r_fast}");
    }

    /// The coarse stage's actual job is producing a shortlist; the fast scan's
    /// shortlist must overlap the reference scan's heavily. (Raw per-vector
    /// score deltas are allowed to drift a few percent from the 4-bit query —
    /// the binding end-to-end guarantee is the rerank parity test above.)
    #[test]
    fn fast_scan_shortlist_overlaps_reference() {
        let vs = synthetic_clusters(22, 128, 300, 8, 0.4);
        let queries = synthetic_clusters(23, 128, 5, 8, 0.4);
        let fast = RaBitQ::build(&vs, 1, 3);
        let refr = RaBitQ::build(&vs, 1, 3).with_fast_scan(false);
        for qi in 0..queries.count() {
            let q = queries.get(qi);
            let shortlist = 64;
            let sf = fast.search(q, shortlist);
            let sr = refr.search(q, shortlist);
            let overlap = sr.iter().filter(|id| sf.contains(id)).count();
            assert!(
                overlap as f64 / shortlist as f64 >= 0.7,
                "fast-scan shortlist diverged: {overlap}/{shortlist} overlap"
            );
        }
    }

    /// Streaming builder must produce byte-identical codes to the batch build.
    #[test]
    fn builder_matches_batch_build() {
        let vs = synthetic_clusters(24, 256, 400, 8, 0.4);
        let batch = RaBitQ::build(&vs, 1, 5);

        let dim = vs.dim;
        let n = vs.count();
        let mut centroid = vec![0.0f32; dim];
        for i in 0..n {
            for d in 0..dim {
                centroid[d] += vs.get(i)[d];
            }
        }
        for x in centroid.iter_mut() {
            *x /= n as f32;
        }
        let mut b = RaBitQBuilder::new(dim, 1, 5, centroid);
        for i in 0..n {
            b.push(vs.get(i));
        }
        let streamed = b.finish();

        assert_eq!(batch.code_bytes(), streamed.code_bytes());
        assert_eq!(batch.resident_bytes(), streamed.resident_bytes());
        let q = vs.get(0);
        assert_eq!(batch.search(q, 10), streamed.search(q, 10));
    }

    /// save -> load round-trip reproduces identical search results (the
    /// rotation is reconstructed from (seed, dim), everything else is stored).
    #[test]
    fn save_load_roundtrip_is_exact() {
        let all = synthetic_clusters(25, 256, 520, 8, 0.4);
        let vs = all.slice_rows(0, 500);
        let queries = all.slice_rows(500, 20);
        let rq = RaBitQ::build(&vs, 1, 7);

        let dir = std::env::temp_dir().join(format!("kortex_rabitq_io_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        rq.save(&dir).unwrap();
        let loaded = RaBitQ::load(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(rq.count(), loaded.count());
        assert_eq!(rq.code_bytes(), loaded.code_bytes());
        for qi in 0..queries.count() {
            let q = queries.get(qi);
            assert_eq!(rq.search(q, 10), loaded.search(q, 10));
        }
    }
}
