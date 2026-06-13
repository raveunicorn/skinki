//! IVF (inverted-file) index with per-list 1-bit RaBitQ residual codes.
//!
//! ## Why IVF
//!
//! The flat 1-bit RaBitQ scan (see [`crate::quant::RaBitQ`]) quantizes
//! residuals against a single GLOBAL dataset centroid. On adversarial
//! geometry (dense clusters far from that centroid) the codes can't
//! discriminate *within* a cluster: the rerank shortlist has to approach the
//! cluster size to reach recall 1.0, blowing the latency budget.
//!
//! IVF fixes discrimination at the root by clustering vectors into `nlist`
//! lists with their own centroids, and quantizing `(v - list_centroid)`
//! instead of `(v - global_centroid)`. Residuals from the *local* centroid
//! are exactly the within-cluster differences the 1-bit codes need to rank.
//! Search then (a) ranks all list centroids exactly against the query (cheap:
//! `nlist` dots), and (b) scans only the `nprobe` best lists' residual codes —
//! cutting the 1-bit scan from "the whole dataset" to "a few lists".
//!
//! Everything is deterministic (seeded SplitMix64 `Rng`), matching the rest of
//! `kortex-vector`.

use crate::exact::select_top_k;
use crate::quant::{quantize_rotated_query, score_bit1_fast_at, Bit1Encoder};
use crate::{dot, Rng, VectorSet};

/// Magic + version for the on-disk IVF index format (see [`IvfRaBitQ::save`]).
const IVF_MAGIC: &[u8; 8] = b"KXIVF001";

/// k-means iterations for both the group-level and per-group training passes.
const KMEANS_ITERS: usize = 8;

// ---------------------------------------------------------------------------
// k-means (full-dim, deterministic)
// ---------------------------------------------------------------------------

/// Squared L2 distance between `a` and `b`.
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Index of the nearest centroid (by L2) to `v` among `centroids` (k rows of
/// `vs.dim`).
fn nearest(v: &[f32], centroids: &VectorSet) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..centroids.count() {
        let d = l2_sq(v, centroids.get(c));
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

/// Plain full-dimension k-means, modeled on `quant::kmeans_subspace`: init from
/// distinct random sample rows (reseeding on duplicates), `KMEANS_ITERS`
/// iterations of L2-nearest assignment + mean update, empty clusters reseeded
/// from a random sample row. `k` must be `>= 1` and `<= rows.count()`.
fn kmeans(rows: &VectorSet, k: usize, seed: u64) -> VectorSet {
    let dim = rows.dim;
    let n = rows.count();
    debug_assert!(k >= 1 && k <= n, "kmeans: k={k} out of range for n={n}");
    let mut r = Rng::new(seed ^ 0xA17F);

    // Init: distinct random sample rows (redraw on duplicate index).
    let mut centroids = VectorSet::new(dim);
    let mut used = std::collections::HashSet::new();
    for _ in 0..k {
        let mut idx = r.below(n);
        // Bounded redraw for distinctness; if the set is exhausted (k == n)
        // this loop falls through and allows a repeat, which is harmless.
        let mut tries = 0;
        while used.contains(&idx) && tries < n {
            idx = r.below(n);
            tries += 1;
        }
        used.insert(idx);
        centroids.push(rows.get(idx));
    }

    let mut counts = vec![0u32; k];
    let mut sums = vec![0.0f32; k * dim];
    for _ in 0..KMEANS_ITERS {
        for x in sums.iter_mut() {
            *x = 0.0;
        }
        for c in counts.iter_mut() {
            *c = 0;
        }
        for i in 0..n {
            let v = rows.get(i);
            let c = nearest(v, &centroids);
            counts[c] += 1;
            let dst = &mut sums[c * dim..(c + 1) * dim];
            for d in 0..dim {
                dst[d] += v[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                let dst = &mut centroids.data[c * dim..(c + 1) * dim];
                for d in 0..dim {
                    dst[d] = sums[c * dim + d] * inv;
                }
            } else {
                // Reseed an empty centroid from a random sample row.
                let idx = r.below(n);
                centroids.data[c * dim..(c + 1) * dim].copy_from_slice(rows.get(idx));
            }
        }
    }
    centroids
}

// ---------------------------------------------------------------------------
// IvfBuilder — hierarchical k-means training + two-pass build
// ---------------------------------------------------------------------------

/// Trained IVF list centroids, plus the group structure used to assign new
/// vectors to a list, plus the running build-time state for the two-pass
/// streaming build (`assign` -> `finalize_layout` -> `encode` -> `finish`).
pub struct IvfBuilder {
    dim: usize,
    nlist: usize,
    seed: u64,
    /// All `nlist` list centroids, concatenated in group order.
    centroids: VectorSet,
    /// Group centers from the top-level k-means (len `g`).
    group_centers: VectorSet,
    /// `[start, end)` range into `centroids` for each group, in group order.
    group_ranges: Vec<(usize, usize)>,
    encoder: Bit1Encoder,

    // Build-time state.
    /// Pass A: list assigned to each vector, in input order.
    assignments: Vec<u32>,
    /// Pass A->layout: count of vectors per list.
    counts: Vec<u32>,
    /// CSR offsets, len nlist+1 (filled by `finalize_layout`).
    offsets: Vec<u32>,
    /// Pass B write cursors per list (slot index, advances as vectors land).
    cursors: Vec<u32>,
    /// Pass B: count of `encode` calls so far == index into `assignments`.
    encoded: usize,
    /// Pass B output, pre-sized by `finalize_layout`.
    ids: Vec<u32>,
    factors: Vec<f32>,
    pc: Vec<u32>,
    codes: Vec<u8>,
}

/// Heuristic default for `nlist` when the caller passes 0: a few nearest
/// neighbors per probed list at typical recall settings wants
/// `nlist ~ sqrt(n)`-ish; `4*sqrt(n)` keeps lists small (good discrimination)
/// without exploding centroid-scan cost. Clamped to a sane range and to the
/// available training sample.
fn auto_nlist(expected_n: usize, sample_count: usize) -> usize {
    let heuristic = (4.0 * (expected_n as f64).sqrt()) as usize;
    heuristic.clamp(64, 65536).min(sample_count.max(1))
}

impl IvfBuilder {
    /// Train list centroids on a sample via hierarchical k-means:
    /// 1. `g = ceil(sqrt(nlist))` top-level groups via `kmeans(sample, g, ...)`.
    /// 2. Partition the sample by nearest group center.
    /// 3. Split `nlist` across groups proportional to their sample share
    ///    (deterministic rounding, see [`split_ki`]).
    /// 4. Per group, `kmeans(group_rows, ki, ...)` -> that group's list
    ///    centroids, concatenated in group order.
    ///
    /// `nlist == 0` triggers [`auto_nlist`]. `nlist` is clamped to
    /// `sample.count()` (can't have more lists than training rows).
    pub fn train(
        dim: usize,
        nlist: usize,
        seed: u64,
        sample: &VectorSet,
        expected_n: usize,
    ) -> Self {
        assert_eq!(sample.dim, dim, "sample dim mismatch");
        let n = sample.count();
        assert!(n >= 1, "training sample must be non-empty");
        let nlist = if nlist == 0 {
            auto_nlist(expected_n, n)
        } else {
            nlist.min(n)
        }
        .max(1);

        let g = (nlist as f64).sqrt().ceil() as usize;
        let g = g.clamp(1, nlist).min(n);

        let group_centers = kmeans(sample, g, seed ^ 0xA11);

        // Partition sample row indices by nearest group center.
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); g];
        for i in 0..n {
            let gi = nearest(sample.get(i), &group_centers);
            groups[gi].push(i);
        }

        let sizes: Vec<usize> = groups.iter().map(|gi| gi.len()).collect();
        let kis = split_ki(nlist, &sizes);

        // Per-group k-means -> concatenate into the global centroid list.
        let mut centroids = VectorSet::new(dim);
        let mut group_ranges = Vec::with_capacity(g);
        for (gi, rows_idx) in groups.iter().enumerate() {
            let ki = kis[gi];
            let start = centroids.count();
            if ki == 0 || rows_idx.is_empty() {
                // No sample rows landed in this group: ki was forced to 0 by
                // the redistribution (only possible when sizes[gi] == 0, in
                // which case split_ki never assigns it > 0). Nothing to do.
                group_ranges.push((start, start));
                continue;
            }
            let mut group_rows = VectorSet::new(dim);
            for &idx in rows_idx {
                group_rows.push(sample.get(idx));
            }
            let group_centroids = kmeans(&group_rows, ki, seed ^ 0xB22 ^ (gi as u64));
            for c in 0..group_centroids.count() {
                centroids.push(group_centroids.get(c));
            }
            group_ranges.push((start, centroids.count()));
        }
        let nlist = centroids.count();

        let encoder = Bit1Encoder::new(dim, seed);

        IvfBuilder {
            dim,
            nlist,
            seed,
            centroids,
            group_centers,
            group_ranges,
            encoder,
            assignments: Vec::new(),
            counts: vec![0u32; nlist],
            offsets: Vec::new(),
            cursors: Vec::new(),
            encoded: 0,
            ids: Vec::new(),
            factors: Vec::new(),
            pc: Vec::new(),
            codes: Vec::new(),
        }
    }

    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// Pass A: assign one vector (in streaming order) to its nearest list.
    /// Two-level approximate assignment — nearest group center, then nearest
    /// list centroid within that group's centroid range — avoiding an exact
    /// `n * nlist` scan. Search-side list ranking (in [`IvfRaBitQ::shortlist`])
    /// stays exact, so this approximation only affects which list a vector
    /// lands in, not how lists are ranked at query time.
    pub fn assign(&mut self, v: &[f32]) {
        let gi = nearest(v, &self.group_centers);
        let (start, end) = self.group_ranges[gi];
        let list = if start < end {
            start + nearest_in_range(v, &self.centroids, start, end)
        } else {
            // Empty group (can only happen if its ki == 0, i.e. it had no
            // sample rows): fall back to a global nearest-centroid search.
            nearest(v, &self.centroids)
        };
        self.counts[list] += 1;
        self.assignments.push(list as u32);
    }

    /// Between passes: turn per-list counts into CSR offsets and pre-size the
    /// pass-B output arrays.
    pub fn finalize_layout(&mut self) {
        let mut offsets = vec![0u32; self.nlist + 1];
        for l in 0..self.nlist {
            offsets[l + 1] = offsets[l] + self.counts[l];
        }
        let total = offsets[self.nlist] as usize;
        self.cursors = offsets[..self.nlist].to_vec();
        self.ids = vec![0u32; total];
        self.factors = vec![0.0f32; total];
        self.pc = vec![0u32; total];
        self.codes = vec![0u8; total * self.encoder.bytes_per_vec()];
        self.offsets = offsets;
    }

    /// Pass B: encode one vector (same order as `assign`) into its reserved
    /// slot — residual against its assigned list centroid via [`Bit1Encoder`].
    pub fn encode(&mut self, v: &[f32]) {
        let row = self.encoded;
        self.encoded += 1;
        let list = self.assignments[row] as usize;
        let slot = self.cursors[list] as usize;
        self.cursors[list] += 1;

        let bpv = self.encoder.bytes_per_vec();
        let centroid = self.centroids.get(list);
        let out = &mut self.codes[slot * bpv..(slot + 1) * bpv];
        let (factor, pc) = self.encoder.encode_residual_into(v, centroid, out);
        self.factors[slot] = factor;
        self.pc[slot] = pc;
        self.ids[slot] = row as u32;
    }

    pub fn finish(self) -> IvfRaBitQ {
        for l in 0..self.nlist {
            assert_eq!(
                self.cursors[l],
                self.offsets[l + 1],
                "IVF list {l} not exactly filled: cursor {} != end {}",
                self.cursors[l],
                self.offsets[l + 1]
            );
        }
        let count = self.ids.len();
        IvfRaBitQ {
            dim: self.dim,
            nlist: self.nlist,
            seed: self.seed,
            count,
            centroids: self.centroids,
            offsets: self.offsets,
            ids: self.ids,
            factors: self.factors,
            pc: self.pc,
            codes: self.codes,
            encoder: self.encoder,
        }
    }
}

/// Nearest centroid index (relative to `start`, i.e. in `0..(end-start)`)
/// among `centroids.get(start..end)`.
fn nearest_in_range(v: &[f32], centroids: &VectorSet, start: usize, end: usize) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in start..end {
        let d = l2_sq(v, centroids.get(c));
        if d < best_d {
            best_d = d;
            best = c - start;
        }
    }
    best
}

/// Split `nlist` slots across groups of the given `sizes`, proportional to
/// `size_i / sum(sizes)`, with **deterministic** rounding so `sum(ki) ==
/// nlist` exactly:
///
/// 1. `ki = round(nlist * size_i / total)`.
/// 2. Fix the rounding remainder by adding/subtracting 1, one group at a
///    time, in order of `(size desc, index asc)`.
/// 3. Clamp each `ki` to `[1, max(1, size_i)]` (every non-empty-by-design
///    group gets at least one list; a group can't have more lists than
///    sample rows). Empty groups (`size_i == 0`) are clamped to `0` instead —
///    there's nothing to train list centroids from.
/// 4. If clamping changed the sum, redistribute the remainder again, same
///    deterministic order, only touching groups that still have headroom.
fn split_ki(nlist: usize, sizes: &[usize]) -> Vec<usize> {
    let g = sizes.len();
    let total: usize = sizes.iter().sum();
    if total == 0 {
        // Degenerate (shouldn't happen: training sample is non-empty), but
        // keep the function total: spread evenly.
        let mut out = vec![0usize; g];
        for (i, o) in out.iter_mut().enumerate().take(nlist) {
            let _ = i;
            *o = 1;
        }
        return out;
    }

    // Deterministic priority order: (size desc, index asc).
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));

    // Bounds: a group with size 0 is forced to ki=0 (nothing to train on);
    // every other group must keep at least 1 list and at most `size` lists.
    let lo = |s: usize| -> i64 {
        if s == 0 {
            0
        } else {
            1
        }
    };
    let hi = |s: usize| -> i64 { (s as i64).max(1) };

    let mut ki: Vec<i64> = sizes
        .iter()
        .map(|&s| ((nlist as f64) * (s as f64) / (total as f64)).round() as i64)
        .collect();

    // Initial rounding fix: bring sum(ki) to nlist exactly, bounded by [lo,hi]
    // even on this first pass so later clamping never has to undo it.
    for i in 0..g {
        ki[i] = ki[i].clamp(lo(sizes[i]), hi(sizes[i]));
    }
    redistribute(&mut ki, nlist as i64, &order, &lo, &hi, sizes);
    // Clamp + redistribute again is a no-op once feasible, but cheap and
    // makes the function robust to any future change in the rounding step.
    for i in 0..g {
        ki[i] = ki[i].clamp(lo(sizes[i]), hi(sizes[i]));
    }
    redistribute(&mut ki, nlist as i64, &order, &lo, &hi, sizes);

    let sum: i64 = ki.iter().sum();
    debug_assert_eq!(sum, nlist as i64, "split_ki must sum to nlist");

    ki.into_iter().map(|k| k.max(0) as usize).collect()
}

/// Add/subtract 1 from groups in `order` (cycling) until `ki.sum() == target`,
/// respecting per-group bounds `[lo(sizes[i]), hi(sizes[i])]`. Terminates: a
/// full pass over `order` with zero progress means no group has headroom,
/// which only happens if `target` is outside `[sum(lo), sum(hi)]` — and
/// `sum(lo) <= g <= nlist <= sum(hi)` always holds for our inputs (every
/// group has `hi >= 1`, and `nlist <= total = sum(sizes) <= sum(hi)`).
fn redistribute(
    ki: &mut [i64],
    target: i64,
    order: &[usize],
    lo: &impl Fn(usize) -> i64,
    hi: &impl Fn(usize) -> i64,
    sizes: &[usize],
) {
    let mut diff = target - ki.iter().sum::<i64>();
    let g = order.len();
    if diff == 0 || g == 0 {
        return;
    }
    let mut idx = 0usize;
    let mut since_progress = 0usize;
    while diff != 0 && since_progress < g {
        let i = order[idx % g];
        if diff > 0 {
            if ki[i] < hi(sizes[i]) {
                ki[i] += 1;
                diff -= 1;
                since_progress = 0;
            } else {
                since_progress += 1;
            }
        } else if ki[i] > lo(sizes[i]) {
            ki[i] -= 1;
            diff += 1;
            since_progress = 0;
        } else {
            since_progress += 1;
        }
        idx += 1;
    }
    debug_assert_eq!(diff, 0, "redistribute could not reach target");
}

// ---------------------------------------------------------------------------
// IvfRaBitQ — the trained, encoded index
// ---------------------------------------------------------------------------

pub struct IvfRaBitQ {
    dim: usize,
    nlist: usize,
    seed: u64,
    count: usize,
    /// `nlist` rows of `dim`.
    centroids: VectorSet,
    /// CSR offsets into the grouped arrays, len `nlist + 1`.
    offsets: Vec<u32>,
    /// Original vector id per slot, len `count`, grouped by list.
    ids: Vec<u32>,
    /// Per-slot RaBitQ ranking factor, grouped by list.
    factors: Vec<f32>,
    /// Per-slot 1-bit code popcount, grouped by list.
    pc: Vec<u32>,
    /// `count * bytes_per_vec` packed 1-bit residual codes, grouped by list.
    codes: Vec<u8>,
    /// Shared rotation (built from `(dim, seed)`, reconstructible on load).
    encoder: Bit1Encoder,
}

impl IvfRaBitQ {
    pub fn count(&self) -> usize {
        self.count
    }

    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// Bytes that must stay resident in RAM at query time: codes + per-slot
    /// factors/popcounts/ids + the list centroids + CSR offsets. The `ids`
    /// array (4 B/vector) is an IVF-specific cost the flat index doesn't pay
    /// — counted honestly here.
    pub fn resident_bytes(&self) -> usize {
        self.codes.len()
            + self.factors.len() * 4
            + self.pc.len() * 4
            + self.ids.len() * 4
            + self.centroids.data.len() * 4
            + self.offsets.len() * 4
    }

    /// Resident bytes **per indexed vector** — the part of [`resident_bytes`]
    /// that scales linearly with `n`: the 1-bit code plus its per-slot ranking
    /// factor (4 B), popcount (4 B), and original-id (4 B). N-independent.
    pub fn per_vec_resident_bytes(&self) -> usize {
        self.encoder.bytes_per_vec() + 4 + 4 + 4
    }

    /// Honest projection of resident RAM to a target corpus size `n_target`.
    ///
    /// Projecting the *measured* total linearly (`total * n_target / n`) is
    /// correct for the flat index — every byte there scales with `n` — but
    /// **over-counts IVF at scale**: the per-slot arrays scale with `n`, yet the
    /// centroid table and CSR offsets scale with `nlist ~ sqrt(n)`. Measured at
    /// a small `n` the centroid table dominates (tens of B/vec), so a linear
    /// projection invents RAM that the sqrt-growth never spends. This splits the
    /// two terms: linear per-vector cost, plus the centroid/offset cost at the
    /// `nlist` [`auto_nlist`] would pick for `n_target`.
    pub fn resident_bytes_at(&self, n_target: usize) -> usize {
        let nlist_target = auto_nlist(n_target, n_target);
        self.per_vec_resident_bytes() * n_target
            + nlist_target * self.dim * 4 // centroids
            + (nlist_target + 1) * 4 // CSR offsets
    }

    /// Coarse shortlist of original vector ids, best-first, scanning only the
    /// `nprobe` lists whose centroids best match `query`.
    ///
    /// `<v,q> = <c_l,q> + <r,q>`, where `r = v - c_l` is the residual the
    /// 1-bit code approximates `<r,q>` for. So a slot's score is the exact
    /// list-centroid term `s_l = <c_l,q>` plus the 1-bit residual estimate.
    pub fn shortlist(&self, query: &[f32], nprobe: usize, shortlist_len: usize) -> Vec<u32> {
        let y = self.encoder.rotator().apply(query);
        let qq = quantize_rotated_query(&y, self.encoder.padded());
        let bpv = self.encoder.bytes_per_vec();

        // Exact list ranking: nlist dots (cheap — ~2M flops at nlist=8192,
        // dim=256 — versus scanning the whole dataset).
        let mut list_scores: Vec<(f32, u32)> = (0..self.nlist)
            .map(|l| (dot(self.centroids.get(l), query), l as u32))
            .collect();
        let probed = select_top_k(&mut list_scores, nprobe.min(self.nlist));

        // Re-derive each probed list's exact centroid score (select_top_k
        // discards it) and scan that list's residual codes.
        let mut scored: Vec<(f32, u32)> = Vec::new();
        for &l in &probed {
            let l = l as usize;
            let s_l = dot(self.centroids.get(l), query);
            let start = self.offsets[l] as usize;
            let end = self.offsets[l + 1] as usize;
            for slot in start..end {
                let est = score_bit1_fast_at(
                    &self.codes,
                    bpv,
                    slot,
                    self.factors[slot],
                    self.pc[slot],
                    &qq,
                );
                scored.push((s_l + est, self.ids[slot]));
            }
        }
        select_top_k(&mut scored, shortlist_len)
    }

    /// Persist to `dir/ivf.idx` (little-endian, versioned). The rotation is
    /// reconstructed from `(dim, seed)` on load; group structure (only needed
    /// for build-time assignment) is not persisted.
    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::with_capacity(
            8 + 4
                + 4
                + 4
                + 8
                + self.centroids.data.len() * 4
                + self.offsets.len() * 4
                + self.ids.len() * 4
                + self.factors.len() * 4
                + self.pc.len() * 4
                + self.codes.len(),
        );
        buf.extend_from_slice(IVF_MAGIC);
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.nlist as u32).to_le_bytes());
        buf.extend_from_slice(&(self.count as u32).to_le_bytes());
        buf.extend_from_slice(&self.seed.to_le_bytes());
        for x in &self.centroids.data {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for o in &self.offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        for id in &self.ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        for f in &self.factors {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        for p in &self.pc {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        buf.extend_from_slice(&self.codes);
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("ivf.idx"), buf)
    }

    pub fn load(dir: &std::path::Path) -> std::io::Result<Self> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let buf = std::fs::read(dir.join("ivf.idx"))?;
        const HEADER: usize = 8 + 4 + 4 + 4 + 8;
        if buf.len() < HEADER || &buf[0..8] != IVF_MAGIC {
            return Err(bad("not a kortex ivf index"));
        }
        let rd_u32 = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let dim = rd_u32(8) as usize;
        let nlist = rd_u32(12) as usize;
        let count = rd_u32(16) as usize;
        let seed = u64::from_le_bytes(buf[20..28].try_into().unwrap());

        let encoder = Bit1Encoder::new(dim, seed);
        let bpv = encoder.bytes_per_vec();

        let mut off = HEADER;
        let centroids_len = nlist * dim;
        let offsets_len = nlist + 1;
        let need = off
            + centroids_len * 4
            + offsets_len * 4
            + count * 4 // ids
            + count * 4 // factors
            + count * 4 // pc
            + count * bpv; // codes
        if buf.len() != need {
            return Err(bad("ivf index truncated or corrupt"));
        }

        let mut centroids = VectorSet::new(dim);
        centroids.data.reserve(centroids_len);
        for _ in 0..centroids_len {
            centroids
                .data
                .push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        let mut offsets = Vec::with_capacity(offsets_len);
        for _ in 0..offsets_len {
            offsets.push(rd_u32(off));
            off += 4;
        }
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(rd_u32(off));
            off += 4;
        }
        let mut factors = Vec::with_capacity(count);
        for _ in 0..count {
            factors.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        let mut pc = Vec::with_capacity(count);
        for _ in 0..count {
            pc.push(rd_u32(off));
            off += 4;
        }
        let codes = buf[off..].to_vec();

        Ok(IvfRaBitQ {
            dim,
            nlist,
            seed,
            count,
            centroids,
            offsets,
            ids,
            factors,
            pc,
            codes,
            encoder,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::synthetic_clusters;
    use crate::exact::top_k;
    use crate::quant::{FloatStore, Quantizer};
    use crate::search::ivf_two_stage_search;

    /// Build an IVF index over `vs` end-to-end (train -> assign -> finalize ->
    /// encode -> finish), driven through the streaming-builder API the
    /// harness uses.
    fn build_ivf(vs: &VectorSet, nlist: usize, seed: u64) -> IvfRaBitQ {
        let mut b = IvfBuilder::train(vs.dim, nlist, seed, vs, vs.count());
        for i in 0..vs.count() {
            b.assign(vs.get(i));
        }
        b.finalize_layout();
        for i in 0..vs.count() {
            b.encode(vs.get(i));
        }
        b.finish()
    }

    #[test]
    fn csr_integrity() {
        let vs = synthetic_clusters(101, 64, 2000, 16, 0.4);
        let ivf = build_ivf(&vs, 32, 7);

        assert_eq!(ivf.offsets[0], 0);
        assert_eq!(*ivf.offsets.last().unwrap(), vs.count() as u32);
        for w in ivf.offsets.windows(2) {
            assert!(w[0] <= w[1], "offsets must be monotonic non-decreasing");
        }

        let mut ids = ivf.ids.clone();
        ids.sort_unstable();
        let expected: Vec<u32> = (0..vs.count() as u32).collect();
        assert_eq!(ids, expected, "ids must be a permutation of 0..n");
    }

    #[test]
    fn determinism() {
        let vs = synthetic_clusters(102, 64, 2000, 16, 0.4);
        let a = build_ivf(&vs, 32, 11);
        let b = build_ivf(&vs, 32, 11);

        assert_eq!(a.codes, b.codes);
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.offsets, b.offsets);
        assert_eq!(a.factors, b.factors);
        assert_eq!(a.pc, b.pc);
        assert_eq!(a.centroids.data, b.centroids.data);
    }

    #[test]
    fn save_load_roundtrip() {
        let all = synthetic_clusters(103, 128, 2050, 16, 0.4);
        let vs = all.slice_rows(0, 2000);
        let queries = all.slice_rows(2000, 20);
        let ivf = build_ivf(&vs, 32, 13);

        let dir = std::env::temp_dir().join(format!("kortex_ivf_io_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        ivf.save(&dir).unwrap();
        let loaded = IvfRaBitQ::load(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(ivf.count(), loaded.count());
        assert_eq!(ivf.nlist(), loaded.nlist());
        for qi in 0..queries.count() {
            let q = queries.get(qi);
            assert_eq!(
                ivf.shortlist(q, ivf.nlist(), 50),
                loaded.shortlist(q, loaded.nlist(), 50)
            );
        }
    }

    fn mean_recall_ivf(
        ivf: &IvfRaBitQ,
        precise: &dyn Quantizer,
        vs: &VectorSet,
        queries: &VectorSet,
        k: usize,
        nprobe: usize,
        refine: usize,
    ) -> f64 {
        let mut acc = 0.0;
        let n = queries.count();
        for qi in 0..n {
            let q = queries.get(qi);
            let truth = top_k(vs, q, k);
            let got = ivf_two_stage_search(ivf, precise, q, k, nprobe, refine);
            acc += crate::search::recall(&got, &truth);
        }
        if n == 0 {
            0.0
        } else {
            acc / n as f64
        }
    }

    #[test]
    fn recall_on_separable_data() {
        let all = synthetic_clusters(104, 128, 2050, 16, 0.45);
        let vs = all.slice_rows(0, 2000);
        let queries = all.slice_rows(2000, 50);
        let ivf = build_ivf(&vs, 32, 17);
        let precise = FloatStore::build(&vs);

        let r_full = mean_recall_ivf(&ivf, &precise, &vs, &queries, 10, ivf.nlist(), 100);
        assert!(r_full >= 0.95, "nprobe=nlist recall too low: {r_full}");

        let nprobe = (ivf.nlist() / 8).max(1);
        let r_partial = mean_recall_ivf(&ivf, &precise, &vs, &queries, 10, nprobe, 100);
        assert!(
            r_partial >= 0.85,
            "nprobe={nprobe} recall too low: {r_partial}"
        );
    }

    #[test]
    fn resident_projection_is_sublinear_in_centroids() {
        // Two indexes over the same per-vector cost but different n: the honest
        // 5M projection must NOT scale the (sqrt-growing) centroid table
        // linearly. A naive `total * 5M/n` from a small index over-counts; the
        // split projection lands close to the true per-vec-dominated cost.
        let vs = synthetic_clusters(106, 256, 4000, 16, 0.4);
        let ivf = build_ivf(&vs, 0, 23); // auto nlist

        let per_vec = ivf.per_vec_resident_bytes();
        let proj_5m = ivf.resident_bytes_at(5_000_000);
        let naive_linear = (ivf.resident_bytes() as f64 * 5_000_000.0 / vs.count() as f64) as usize;

        // The honest projection is dominated by the linear per-vec term...
        assert!(proj_5m >= per_vec * 5_000_000);
        // ...and the centroid overhead at 5M is small (< 5 B/vec here), so the
        // honest projection is well under the naive linear blow-up that drags
        // the small-N centroid table across all 5M vectors.
        assert!(
            proj_5m < per_vec * 5_000_000 + 5 * 5_000_000,
            "centroid overhead at 5M unexpectedly large: {proj_5m} vs {}",
            per_vec * 5_000_000
        );
        assert!(
            (proj_5m as f64) < 0.5 * naive_linear as f64,
            "split projection {proj_5m} should be far below naive linear {naive_linear}"
        );
    }

    #[test]
    fn nprobe_full_equals_flat_quality() {
        let all = synthetic_clusters(105, 64, 1050, 16, 0.4);
        let vs = all.slice_rows(0, 1000);
        let queries = all.slice_rows(1000, 10);
        let ivf = build_ivf(&vs, 16, 19);
        let precise = FloatStore::build(&vs);

        for qi in 0..queries.count() {
            let q = queries.get(qi);
            let truth = top_k(&vs, q, 10);
            let got = ivf_two_stage_search(&ivf, &precise, q, 10, ivf.nlist(), vs.count());
            assert_eq!(
                got, truth,
                "full-probe + full-rerank must match exact top-k"
            );
        }
    }
}
