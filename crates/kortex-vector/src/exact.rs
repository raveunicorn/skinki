//! Exact brute-force top-k over float32 vectors.
//!
//! This is the *ground truth* for Stage 1: every compressed codec's recall is
//! measured against the neighbors this returns. It is intentionally simple and
//! obviously correct — speed is not the point here, fidelity is.

use crate::{dot, VectorSet};

/// Return the indices of the `k` highest inner-product matches to `query`,
/// best first. Ties are broken by smaller index for determinism.
pub fn top_k(data: &VectorSet, query: &[f32], k: usize) -> Vec<u32> {
    let n = data.count();
    let mut scored: Vec<(f32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        scored.push((dot(data.get(i), query), i as u32));
    }
    select_top_k(&mut scored, k)
}

/// Return the `k` best ids from `scored` (score desc, ties by id asc).
///
/// Uses an O(n) partition (`select_nth_unstable_by`) before sorting only the
/// head — a full sort of millions of candidates per query was the dominant
/// query cost at scale (~100 ms at 1M), not the scan itself.
pub fn select_top_k(scored: &mut [(f32, u32)], k: usize) -> Vec<u32> {
    let n = scored.len();
    if k == 0 || n == 0 {
        return Vec::new();
    }
    let cmp = |a: &(f32, u32), b: &(f32, u32)| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    };
    let k = k.min(n);
    if k < n {
        scored.select_nth_unstable_by(k - 1, cmp);
    }
    let head = &mut scored[..k];
    head.sort_by(cmp);
    head.iter().map(|&(_, id)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_finds_the_identical_vector_first() {
        let mut vs = VectorSet::new(3);
        vs.push(&[1.0, 0.0, 0.0]);
        vs.push(&[0.0, 1.0, 0.0]);
        vs.push(&[0.9, 0.1, 0.0]);
        let res = top_k(&vs, &[1.0, 0.0, 0.0], 2);
        assert_eq!(res[0], 0);
        assert_eq!(res[1], 2);
    }

    #[test]
    fn ties_break_by_index() {
        let mut s = vec![(1.0f32, 5u32), (1.0, 2), (1.0, 9)];
        let res = select_top_k(&mut s, 3);
        assert_eq!(res, vec![2, 5, 9]);
    }
}
