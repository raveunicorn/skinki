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

/// Sort `scored` (score, id) descending by score, then ascending by id, and
/// return the first `k` ids. Shared by the quantizers' rerankers.
pub fn select_top_k(scored: &mut [(f32, u32)], k: usize) -> Vec<u32> {
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.iter().take(k).map(|&(_, id)| id).collect()
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
