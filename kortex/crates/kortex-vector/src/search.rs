//! Retrieval strategies and recall measurement.
//!
//! Single-stage search is just `Quantizer::search`. The interesting strategy is
//! **two-stage**: a cheap, highly-compressed codec produces a shortlist of
//! candidates with one fast scan, then a more precise codec re-scores only that
//! shortlist. This is how we buy back the recall lost to aggressive compression
//! without paying full-precision RAM or latency.

use crate::exact::{select_top_k, top_k};
use crate::quant::Quantizer;
use crate::VectorSet;

/// Two-stage retrieval: `coarse` shortlists `refine` candidates, `precise`
/// re-ranks them down to top-`k`.
pub fn two_stage_search(
    coarse: &dyn Quantizer,
    precise: &dyn Quantizer,
    query: &[f32],
    k: usize,
    refine: usize,
) -> Vec<u32> {
    let shortlist = coarse.search(query, refine.max(k));
    let scores = precise.scores_subset(query, &shortlist);
    let mut scored: Vec<(f32, u32)> = scores.into_iter().zip(shortlist.iter().copied()).collect();
    select_top_k(&mut scored, k)
}

/// recall@k of `got` against the exact `truth` (set overlap / k).
pub fn recall(got: &[u32], truth: &[u32]) -> f64 {
    if truth.is_empty() {
        return 0.0;
    }
    let hits = truth.iter().filter(|t| got.contains(t)).count();
    hits as f64 / truth.len() as f64
}

/// Mean recall@k of a single-stage quantizer over a query set, vs exact float32.
pub fn mean_recall_single(
    q: &dyn Quantizer,
    base: &VectorSet,
    queries: &VectorSet,
    k: usize,
) -> f64 {
    let mut acc = 0.0;
    let n = queries.count();
    for qi in 0..n {
        let query = queries.get(qi);
        let truth = top_k(base, query, k);
        acc += recall(&q.search(query, k), &truth);
    }
    if n == 0 {
        0.0
    } else {
        acc / n as f64
    }
}

/// Mean recall@k of a two-stage pipeline over a query set, vs exact float32.
pub fn mean_recall_two_stage(
    coarse: &dyn Quantizer,
    precise: &dyn Quantizer,
    base: &VectorSet,
    queries: &VectorSet,
    k: usize,
    refine: usize,
) -> f64 {
    let mut acc = 0.0;
    let n = queries.count();
    for qi in 0..n {
        let query = queries.get(qi);
        let truth = top_k(base, query, k);
        let got = two_stage_search(coarse, precise, query, k, refine);
        acc += recall(&got, &truth);
    }
    if n == 0 {
        0.0
    } else {
        acc / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::synthetic_clusters;
    use crate::quant::{FloatStore, RaBitQ};

    #[test]
    fn recall_overlap() {
        assert!((recall(&[1, 2, 3], &[2, 3, 4]) - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(recall(&[], &[1]), 0.0);
    }

    #[test]
    fn two_stage_recovers_recall_over_coarse_alone() {
        let all = synthetic_clusters(11, 256, 1550, 12, 0.4);
        let vs = all.slice_rows(0, 1500);
        let queries = all.slice_rows(1500, 50);
        let coarse = RaBitQ::build(&vs, 1, 5);
        let precise = FloatStore::build(&vs);

        let single = mean_recall_single(&coarse, &vs, &queries, 10);
        let two = mean_recall_two_stage(&coarse, &precise, &vs, &queries, 10, 100);
        // Reranking a 1-bit shortlist with float should match exact (recall ~1)
        // and never be worse than the coarse stage alone.
        assert!(two >= single, "two-stage {two} < coarse {single}");
        assert!(two > 0.95, "two-stage recall too low: {two}");
    }
}
