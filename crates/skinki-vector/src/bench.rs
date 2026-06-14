//! Stage 1 benchmark orchestration.
//!
//! Runs the codec matrix over a base vector set + query set, measuring recall vs
//! exact float32, per-vector footprint (projected to 1M/5M vectors), and query
//! latency. Emits a serde report plus a console table, and applies the Stage 1
//! decision gate (recall >= 95% within the RAM budget).

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::exact::top_k;
use crate::quant::{FloatStore, ProductQuantizer, Quantizer, RaBitQ, ScalarI8};
use crate::search::{recall, two_stage_search};
use crate::store::CodeStore;
use crate::VectorSet;
use skinki_telemetry::LatencySummary;

const MIB: f64 = (1usize << 20) as f64;

/// Stage 1 fitness budgets (see ROADMAP.md).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budgets {
    pub recall_at_k: f64,
    pub idle_ram_mb_at_5m: f64,
    pub p95_ms: f64,
}

impl Default for Budgets {
    fn default() -> Self {
        Budgets {
            recall_at_k: 0.95,
            idle_ram_mb_at_5m: 250.0,
            p95_ms: 150.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecResult {
    pub name: String,
    pub dim: usize,
    pub bytes_per_vector: f64,
    pub compression_x: f64,
    pub recall_at_k: f64,
    pub build_ms: f64,
    pub latency: LatencySummary,
    pub projected_ram_mb_1m: f64,
    pub projected_ram_mb_5m: f64,
    pub gate_pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoStageResult {
    pub coarse: String,
    pub precise: String,
    pub refine: usize,
    pub resident_bytes_per_vector: f64,
    pub recall_at_k: f64,
    pub latency: LatencySummary,
    pub projected_resident_mb_5m: f64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmapCheck {
    pub supported: bool,
    pub code_bytes: usize,
    pub recall_matches_ram: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub vectors: usize,
    pub queries: usize,
    pub k: usize,
    pub dim_full: usize,
    pub budgets: Budgets,
    pub single: Vec<CodecResult>,
    pub two_stage: Vec<TwoStageResult>,
    pub mmap: MmapCheck,
}

fn projected_mb(bytes_per_vector: f64, n: f64) -> f64 {
    bytes_per_vector * n / MIB
}

/// Time single-stage queries and gather latencies.
fn timed_single(
    q: &dyn Quantizer,
    queries: &VectorSet,
    k: usize,
) -> (Vec<Vec<u32>>, Vec<Duration>) {
    let mut results = Vec::with_capacity(queries.count());
    let mut durations = Vec::with_capacity(queries.count());
    for qi in 0..queries.count() {
        let query = queries.get(qi);
        let start = Instant::now();
        let got = q.search(query, k);
        durations.push(start.elapsed());
        results.push(got);
    }
    (results, durations)
}

fn evaluate_codec(
    q: &dyn Quantizer,
    base: &VectorSet,
    queries: &VectorSet,
    truth: &[Vec<u32>],
    k: usize,
    build_ms: f64,
    budgets: &Budgets,
) -> CodecResult {
    let dim = base.dim;
    let bpv = q.bytes_per_vector();
    let (results, durations) = timed_single(q, queries, k);
    let mut racc = 0.0;
    for (got, t) in results.iter().zip(truth.iter()) {
        racc += recall(got, t);
    }
    let recall_at_k = if queries.count() == 0 {
        0.0
    } else {
        racc / queries.count() as f64
    };
    let latency = LatencySummary::from_durations(&durations);
    let ram_5m = projected_mb(bpv, 5_000_000.0);
    let gate_pass = recall_at_k >= budgets.recall_at_k && ram_5m <= budgets.idle_ram_mb_at_5m;
    CodecResult {
        name: q.name(),
        dim,
        bytes_per_vector: bpv,
        compression_x: (dim * 4) as f64 / bpv,
        recall_at_k,
        build_ms,
        latency,
        projected_ram_mb_1m: projected_mb(bpv, 1_000_000.0),
        projected_ram_mb_5m: ram_5m,
        gate_pass,
    }
}

fn build_timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let v = f();
    (v, start.elapsed().as_secs_f64() * 1000.0)
}

/// Run the full Stage 1 matrix at one dimensionality.
pub fn run_matrix(
    base: &VectorSet,
    queries: &VectorSet,
    k: usize,
    seed: u64,
    budgets: &Budgets,
) -> BenchReport {
    let dim = base.dim;

    // Exact ground truth (computed once, reused everywhere).
    let truth: Vec<Vec<u32>> = (0..queries.count())
        .map(|qi| top_k(base, queries.get(qi), k))
        .collect();

    let mut single = Vec::new();

    let (float, t) = build_timed(|| FloatStore::build(base));
    single.push(evaluate_codec(&float, base, queries, &truth, k, t, budgets));

    let (i8, t) = build_timed(|| ScalarI8::build(base));
    single.push(evaluate_codec(&i8, base, queries, &truth, k, t, budgets));

    // PQ subspace counts that divide the dim (4x, 8x, 16x compression).
    for &m in &[dim / 4, dim / 8, dim / 16] {
        if m == 0 || !dim.is_multiple_of(m) {
            continue;
        }
        let (pq, t) = build_timed(|| ProductQuantizer::build(base, m, seed));
        single.push(evaluate_codec(&pq, base, queries, &truth, k, t, budgets));
    }

    for &bits in &[1u8, 3, 5, 7] {
        let (rq, t) = build_timed(|| RaBitQ::build(base, bits, seed));
        single.push(evaluate_codec(&rq, base, queries, &truth, k, t, budgets));
    }

    // Two-stage pipelines.
    let coarse = RaBitQ::build(base, 1, seed);
    let precise_mb = RaBitQ::build(base, 7, seed);
    let precise_fp = FloatStore::build(base);
    let refine = (k * 16).max(64);

    let two_stage = vec![
        eval_two_stage(
            &coarse,
            &precise_mb,
            base,
            queries,
            &truth,
            k,
            refine,
            coarse.bytes_per_vector() + precise_mb.bytes_per_vector(),
            "both codes resident (1-bit scan + 7-bit rerank)",
        ),
        eval_two_stage(
            &coarse,
            &precise_fp,
            base,
            queries,
            &truth,
            k,
            refine,
            coarse.bytes_per_vector(),
            "float reranked from disk/mmap for candidates only; resident = 1-bit",
        ),
    ];

    let mmap = mmap_check(&coarse);

    BenchReport {
        vectors: base.count(),
        queries: queries.count(),
        k,
        dim_full: dim,
        budgets: budgets.clone(),
        single,
        two_stage,
        mmap,
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_two_stage(
    coarse: &dyn Quantizer,
    precise: &dyn Quantizer,
    _base: &VectorSet,
    queries: &VectorSet,
    truth: &[Vec<u32>],
    k: usize,
    refine: usize,
    resident_bpv: f64,
    note: &str,
) -> TwoStageResult {
    let mut racc = 0.0;
    let mut durations = Vec::with_capacity(queries.count());
    for qi in 0..queries.count() {
        let query = queries.get(qi);
        let start = Instant::now();
        let got = two_stage_search(coarse, precise, query, k, refine);
        durations.push(start.elapsed());
        racc += recall(&got, &truth[qi]);
    }
    let recall_at_k = if queries.count() == 0 {
        0.0
    } else {
        racc / queries.count() as f64
    };
    TwoStageResult {
        coarse: coarse.name(),
        precise: precise.name(),
        refine,
        resident_bytes_per_vector: resident_bpv,
        recall_at_k,
        latency: LatencySummary::from_durations(&durations),
        projected_resident_mb_5m: projected_mb(resident_bpv, 5_000_000.0),
        note: note.to_string(),
    }
}

/// Persist the 1-bit code buffer, mmap it read-only, and confirm the mapped
/// bytes are byte-identical to the in-RAM buffer — proving the production cold
/// path (index on disk, demand-paged) serves the same codes the scan reads.
fn mmap_check(coarse: &RaBitQ) -> MmapCheck {
    let bytes = coarse.code_bytes().to_vec();
    let path = std::env::temp_dir().join(format!("skinki_codes_{}.bin", std::process::id()));
    match CodeStore::mmap_from(&path, &bytes) {
        Ok(store) => {
            let matches = store.as_slice() == bytes.as_slice();
            let supported = store.is_mmap();
            drop(store);
            let _ = std::fs::remove_file(&path);
            MmapCheck {
                supported,
                code_bytes: bytes.len(),
                recall_matches_ram: matches,
                note: if supported {
                    "1-bit codes served from a read-only mmap; bytes identical to RAM".into()
                } else {
                    "mmap unsupported on this target; fell back to RAM".into()
                },
            }
        }
        Err(e) => MmapCheck {
            supported: false,
            code_bytes: bytes.len(),
            recall_matches_ram: false,
            note: format!("mmap failed: {e}"),
        },
    }
}

/// Machine-checkable gate: does any config meet recall AND the RAM budget?
///
/// A single-stage codec passes if `gate_pass` (recall + projected RAM). A
/// two-stage pipeline passes if it meets the recall budget AND its resident
/// footprint at 5M is within the RAM budget. This is what CI asserts.
pub fn passes_gate(report: &BenchReport) -> bool {
    let single = report.single.iter().any(|c| c.gate_pass);
    let two = report.two_stage.iter().any(|t| {
        t.recall_at_k >= report.budgets.recall_at_k
            && t.projected_resident_mb_5m <= report.budgets.idle_ram_mb_at_5m
    });
    single || two
}

/// Apply the gate and produce a human-readable verdict string.
pub fn verdict(report: &BenchReport) -> String {
    let passing: Vec<&CodecResult> = report.single.iter().filter(|c| c.gate_pass).collect();
    let best_two = report
        .two_stage
        .iter()
        .filter(|t| t.recall_at_k >= report.budgets.recall_at_k)
        .min_by(|a, b| {
            a.projected_resident_mb_5m
                .partial_cmp(&b.projected_resident_mb_5m)
                .unwrap()
        });

    let mut s = String::new();
    if passing.is_empty() && best_two.is_none() {
        s.push_str("GATE: FAIL (single-stage) — no codec hits recall + RAM alone.\n");
    }
    if let Some(t) = best_two {
        s.push_str(&format!(
            "GATE: PASS via two-stage — {} -> {} reaches recall {:.3} at {:.1} MB resident /5M (budget {:.0} MB).\n",
            t.coarse, t.precise, t.recall_at_k, t.projected_resident_mb_5m, report.budgets.idle_ram_mb_at_5m
        ));
    }
    for c in &passing {
        s.push_str(&format!(
            "  single-stage PASS: {} recall {:.3}, {:.1} MB /5M\n",
            c.name, c.recall_at_k, c.projected_ram_mb_5m
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::synthetic_clusters;

    #[test]
    fn gate_passes_on_separable_data() {
        // A small, clearly-clustered set: the two-stage pipeline must clear the
        // recall + RAM gate. This is the contract CI's --assert-gate relies on.
        // The gate verdict is N-independent (footprint is per-vector), so a
        // small N keeps the test fast while still validating the contract.
        let all = synthetic_clusters(7, 256, 700, 16, 0.4);
        let base = all.slice_rows(0, 600);
        let queries = all.slice_rows(600, 40);
        let report = run_matrix(&base, &queries, 10, 7, &Budgets::default());
        assert!(
            passes_gate(&report),
            "expected gate to pass:\n{}",
            verdict(&report)
        );
    }

    #[test]
    fn gate_fails_under_impossible_ram_budget() {
        let all = synthetic_clusters(8, 256, 640, 16, 0.4);
        let base = all.slice_rows(0, 600);
        let queries = all.slice_rows(600, 40);
        let tight = Budgets {
            recall_at_k: 0.95,
            idle_ram_mb_at_5m: 1.0, // 1 MB / 5M vectors is unattainable
            p95_ms: 150.0,
        };
        let report = run_matrix(&base, &queries, 10, 8, &tight);
        assert!(
            !passes_gate(&report),
            "gate should fail under a 1 MB budget"
        );
    }
}
