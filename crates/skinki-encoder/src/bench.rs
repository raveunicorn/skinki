//! T0 kill-switch bench harness. Runs the `gemm` kernel at 384-class shapes
//! (the ones the T2 forward pass will execute) under sustained load and
//! reports achieved GFLOP/s. See `specs/STAGE_1C_B_PURE_RUST_ENCODER.md`
//! §1–2 for the budget and the bar (≥ 40 GFLOP/s sustained, 4 threads).
//!
//! The harness is deliberately self-contained and dependency-free: it uses
//! `std::time::Instant`, a deterministic LCG for inputs, and
//! `std::thread::scope`-partitioned execution through `crate::gemm`.

use std::time::{Duration, Instant};

/// T0 budget from §2 of the spec, in GFLOP/s sustained (4-thread minimum).
pub const T0_GATE_GFLOPS: f64 = 40.0;

/// The four 384-class shapes the BERT forward pass actually multiplies:
/// query/turn × hidden × {hidden, ffn}. `M` ∈ {32 (query), 128 (turn)},
/// `N` ∈ {384 (hidden), 1536 (ffn)}, `K = 384` (hidden) throughout.
pub const SHAPES: &[(usize, usize, usize)] = &[
    (32, 384, 384),
    (32, 1536, 384),
    (128, 384, 384),
    (128, 1536, 384),
];

/// One row of the bench report.
#[derive(Debug, Clone, Copy)]
pub struct ShapeResult {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub threads: usize,
    /// Aggregate GFLOP/s = (2·M·N·K·iters) / wall_seconds.
    pub gflops_mean: f64,
    /// 5th percentile of per-window GFLOP/s — the honest sustained number
    /// (worst window after throttling).
    pub gflops_p5: f64,
    pub iters: u64,
}

impl ShapeResult {
    /// Gate is judged on the **minimum** sustained (p5) GFLOP/s across all
    /// configured shapes at the chosen thread count (4 by §2). Returns the
    /// (shape, threads) the worst-case came from.
    pub fn worst_p5(results: &[ShapeResult]) -> Option<(f64, (usize, usize, usize, usize))> {
        results
            .iter()
            .map(|r| (r.gflops_p5, (r.m, r.n, r.k, r.threads)))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    }
}

/// Bench configuration.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Shapes to measure. Default [`SHAPES`].
    pub shapes: Vec<(usize, usize, usize)>,
    /// Thread counts to sweep. Default `[1, 2, 4]`.
    pub threads: Vec<usize>,
    /// Sustained wall-clock duration per (shape, threads) measurement.
    pub duration: Duration,
    /// Per-window length for p5 calculation.
    pub window: Duration,
    /// Warmup time per measurement (not counted).
    pub warmup: Duration,
}

impl Default for BenchConfig {
    fn default() -> Self {
        // Short-run default (~10 s/shape/threads). `--full-run` raises this
        // to the spec's honest 10 minutes for the final D1 numbers.
        BenchConfig {
            shapes: SHAPES.to_vec(),
            threads: vec![1, 2, 4],
            duration: Duration::from_secs(10),
            window: Duration::from_millis(500),
            warmup: Duration::from_millis(500),
        }
    }
}

impl BenchConfig {
    /// The spec's honest 10-minute sustained run (D1 / final numbers).
    pub fn full_run() -> Self {
        BenchConfig {
            duration: Duration::from_secs(600),
            window: Duration::from_secs(5),
            warmup: Duration::from_secs(2),
            ..BenchConfig::default()
        }
    }

    /// CI smoke (fast, no gate assertions possible — CI is not an M1 Air).
    pub fn smoke() -> Self {
        BenchConfig {
            shapes: vec![(128, 384, 384), (128, 1536, 384)],
            threads: vec![1, 4],
            duration: Duration::from_secs(3),
            window: Duration::from_millis(250),
            warmup: Duration::from_millis(250),
        }
    }
}

/// Run the configured bench, returning one `ShapeResult` per (shape, threads).
/// Prints live progress to stdout (one line per measurement, refreshed as it
/// runs) so a long `full_run` is not silent for an hour.
pub fn run(cfg: &BenchConfig) -> Vec<ShapeResult> {
    let total = cfg.threads.len() * cfg.shapes.len();
    let mut out = Vec::with_capacity(total);
    let mut idx = 0;
    for &threads in &cfg.threads {
        for &(m, n, k) in &cfg.shapes {
            idx += 1;
            println!(
                "[{idx}/{total}] M={m:>4} N={n:>5} K={k:>4} threads={threads:>2}  \
                 ({:.0}s + {:.0}s warmup) ...",
                cfg.duration.as_secs_f64(),
                cfg.warmup.as_secs_f64(),
            );
            // Flush so the "..." line is visible before the (long) run.
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let r = measure_shape(m, n, k, threads, cfg);
            println!(
                "    -> mean {gflops_mean:>8.2} GF/s   p5 {gflops_p5:>8.2} GF/s   iters {iters}",
                gflops_mean = r.gflops_mean,
                gflops_p5 = r.gflops_p5,
                iters = r.iters
            );
            out.push(r);
        }
    }
    out
}

/// Measure one (shape, threads) combination to `cfg.duration` sustained.
/// Prints a live per-window GFLOP/s ticker so the long sustained run is not
/// silent.
fn measure_shape(m: usize, n: usize, k: usize, threads: usize, cfg: &BenchConfig) -> ShapeResult {
    let (a, b) = seeded_inputs(m, n, k);
    let mut c = vec![0.0f32; m * n];

    // Warmup — fill the cache, let the CPU settle; not counted.
    let warmup_end = Instant::now() + cfg.warmup;
    while Instant::now() < warmup_end {
        crate::gemm(m, n, k, &a, &b, &mut c, threads).ok();
    }

    // Sustained run. We collect per-window GFLOP/s so the report can show
    // the honest worst-case window (throttling) instead of just the mean.
    let mut windows: Vec<f64> = Vec::new();
    let mut iters: u64 = 0;
    let start = Instant::now();
    let end = start + cfg.duration;
    let mut window_start = start;
    let mut window_iters: u64 = 0;
    while Instant::now() < end {
        crate::gemm(m, n, k, &a, &b, &mut c, threads).ok();
        window_iters += 1;
        iters += 1;
        let now = Instant::now();
        if now - window_start >= cfg.window {
            let secs = (now - window_start).as_secs_f64().max(1e-9);
            let flops = 2.0 * (m as f64) * (n as f64) * (k as f64) * (window_iters as f64);
            let g = flops / secs / 1e9;
            windows.push(g);
            // Live ticker: elapsed / total | current window GF/s.
            let elapsed = (now - start).as_secs_f64();
            let total = cfg.duration.as_secs_f64();
            eprint!(
                "\r    [{elapsed:>5.0}/{total:>4.0}s] window {g:>7.2} GF/s  ({iters} iters)   "
            );
            use std::io::Write;
            let _ = std::io::stderr().flush();
            window_start = now;
            window_iters = 0;
        }
    }
    eprintln!(); // newline after the ticker

    // Insurance against a future compiler proving `c` unused and eliding the
    // work — today's codegen keeps the stores, but a bench that can silently
    // measure nothing is not a bench.
    std::hint::black_box(&c);

    let total_secs = (Instant::now() - start).as_secs_f64().max(1e-9);
    let total_flops = 2.0 * (m as f64) * (n as f64) * (k as f64) * (iters as f64);
    let gflops_mean = total_flops / total_secs / 1e9;
    let gflops_p5 = percentile(&windows, 5.0).unwrap_or(gflops_mean);

    ShapeResult {
        m,
        n,
        k,
        threads,
        gflops_mean,
        gflops_p5,
        iters,
    }
}

/// Linear-interpolated percentile (R-7). `p` in [0, 100].
fn percentile(xs: &[f64], p: f64) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (p / 100.0) * (v.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        Some(v[lo])
    } else {
        let frac = rank - lo as f64;
        Some(v[lo] * (1.0 - frac) + v[hi] * frac)
    }
}

/// Deterministic LCG inputs — no `rand`, byte-reproducible across platforms.
/// Shared with the gemm unit tests (single source for the probe inputs).
pub(crate) fn seeded_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as i32 as f32) / (1u64 << 31) as f32 - 1.0
    };
    let a = (0..m * k).map(|_| next()).collect::<Vec<_>>();
    let b = (0..k * n).map(|_| next()).collect::<Vec<_>>();
    (a, b)
}

/// Render a results table to a String (one line per result).
pub fn format_table(results: &[ShapeResult]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(
        out,
        "  M     N     K    threads   mean GF/s   p5 GF/s    iters"
    )
    .ok();
    for r in results {
        writeln!(
            out,
            "{:>3}  {:>5} {:>5}     {:>2}     {:>8.2}   {:>8.2}  {:>8}",
            r.m, r.n, r.k, r.threads, r.gflops_mean, r.gflops_p5, r.iters
        )
        .ok();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_basic() {
        let xs: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        // p5 of {1..10} via R-7 linear interpolation ≈ 1.45.
        let p = percentile(&xs, 5.0).unwrap();
        assert!((p - 1.45).abs() < 1e-9, "got {p}");
    }

    #[test]
    fn worst_p5_picks_min() {
        let rs = vec![
            ShapeResult {
                m: 1,
                n: 1,
                k: 1,
                threads: 1,
                gflops_mean: 50.0,
                gflops_p5: 30.0,
                iters: 0,
            },
            ShapeResult {
                m: 1,
                n: 1,
                k: 1,
                threads: 4,
                gflops_mean: 90.0,
                gflops_p5: 70.0,
                iters: 0,
            },
        ];
        let (val, _) = ShapeResult::worst_p5(&rs).unwrap();
        assert_eq!(val, 30.0);
    }

    #[test]
    fn smoke_config_runs_quickly() {
        // A single shape/threads smoke run completes in well under 10 s.
        let start = Instant::now();
        let cfg = BenchConfig {
            shapes: vec![(32, 384, 384)],
            threads: vec![1],
            duration: Duration::from_millis(800),
            window: Duration::from_millis(100),
            warmup: Duration::from_millis(100),
        };
        let results = run(&cfg);
        assert_eq!(results.len(), 1);
        let r = results[0];
        assert!(r.gflops_mean > 0.0);
        assert!(r.iters > 0);
        assert!(start.elapsed() < Duration::from_secs(5));
    }
}
