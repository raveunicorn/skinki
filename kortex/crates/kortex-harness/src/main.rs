#![forbid(unsafe_code)]
//! Kortex harness CLI.
//!
//! Subcommands:
//!   generate       — write a deterministic synthetic corpus to disk
//!   eval           — score a system (BM25 baseline) over a corpus file
//!   demo           — generate + eval in one shot (no files), print the report
//!   compress-bench — Stage 1: benchmark vector-compression codecs vs exact

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use kortex_baseline::Bm25;
use kortex_corpus::{generate, Corpus, EntryId, GenConfig};
use kortex_eval::{
    answer_in_entries, ndcg_at_k, precision_at_k, recall_at_k, score_insights, Latency, Report,
    RetrievalScores, RetrievalSystem,
};
use kortex_store::{derive_units, RawEvent, Source, Store};
use kortex_telemetry::{peak_rss_bytes, LatencySummary};
use kortex_vector::bench::{passes_gate, run_matrix, verdict, BenchReport, Budgets};
use kortex_vector::embed::{synthetic_clusters, StaticHashEmbedder};

use kortex_vector::VectorSet;

#[derive(Parser)]
#[command(
    name = "kortex",
    about = "Kortex Stage 0 — synthetic corpus + eval harness"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate a deterministic synthetic corpus and write it to a JSON file.
    Generate {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 5)]
        years: u32,
        /// Average routine entries per day (crank to stress scale).
        #[arg(long, default_value_t = 2)]
        entries_per_day: u32,
        #[arg(long)]
        out: PathBuf,
    },
    /// Evaluate the BM25 baseline over a corpus file.
    Eval {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long)]
        report_out: Option<PathBuf>,
    },
    /// Generate a corpus in memory and evaluate it immediately.
    Demo {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 3)]
        years: u32,
        /// Average routine entries per day (crank to stress scale).
        #[arg(long, default_value_t = 2)]
        entries_per_day: u32,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// Stage 2: benchmark L0 store (overhead, random-access, ingest throughput).
    StoreBench {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 5)]
        years: u32,
        #[arg(long, default_value_t = 2)]
        entries_per_day: u32,
        #[arg(long)]
        report_out: Option<PathBuf>,
        /// Exit non-zero if budgets miss (for CI).
        #[arg(long, default_value_t = false)]
        assert_gate: bool,
    },
    /// Stage 1: benchmark compression codecs (int8/PQ/RaBitQ) vs exact float32.
    CompressBench {
        /// Vector source: "corpus" (static-embed Stage 0 text) or "synthetic".
        #[arg(long, default_value = "corpus")]
        source: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 5)]
        years: u32,
        #[arg(long, default_value_t = 2)]
        entries_per_day: u32,
        /// Full embedding dimensionality (also benched truncated to 256).
        #[arg(long, default_value_t = 768)]
        dim: usize,
        /// Cap on base vectors (exact ground truth is O(queries*vectors*dim)).
        #[arg(long, default_value_t = 20000)]
        vectors: usize,
        /// Number of held-out queries.
        #[arg(long, default_value_t = 200)]
        queries: usize,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long)]
        report_out: Option<PathBuf>,
        /// Exit non-zero if no config meets the Stage 1 gate (for CI).
        #[arg(long, default_value_t = false)]
        assert_gate: bool,
    },
}

struct QItem<'a> {
    question: &'a str,
    relevant: &'a [EntryId],
    answer: &'a str,
}

fn score_set(
    system: &dyn RetrievalSystem,
    corpus: &Corpus,
    items: &[QItem<'_>],
    k: usize,
    durations: &mut Vec<Duration>,
) -> RetrievalScores {
    let (mut recall, mut precision, mut ndcg, mut answer_hits) = (0.0, 0.0, 0.0, 0.0);
    for item in items {
        let start = Instant::now();
        let retrieved = system.search(item.question, k);
        durations.push(start.elapsed());

        recall += recall_at_k(&retrieved, item.relevant, k);
        precision += precision_at_k(&retrieved, item.relevant, k);
        ndcg += ndcg_at_k(&retrieved, item.relevant, k);
        if answer_in_entries(corpus, &retrieved, item.answer) {
            answer_hits += 1.0;
        }
    }
    let n = items.len().max(1) as f64;
    RetrievalScores {
        queries: items.len(),
        k,
        recall_at_k: recall / n,
        precision_at_k: precision / n,
        ndcg_at_k: ndcg / n,
        answer_in_topk: answer_hits / n,
    }
}

fn run_eval(corpus: &Corpus, k: usize) -> Report {
    let mut system = Bm25::new();
    system.index(corpus);

    let mut durations: Vec<Duration> = Vec::new();

    let recall_items: Vec<QItem> = corpus
        .ground_truth
        .recall
        .iter()
        .map(|q| QItem {
            question: &q.question,
            relevant: &q.relevant_entries,
            answer: &q.answer,
        })
        .collect();
    let recall = score_set(&system, corpus, &recall_items, k, &mut durations);

    let multihop_items: Vec<QItem> = corpus
        .ground_truth
        .multi_hop
        .iter()
        .map(|q| QItem {
            question: &q.question,
            relevant: &q.hop_entries,
            answer: &q.answer,
        })
        .collect();
    let multi_hop = score_set(&system, corpus, &multihop_items, k, &mut durations);

    let discovered = system.discover_insights();
    let insight = score_insights(&discovered, &corpus.ground_truth.insights);

    let summary = LatencySummary::from_durations(&durations);
    let latency = Latency {
        count: summary.count,
        p50_ms: summary.p50_ms,
        p95_ms: summary.p95_ms,
        mean_ms: summary.mean_ms,
        max_ms: summary.max_ms,
    };

    Report {
        system: system.name().to_string(),
        corpus_entries: corpus.entries.len(),
        recall,
        multi_hop,
        insight,
        latency: Some(latency),
        peak_rss_bytes: peak_rss_bytes(),
    }
}

fn load_corpus(path: &PathBuf) -> Result<Corpus> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading corpus {}", path.display()))?;
    let corpus: Corpus = serde_json::from_slice(&bytes).context("parsing corpus JSON")?;
    Ok(corpus)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Generate {
            seed,
            years,
            entries_per_day,
            out,
        } => {
            let corpus = generate(&GenConfig {
                seed,
                years,
                entries_per_day,
            });
            let file = std::fs::File::create(&out)
                .with_context(|| format!("creating {}", out.display()))?;
            serde_json::to_writer(file, &corpus).context("writing corpus")?;
            println!(
                "Generated corpus: {} entries over {} years (seed {}) -> {}",
                corpus.entries.len(),
                years,
                seed,
                out.display()
            );
            print_ground_truth_summary(&corpus);
        }
        Cmd::Eval {
            corpus,
            k,
            report_out,
        } => {
            let corpus = load_corpus(&corpus)?;
            let report = run_eval(&corpus, k);
            print!("{report}");
            if let Some(path) = report_out {
                let file = std::fs::File::create(&path)
                    .with_context(|| format!("creating {}", path.display()))?;
                serde_json::to_writer_pretty(file, &report).context("writing report")?;
                println!("\nReport written to {}", path.display());
            }
        }
        Cmd::Demo {
            seed,
            years,
            entries_per_day,
            k,
        } => {
            let corpus = generate(&GenConfig {
                seed,
                years,
                entries_per_day,
            });
            print_ground_truth_summary(&corpus);
            let report = run_eval(&corpus, k);
            println!();
            print!("{report}");
            println!(
                "\nReading: BM25 is expected to score well on recall but poorly on multi-hop,\n\
                 and to surface zero insights. Those gaps are the targets for Stages 1-5."
            );
        }
        Cmd::StoreBench {
            seed,
            years,
            entries_per_day,
            report_out,
            assert_gate,
        } => {
            let report = run_store_bench(seed, years, entries_per_day)?;
            print_store_bench(&report);
            if let Some(path) = report_out {
                let file = std::fs::File::create(&path)
                    .with_context(|| format!("creating {}", path.display()))?;
                serde_json::to_writer_pretty(file, &report).context("writing report")?;
                println!("Report written to {}", path.display());
            }
            if assert_gate {
                let pass = report.content_overhead <= 1.25
                    && report.index_bytes_per_unit <= 24.0
                    && report.p95_us >= 0.0
                    && report.p95_us < 1_000.0
                    && report.ingest_units_per_sec >= 50_000.0;
                if pass {
                    println!("\nGATE: PASS (store-bench --assert-gate satisfied)");
                } else {
                    anyhow::bail!(
                        "Stage 2 gate FAILED: content_overhead={:.3}x (budget 1.25x), index={:.1} B/unit (budget 24), p95={:.0}us (budget <1000us), ingest={:.0} units/s (budget >=50k)",
                        report.content_overhead,
                        report.index_bytes_per_unit,
                        report.p95_us,
                        report.ingest_units_per_sec
                    );
                }
            }
        }
        Cmd::CompressBench {
            source,
            seed,
            years,
            entries_per_day,
            dim,
            vectors,
            queries,
            k,
            report_out,
            assert_gate,
        } => {
            let (base, qset, label) =
                build_vectors(&source, seed, years, entries_per_day, dim, vectors, queries);
            println!(
                "Stage 1 compression bench | source: {label} | {} base vecs, {} queries, dim {}",
                base.count(),
                qset.count(),
                base.dim
            );
            let budgets = Budgets::default();

            // Co-design: full dim, and Matryoshka-truncated to 256.
            let mut reports: Vec<BenchReport> = Vec::new();
            reports.push(run_matrix(&base, &qset, k, seed, &budgets));
            if base.dim > 256 {
                let base256 = base.truncate_dims(256);
                let q256 = qset.truncate_dims(256);
                reports.push(run_matrix(&base256, &q256, k, seed, &budgets));
            }

            for r in &reports {
                print_compress_report(r);
            }
            println!("\nPeak RSS during bench: {:.1} MB", rss_mb());

            if let Some(path) = report_out {
                let file = std::fs::File::create(&path)
                    .with_context(|| format!("creating {}", path.display()))?;
                serde_json::to_writer_pretty(file, &reports).context("writing report")?;
                println!("Report written to {}", path.display());
            }

            if assert_gate {
                let any_pass = reports.iter().any(passes_gate);
                if any_pass {
                    println!("\nGATE: PASS (assert-gate satisfied)");
                } else {
                    anyhow::bail!(
                        "Stage 1 gate FAILED: no config met recall {:.2} within {:.0} MB /5M",
                        Budgets::default().recall_at_k,
                        Budgets::default().idle_ram_mb_at_5m
                    );
                }
            }
        }
    }
    Ok(())
}

fn rss_mb() -> f64 {
    peak_rss_bytes().unwrap_or(0) as f64 / 1_048_576.0
}

/// Build a base vector set + a held-out query set from the chosen source.
fn build_vectors(
    source: &str,
    seed: u64,
    years: u32,
    entries_per_day: u32,
    dim: usize,
    cap: usize,
    queries: usize,
) -> (VectorSet, VectorSet, String) {
    match source {
        "synthetic" => {
            let total = cap + queries;
            let all = synthetic_clusters(seed, dim, total, 24, 0.45);
            let base = all.slice_rows(0, cap);
            let qset = all.slice_rows(cap, queries);
            (base, qset, format!("synthetic (dim {dim}, 24 clusters)"))
        }
        _ => {
            let corpus = generate(&GenConfig {
                seed,
                years,
                entries_per_day,
            });
            let embedder = StaticHashEmbedder::new(dim);
            let mut all = embedder.embed_corpus(&corpus);
            // Subsample to the cap (deterministic stride) to bound exact search.
            if all.count() > cap + queries {
                all = stride_subsample(&all, cap + queries);
            }
            let n = all.count();
            let q = queries.min(n / 5).max(1);
            let base = all.slice_rows(0, n - q);
            let qset = all.slice_rows(n - q, q);
            (
                base,
                qset,
                format!("corpus static-embed (dim {dim}, {years}y)"),
            )
        }
    }
}

fn stride_subsample(vs: &VectorSet, target: usize) -> VectorSet {
    let n = vs.count();
    if n <= target {
        return vs.clone();
    }
    let stride = n / target;
    let mut out = VectorSet::new(vs.dim);
    let mut i = 0;
    while i < n && out.count() < target {
        out.push(vs.get(i));
        i += stride;
    }
    out
}

fn print_compress_report(r: &BenchReport) {
    println!(
        "\n=== Compression matrix @ dim {} (vs exact float32, recall@{}) ===",
        r.dim_full, r.k
    );
    println!(
        "{:<14} {:>6} {:>8} {:>9} {:>10} {:>11} {:>6}",
        "codec", "bytes", "compr", "recall", "p95_ms", "RAM/5M_MB", "gate"
    );
    for c in &r.single {
        println!(
            "{:<14} {:>6.0} {:>7.1}x {:>9.3} {:>10.3} {:>11.1} {:>6}",
            c.name,
            c.bytes_per_vector,
            c.compression_x,
            c.recall_at_k,
            c.latency.p95_ms,
            c.projected_ram_mb_5m,
            if c.gate_pass { "PASS" } else { "-" }
        );
    }
    println!("two-stage (coarse -> precise):");
    for t in &r.two_stage {
        println!(
            "  {:<22} refine={:<4} recall={:.3} p95={:.3}ms resident/5M={:.1}MB",
            format!("{}->{}", t.coarse, t.precise),
            t.refine,
            t.recall_at_k,
            t.latency.p95_ms,
            t.projected_resident_mb_5m
        );
        println!("      note: {}", t.note);
    }
    println!(
        "mmap: supported={}, code_bytes={}, bytes_identical_to_ram={} ({})",
        r.mmap.supported, r.mmap.code_bytes, r.mmap.recall_matches_ram, r.mmap.note
    );
    print!("{}", verdict(r));
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoreBenchReport {
    corpus_entries: usize,
    event_count: usize,
    unit_count: usize,
    raw_text_bytes: u64,
    event_store_bytes: u64,
    unit_store_bytes: u64,
    store_bytes: u64,
    content_overhead: f64,
    index_bytes_per_unit: f64,
    p50_us: f64,
    p95_us: f64,
    ingest_elapsed_secs: f64,
    ingest_units_per_sec: f64,
}

// FIX 5: deterministic shuffle using the same SplitMix64 as the project.
fn shuffle_det(seed: u64, items: &mut [kortex_store::UnitId]) {
    // Simple portable SplitMix64 shuffle — same constants as kortex-corpus::Rng.
    let mut state = seed;
    for i in (1..items.len()).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let j = (z ^ (z >> 31)) as usize % (i + 1);
        items.swap(i, j);
    }
}

fn run_store_bench(seed: u64, years: u32, entries_per_day: u32) -> Result<StoreBenchReport> {
    let corpus = generate(&GenConfig {
        seed,
        years,
        entries_per_day,
    });

    let dir = std::env::temp_dir().join(format!("kortex_store_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut store = Store::open(&dir).context("opening store")?;

    let ingest_start = Instant::now();
    let mut total_units = 0usize;
    for entry in &corpus.entries {
        let ev = RawEvent {
            source: if entry.kind == kortex_corpus::EntryKind::Voice {
                Source::Voice
            } else {
                Source::Text
            },
            created_utc_secs: entry.day as i64 * 86400,
            text: entry.text.clone(),
        };
        let event_id = store.append_event(&ev)?;
        let units = derive_units(event_id, &entry.text);
        total_units += units.len();
        for u in &units {
            store.append_unit(u)?;
        }
    }
    let ingest_elapsed = ingest_start.elapsed();
    store.sync()?;

    let ingest_elapsed_secs = ingest_elapsed.as_secs_f64();
    let ingest_units_per_sec = total_units as f64 / ingest_elapsed_secs.max(0.001);

    // FIX 1: two explicit metrics, both checked by the gate.
    let content_overhead = if store.raw_text_bytes() > 0 {
        store.event_store_bytes() as f64 / store.raw_text_bytes() as f64
    } else {
        0.0
    };
    let unit_count = store.unit_count();
    let index_bytes_per_unit = if unit_count > 0 {
        store.unit_store_bytes() as f64 / unit_count as f64
    } else {
        0.0
    };

    // FIX 5: random-access measured in random order (deterministic shuffle).
    let mut uids: Vec<kortex_store::UnitId> = store.units().map(|(id, _)| id).collect();
    shuffle_det(seed, &mut uids);
    let mut latencies_us: Vec<f64> = Vec::new();
    for &uid in &uids {
        let start = Instant::now();
        let _ = store.unit_text(uid);
        latencies_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies_us.len();
    let pct = |p: f64| {
        let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n.saturating_sub(1));
        latencies_us[idx]
    };
    let p50_us = if n > 0 { pct(0.50) } else { 0.0 };
    let p95_us = if n > 0 { pct(0.95) } else { 0.0 };

    let _ = std::fs::remove_dir_all(&dir);

    Ok(StoreBenchReport {
        corpus_entries: corpus.entries.len(),
        event_count: store.event_count(),
        unit_count,
        raw_text_bytes: store.raw_text_bytes(),
        event_store_bytes: store.event_store_bytes(),
        unit_store_bytes: store.unit_store_bytes(),
        store_bytes: store.store_bytes(),
        content_overhead,
        index_bytes_per_unit,
        p50_us,
        p95_us,
        ingest_elapsed_secs,
        ingest_units_per_sec,
    })
}

fn print_store_bench(r: &StoreBenchReport) {
    println!("\n=== Kortex Stage 2 — Store Benchmark ===");
    println!("corpus entries    : {}", r.corpus_entries);
    println!("events stored     : {}", r.event_count);
    println!("units stored      : {}", r.unit_count);
    println!(
        "raw text bytes    : {} ({:.1} MB)",
        r.raw_text_bytes,
        r.raw_text_bytes as f64 / 1_048_576.0
    );
    println!(
        "event store bytes : {} ({:.1} MB)",
        r.event_store_bytes,
        r.event_store_bytes as f64 / 1_048_576.0
    );
    println!(
        "unit store bytes  : {} ({:.1} MB)",
        r.unit_store_bytes,
        r.unit_store_bytes as f64 / 1_048_576.0
    );
    println!(
        "content overhead  : {:.3}x  (budget: <= 1.25x)",
        r.content_overhead
    );
    println!(
        "index B/unit      : {:.1}  (budget: <= 24.0)",
        r.index_bytes_per_unit
    );
    println!("get_unit p50      : {:.1} us", r.p50_us);
    println!(
        "get_unit p95      : {:.1} us  (budget: < 1000 us)",
        r.p95_us
    );
    println!(
        "ingest throughput : {:.0} units/s  (budget: >= 50000 units/s, {:.3}s)",
        r.ingest_units_per_sec, r.ingest_elapsed_secs
    );
    let ok_content = r.content_overhead <= 1.25;
    let ok_index = r.index_bytes_per_unit <= 24.0;
    let ok_p95 = r.p95_us >= 0.0 && r.p95_us < 1_000.0;
    let ok_ingest = r.ingest_units_per_sec >= 50_000.0;
    println!();
    println!(
        "gate check: content={} index={} p95={} ingest={}",
        if ok_content { "PASS" } else { "FAIL" },
        if ok_index { "PASS" } else { "FAIL" },
        if ok_p95 { "PASS" } else { "FAIL" },
        if ok_ingest { "PASS" } else { "FAIL" },
    );
}

fn print_ground_truth_summary(corpus: &Corpus) {
    let gt = &corpus.ground_truth;
    println!(
        "Planted ground truth: {} recall, {} multi-hop, {} temporal, {} contradictions, {} insights ({} entities)",
        gt.recall.len(),
        gt.multi_hop.len(),
        gt.temporal.len(),
        gt.contradictions.len(),
        gt.insights.len(),
        gt.entities.len(),
    );
}
