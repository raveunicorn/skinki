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
use kortex_corpus::{generate, Corpus, Difficulty, EntryId, GenConfig};
use kortex_eval::{
    answer_in_entries, ndcg_at_k, precision_at_k, recall_at_k, score_insights, Latency, Report,
    RetrievalScores, RetrievalSystem,
};
use kortex_store::{derive_units, RawEvent, Source, Store};
use kortex_telemetry::{peak_rss_bytes, LatencySummary};
use kortex_vector::bench::{passes_gate, run_matrix, verdict, BenchReport, Budgets};
use kortex_vector::embed::{synthetic_clusters, ClusterSampler, StaticHashEmbedder};
use kortex_vector::quant::{RaBitQ, RaBitQBuilder};
use kortex_vector::search::{recall as recall_overlap, two_stage_search};
use kortex_vector::store::{available_disk_bytes, FloatMmapStore};

use kortex_sleep::{check_gate, run_sim, SimConfig, SimResult, StubJob};
use kortex_vector::{dot, VectorSet};

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
        /// Corpus hardness: v2 (default) or v1 (legacy single-template).
        #[arg(long, default_value = "v2")]
        difficulty: String,
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
        /// Corpus hardness: v2 (default) or v1 (legacy single-template).
        #[arg(long, default_value = "v2")]
        difficulty: String,
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
        /// Corpus hardness: v2 (default) or v1 (legacy single-template).
        #[arg(long, default_value = "v2")]
        difficulty: String,
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
        /// Corpus hardness: v2 (default) or v1 (legacy single-template).
        #[arg(long, default_value = "v2")]
        difficulty: String,
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
        /// Raw little-endian f32 file, row-major, row length = --dim;
        /// overrides --source. The last --queries rows are held out as
        /// queries; --vectors caps the base set (no subsampling).
        #[arg(long)]
        vectors_file: Option<PathBuf>,
    },
    /// Stage 1 at real scale: stream-build a 1M-5M vector index on disk and
    /// measure actual two-stage latency, recall, resident RAM and cold open —
    /// no projections.
    ScaleBench {
        /// Base vector count: "1m", "5m", "500k", or a raw integer.
        #[arg(long, default_value = "1m")]
        scale: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 256)]
        dim: usize,
        #[arg(long, default_value_t = 64)]
        clusters: usize,
        /// Held-out queries (also the sample for exact ground truth).
        #[arg(long, default_value_t = 50)]
        queries: usize,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Coarse-stage shortlist size(s) fed to the float rerank; a comma
        /// list reuses one build/truth pass across all settings.
        #[arg(long, default_value = "1024,4096,16384,65536")]
        refine: String,
        /// Working directory for the multi-GB float file (default: temp dir).
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Keep the working directory after the run.
        #[arg(long, default_value_t = false)]
        keep: bool,
        #[arg(long)]
        report_out: Option<PathBuf>,
        /// Exit non-zero unless recall/latency/RAM budgets hold at this scale.
        #[arg(long, default_value_t = false)]
        assert_gate: bool,
        /// Raw little-endian f32 file, row-major, row length = --dim;
        /// overrides --scale (no synthetic generation, no disk-space check).
        /// The last --queries rows are held out as queries.
        #[arg(long)]
        vectors_file: Option<PathBuf>,
    },
    /// Stage 4: simulate the sleep consolidation scheduler over a scripted
    /// timeline with stub jobs, verifying all six gate metrics.
    SleepSim {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        report_out: Option<PathBuf>,
        /// Exit non-zero if any gate metric fails (for CI).
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
    let insight = score_insights(
        &discovered,
        &corpus.ground_truth.insights,
        &corpus.ground_truth.negative_bridges,
    );

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

fn parse_difficulty(s: &str) -> Result<Difficulty> {
    match s.to_ascii_lowercase().as_str() {
        "v1" => Ok(Difficulty::V1),
        "v2" => Ok(Difficulty::V2),
        other => anyhow::bail!("unknown difficulty '{other}' (expected v1 or v2)"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Generate {
            seed,
            years,
            entries_per_day,
            difficulty,
            out,
        } => {
            let corpus = generate(&GenConfig {
                seed,
                years,
                entries_per_day,
                difficulty: parse_difficulty(&difficulty)?,
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
            difficulty,
            k,
        } => {
            let corpus = generate(&GenConfig {
                seed,
                years,
                entries_per_day,
                difficulty: parse_difficulty(&difficulty)?,
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
            difficulty,
            report_out,
            assert_gate,
        } => {
            let report =
                run_store_bench(seed, years, entries_per_day, parse_difficulty(&difficulty)?)?;
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
                    && report.ingest_units_per_sec >= 50_000.0
                    && report.durable_events_per_sec >= 100.0
                    && report.reopen_ms < 1_000.0;
                if pass {
                    println!("\nGATE: PASS (store-bench --assert-gate satisfied)");
                } else {
                    anyhow::bail!(
                        "Stage 2 gate FAILED: content_overhead={:.3}x (budget 1.25x), index={:.1} B/unit (budget 24), p95={:.0}us (budget <1000us), ingest={:.0} units/s (budget >=50k), durable={:.0} events/s (budget >=100), reopen={:.1}ms (budget <1000ms)",
                        report.content_overhead,
                        report.index_bytes_per_unit,
                        report.p95_us,
                        report.ingest_units_per_sec,
                        report.durable_events_per_sec,
                        report.reopen_ms
                    );
                }
            }
        }
        Cmd::CompressBench {
            source,
            seed,
            years,
            entries_per_day,
            difficulty,
            dim,
            vectors,
            queries,
            k,
            report_out,
            assert_gate,
            vectors_file,
        } => {
            let (base, qset, label) = match &vectors_file {
                Some(path) => load_vectors_file(path, dim, vectors, queries)?,
                None => build_vectors(
                    &source,
                    seed,
                    years,
                    entries_per_day,
                    parse_difficulty(&difficulty)?,
                    dim,
                    vectors,
                    queries,
                ),
            };
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
        Cmd::ScaleBench {
            scale,
            seed,
            dim,
            clusters,
            queries,
            k,
            refine,
            dir,
            keep,
            report_out,
            assert_gate,
            vectors_file,
        } => {
            let n = parse_scale(&scale)?;
            let refines: Vec<usize> = refine
                .split(',')
                .map(|s| s.trim().parse::<usize>().context("bad --refine entry"))
                .collect::<Result<_>>()?;
            anyhow::ensure!(!refines.is_empty(), "--refine must list at least one size");
            let report = run_scale_bench(
                n,
                seed,
                dim,
                clusters,
                queries,
                k,
                &refines,
                dir,
                keep,
                vectors_file,
            )?;
            print_scale_report(&report);
            if let Some(path) = report_out {
                let file = std::fs::File::create(&path)
                    .with_context(|| format!("creating {}", path.display()))?;
                serde_json::to_writer_pretty(file, &report).context("writing report")?;
                println!("Report written to {}", path.display());
            }
            if assert_gate {
                let budgets = Budgets::default();
                let pass = report.runs.iter().find(|r| {
                    r.recall_at_k >= budgets.recall_at_k
                        && r.latency.p95_ms <= budgets.p95_ms
                        && report.resident_mb_at_5m <= budgets.idle_ram_mb_at_5m
                });
                if let Some(r) = pass {
                    println!(
                        "\nGATE: PASS at n={} via refine={} (recall {:.3}, p95 {:.1}ms, resident@5M {:.1}MB)",
                        report.vectors,
                        r.refine,
                        r.recall_at_k,
                        r.latency.p95_ms,
                        report.resident_mb_at_5m
                    );
                } else {
                    anyhow::bail!(
                        "Stage 1 scale gate FAILED at n={}: no refine in {:?} met recall>={:.2} with p95<={:.0}ms and resident@5M<={:.0}MB",
                        report.vectors,
                        refines,
                        budgets.recall_at_k,
                        budgets.p95_ms,
                        budgets.idle_ram_mb_at_5m
                    );
                }
            }
        }
        Cmd::SleepSim {
            seed,
            report_out,
            assert_gate,
        } => {
            let result = run_sleep_sim(seed)?;
            print_sleep_sim(&result);
            if let Some(path) = report_out {
                let file = std::fs::File::create(&path)
                    .with_context(|| format!("creating {}", path.display()))?;
                serde_json::to_writer_pretty(file, &result).context("writing report")?;
                println!("Report written to {}", path.display());
            }
            if assert_gate {
                let verdict = check_gate(&result);
                if verdict.passed {
                    println!("\nGATE: PASS (sleep-sim --assert-gate satisfied)");
                } else {
                    anyhow::bail!("Stage 4 gate FAILED: {}", verdict.failures.join("; "));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 1 scale bench — measured, not projected
// ---------------------------------------------------------------------------

fn parse_scale(s: &str) -> Result<usize> {
    let t = s.to_ascii_lowercase();
    let parsed = if let Some(m) = t.strip_suffix('m') {
        m.parse::<usize>().map(|v| v * 1_000_000)
    } else if let Some(kk) = t.strip_suffix('k') {
        kk.parse::<usize>().map(|v| v * 1_000)
    } else {
        t.parse::<usize>()
    };
    let n = parsed.with_context(|| format!("cannot parse scale '{s}' (try 1m, 5m, 500k)"))?;
    anyhow::ensure!(n >= 1_000, "scale {n} too small to be meaningful");
    Ok(n)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScaleRefineRun {
    refine: usize,
    recall_at_k: f64,
    latency: LatencySummary,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScaleBenchReport {
    vectors: usize,
    dim: usize,
    queries: usize,
    k: usize,
    float_file_bytes: u64,
    gen_secs: f64,
    build_secs: f64,
    truth_secs: f64,
    /// One timed run per --refine setting, sharing the same build/truth.
    runs: Vec<ScaleRefineRun>,
    cold_open_first_query_ms: f64,
    resident_index_bytes: u64,
    resident_index_mb: f64,
    /// Linear extrapolation to 5M vectors — sound for RAM because the index
    /// is strictly per-vector (codes + factor + popcount), unlike latency,
    /// which is only ever reported as measured.
    resident_mb_at_5m: f64,
    peak_rss_mb: f64,
}

/// Stream rows of a raw little-endian f32 file through a callback. Stops
/// after `limit` rows when `Some` (used to exclude held-out query rows that
/// live at the tail of a user-supplied vectors file).
fn stream_rows(
    path: &std::path::Path,
    dim: usize,
    limit: Option<u32>,
    mut f: impl FnMut(u32, &[f32]),
) -> Result<()> {
    use std::io::Read;
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = std::io::BufReader::with_capacity(1 << 22, file);
    let row_bytes = dim * 4;
    let mut buf = vec![0u8; row_bytes];
    let mut row = vec![0f32; dim];
    let mut id: u32 = 0;
    loop {
        if let Some(limit) = limit {
            if id >= limit {
                break;
            }
        }
        match reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        for (d, r) in row.iter_mut().enumerate() {
            *r = f32::from_le_bytes(buf[d * 4..d * 4 + 4].try_into().unwrap());
        }
        f(id, &row);
        id += 1;
    }
    Ok(())
}

/// Maintain a top-k set with `top[0]` always the current worst entry, using
/// the same ordering as `select_top_k` (score desc, then id asc).
fn top_push(top: &mut Vec<(f32, u32)>, k: usize, s: f32, id: u32) {
    let worse = |a: (f32, u32), b: (f32, u32)| a.0 < b.0 || (a.0 == b.0 && a.1 > b.1);
    if top.len() < k {
        top.push((s, id));
        if top.len() == k {
            // Establish the invariant: worst at index 0.
            let mut mi = 0;
            for i in 1..top.len() {
                if worse(top[i], top[mi]) {
                    mi = i;
                }
            }
            top.swap(0, mi);
        }
        return;
    }
    if worse(top[0], (s, id)) {
        top[0] = (s, id);
        let mut mi = 0;
        for i in 1..top.len() {
            if worse(top[i], top[mi]) {
                mi = i;
            }
        }
        top.swap(0, mi);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_scale_bench(
    n: usize,
    seed: u64,
    dim: usize,
    clusters: usize,
    num_queries: usize,
    k: usize,
    refines: &[usize],
    dir: Option<PathBuf>,
    keep: bool,
    vectors_file: Option<PathBuf>,
) -> Result<ScaleBenchReport> {
    use std::io::Write;

    // When a real-embedding file is supplied: it already lives on disk (so no
    // synthetic generation, no disk-space check), and it's the user's file
    // (so we never delete it). `n`, `fpath`, `centroid` and `queries` are all
    // derived from it instead of from the synthetic sampler.
    let (dir, created_dir, fpath, n, centroid, queries, gen_secs) = if let Some(vfile) =
        vectors_file
    {
        println!(
            "scale-bench: --vectors-file given ({}), ignoring --scale; dim={dim}",
            vfile.display()
        );
        let row_bytes = dim * 4;
        let file_len = std::fs::metadata(&vfile)
            .with_context(|| format!("stat {}", vfile.display()))?
            .len();
        anyhow::ensure!(
            row_bytes > 0 && file_len % row_bytes as u64 == 0,
            "vectors file {} has {file_len} bytes, not a multiple of dim*4={row_bytes}",
            vfile.display()
        );
        let total = (file_len / row_bytes as u64) as usize;
        anyhow::ensure!(
            total > num_queries,
            "vectors file {} has only {total} rows, need more than --queries={num_queries}",
            vfile.display()
        );
        let n = total - num_queries;
        anyhow::ensure!(
                n >= 1000,
                "vectors file {} yields only n={n} base rows after holding out {num_queries} queries (need >= 1000)",
                vfile.display()
            );

        let (dir, created_dir) = match dir {
            Some(d) => (d, false),
            None => (
                std::env::temp_dir().join(format!("kortex_scale_{}", std::process::id())),
                true,
            ),
        };
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

        let t0 = Instant::now();
        // Held-out queries: the LAST num_queries rows. One full pass over
        // the file, keeping only rows with id >= n — simple and correct,
        // and the file is read again in passes 1/2 anyway.
        let mut queries = VectorSet::new(dim);
        stream_rows(&vfile, dim, None, |id, row| {
            if id as usize >= n {
                queries.push(row);
            }
        })?;
        anyhow::ensure!(
            queries.count() == num_queries,
            "expected {num_queries} held-out query rows, got {}",
            queries.count()
        );

        // Centroid over the first n rows, accumulated in f64 (matches the
        // synthetic path's precision discipline).
        let mut centroid_sum = vec![0f64; dim];
        stream_rows(&vfile, dim, Some(n as u32), |_, row| {
            for (d, x) in row.iter().enumerate() {
                centroid_sum[d] += *x as f64;
            }
        })?;
        let centroid: Vec<f32> = centroid_sum.iter().map(|s| (s / n as f64) as f32).collect();
        let gen_secs = t0.elapsed().as_secs_f64();
        println!("pass 0 (read centroid + queries from file): {gen_secs:.1}s");

        (dir, created_dir, vfile, n, centroid, queries, gen_secs)
    } else {
        let dir = dir.unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kortex_scale_{}", std::process::id()))
        });
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let float_file_bytes = (n * dim * 4) as u64;
        if let Some(avail) = available_disk_bytes(&dir) {
            let need = float_file_bytes + float_file_bytes / 10;
            anyhow::ensure!(
                avail > need,
                "scale-bench needs ~{:.1} GB free at {}, only {:.1} GB available",
                need as f64 / 1e9,
                dir.display(),
                avail as f64 / 1e9
            );
        }
        let fpath = dir.join("base.f32");
        println!(
            "scale-bench: n={n}, dim={dim}, float file {:.2} GB at {}",
            float_file_bytes as f64 / 1e9,
            fpath.display()
        );

        // Pass 0 — stream-generate base vectors to disk; accumulate the
        // centroid in f64 so 5M additions don't lose precision.
        let t0 = Instant::now();
        let mut sampler = ClusterSampler::new(seed, dim, clusters, 0.45);
        let mut centroid_sum = vec![0f64; dim];
        {
            let file = std::fs::File::create(&fpath)
                .with_context(|| format!("creating {}", fpath.display()))?;
            let mut w = std::io::BufWriter::with_capacity(1 << 20, file);
            let mut v = vec![0f32; dim];
            for _ in 0..n {
                sampler.fill(&mut v);
                for (d, x) in v.iter().enumerate() {
                    centroid_sum[d] += *x as f64;
                }
                for x in &v {
                    w.write_all(&x.to_le_bytes())?;
                }
            }
            w.flush()?;
        }
        // Held-out queries: the next draws from the same distribution.
        let mut queries = VectorSet::new(dim);
        {
            let mut v = vec![0f32; dim];
            for _ in 0..num_queries {
                sampler.fill(&mut v);
                queries.push(&v);
            }
        }
        let centroid: Vec<f32> = centroid_sum.iter().map(|s| (s / n as f64) as f32).collect();
        let gen_secs = t0.elapsed().as_secs_f64();
        println!("pass 0 (generate -> disk): {gen_secs:.1}s");

        (dir, true, fpath, n, centroid, queries, gen_secs)
    };
    let float_file_bytes = (n * dim * 4) as u64;

    // Pass 1 — stream the file into the incremental index builder. Excludes
    // held-out query rows (which live past index n in a user file).
    let t1 = Instant::now();
    let mut builder = RaBitQBuilder::new(dim, 1, seed, centroid);
    stream_rows(&fpath, dim, Some(n as u32), |_, row| builder.push(row))?;
    let rq = builder.finish();
    let resident_index_bytes = rq.resident_bytes() as u64;
    rq.save(&dir)?;
    drop(rq);
    let build_secs = t1.elapsed().as_secs_f64();
    println!("pass 1 (build 1-bit index): {build_secs:.1}s");

    // Pass 2 — exact ground truth by streaming the base portion once.
    let t2 = Instant::now();
    let mut tops: Vec<Vec<(f32, u32)>> = vec![Vec::with_capacity(k); num_queries];
    stream_rows(&fpath, dim, Some(n as u32), |id, row| {
        for (qi, top) in tops.iter_mut().enumerate() {
            top_push(top, k, dot(queries.get(qi), row), id);
        }
    })?;
    let truth: Vec<Vec<u32>> = tops
        .into_iter()
        .map(|mut t| {
            t.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            t.into_iter().map(|(_, id)| id).collect()
        })
        .collect();
    let truth_secs = t2.elapsed().as_secs_f64();
    println!("pass 2 (exact ground truth): {truth_secs:.1}s");

    // Cold-ish open: load the saved index from disk, mmap the float file, run
    // the first query end-to-end. (Page cache is warm from the build — a true
    // cold start needs a reboot/purge — so treat this as a lower bound.)
    let t3 = Instant::now();
    let rq = RaBitQ::load(&dir).context("loading saved index")?;
    let fmm = FloatMmapStore::open(&fpath, dim).context("mmap of float file")?;
    let _ = two_stage_search(&rq, &fmm, queries.get(0), k, refines[0]);
    let cold_open_first_query_ms = t3.elapsed().as_secs_f64() * 1000.0;

    // Timed queries: full coarse scan over n + float rerank from mmap, once
    // per refine setting (the expensive passes above are shared).
    let mut runs = Vec::with_capacity(refines.len());
    for &refine in refines {
        let mut durations: Vec<Duration> = Vec::with_capacity(num_queries);
        let mut racc = 0.0;
        for (qi, t) in truth.iter().enumerate() {
            let q = queries.get(qi);
            let start = Instant::now();
            let got = two_stage_search(&rq, &fmm, q, k, refine);
            durations.push(start.elapsed());
            racc += recall_overlap(&got, t);
        }
        let recall_at_k = racc / num_queries.max(1) as f64;
        let latency = LatencySummary::from_durations(&durations);
        println!(
            "refine {refine:>6}: recall@{k}={recall_at_k:.3}  p50={:.1}ms p95={:.1}ms",
            latency.p50_ms, latency.p95_ms
        );
        runs.push(ScaleRefineRun {
            refine,
            recall_at_k,
            latency,
        });
    }

    // Drop the mmap before any cleanup that might remove files under it.
    drop(fmm);
    drop(rq);

    // Only remove the working dir if we created it (never the user's
    // --vectors-file, which lives elsewhere and is never copied here).
    if !keep && created_dir {
        let _ = std::fs::remove_dir_all(&dir);
    } else if keep {
        println!("kept working dir: {}", dir.display());
    }

    let resident_index_mb = resident_index_bytes as f64 / 1_048_576.0;
    Ok(ScaleBenchReport {
        vectors: n,
        dim,
        queries: num_queries,
        k,
        float_file_bytes,
        gen_secs,
        build_secs,
        truth_secs,
        runs,
        cold_open_first_query_ms,
        resident_index_bytes,
        resident_index_mb,
        resident_mb_at_5m: resident_index_mb * (5_000_000.0 / n as f64),
        peak_rss_mb: rss_mb(),
    })
}

fn print_scale_report(r: &ScaleBenchReport) {
    println!("\n=== Kortex Stage 1 — Scale Benchmark (measured) ===");
    println!("vectors            : {} (dim {})", r.vectors, r.dim);
    println!(
        "pipeline           : 1-bit RaBitQ popcount scan -> float32 rerank from mmap; {} held-out queries",
        r.queries
    );
    println!(
        "{:<10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "refine", "recall@k", "p50_ms", "p95_ms", "mean_ms", "max_ms"
    );
    for run in &r.runs {
        println!(
            "{:<10} {:>10.3} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            run.refine,
            run.recall_at_k,
            run.latency.p50_ms,
            run.latency.p95_ms,
            run.latency.mean_ms,
            run.latency.max_ms
        );
    }
    println!(
        "cold open+query    : {:.1} ms (load index + mmap + first search; warm page cache -> lower bound)",
        r.cold_open_first_query_ms
    );
    println!(
        "resident index     : {:.1} MB measured at n={} ({:.1} B/vec) -> {:.1} MB at 5M (linear)",
        r.resident_index_mb,
        r.vectors,
        r.resident_index_bytes as f64 / r.vectors as f64,
        r.resident_mb_at_5m
    );
    println!(
        "float file (disk)  : {:.2} GB, rerank served via mmap",
        r.float_file_bytes as f64 / 1e9
    );
    println!(
        "peak RSS (process) : {:.1} MB (includes build + truth passes; mmap-touched file pages are clean and reclaimable under memory pressure — the index keeps only {:.1} MB hot)",
        r.peak_rss_mb, r.resident_index_mb
    );
    println!(
        "timings            : generate {:.1}s, build {:.1}s, exact truth {:.1}s",
        r.gen_secs, r.build_secs, r.truth_secs
    );
}

fn rss_mb() -> f64 {
    peak_rss_bytes().unwrap_or(0) as f64 / 1_048_576.0
}

/// Build a base vector set + a held-out query set from the chosen source.
#[allow(clippy::too_many_arguments)]
fn build_vectors(
    source: &str,
    seed: u64,
    years: u32,
    entries_per_day: u32,
    difficulty: Difficulty,
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
                difficulty,
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

/// Load base + held-out query vectors from a raw little-endian f32 file
/// (row-major, row length = `dim`). The last `queries` rows are held out;
/// `cap` bounds the base set from the front (no subsampling — real-embedding
/// runs want the actual rows, not a stride sample).
fn load_vectors_file(
    path: &std::path::Path,
    dim: usize,
    cap: usize,
    queries: usize,
) -> Result<(VectorSet, VectorSet, String)> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading vectors file {}", path.display()))?;
    let row_bytes = dim * 4;
    anyhow::ensure!(
        row_bytes > 0 && bytes.len() % row_bytes == 0,
        "vectors file {} has {} bytes, not a multiple of dim*4={row_bytes}",
        path.display(),
        bytes.len()
    );
    let total = bytes.len() / row_bytes;
    anyhow::ensure!(
        total >= queries + 10,
        "vectors file {} has only {total} rows, need at least queries+10={}",
        path.display(),
        queries + 10
    );

    let read_row = |i: usize, out: &mut [f32]| {
        let start = i * row_bytes;
        for (d, x) in out.iter_mut().enumerate() {
            let off = start + d * 4;
            *x = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        }
    };

    let n_base_rows = total - queries;
    let n_base = n_base_rows.min(cap);
    let mut base = VectorSet::new(dim);
    let mut row = vec![0f32; dim];
    for i in 0..n_base {
        read_row(i, &mut row);
        base.push(&row);
    }

    let mut qset = VectorSet::new(dim);
    for i in n_base_rows..total {
        read_row(i, &mut row);
        qset.push(&row);
    }

    Ok((
        base,
        qset,
        format!("vectors-file {} (dim {dim})", path.display()),
    ))
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
    /// Buffered ingest: appends with a single sync() at the end.
    ingest_units_per_sec: f64,
    /// Durable ingest: fsync after every event (the worst-case capture mode).
    durable_events_per_sec: f64,
    /// Cold reopen of the full store + one random read (Stage 2B headline:
    /// must not scan the whole history).
    reopen_ms: f64,
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

fn run_store_bench(
    seed: u64,
    years: u32,
    entries_per_day: u32,
    difficulty: Difficulty,
) -> Result<StoreBenchReport> {
    let corpus = generate(&GenConfig {
        seed,
        years,
        entries_per_day,
        difficulty,
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

    // Snapshot accounting, then measure a cold reopen of the same store: the
    // Stage 2B contract is that open() scans at most one segment tail (dedup
    // runs + counts meta carry the rest), so this must stay fast no matter
    // how much history accumulated.
    let event_count = store.event_count();
    let raw_text_bytes = store.raw_text_bytes();
    let event_store_bytes = store.event_store_bytes();
    let unit_store_bytes = store.unit_store_bytes();
    let store_bytes = store.store_bytes();
    let probe_uid = uids.first().copied();
    drop(store);
    let reopen_start = Instant::now();
    let reopened = Store::open(&dir).context("reopening store")?;
    if let Some(uid) = probe_uid {
        let _ = reopened.unit_text(uid);
    }
    let reopen_ms = reopen_start.elapsed().as_secs_f64() * 1000.0;
    anyhow::ensure!(
        reopened.event_count() == event_count && reopened.unit_count() == unit_count,
        "reopen lost data: events {} -> {}, units {} -> {}",
        event_count,
        reopened.event_count(),
        unit_count,
        reopened.unit_count()
    );
    drop(reopened);
    let _ = std::fs::remove_dir_all(&dir);

    // Durable ingest: a separate small store, fsync after every event. This
    // is the worst-case capture mode; human capture is a few events/minute,
    // so triple-digit events/sec leaves orders of magnitude of margin.
    let ddir = std::env::temp_dir().join(format!("kortex_store_bench_d_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ddir);
    let mut dstore = Store::open(&ddir).context("opening durable store")?;
    let durable_n = 200usize;
    let durable_start = Instant::now();
    for i in 0..durable_n {
        dstore.append_event(&RawEvent {
            source: Source::Text,
            created_utc_secs: i as i64,
            text: format!("durable capture event {i}"),
        })?;
        dstore.sync()?;
    }
    let durable_events_per_sec = durable_n as f64 / durable_start.elapsed().as_secs_f64().max(1e-9);
    drop(dstore);
    let _ = std::fs::remove_dir_all(&ddir);

    Ok(StoreBenchReport {
        corpus_entries: corpus.entries.len(),
        event_count,
        unit_count,
        raw_text_bytes,
        event_store_bytes,
        unit_store_bytes,
        store_bytes,
        content_overhead,
        index_bytes_per_unit,
        p50_us,
        p95_us,
        ingest_elapsed_secs,
        ingest_units_per_sec,
        durable_events_per_sec,
        reopen_ms,
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
        "ingest (buffered) : {:.0} units/s  (budget: >= 50000 units/s, {:.3}s)",
        r.ingest_units_per_sec, r.ingest_elapsed_secs
    );
    println!(
        "ingest (durable)  : {:.0} events/s, fsync per event  (budget: >= 100/s)",
        r.durable_events_per_sec
    );
    println!(
        "cold reopen       : {:.1} ms incl. one random read  (budget: < 1000 ms)",
        r.reopen_ms
    );
    let ok_content = r.content_overhead <= 1.25;
    let ok_index = r.index_bytes_per_unit <= 24.0;
    let ok_p95 = r.p95_us >= 0.0 && r.p95_us < 1_000.0;
    let ok_ingest = r.ingest_units_per_sec >= 50_000.0;
    let ok_durable = r.durable_events_per_sec >= 100.0;
    let ok_reopen = r.reopen_ms < 1_000.0;
    println!();
    println!(
        "gate check: content={} index={} p95={} ingest={} durable={} reopen={}",
        if ok_content { "PASS" } else { "FAIL" },
        if ok_index { "PASS" } else { "FAIL" },
        if ok_p95 { "PASS" } else { "FAIL" },
        if ok_ingest { "PASS" } else { "FAIL" },
        if ok_durable { "PASS" } else { "FAIL" },
        if ok_reopen { "PASS" } else { "FAIL" },
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

// ---------------------------------------------------------------------------
// Stage 4: Sleep simulation CLI
// ---------------------------------------------------------------------------

/// Simple deterministic SplitMix64 PRNG — same constants as kortex-corpus.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn run_sleep_sim(seed: u64) -> anyhow::Result<SimResult> {
    use kortex_sleep::TimelineSegment;

    let mut rng = seed;

    // Build a scripted week timeline from the seed — deterministic pattern
    // of alternating active and blocked windows.
    let mut timeline: Vec<TimelineSegment> = Vec::new();
    let mut tick = 1u64;
    // Active: overnight (idle + power on), 1h = ~7200 ticks (at 0.5s/tick)
    // Simulate a week with ~2 active windows per day (night + afternoon idle).
    for _day in 0..7u64 {
        // Morning: active for ~7200 ticks (simulated 1h)
        let dur = 7200;
        timeline.push(TimelineSegment {
            tick_start: tick,
            tick_end: tick + dur - 1,
            on_power: true,
            user_idle: true,
            thermal_ok: true,
        });
        tick += dur;

        // Mid-day: blocked (user active, on battery or mixed)
        let dur = 3600;
        timeline.push(TimelineSegment {
            tick_start: tick,
            tick_end: tick + dur - 1,
            on_power: false,
            user_idle: false,
            thermal_ok: false,
        });
        tick += dur;

        // Night: active again
        let dur = 14400;
        timeline.push(TimelineSegment {
            tick_start: tick,
            tick_end: tick + dur - 1,
            on_power: true,
            user_idle: true,
            thermal_ok: true,
        });
        tick += dur;
    }
    // Final drain window
    timeline.push(TimelineSegment {
        tick_start: tick,
        tick_end: tick + 100000,
        on_power: true,
        user_idle: true,
        thermal_ok: true,
    });

    // Generate a deterministic backlog of stub jobs from the seed.
    let num_jobs = 5 + (splitmix64(&mut rng) % 11) as usize; // 5-15 jobs
    let mut jobs: Vec<StubJob> = Vec::new();
    for i in 0..num_jobs {
        let priority = (splitmix64(&mut rng) % 10) as u8 + 1; // 1-10
        let total_work = 50 + splitmix64(&mut rng) % 451; // 50-500
        let items_per_step = 5 + splitmix64(&mut rng) % 26; // 5-30
        jobs.push(StubJob::new(
            format!("sim_job_{i}"),
            priority,
            total_work,
            items_per_step,
        ));
    }

    Ok(run_sim(SimConfig { timeline, jobs }))
}

fn print_sleep_sim(result: &SimResult) {
    println!("\n=== Kortex Stage 4 — Sleep Simulation ===");
    println!("total work         : {} items", result.total_work);
    println!(
        "completed          : {} ({:.1}%)",
        result.completed_work,
        result.completed_work as f64 / result.total_work.max(1) as f64 * 100.0
    );
    println!("work during blocked: {}", result.work_during_blocked);
    println!("total ticks        : {}", result.total_ticks);

    let ran_ticks = result.trace.iter().filter(|e| e.action == "ran").count();
    let blocked_ticks = result
        .trace
        .iter()
        .filter(|e| e.action == "blocked")
        .count();
    let drained_at = result
        .trace
        .iter()
        .find(|e| e.action == "drained")
        .map(|e| e.tick);
    println!(
        "trace              : {ran_ticks} ran, {blocked_ticks} blocked, drained at tick {:?}",
        drained_at
    );

    // Print the first and last few trace entries for inspection.
    println!();
    println!(
        "{:<8} {:<8} {:<10} {:>10} {:>10}",
        "tick", "action", "job", "work_done", "pending"
    );
    for entry in result.trace.iter().take(5) {
        println!(
            "{:<8} {:<8} {:<10} {:>10} {:>10}",
            entry.tick,
            entry.action,
            entry.job_id.as_deref().unwrap_or("-"),
            entry.work_done.map(|w| w.to_string()).unwrap_or_default(),
            entry.pending,
        );
    }
    if result.trace.len() > 10 {
        println!("  ...  ");
    }
    for entry in result.trace.iter().rev().take(5).rev() {
        println!(
            "{:<8} {:<8} {:<10} {:>10} {:>10}",
            entry.tick,
            entry.action,
            entry.job_id.as_deref().unwrap_or("-"),
            entry.work_done.map(|w| w.to_string()).unwrap_or_default(),
            entry.pending,
        );
    }

    // Gate summary
    let verdict = check_gate(result);
    if verdict.passed {
        println!("\nGATE CHECK: PASS — all six metrics satisfied.");
    } else {
        println!("\nGATE CHECK: FAIL");
        for f in &verdict.failures {
            println!("  - {f}");
        }
    }
}
