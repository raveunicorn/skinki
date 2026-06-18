#![forbid(unsafe_code)]
//! skinki harness CLI.
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

use skinki_baseline::Bm25;
use skinki_corpus::{generate, Corpus, Difficulty, EntryId, GenConfig};
use skinki_eval::{
    answer_in_entries, ndcg_at_k, precision_at_k, recall_at_k, score_insights, Latency, Report,
    RetrievalScores, RetrievalSystem,
};
use skinki_store::{derive_units, RawEvent, Source, Store};
use skinki_telemetry::{peak_rss_bytes, LatencySummary};
use skinki_vector::bench::{passes_gate, run_matrix, verdict, BenchReport, Budgets};
use skinki_vector::embed::{synthetic_clusters, ClusterSampler, Embedder, StaticHashEmbedder};
use skinki_vector::ivf::{IvfBuilder, IvfRaBitQ};
use skinki_vector::quant::{RaBitQ, RaBitQBuilder};
use skinki_vector::search::{ivf_two_stage_search, recall as recall_overlap, two_stage_search};
use skinki_vector::store::{available_disk_bytes, FloatMmapStore};

use skinki_graph::{
    assemble_context, ArtifactLog, GraphRetriever, LlmExtraction, RelationRetriever,
};
use skinki_ledger::{score_staleness, ContentHash, Derivation, Ledger, MethodStamp};
use skinki_sleep::{check_gate, run_sim, SimConfig, SimResult, StubJob};
use skinki_vector::{dot, VectorSet};

mod llm_graph;
mod locomo;
use locomo::{load_locomo, LocomoSample};

#[derive(Parser)]
#[command(
    name = "skinki",
    about = "skinki Stage 0 — synthetic corpus + eval harness"
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
        /// Index type: "flat" (1-bit RaBitQ, global centroid) or "ivf"
        /// (per-list 1-bit RaBitQ residual codes).
        #[arg(long, default_value = "flat")]
        index: String,
        /// Comma list of nprobe settings (ivf only); cross-producted with
        /// --refine.
        #[arg(long, default_value = "16,64,256")]
        nprobe: String,
        /// IVF list count; 0 = auto heuristic (see `IvfBuilder::train`).
        #[arg(long, default_value_t = 0)]
        nlist: usize,
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
    /// Derivation Ledger: build a derivation DAG over the corpus's planted
    /// conclusions, supersede each planted contradiction, and measure whether
    /// staleness propagates to exactly the dependent conclusions — versus a
    /// fact-storage baseline that detects nothing.
    LedgerBench {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 5)]
        years: u32,
        #[arg(long, default_value_t = 6)]
        entries_per_day: u32,
        #[arg(long, default_value = "v2")]
        difficulty: String,
        #[arg(long)]
        report_out: Option<PathBuf>,
        /// Exit non-zero if the ledger doesn't reach recall 1.0 at 0
        /// over-invalidation with real signal (for CI).
        #[arg(long, default_value_t = false)]
        assert_gate: bool,
    },
    /// Stage 3 MVP: score the deterministic co-mention graph retriever
    /// alongside BM25, printing a single-hop / multi-hop contrast table.
    GraphEval {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 5)]
        years: u32,
        #[arg(long, default_value_t = 6)]
        entries_per_day: u32,
        #[arg(long, default_value = "v2")]
        difficulty: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Exit non-zero unless the relation retriever clears the Stage 3 gate
        /// (multi-hop recall@k >= 0.50, ans@k >= 0.60, no single-hop regression).
        #[arg(long, default_value_t = false)]
        assert_gate: bool,
    },
    /// DEV-ONLY: score retrievers on the LoCoMo10 real-conversation benchmark
    /// (real multi-session dialogue + memory QA — not the synthetic corpus).
    /// No gate: this is a measurement tool, not a CI check.
    LocomoEval {
        /// Path to locomo10.json (not checked into the repo).
        #[arg(long)]
        path: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// "all" (concatenate all 10 samples) or a 0-based sample index.
        #[arg(long, default_value = "all")]
        sample: String,
        /// Embedding dimension for the static (lexical) semantic retriever.
        #[arg(long, default_value_t = 256)]
        dim: usize,
        /// Optional precomputed entry embeddings: flat little-endian f32,
        /// `dim * N` floats, one row per corpus entry in entry-id order (the
        /// real-model replay slot, e.g. EmbeddingGemma).
        #[arg(long)]
        embeddings_file: Option<PathBuf>,
        /// Optional precomputed QUERY embeddings (same format), one row per
        /// recall query in query order. Required alongside `--embeddings-file`
        /// to score `semantic-real` (docs and queries must share a space).
        #[arg(long)]
        query_embeddings_file: Option<PathBuf>,
        /// Dump the canonical entry/query texts as JSON arrays to this dir
        /// (`entries.json`, `queries.json`) and exit — the input to
        /// `tools/export-embeddings-gemma.py`. No eval is run.
        #[arg(long)]
        dump_texts: Option<PathBuf>,
        /// Optional LLM extraction artifact log (JSON-lines from
        /// `tools/extract-graph-llm.py`): adds two graph columns built from the
        /// same log — `llm-graph+bm25` (entity co-mention, the honest negative)
        /// and `llm-facts+bm25` (typed-fact walk + coref + structural gate, the
        /// real-text analogue of the synthetic `RelationRetriever` win).
        #[arg(long)]
        graph_artifacts: Option<PathBuf>,
        /// Only score QA of this LoCoMo category (e.g. 2 = multi-hop — the
        /// regime the graph is meant to help). Default: all categories.
        #[arg(long)]
        category: Option<i64>,
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
            index,
            nprobe,
            nlist,
        } => {
            let n = parse_scale(&scale)?;
            let refines: Vec<usize> = refine
                .split(',')
                .map(|s| s.trim().parse::<usize>().context("bad --refine entry"))
                .collect::<Result<_>>()?;
            anyhow::ensure!(!refines.is_empty(), "--refine must list at least one size");
            let index = match index.to_ascii_lowercase().as_str() {
                "flat" => IndexKind::Flat,
                "ivf" => IndexKind::Ivf,
                other => anyhow::bail!("unknown --index '{other}' (expected flat or ivf)"),
            };
            let nprobes: Vec<usize> = if index == IndexKind::Ivf {
                nprobe
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().context("bad --nprobe entry"))
                    .collect::<Result<_>>()?
            } else {
                Vec::new()
            };
            if index == IndexKind::Ivf {
                anyhow::ensure!(!nprobes.is_empty(), "--nprobe must list at least one value");
            }
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
                index,
                &nprobes,
                nlist,
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
        Cmd::LedgerBench {
            seed,
            years,
            entries_per_day,
            difficulty,
            report_out,
            assert_gate,
        } => {
            let corpus = generate(&GenConfig {
                seed,
                years,
                entries_per_day,
                difficulty: parse_difficulty(&difficulty)?,
            });
            let report = run_ledger_bench(&corpus);
            print_ledger_bench(&report);
            if let Some(path) = report_out {
                let file = std::fs::File::create(&path)
                    .with_context(|| format!("creating {}", path.display()))?;
                serde_json::to_writer_pretty(file, &report).context("writing report")?;
                println!("Report written to {}", path.display());
            }
            if assert_gate {
                // The exact-hash policy must be perfect on this construction:
                // catch every dependent (recall 1.0), flag nothing independent
                // (over 0.0), and there must be real signal to catch.
                let ok = report.ledger_invalidation_recall >= 1.0
                    && report.ledger_over_invalidation <= 0.0
                    && report.contradictions_with_dependents > 0
                    && report.ledger_invalidation_recall > report.baseline_invalidation_recall;
                if ok {
                    println!("\nGATE: PASS (ledger-bench --assert-gate satisfied)");
                } else {
                    anyhow::bail!(
                        "Ledger gate FAILED: recall={:.3} (want 1.0), over={:.3} (want 0.0), contradictions_with_dependents={} (want >0), baseline_recall={:.3}",
                        report.ledger_invalidation_recall,
                        report.ledger_over_invalidation,
                        report.contradictions_with_dependents,
                        report.baseline_invalidation_recall
                    );
                }
            }
        }
        Cmd::GraphEval {
            seed,
            years,
            entries_per_day,
            difficulty,
            k,
            assert_gate,
        } => {
            let corpus = generate(&GenConfig {
                seed,
                years,
                entries_per_day,
                difficulty: parse_difficulty(&difficulty)?,
            });
            run_graph_eval(&corpus, k, assert_gate)?;
        }
        Cmd::LocomoEval {
            path,
            k,
            sample,
            dim,
            embeddings_file,
            query_embeddings_file,
            dump_texts,
            graph_artifacts,
            category,
        } => {
            let sample = parse_locomo_sample(&sample)?;
            let corpus = load_locomo(&path, sample, category)?;
            if let Some(dir) = dump_texts {
                dump_locomo_texts(&corpus, &dir)?;
            } else {
                run_locomo_eval(
                    &corpus,
                    k,
                    dim,
                    embeddings_file.as_deref(),
                    query_embeddings_file.as_deref(),
                    graph_artifacts.as_deref(),
                )?;
            }
        }
    }
    Ok(())
}

/// Parse `--sample`: "all" or a 0-based integer index.
fn parse_locomo_sample(s: &str) -> Result<LocomoSample> {
    if s.eq_ignore_ascii_case("all") {
        Ok(LocomoSample::All)
    } else {
        let n: usize = s
            .parse()
            .with_context(|| format!("--sample must be 'all' or an integer, got '{s}'"))?;
        Ok(LocomoSample::One(n))
    }
}

// ---------------------------------------------------------------------------
// Stage 3 MVP — graph-eval: deterministic co-mention graph vs BM25
// ---------------------------------------------------------------------------

/// Sorted set of ground-truth Person names that occur (case-insensitive
/// substring match) in `text`. Used by the Stage 3 D2 ORACLE to resolve a
/// coreference rec edge's recommender from hop A's text — this stands in for
/// what an LLM tier would infer, but is itself a deterministic function of
/// the planted ground truth (no inference at gate time).
fn person_names_in(text: &str, corpus: &Corpus) -> std::collections::BTreeSet<String> {
    let text_lower = text.to_lowercase();
    corpus
        .ground_truth
        .entities
        .iter()
        .filter(|e| e.kind == skinki_corpus::EntityKind::Person)
        .map(|e| e.name.to_lowercase())
        .filter(|n| text_lower.contains(n.as_str()))
        .collect()
}

fn run_graph_eval(corpus: &Corpus, k: usize, assert_gate: bool) -> anyhow::Result<()> {
    let mut bm25 = Bm25::new();
    bm25.index(corpus);

    let mut graph = GraphRetriever::new();
    graph.index(corpus);

    let mut relation = RelationRetriever::new();
    relation.index(corpus);

    // --- D2: build the ORACLE artifact log -----------------------------
    //
    // For each planted multi-hop chain [hopA, hopB]: if hopB is the
    // ambiguous coreference rec form (`relation.selects_for_llm_tier`), the
    // recommender Q is the person(s) named in hopA's text that are NOT named
    // in the question itself (the question already names the querying
    // person; Q is the *other* person hopA introduces). This is a
    // deterministic function of planted ground truth — it stands in for the
    // LLM tier, not a live model — and is written/replayed through the real
    // T6 artifact-log path.
    let artifact_path = std::env::temp_dir().join(format!(
        "skinki_graph_eval_artifacts_{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&artifact_path);
    for mh in &corpus.ground_truth.multi_hop {
        if mh.hop_entries.len() != 2 {
            continue;
        }
        let (hop_a, hop_b) = (mh.hop_entries[0], mh.hop_entries[1]);
        let Some(hop_b_text) = corpus.entry_text(hop_b) else {
            continue;
        };
        if !relation.selects_for_llm_tier(hop_b_text) {
            continue;
        }
        let Some(hop_a_text) = corpus.entry_text(hop_a) else {
            continue;
        };
        let hop_a_persons = person_names_in(hop_a_text, corpus);
        let question_persons = person_names_in(&mh.question, corpus);
        let resolved_by: Vec<String> = hop_a_persons
            .into_iter()
            .filter(|p| !question_persons.contains(p))
            .collect();
        if resolved_by.is_empty() {
            continue;
        }
        ArtifactLog::append(
            &artifact_path,
            &LlmExtraction {
                entry: hop_b,
                resolved_by,
                model_version: 1,
            },
        )?;
    }
    let replayed = if artifact_path.exists() {
        ArtifactLog::replay(&artifact_path)?
    } else {
        Vec::new()
    };
    let _ = std::fs::remove_file(&artifact_path);

    let mut relation_llm = RelationRetriever::new();
    relation_llm.index_with_artifacts(corpus, &replayed);

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

    let mut bm25_durations: Vec<Duration> = Vec::new();
    let mut graph_durations: Vec<Duration> = Vec::new();
    let mut relation_durations: Vec<Duration> = Vec::new();
    let mut relation_llm_durations: Vec<Duration> = Vec::new();

    let bm25_recall = score_set(&bm25, corpus, &recall_items, k, &mut bm25_durations);
    let bm25_multihop = score_set(&bm25, corpus, &multihop_items, k, &mut bm25_durations);
    let graph_recall = score_set(&graph, corpus, &recall_items, k, &mut graph_durations);
    let graph_multihop = score_set(&graph, corpus, &multihop_items, k, &mut graph_durations);
    let relation_recall = score_set(&relation, corpus, &recall_items, k, &mut relation_durations);
    let relation_multihop = score_set(
        &relation,
        corpus,
        &multihop_items,
        k,
        &mut relation_durations,
    );
    let relation_llm_recall = score_set(
        &relation_llm,
        corpus,
        &recall_items,
        k,
        &mut relation_llm_durations,
    );
    let relation_llm_multihop = score_set(
        &relation_llm,
        corpus,
        &multihop_items,
        k,
        &mut relation_llm_durations,
    );

    println!("=== skinki Stage 3 (MVP) — Graph vs BM25 contrast ===");
    println!(
        "corpus           : {} entries (seed {}, {} years, {} entries/day, {})",
        corpus.entries.len(),
        corpus.meta.seed,
        corpus.meta.years,
        corpus.entries.len() as f64 / (corpus.meta.years.max(1) as f64 * 365.0),
        match corpus.meta.difficulty {
            Difficulty::V1 => "v1",
            Difficulty::V2 => "v2",
        }
    );
    println!(
        "queries          : single-hop recall={}, multi-hop={}",
        recall_items.len(),
        multihop_items.len()
    );
    println!();
    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>9}",
        "metric", "bm25", "graph", "relation", "rel+llm"
    );
    println!(
        "{:<28} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
        format!("single-hop recall@{k}"),
        bm25_recall.recall_at_k,
        graph_recall.recall_at_k,
        relation_recall.recall_at_k,
        relation_llm_recall.recall_at_k
    );
    println!(
        "{:<28} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
        format!("single-hop ans@{k}"),
        bm25_recall.answer_in_topk,
        graph_recall.answer_in_topk,
        relation_recall.answer_in_topk,
        relation_llm_recall.answer_in_topk
    );
    println!(
        "{:<28} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
        format!("multi-hop  recall@{k}"),
        bm25_multihop.recall_at_k,
        graph_multihop.recall_at_k,
        relation_multihop.recall_at_k,
        relation_llm_multihop.recall_at_k
    );
    println!(
        "{:<28} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
        format!("multi-hop  ans@{k}"),
        bm25_multihop.answer_in_topk,
        graph_multihop.answer_in_topk,
        relation_multihop.answer_in_topk,
        relation_llm_multihop.answer_in_topk
    );
    println!(
        "\nrelation ledger  : {} edge derivations (provenance-pinned -> incremental re-extraction + staleness, T7)",
        relation.ledger().len()
    );

    // D2 — LLM-tier selection policy: the fraction of corpus units routed to
    // the (simulated) LLM tier, and the multi-hop recall@k lift it buys over
    // the deterministic relation tier alone.
    let tier1_share = relation.llm_tier_share(corpus);
    let mh_lift = relation_llm_multihop.recall_at_k - relation_multihop.recall_at_k;
    println!(
        "llm tier         : selected {} / {} units ({:.2}% of corpus) -> tier-1 share {:.4}",
        corpus
            .entries
            .iter()
            .filter(|e| relation.selects_for_llm_tier(&e.text))
            .count(),
        corpus.entries.len(),
        tier1_share * 100.0,
        tier1_share
    );
    println!("multi-hop recall@{k} lift (rel+llm - relation): {mh_lift:+.3}");

    // T8 — graph RAM telemetry: measure the graph structures and project to 5M
    // (edges scale ~linearly with entries). Budget: <= ~120 MB, leaving room for
    // the Stage-1 vector index within the 250 MB idle envelope.
    let graph_bytes = relation.graph_resident_bytes();
    let n = corpus.entries.len().max(1);
    let graph_5m_mb = (graph_bytes as f64 / n as f64) * 5_000_000.0 / 1_048_576.0;
    println!(
        "relation graph   : {:.2} MB at n={} ({:.1} B/entry) -> {:.1} MB at 5M (budget <=120)",
        graph_bytes as f64 / 1_048_576.0,
        n,
        graph_bytes as f64 / n as f64,
        graph_5m_mb
    );

    // --- Stage 3C: context-sufficiency at a fixed token budget ---------
    //
    // Instead of dumping top-k chunks, assemble a small, dense, cited, dated
    // package within `TOKEN_BUDGET` and measure whether the planted answer is
    // derivable from it (proxy: substring match, same as `answer_in_topk`,
    // but against the *budgeted package* rather than the raw top-k). Compare
    // the relation-graph assembler against a naive top-k BM25 "dump" at the
    // SAME budget.
    const TOKEN_BUDGET: usize = 512;
    let mut rel_hits = 0usize;
    let mut bm25_hits = 0usize;
    let mut rel_tokens_sum = 0usize;
    for mh in &corpus.ground_truth.multi_hop {
        let rel_pkg = assemble_context(&relation, corpus, &mh.question, TOKEN_BUDGET);
        let bm25_pkg = assemble_context(&bm25, corpus, &mh.question, TOKEN_BUDGET);
        if rel_pkg.contains_answer(&mh.answer) {
            rel_hits += 1;
        }
        if bm25_pkg.contains_answer(&mh.answer) {
            bm25_hits += 1;
        }
        rel_tokens_sum += rel_pkg.est_tokens;
    }
    let mh_n = corpus.ground_truth.multi_hop.len().max(1) as f64;
    let rel_sufficiency = rel_hits as f64 / mh_n;
    let bm25_sufficiency = bm25_hits as f64 / mh_n;
    let rel_mean_tokens = rel_tokens_sum as f64 / mh_n;
    println!(
        "context ({TOKEN_BUDGET} tok): sufficiency relation={rel_sufficiency:.3} vs bm25-dump={bm25_sufficiency:.3} (mean pkg {rel_mean_tokens:.0} tok)"
    );

    if assert_gate {
        // Stage 3 gate: the relation retriever must clear the multi-hop bars and
        // not regress single-hop recall below BM25 (the deterministic-tier
        // verdict; recall is deterministic so a fixed corpus is a stable gate).
        const MULTIHOP_RECALL_MIN: f64 = 0.50;
        const MULTIHOP_ANS_MIN: f64 = 0.60;
        const GRAPH_RAM_5M_MAX_MB: f64 = 120.0;
        // D2: the LLM tier must stay cheap (selects a small minority of units).
        // We do NOT gate on a positive lift: the oracle-ceiling measurement shows
        // the LLM tier does not reliably beat the deterministic venue+temporal
        // bridge on this corpus (it can even hurt, via person-name-collision
        // cross-talk) — an honest "not earned yet" result, surfaced as the printed
        // `mh_lift`, not a pass/fail. The gate guards the deterministic tier and
        // the tier-1 cost.
        const TIER1_SHARE_MAX: f64 = 0.05;
        // Stage 3C: the relation context package must be at least as sufficient
        // as a naive BM25 dump at the same budget, clear a 50% floor, and stay
        // within the token budget on average.
        const CTX_SUFFICIENCY_MIN: f64 = 0.50;
        let mh_recall = relation_multihop.recall_at_k;
        let mh_ans = relation_multihop.answer_in_topk;
        let sh_ok = relation_recall.recall_at_k >= bm25_recall.recall_at_k;
        let ram_ok = graph_5m_mb <= GRAPH_RAM_5M_MAX_MB;
        let tier1_ok = tier1_share <= TIER1_SHARE_MAX;
        let ctx_ok = rel_sufficiency >= CTX_SUFFICIENCY_MIN
            && rel_sufficiency >= bm25_sufficiency
            && rel_mean_tokens <= TOKEN_BUDGET as f64;
        if mh_recall >= MULTIHOP_RECALL_MIN
            && mh_ans >= MULTIHOP_ANS_MIN
            && sh_ok
            && ram_ok
            && tier1_ok
            && ctx_ok
        {
            println!(
                "\nGATE: PASS (relation multi-hop recall@{k}={mh_recall:.3} ans@{k}={mh_ans:.3}; single-hop recall {:.3} >= bm25 {:.3}; graph {graph_5m_mb:.1} MB @5M; tier-1 share {:.4} <= {TIER1_SHARE_MAX}; LLM-tier lift {mh_lift:+.3} — informational, not earned; context sufficiency relation={rel_sufficiency:.3} >= bm25-dump={bm25_sufficiency:.3} >= {CTX_SUFFICIENCY_MIN}, mean pkg {rel_mean_tokens:.0} <= {TOKEN_BUDGET} tok)",
                relation_recall.recall_at_k,
                bm25_recall.recall_at_k,
                tier1_share,
            );
        } else {
            anyhow::bail!(
                "Stage 3 gate FAILED: relation multi-hop recall@{k}={mh_recall:.3} (want >={MULTIHOP_RECALL_MIN}), ans@{k}={mh_ans:.3} (want >={MULTIHOP_ANS_MIN}), single-hop recall {:.3} vs bm25 {:.3} (no regress), graph {graph_5m_mb:.1} MB @5M (want <={GRAPH_RAM_5M_MAX_MB}), tier-1 share {:.4} (want <={TIER1_SHARE_MAX}), context sufficiency relation={rel_sufficiency:.3} (want >={CTX_SUFFICIENCY_MIN} and >= bm25-dump={bm25_sufficiency:.3}), mean pkg {rel_mean_tokens:.0} tok (want <={TOKEN_BUDGET})",
                relation_recall.recall_at_k,
                bm25_recall.recall_at_k,
                tier1_share,
            );
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

/// Index variant exercised by `scale-bench`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexKind {
    /// 1-bit RaBitQ against a single global centroid (Stage 1 baseline).
    Flat,
    /// Per-list 1-bit RaBitQ residual codes (this module).
    Ivf,
}

impl IndexKind {
    fn label(&self) -> &'static str {
        match self {
            IndexKind::Flat => "flat (1-bit RaBitQ, global centroid)",
            IndexKind::Ivf => "ivf (per-list 1-bit RaBitQ residual codes)",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScaleRefineRun {
    /// `None` for the flat index (no probing); `Some(nprobe)` for ivf.
    nprobe: Option<usize>,
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
    /// One timed run per (nprobe, refine) setting (flat: per refine only),
    /// sharing the same build/truth.
    runs: Vec<ScaleRefineRun>,
    cold_open_first_query_ms: f64,
    resident_index_bytes: u64,
    resident_index_mb: f64,
    /// Linear extrapolation to 5M vectors — sound for RAM because the index
    /// is strictly per-vector (codes + factor + popcount [+ ids for ivf]),
    /// unlike latency, which is only ever reported as measured.
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
    index: IndexKind,
    nprobes: &[usize],
    nlist: usize,
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
                std::env::temp_dir().join(format!("skinki_scale_{}", std::process::id())),
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
            std::env::temp_dir().join(format!("skinki_scale_{}", std::process::id()))
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
    println!("scale-bench: index = {}", index.label());

    // Pass 1 — build the index. The flat path is unchanged: stream the file
    // into the incremental 1-bit RaBitQ builder against the global centroid.
    // The ivf path additionally needs a training sample (pass 1a) before the
    // two assign/encode streaming passes (1b/1c).
    let t1 = Instant::now();
    // `resident_at_5m` is the budget-relevant projection. For flat every byte
    // scales with n, so the projection is linear. IVF projects honestly via
    // `resident_bytes_at` (centroid table grows as sqrt(n), not linearly), so a
    // small-N gate run reports the same ~5M RAM the at-scale runs measure.
    let (resident_index_bytes, resident_5m_bytes) = match index {
        IndexKind::Flat => {
            let mut builder = RaBitQBuilder::new(dim, 1, seed, centroid);
            stream_rows(&fpath, dim, Some(n as u32), |_, row| builder.push(row))?;
            let rq = builder.finish();
            let resident = rq.resident_bytes() as u64;
            let at_5m = (resident as f64 * 5_000_000.0 / n as f64) as u64;
            rq.save(&dir)?;
            drop(rq);
            (resident, at_5m)
        }
        IndexKind::Ivf => {
            // Pass 1a — sample pass: every stride-th row, capped at 100k, for
            // training the IVF list centroids.
            let stride = (n / 100_000).max(1);
            let mut sample = VectorSet::new(dim);
            stream_rows(&fpath, dim, Some(n as u32), |id, row| {
                if (id as usize).is_multiple_of(stride) && sample.count() < 100_000 {
                    sample.push(row);
                }
            })?;
            println!(
                "pass 1a (training sample): {} rows (stride {stride})",
                sample.count()
            );

            let mut ivf_builder = IvfBuilder::train(dim, nlist, seed, &sample, n);
            drop(sample);
            println!(
                "pass 1a (train IVF centroids): nlist={}",
                ivf_builder.nlist()
            );

            // Pass 1b — assign each base row to its nearest list.
            stream_rows(&fpath, dim, Some(n as u32), |_, row| {
                ivf_builder.assign(row)
            })?;
            ivf_builder.finalize_layout();

            // Pass 1c — encode each base row's residual into its list slot.
            stream_rows(&fpath, dim, Some(n as u32), |_, row| {
                ivf_builder.encode(row)
            })?;
            let ivf = ivf_builder.finish();
            let resident = ivf.resident_bytes() as u64;
            let at_5m = ivf.resident_bytes_at(5_000_000) as u64;
            ivf.save(&dir)?;
            drop(ivf);
            (resident, at_5m)
        }
    };
    let build_secs = t1.elapsed().as_secs_f64();
    println!("pass 1 (build index): {build_secs:.1}s");

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
    let fmm = FloatMmapStore::open(&fpath, dim).context("mmap of float file")?;
    let (rq, ivf): (Option<RaBitQ>, Option<IvfRaBitQ>) = match index {
        IndexKind::Flat => {
            let rq = RaBitQ::load(&dir).context("loading saved index")?;
            let _ = two_stage_search(&rq, &fmm, queries.get(0), k, refines[0]);
            (Some(rq), None)
        }
        IndexKind::Ivf => {
            let ivf = IvfRaBitQ::load(&dir).context("loading saved index")?;
            let _ = ivf_two_stage_search(&ivf, &fmm, queries.get(0), k, nprobes[0], refines[0]);
            (None, Some(ivf))
        }
    };
    let cold_open_first_query_ms = t3.elapsed().as_secs_f64() * 1000.0;

    // Timed queries: full coarse scan over n + float rerank from mmap, once
    // per --refine setting (flat), or once per (nprobe, refine) pair (ivf) —
    // the expensive build/truth passes above are shared across all settings.
    let probe_settings: Vec<Option<usize>> = match index {
        IndexKind::Flat => vec![None],
        IndexKind::Ivf => nprobes.iter().map(|&p| Some(p)).collect(),
    };
    let mut runs = Vec::with_capacity(probe_settings.len() * refines.len());
    for &nprobe in &probe_settings {
        for &refine in refines {
            let mut durations: Vec<Duration> = Vec::with_capacity(num_queries);
            let mut racc = 0.0;
            for (qi, t) in truth.iter().enumerate() {
                let q = queries.get(qi);
                let start = Instant::now();
                let got = match (&rq, &ivf, nprobe) {
                    (Some(rq), _, _) => two_stage_search(rq, &fmm, q, k, refine),
                    (_, Some(ivf), Some(nprobe)) => {
                        ivf_two_stage_search(ivf, &fmm, q, k, nprobe, refine)
                    }
                    _ => unreachable!("index/probe mismatch"),
                };
                durations.push(start.elapsed());
                racc += recall_overlap(&got, t);
            }
            let recall_at_k = racc / num_queries.max(1) as f64;
            let latency = LatencySummary::from_durations(&durations);
            match nprobe {
                Some(nprobe) => println!(
                    "nprobe {nprobe:>5} refine {refine:>6}: recall@{k}={recall_at_k:.3}  p50={:.1}ms p95={:.1}ms",
                    latency.p50_ms, latency.p95_ms
                ),
                None => println!(
                    "refine {refine:>6}: recall@{k}={recall_at_k:.3}  p50={:.1}ms p95={:.1}ms",
                    latency.p50_ms, latency.p95_ms
                ),
            }
            runs.push(ScaleRefineRun {
                nprobe,
                refine,
                recall_at_k,
                latency,
            });
        }
    }

    // Drop the mmap and index handles before any cleanup that might remove
    // files under them.
    drop(fmm);
    drop(rq);
    drop(ivf);

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
        resident_mb_at_5m: resident_5m_bytes as f64 / 1_048_576.0,
        peak_rss_mb: rss_mb(),
    })
}

fn print_scale_report(r: &ScaleBenchReport) {
    println!("\n=== skinki Stage 1 — Scale Benchmark (measured) ===");
    println!("vectors            : {} (dim {})", r.vectors, r.dim);
    println!(
        "pipeline           : 1-bit RaBitQ popcount scan -> float32 rerank from mmap; {} held-out queries",
        r.queries
    );
    println!(
        "{:<8} {:<10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "nprobe", "refine", "recall@k", "p50_ms", "p95_ms", "mean_ms", "max_ms"
    );
    for run in &r.runs {
        let nprobe = run
            .nprobe
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<8} {:<10} {:>10.3} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            nprobe,
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
        "resident index     : {:.1} MB measured at n={} ({:.1} B/vec) -> {:.1} MB at 5M (flat: linear; ivf: per-vec linear + sqrt(n) centroids)",
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
fn shuffle_det(seed: u64, items: &mut [skinki_store::UnitId]) {
    // Simple portable SplitMix64 shuffle — same constants as skinki-corpus::Rng.
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

    let dir = std::env::temp_dir().join(format!("skinki_store_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut store = Store::open(&dir).context("opening store")?;

    let ingest_start = Instant::now();
    let mut total_units = 0usize;
    for entry in &corpus.entries {
        let ev = RawEvent {
            source: if entry.kind == skinki_corpus::EntryKind::Voice {
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
    let mut uids: Vec<skinki_store::UnitId> = store.units().map(|(id, _)| id).collect();
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
    let ddir = std::env::temp_dir().join(format!("skinki_store_bench_d_{}", std::process::id()));
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
    println!("\n=== skinki Stage 2 — Store Benchmark ===");
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
// Stage 4 — sleep consolidation simulation
// ---------------------------------------------------------------------------

/// Simple deterministic SplitMix64 PRNG — same constants as skinki-corpus.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn run_sleep_sim(seed: u64) -> anyhow::Result<SimResult> {
    use skinki_sleep::TimelineSegment;

    let mut rng = seed;

    // Build a scripted week timeline from the seed — deterministic pattern
    // of alternating active and blocked windows.
    let mut timeline: Vec<TimelineSegment> = Vec::new();
    let mut tick = 1u64;
    // A week of day cycles. Window sizes are chosen so a realistic backlog
    // spans several days: the policy must pause through every mid-day blocked
    // window and resume at night, so the headline gate exercises all six
    // metrics — not just draining inside the first active window.
    for _day in 0..7u64 {
        // Morning: active (on power, user idle).
        let dur = 400;
        timeline.push(TimelineSegment {
            tick_start: tick,
            tick_end: tick + dur - 1,
            on_power: true,
            user_idle: true,
            thermal_ok: true,
        });
        tick += dur;

        // Mid-day: blocked (user active, on battery or throttling).
        let dur = 300;
        timeline.push(TimelineSegment {
            tick_start: tick,
            tick_end: tick + dur - 1,
            on_power: false,
            user_idle: false,
            thermal_ok: false,
        });
        tick += dur;

        // Night: active again (longer idle window).
        let dur = 800;
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

    // A backlog large enough to span multiple days of active windows, so the
    // policy is forced to stop at every blocked window and resume afterwards.
    let num_jobs = 15 + (splitmix64(&mut rng) % 11) as usize; // 15-25 jobs
    let mut jobs: Vec<StubJob> = Vec::new();
    for i in 0..num_jobs {
        let priority = (splitmix64(&mut rng) % 10) as u8 + 1; // 1-10
        let total_work = 2000 + splitmix64(&mut rng) % 3001; // 2000-5000
        let items_per_step = 15 + splitmix64(&mut rng) % 11; // 15-25
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
    println!("\n=== skinki Stage 4 — Sleep Simulation ===");
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

// ---------------------------------------------------------------------------
// Derivation Ledger benchmark — staleness propagation on planted contradictions
// ---------------------------------------------------------------------------

// Method ids for the derivation DAG built from corpus ground truth. Each kind
// of conclusion is produced by a distinct (versioned) "method".
const M_BELIEF: u32 = 1;
const M_MULTIHOP: u32 = 2;
const M_RECALL: u32 = 3;
const M_INSIGHT: u32 = 4;
const M_TEMPORAL: u32 = 5;

#[derive(serde::Serialize, serde::Deserialize)]
struct LedgerBenchReport {
    corpus_entries: usize,
    /// Total derivations in the DAG (beliefs + higher-order conclusions).
    derivations: usize,
    contradictions: usize,
    /// Contradictions whose superseded entry actually feeds a conclusion — the
    /// ones that exercise propagation.
    contradictions_with_dependents: usize,
    /// Mean conclusions a single reversal invalidates (over the above).
    mean_fanout: f64,
    max_fanout: usize,
    /// Fraction of genuinely-stale conclusions the ledger catches (want 1.0).
    ledger_invalidation_recall: f64,
    /// Fraction the ledger wrongly flags as stale (want 0.0).
    ledger_over_invalidation: f64,
    /// What a fact-storage memory (no provenance) catches — structurally 0.
    baseline_invalidation_recall: f64,
}

/// Build a derivation DAG from the corpus's planted ground truth, then for each
/// planted contradiction supersede the "before" entry and measure whether
/// staleness reaches exactly the dependent conclusions. The DAG is one tier
/// (entry premises -> conclusions); transitive propagation is covered by
/// `skinki-ledger`'s unit tests. The value shown here is scale, isolation, and
/// the contrast against a provenance-free baseline.
fn run_ledger_bench(corpus: &Corpus) -> LedgerBenchReport {
    use std::collections::{BTreeMap, BTreeSet};

    // Premise hash per entry: a one-byte change to the entry text moves it,
    // which is the signal a reversal rides on.
    let premise: BTreeMap<u64, ContentHash> = corpus
        .entries
        .iter()
        .map(|e| (e.id, ContentHash::of(e.text.as_bytes())))
        .collect();
    let prem = |id: u64| premise.get(&id).copied();
    let stamp = |id| MethodStamp::new(id, 1);
    let inputs_of = |ids: &[u64]| ids.iter().filter_map(|&id| prem(id)).collect::<Vec<_>>();

    let gt = &corpus.ground_truth;
    let mut ledger = Ledger::new();

    // Each planted contradiction models a belief the engine formed FROM the
    // "before" entry — exactly the thing the later reversal must invalidate.
    for c in &gt.contradictions {
        if let Some(p) = prem(c.entry_before) {
            ledger.record(Derivation::new(
                ContentHash::of(format!("belief:{}", c.id).as_bytes()),
                vec![p],
                stamp(M_BELIEF),
            ));
        }
    }
    // Higher-order conclusions that cite source entries — richer DAG with shared
    // premises, so propagation must stay isolated to the right dependents.
    for q in &gt.multi_hop {
        ledger.record(Derivation::new(
            ContentHash::of(format!("multihop:{}", q.id).as_bytes()),
            inputs_of(&q.hop_entries),
            stamp(M_MULTIHOP),
        ));
    }
    for q in &gt.recall {
        ledger.record(Derivation::new(
            ContentHash::of(format!("recall:{}", q.id).as_bytes()),
            inputs_of(&q.relevant_entries),
            stamp(M_RECALL),
        ));
    }
    for ib in &gt.insights {
        ledger.record(Derivation::new(
            ContentHash::of(format!("insight:{}", ib.id).as_bytes()),
            inputs_of(&ib.supporting_entries),
            stamp(M_INSIGHT),
        ));
    }
    for t in &gt.temporal {
        let mut ids = t.lead_entries.clone();
        ids.extend_from_slice(&t.trail_entries);
        ledger.record(Derivation::new(
            ContentHash::of(format!("temporal:{}", t.id).as_bytes()),
            inputs_of(&ids),
            stamp(M_TEMPORAL),
        ));
    }

    let empty_versions: BTreeMap<u32, u64> = BTreeMap::new();
    let mut with_dep = 0usize;
    let (mut recall_sum, mut over_sum, mut base_sum) = (0.0, 0.0, 0.0);
    let (mut fanout_sum, mut max_fanout) = (0usize, 0usize);

    for c in &gt.contradictions {
        let Some(p) = prem(c.entry_before) else {
            continue;
        };
        // Independent oracle: every conclusion whose inputs cite the superseded
        // premise. Computed directly from membership, not via stale_closure.
        let truth: BTreeSet<ContentHash> = ledger
            .records()
            .iter()
            .filter(|d| d.inputs.contains(&p))
            .map(|d| d.output)
            .collect();
        if truth.is_empty() {
            continue;
        }
        with_dep += 1;

        let changed: BTreeSet<ContentHash> = [p].into_iter().collect();
        let flagged = ledger.stale_closure(&changed, &empty_versions);
        let score = score_staleness(&flagged, &truth);
        // A fact-storage memory has no provenance, so it flags nothing.
        let base = score_staleness(&BTreeSet::new(), &truth);

        recall_sum += score.invalidation_recall;
        over_sum += score.over_invalidation;
        base_sum += base.invalidation_recall;
        fanout_sum += truth.len();
        max_fanout = max_fanout.max(truth.len());
    }

    let mean = |s: f64| {
        if with_dep == 0 {
            0.0
        } else {
            s / with_dep as f64
        }
    };
    LedgerBenchReport {
        corpus_entries: corpus.entries.len(),
        derivations: ledger.len(),
        contradictions: gt.contradictions.len(),
        contradictions_with_dependents: with_dep,
        mean_fanout: mean(fanout_sum as f64),
        max_fanout,
        ledger_invalidation_recall: mean(recall_sum),
        ledger_over_invalidation: mean(over_sum),
        baseline_invalidation_recall: mean(base_sum),
    }
}

fn print_ledger_bench(r: &LedgerBenchReport) {
    println!("\n=== skinki — Derivation Ledger benchmark (planted contradictions) ===");
    println!("corpus entries     : {}", r.corpus_entries);
    println!("derivations (DAG)  : {}", r.derivations);
    println!(
        "contradictions     : {} ({} with derived dependents)",
        r.contradictions, r.contradictions_with_dependents
    );
    println!(
        "fan-out per reversal: mean {:.1}, max {} conclusions invalidated",
        r.mean_fanout, r.max_fanout
    );
    println!(
        "invalidation-recall : ledger {:.3}  vs  fact-storage baseline {:.3}",
        r.ledger_invalidation_recall, r.baseline_invalidation_recall
    );
    println!(
        "over-invalidation   : {:.3}  (0 = no independent conclusion wrongly flagged)",
        r.ledger_over_invalidation
    );
    println!(
        "\nReading: when a belief is reversed, a provenance-free memory detects {:.0}% of\nthe conclusions that silently went stale; the ledger detects {:.0}% and flags\nnothing it shouldn't — staleness the agent would otherwise never notice.",
        r.baseline_invalidation_recall * 100.0,
        r.ledger_invalidation_recall * 100.0
    );
}

// ---------------------------------------------------------------------------
// locomo-eval — dev-only real-text validation against LoCoMo10
// ---------------------------------------------------------------------------

/// Cosine-similarity nearest-neighbor retriever over a fixed set of
/// per-entry embeddings, generic over [`Embedder`]. Vectors produced by
/// [`StaticHashEmbedder`] are already L2-normalized, so cosine == dot. The
/// real-transformer path (where both docs and queries are precomputed) is
/// scored directly by [`locomo_score_precomputed`], not through this type.
struct SemanticRetriever<E: Embedder> {
    embedder: E,
    vectors: Vec<Vec<f32>>,
    ids: Vec<EntryId>,
    name: String,
}

impl<E: Embedder> SemanticRetriever<E> {
    fn new(embedder: E, name: &str) -> Self {
        SemanticRetriever {
            embedder,
            vectors: Vec::new(),
            ids: Vec::new(),
            name: name.to_string(),
        }
    }
}

impl<E: Embedder> RetrievalSystem for SemanticRetriever<E> {
    fn name(&self) -> &str {
        &self.name
    }

    fn index(&mut self, corpus: &Corpus) {
        self.vectors.clear();
        self.ids.clear();
        self.vectors.reserve(corpus.entries.len());
        self.ids.reserve(corpus.entries.len());
        for e in &corpus.entries {
            self.vectors.push(self.embedder.embed(&e.text));
            self.ids.push(e.id);
        }
    }

    fn search(&self, query: &str, k: usize) -> Vec<EntryId> {
        let qv = self.embedder.embed(query);
        let mut scored: Vec<(f32, EntryId)> = self
            .vectors
            .iter()
            .zip(self.ids.iter())
            .map(|(v, &id)| (dot(&qv, v), id))
            .collect();
        // Sort by score descending, tie-break by ascending id for determinism.
        scored.sort_by(|a, b| match b.0.partial_cmp(&a.0) {
            Some(std::cmp::Ordering::Equal) | None => a.1.cmp(&b.1),
            Some(ord) => ord,
        });
        scored.truncate(k);
        scored.into_iter().map(|(_, id)| id).collect()
    }
}

/// Read a flat little-endian f32 embeddings file (`dim * n` floats, one row
/// per corpus entry in entry-id order) into per-entry vectors.
fn read_embeddings_file(path: &std::path::Path, dim: usize, n: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading embeddings file {}", path.display()))?;
    let expected = dim * n * std::mem::size_of::<f32>();
    anyhow::ensure!(
        bytes.len() == expected,
        "embeddings file {} has {} bytes, expected dim({dim}) * n({n}) * 4 = {expected}",
        path.display(),
        bytes.len()
    );
    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            let off = (row * dim + j) * 4;
            let f =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            v.push(f);
        }
        out.push(v);
    }
    Ok(out)
}

/// Dump the canonical entry texts and query texts as JSON arrays (entry-id
/// order / query order) — the exact order skinki scores them in, so the
/// embeddings a model produces from these line up byte-for-byte on reload.
fn dump_locomo_texts(corpus: &Corpus, dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let entries: Vec<&str> = corpus.entries.iter().map(|e| e.text.as_str()).collect();
    let queries: Vec<&str> = corpus
        .ground_truth
        .recall
        .iter()
        .map(|q| q.question.as_str())
        .collect();
    let ep = dir.join("entries.json");
    let qp = dir.join("queries.json");
    std::fs::write(&ep, serde_json::to_vec(&entries)?)?;
    std::fs::write(&qp, serde_json::to_vec(&queries)?)?;
    println!(
        "dumped {} entry texts -> {}\ndumped {} query texts -> {}\n\nNext: embed them with tools/export-embeddings-gemma.py, then re-run with\n  --embeddings-file <dir>/entries.f32 --query-embeddings-file <dir>/queries.f32",
        entries.len(),
        ep.display(),
        queries.len(),
        qp.display()
    );
    Ok(())
}

/// Score precomputed query vectors against precomputed entry vectors directly
/// (cosine == dot on L2-normalized rows): the real-model path, where BOTH sides
/// come from the same embedder. `query_vecs[i]` corresponds to
/// `corpus.ground_truth.recall[i]`; `entry_vecs[j]` to `corpus.entries[j]`.
fn locomo_score_precomputed(
    entry_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    corpus: &Corpus,
    k: usize,
) -> RetrievalScores {
    let queries = &corpus.ground_truth.recall;
    let (mut recall, mut ndcg, mut precision, mut answer_hits) = (0.0, 0.0, 0.0, 0.0);
    for (qi, q) in queries.iter().enumerate() {
        let qv = &query_vecs[qi];
        let mut scored: Vec<(f32, EntryId)> = entry_vecs
            .iter()
            .zip(corpus.entries.iter())
            .map(|(v, e)| (dot(qv, v), e.id))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let retrieved: Vec<EntryId> = scored.into_iter().take(k).map(|(_, id)| id).collect();
        recall += recall_at_k(&retrieved, &q.relevant_entries, k);
        precision += precision_at_k(&retrieved, &q.relevant_entries, k);
        ndcg += ndcg_at_k(&retrieved, &q.relevant_entries, k);
        if answer_in_entries(corpus, &retrieved, &q.answer) {
            answer_hits += 1.0;
        }
    }
    let n = queries.len().max(1) as f64;
    RetrievalScores {
        queries: queries.len(),
        k,
        recall_at_k: recall / n,
        precision_at_k: precision / n,
        ndcg_at_k: ndcg / n,
        answer_in_topk: answer_hits / n,
    }
}

/// Score `system` on `corpus.ground_truth.recall` and return the aggregate.
fn locomo_score(system: &dyn RetrievalSystem, corpus: &Corpus, k: usize) -> RetrievalScores {
    let items: Vec<QItem> = corpus
        .ground_truth
        .recall
        .iter()
        .map(|q| QItem {
            question: &q.question,
            relevant: &q.relevant_entries,
            answer: &q.answer,
        })
        .collect();
    let mut durations: Vec<Duration> = Vec::new();
    score_set(system, corpus, &items, k, &mut durations)
}

fn run_locomo_eval(
    corpus: &Corpus,
    k: usize,
    dim: usize,
    embeddings_file: Option<&std::path::Path>,
    query_embeddings_file: Option<&std::path::Path>,
    graph_artifacts: Option<&std::path::Path>,
) -> Result<()> {
    println!("\n=== skinki — locomo-eval (LoCoMo10 real-conversation benchmark) ===");
    println!(
        "corpus: {} entries, {} recall queries (k={k})",
        corpus.entries.len(),
        corpus.ground_truth.recall.len()
    );

    // Collect (column name, scores) so the table grows with whatever real-model
    // artifacts the caller supplies.
    let mut cols: Vec<(String, RetrievalScores)> = Vec::new();

    let mut bm25 = Bm25::new();
    bm25.index(corpus);
    cols.push(("bm25".into(), locomo_score(&bm25, corpus, k)));

    let mut semantic = SemanticRetriever::new(StaticHashEmbedder::new(dim), "semantic-static");
    semantic.index(corpus);
    cols.push(("semantic-static".into(), locomo_score(&semantic, corpus, k)));

    let mut graph = GraphRetriever::new();
    graph.index(corpus);
    cols.push(("graph".into(), locomo_score(&graph, corpus, k)));

    // `semantic-real` needs BOTH entry and query embeddings from the same model
    // (a precomputed doc space is meaningless if queries are embedded by a
    // different embedder). Require both files; score them directly.
    match (embeddings_file, query_embeddings_file) {
        (Some(epath), Some(qpath)) => {
            let entry_vecs = read_embeddings_file(epath, dim, corpus.entries.len())?;
            let query_vecs = read_embeddings_file(qpath, dim, corpus.ground_truth.recall.len())?;
            cols.push((
                "semantic-real".into(),
                locomo_score_precomputed(&entry_vecs, &query_vecs, corpus, k),
            ));
        }
        (Some(_), None) => anyhow::bail!(
            "--embeddings-file requires --query-embeddings-file too (docs and queries must share an embedding space); dump both with --dump-texts and embed via tools/export-embeddings-gemma.py"
        ),
        (None, Some(_)) => anyhow::bail!("--query-embeddings-file requires --embeddings-file too"),
        (None, None) => {}
    }

    // The real-text graph path: rebuild an entity graph from the LLM extraction
    // artifact log (fused with BM25 via RRF) and score it.
    if let Some(path) = graph_artifacts {
        let mut llm_graph = llm_graph::LlmGraphRetriever::from_artifacts(path, true)?;
        llm_graph.index(corpus);
        cols.push(("llm-graph+bm25".into(), locomo_score(&llm_graph, corpus, k)));

        // The typed-fact variant: same artifact log, but the walk hops through
        // typed-fact endpoints (the `facts` field) instead of bare co-mention,
        // with prefix-merge coref and a structural no-regression gate. This is
        // the real-text analogue of the synthetic `RelationRetriever` win —
        // measured here against the co-mention column above (the honest
        // negative) and BM25.
        let mut facts_graph = llm_graph::FactsGraphRetriever::from_artifacts(path, true)?;
        facts_graph.index(corpus);
        cols.push((
            "llm-facts+bm25".into(),
            locomo_score(&facts_graph, corpus, k),
        ));
    }

    // Print: one column per system, one row per metric.
    let w = 16usize;
    print!("\n{:<10}", "metric");
    for (name, _) in &cols {
        print!(" {name:>w$}");
    }
    println!();
    print!("{:<10}", "recall@k");
    for (_, s) in &cols {
        print!(" {:>w$.3}", s.recall_at_k);
    }
    println!();
    print!("{:<10}", "answer@k");
    for (_, s) in &cols {
        print!(" {:>w$.3}", s.answer_in_topk);
    }
    println!();

    println!(
        "\nNote: semantic-static is a lexical (hash-of-tokens) embedder (≈ bm25 on real \
         dialogue). The real semantic lift needs precomputed transformer embeddings \
         (--embeddings-file/--query-embeddings-file, e.g. EmbeddingGemma); the real-text \
         GRAPH lift needs an LLM extraction log (--graph-artifacts, from \
         tools/extract-graph-llm.py). `llm-graph+bm25` is co-mention (the honest \
         negative); `llm-facts+bm25` is the typed-fact walk + coref + gate variant."
    );

    Ok(())
}

#[cfg(test)]
mod locomo_eval_tests {
    use super::*;
    use skinki_corpus::{CorpusMeta, Difficulty, Entry, EntryKind, GroundTruth, RecallQuery};

    fn tiny_corpus() -> Corpus {
        let entries = vec![
            Entry {
                id: 0,
                day: 0,
                date: String::new(),
                kind: EntryKind::Text,
                text: "Alice: I adopted a cat named Whiskers.".to_string(),
            },
            Entry {
                id: 1,
                day: 0,
                date: String::new(),
                kind: EntryKind::Text,
                text: "Bob: That's awesome, congrats!".to_string(),
            },
            Entry {
                id: 2,
                day: 1,
                date: String::new(),
                kind: EntryKind::Text,
                text: "Alice: I went hiking in the mountains.".to_string(),
            },
        ];
        Corpus {
            meta: CorpusMeta {
                seed: 0,
                years: 0,
                num_entries: entries.len(),
                difficulty: Difficulty::V2,
            },
            entries,
            ground_truth: GroundTruth {
                recall: vec![RecallQuery {
                    id: 0,
                    question: "What did Alice name her cat?".to_string(),
                    answer: "Whiskers".to_string(),
                    relevant_entries: vec![0],
                }],
                ..Default::default()
            },
        }
    }

    #[test]
    fn semantic_retriever_ranks_lexically_closest_first() {
        let corpus = tiny_corpus();
        let mut system = SemanticRetriever::new(StaticHashEmbedder::new(64), "semantic-static");
        system.index(&corpus);
        let top = system.search("Alice's cat is named Whiskers", 1);
        assert_eq!(top, vec![0]);
    }

    #[test]
    fn semantic_retriever_scores_via_locomo_score() {
        let corpus = tiny_corpus();
        let mut system = SemanticRetriever::new(StaticHashEmbedder::new(64), "semantic-static");
        system.index(&corpus);
        let scores = locomo_score(&system, &corpus, 10);
        assert_eq!(scores.queries, 1);
        assert!(scores.recall_at_k > 0.0);
    }

    #[test]
    fn parse_locomo_sample_variants() {
        assert!(matches!(
            parse_locomo_sample("all").unwrap(),
            LocomoSample::All
        ));
        assert!(matches!(
            parse_locomo_sample("3").unwrap(),
            LocomoSample::One(3)
        ));
        assert!(parse_locomo_sample("bogus").is_err());
    }
}
