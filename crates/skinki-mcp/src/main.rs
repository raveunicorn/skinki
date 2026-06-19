#![forbid(unsafe_code)]
//! `skinki-mcp` — MCP server entrypoint.
//!
//! Usage: `skinki-mcp --corpus <path.json>`
//!
//! Reads newline-delimited JSON-RPC 2.0 messages from stdin, dispatches them
//! to [`skinki_mcp::Server::handle`], and writes any response as a single
//! compact JSON line to stdout, flushing after each one. All logging goes to
//! stderr — stdout must contain only JSON-RPC protocol messages.

use std::io::{self, BufRead, Write};

use anyhow::{Context, Result};
use skinki_corpus::{Corpus, CorpusMeta, Difficulty, Entry, EntryId, EntryKind};
use skinki_mcp::{parse_line, Server};
use skinki_store::Store;

fn main() -> Result<()> {
    let args = parse_args()?;

    let corpus = if let Some(store_path) = args.store {
        eprintln!("skinki-mcp: opening store at {}", store_path.display());
        let s = Store::open(&store_path)?;
        corpus_from_store(&s)?
    } else {
        let corpus_path = args
            .corpus
            .as_deref()
            .context("--corpus or --store required")?;
        eprintln!("skinki-mcp: loading corpus from {}", corpus_path.display());
        let data = std::fs::read_to_string(corpus_path)
            .with_context(|| format!("reading corpus file {}", corpus_path.display()))?;
        serde_json::from_str(&data)
            .with_context(|| format!("parsing corpus file {}", corpus_path.display()))?
    };

    eprintln!(
        "skinki-mcp: indexed {} entries; ready on stdio",
        corpus.entries.len()
    );
    let server = Server::new(corpus);
    run(&server, io::stdin().lock(), io::stdout().lock())
}

/// Build a Corpus from all events in a store.
fn corpus_from_store(s: &Store) -> Result<Corpus> {
    let mut entries: Vec<Entry> = Vec::new();
    for (uid, unit) in s.units() {
        let text = s.unit_text(uid).unwrap_or_default().to_string();
        let ev_ts = s.event_text(unit.event).map(|_| 0i64).unwrap_or(0);
        // approximation: use unit id as day offset
        entries.push(Entry {
            id: entries.len() as EntryId,
            day: (uid / 7 % 365) as u32, // week-bucket day
            date: chrono_fmt(ev_ts),
            kind: EntryKind::Text,
            text,
        });
    }
    let num_entries = entries.len();
    Ok(Corpus {
        meta: CorpusMeta {
            seed: 0,
            years: 0,
            num_entries,
            difficulty: Difficulty::V2,
        },
        entries,
        ground_truth: Default::default(),
    })
}

fn chrono_fmt(ts: i64) -> String {
    let days = ts / 86400;
    let y = 1970 + days / 365;
    let rem = days % 365;
    let m = rem / 30 + 1;
    let d = rem % 30 + 1;
    format!("{y:04}-{m:02}-{d:02}")
}

struct CliArgs {
    corpus: Option<std::path::PathBuf>,
    store: Option<std::path::PathBuf>,
}

/// Parse `--corpus <path>` or `--store <dir>` from argv.
fn parse_args() -> Result<CliArgs> {
    let mut args = std::env::args().skip(1);
    let mut corpus = None;
    let mut store = None;
    while let Some(arg) = args.next() {
        if arg == "--corpus" {
            corpus = Some(std::path::PathBuf::from(
                args.next().context("--corpus requires a path")?,
            ));
        } else if let Some(path) = arg.strip_prefix("--corpus=") {
            corpus = Some(std::path::PathBuf::from(path));
        } else if arg == "--store" {
            store = Some(std::path::PathBuf::from(
                args.next().context("--store requires a path")?,
            ));
        } else if let Some(path) = arg.strip_prefix("--store=") {
            store = Some(std::path::PathBuf::from(path));
        }
    }
    if corpus.is_none() && store.is_none() {
        anyhow::bail!("usage: skinki-mcp --corpus <path.json> OR --store <dir>");
    }
    Ok(CliArgs { corpus, store })
}

/// The main stdio loop: read one line at a time, parse, dispatch, and write
/// any response. Never crashes on malformed input — parse errors and
/// per-message failures become JSON-RPC error responses (or are logged and
/// skipped for notifications).
fn run(server: &Server, input: impl BufRead, mut output: impl Write) -> Result<()> {
    for line in input.lines() {
        let line = line.context("reading stdin")?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match parse_line(trimmed) {
            Ok(msg) => server.handle(&msg),
            Err(err) => Some(err),
        };

        if let Some(response) = response {
            let text = serde_json::to_string(&response).context("serializing response")?;
            writeln!(output, "{text}").context("writing response")?;
            output.flush().context("flushing stdout")?;
        }
    }
    Ok(())
}
