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
use skinki_corpus::Corpus;
use skinki_mcp::{parse_line, Server};

fn main() -> Result<()> {
    let corpus_path = parse_args()?;

    eprintln!("skinki-mcp: loading corpus from {}", corpus_path);
    let data = std::fs::read_to_string(&corpus_path)
        .with_context(|| format!("reading corpus file {corpus_path}"))?;
    let corpus: Corpus = serde_json::from_str(&data)
        .with_context(|| format!("parsing corpus file {corpus_path}"))?;

    eprintln!(
        "skinki-mcp: indexed {} entries; ready on stdio",
        corpus.entries.len()
    );
    let server = Server::new(corpus);

    run(&server, io::stdin().lock(), io::stdout().lock())
}

/// Parse `--corpus <path>` from argv. Hand-rolled (no extra dep): the only
/// flag this v0 server needs.
fn parse_args() -> Result<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--corpus" {
            return args.next().context("--corpus requires a path argument");
        }
        if let Some(path) = arg.strip_prefix("--corpus=") {
            return Ok(path.to_string());
        }
    }
    anyhow::bail!("usage: skinki-mcp --corpus <path.json>")
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
