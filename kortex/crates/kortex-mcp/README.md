# kortex-mcp

A Model Context Protocol (MCP) server that exposes kortex's graph memory —
multi-hop `search` and budgeted, cited `assemble_context` over a corpus
indexed with `kortex_graph::RelationRetriever` — to any MCP host (Claude
Code, Cursor, etc.) over stdio. It speaks hand-rolled, newline-delimited
JSON-RPC 2.0: no third-party MCP/JSON-RPC dependencies, per `AGENTS.md`'s
minimal-deps rule. v0 loads a single corpus JSON file (produced by
`kortex generate`) at startup and serves it for the process lifetime.

## Quickstart

Generate a corpus, then register the server with your MCP host:

```bash
cargo run --release -p kortex-harness -- generate --years 1 --entries-per-day 2 --out /tmp/corpus.json
cargo build --release -p kortex-mcp
```

Host config (e.g. Claude Code's MCP server config):

```json
{
  "mcpServers": {
    "kortex-memory": {
      "command": "kortex-mcp",
      "args": ["--corpus", "/tmp/corpus.json"]
    }
  }
}
```

## Tools

- `search { query, k? }` — multi-hop graph retrieval; returns matching
  entries as `[<entry_id>] <date>: <text>` lines.
- `assemble_context { query, token_budget? }` — a budgeted, cited context
  package (header line with `est_tokens`, then one `[id] date: text` line per
  cited fact) — feed this to a model instead of raw chunks.
