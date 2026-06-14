#![forbid(unsafe_code)]
//! Stage 6 — `kortex-mcp`: a Model Context Protocol (MCP) server exposing
//! kortex's graph memory (search + budgeted, cited context assembly) to MCP
//! hosts (Claude Code, Cursor, any MCP client) over stdio.
//!
//! ## Why this exists
//!
//! Per the two laws (`AGENTS.md`): intelligence lives in the memory
//! substrate, not the model. This crate is the "memory for agents"
//! distribution channel — any MCP-capable agent can call `search` /
//! `assemble_context` against a kortex corpus and get back dense, dated,
//! provenance-preserving results instead of raw chunks.
//!
//! ## Protocol
//!
//! MCP is JSON-RPC 2.0 over newline-delimited stdio: one compact JSON object
//! per line on stdin (requests/notifications), one compact JSON object per
//! line on stdout (responses). All logging goes to stderr — stdout carries
//! *only* protocol messages. This crate hand-rolls the JSON-RPC envelope
//! (no third-party MCP/JSON-RPC crates, per `AGENTS.md`'s minimal-deps rule):
//! [`Server::handle`] is pure (no I/O) and unit-tested directly; `main`
//! (in `main.rs`) does the actual stdin/stdout loop.

use kortex_corpus::Corpus;
use kortex_eval::RetrievalSystem;
use kortex_graph::{assemble_context, RelationRetriever};
use serde_json::{json, Value};

/// JSON-RPC error codes used by this server (standard JSON-RPC 2.0 codes).
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

/// Protocol version this server speaks if the client doesn't specify one.
const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";

/// The MCP server: an indexed corpus + the graph retriever over it.
///
/// Built once at startup (`new`) and held for the process lifetime; one
/// thread, one stdio loop — no internal mutability needed beyond the index
/// built during construction.
pub struct Server {
    corpus: Corpus,
    retriever: RelationRetriever,
}

impl Server {
    /// Build the server: index `corpus` with [`RelationRetriever`].
    pub fn new(corpus: Corpus) -> Self {
        let mut retriever = RelationRetriever::new();
        retriever.index(&corpus);
        Server { corpus, retriever }
    }

    /// Handle one parsed JSON-RPC message.
    ///
    /// Returns `Some(response)` for requests (messages with an `id`), or
    /// `None` for notifications (no `id`) — including malformed messages
    /// that lack an `id`, per JSON-RPC 2.0 (notifications never get a
    /// response, even error responses).
    ///
    /// Pure / no I/O: makes this directly unit-testable.
    pub fn handle(&self, msg: &Value) -> Option<Value> {
        let id = msg.get("id").cloned();

        let Some(method) = msg.get("method").and_then(Value::as_str) else {
            return id.map(|id| error_response(id, INVALID_REQUEST, "invalid request"));
        };

        match method {
            "initialize" => {
                let id = id?;
                let protocol_version = msg
                    .get("params")
                    .and_then(|p| p.get("protocolVersion"))
                    .and_then(Value::as_str)
                    .unwrap_or(DEFAULT_PROTOCOL_VERSION);
                Some(success_response(
                    id,
                    json!({
                        "protocolVersion": protocol_version,
                        "capabilities": { "tools": {} },
                        "serverInfo": {
                            "name": "kortex-memory",
                            "version": env!("CARGO_PKG_VERSION"),
                        }
                    }),
                ))
            }
            "notifications/initialized" => None,
            "tools/list" => {
                let id = id?;
                Some(success_response(id, json!({ "tools": tool_defs() })))
            }
            "tools/call" => {
                let id = id?;
                Some(self.handle_tool_call(id, msg.get("params")))
            }
            _ => id.map(|id| error_response(id, METHOD_NOT_FOUND, "method not found")),
        }
    }

    /// Dispatch a `tools/call` request to the named tool, building the
    /// `{"content": [...], "isError": ...}` result envelope.
    fn handle_tool_call(&self, id: Value, params: Option<&Value>) -> Value {
        let Some(params) = params else {
            return error_response(id, INVALID_PARAMS, "missing params");
        };
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return error_response(id, INVALID_PARAMS, "missing tool name");
        };
        let empty = json!({});
        let arguments = params.get("arguments").unwrap_or(&empty);

        let result = match name {
            "search" => self.tool_search(arguments),
            "assemble_context" => self.tool_assemble_context(arguments),
            other => Err(format!("unknown tool: {other}")),
        };

        match result {
            Ok(text) => success_response(
                id,
                json!({
                    "content": [{ "type": "text", "text": text }],
                    "isError": false,
                }),
            ),
            Err(text) => success_response(
                id,
                json!({
                    "content": [{ "type": "text", "text": text }],
                    "isError": true,
                }),
            ),
        }
    }

    /// `search`: run `retriever.search(query, k)` and render each hit as
    /// `"[<entry_id>] <date>: <text>"`.
    fn tool_search(&self, arguments: &Value) -> Result<String, String> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing required argument: query".to_string())?;
        let k = arguments.get("k").and_then(Value::as_u64).unwrap_or(10) as usize;

        let ids = self.retriever.search(query, k);
        if ids.is_empty() {
            return Ok("No results.".to_string());
        }

        let mut out = String::new();
        for id in ids {
            if let Some(entry) = self.corpus.entries.get(id as usize) {
                out.push_str(&format!("[{}] {}: {}\n", entry.id, entry.date, entry.text));
            }
        }
        Ok(out.trim_end().to_string())
    }

    /// `assemble_context`: call `kortex_graph::assemble_context` and render
    /// a header line (est_tokens) plus one `"[id] date: text"` line per
    /// cited fact.
    fn tool_assemble_context(&self, arguments: &Value) -> Result<String, String> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing required argument: query".to_string())?;
        let token_budget = arguments
            .get("token_budget")
            .and_then(Value::as_u64)
            .unwrap_or(512) as usize;

        let package = assemble_context(&self.retriever, &self.corpus, query, token_budget);

        let mut out = format!("est_tokens: {}\n", package.est_tokens);
        for fact in &package.facts {
            out.push_str(&format!("[{}] {}: {}\n", fact.entry, fact.date, fact.text));
        }
        Ok(out.trim_end().to_string())
    }
}

/// The `tools/list` tool definitions: `search` and `assemble_context`.
fn tool_defs() -> Value {
    json!([
        {
            "name": "search",
            "description": "Search the memory for entries relevant to a query (multi-hop graph retrieval).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "k": { "type": "integer", "default": 10 }
                },
                "required": ["query"]
            }
        },
        {
            "name": "assemble_context",
            "description": "Assemble a budgeted, cited context package for a query (dense, dated, provenance-preserving — feed this to the model instead of raw chunks).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "token_budget": { "type": "integer", "default": 512 }
                },
                "required": ["query"]
            }
        }
    ])
}

/// Build a `{"jsonrpc":"2.0","id":id,"result":result}` response.
fn success_response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// Build a `{"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}}`
/// response.
fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Parse a single line of input into a JSON-RPC message `Value`, or build a
/// parse-error response if it isn't valid JSON.
///
/// Per JSON-RPC 2.0, a parse error has no associated `id` (the request
/// couldn't be parsed far enough to find one), so the response `id` is
/// `null`.
pub fn parse_line(line: &str) -> Result<Value, Value> {
    serde_json::from_str(line)
        .map_err(|e| error_response(Value::Null, PARSE_ERROR, &format!("parse error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kortex_corpus::{CorpusMeta, Difficulty, Entry, EntryKind, GroundTruth};

    /// A tiny in-memory corpus, enough to exercise `search` / `assemble_context`.
    fn tiny_corpus() -> Corpus {
        let entries = vec![
            Entry {
                id: 0,
                day: 0,
                date: "2018-01-01".to_string(),
                kind: EntryKind::Text,
                text: "Anna recommended the book Dune at the meetup.".to_string(),
            },
            Entry {
                id: 1,
                day: 1,
                date: "2018-01-02".to_string(),
                kind: EntryKind::Text,
                text: "Quiet day. Thought about jazz harmony on a walk.".to_string(),
            },
        ];

        Corpus {
            meta: CorpusMeta {
                seed: 0,
                years: 1,
                num_entries: entries.len(),
                difficulty: Difficulty::V2,
            },
            entries,
            ground_truth: GroundTruth::default(),
        }
    }

    fn server() -> Server {
        Server::new(tiny_corpus())
    }

    #[test]
    fn initialize_returns_server_info() {
        let server = server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "protocolVersion": "2025-06-18" }
        });
        let resp = server.handle(&req).expect("initialize is a request");
        assert_eq!(resp["id"], json!(1));
        assert_eq!(resp["result"]["serverInfo"]["name"], json!("kortex-memory"));
        assert_eq!(resp["result"]["protocolVersion"], json!("2025-06-18"));
    }

    #[test]
    fn tools_list_returns_both_tools() {
        let server = server();
        let req = json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" });
        let resp = server.handle(&req).expect("tools/list is a request");
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"search"));
        assert!(names.contains(&"assemble_context"));
    }

    #[test]
    fn tools_call_search_returns_expected_entry() {
        let server = server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": { "query": "Dune meetup recommendation", "k": 5 }
            }
        });
        let resp = server.handle(&req).expect("tools/call is a request");
        assert_eq!(resp["result"]["isError"], json!(false));
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("Dune"),
            "expected search result to contain entry 0's text, got: {text}"
        );
        assert!(text.contains("[0]"));
        assert!(text.contains("2018-01-01"));
    }

    #[test]
    fn tools_call_assemble_context_returns_cited_facts() {
        let server = server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "assemble_context",
                "arguments": { "query": "Dune meetup recommendation", "token_budget": 512 }
            }
        });
        let resp = server.handle(&req).expect("tools/call is a request");
        assert_eq!(resp["result"]["isError"], json!(false));
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("est_tokens:"));
        assert!(text.contains("Dune"));
    }

    #[test]
    fn notification_returns_none() {
        let server = server();
        let msg = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(server.handle(&msg).is_none());
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let server = server();
        let req = json!({ "jsonrpc": "2.0", "id": 5, "method": "not/a/real/method" });
        let resp = server.handle(&req).expect("has an id, so gets a response");
        assert_eq!(resp["error"]["code"], json!(-32601));
    }

    #[test]
    fn malformed_json_is_parse_error() {
        let err = parse_line("{not valid json").expect_err("malformed JSON should error");
        assert_eq!(err["error"]["code"], json!(-32700));
    }

    #[test]
    fn missing_method_is_invalid_request() {
        let server = server();
        let req = json!({ "jsonrpc": "2.0", "id": 6 });
        let resp = server.handle(&req).expect("has an id, so gets a response");
        assert_eq!(resp["error"]["code"], json!(-32600));
    }
}
