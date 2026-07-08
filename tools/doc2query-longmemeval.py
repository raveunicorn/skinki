#!/usr/bin/env python3
"""Generate candidate questions per entry from LongMemEval dumps via llama-server.

Sleep-time doc2query spike (Stage 1D T6): for every turn's text the local LLM
produces up to 3 short questions a user might later ask that this turn answers.
Output is an append-only artifact log (one JSONL per instance), replayed by the
`bm25+doc2query` column of `longmemeval-eval --pooled` — never re-run inside an
eval/gate.

Crash-safe and resumable — each turn's result is written to its instance's
artifact log IMMEDIATELY (incremental JSONL, append-only). On re-run only the
missing entry indices for each instance are picked up.

Usage:
  1. Start llama-server:
       llama-server -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --port 8081 \
           -ngl 99 --ctx-size 4096 --batch-size 512

  2. Generate (resumable — safe to kill and re-run):
       python3 tools/doc2query-longmemeval.py \
           --dump-dir ./t6_dump --out-dir ./t6_dump \
           --server http://localhost:8081 --workers 8

  Input dump is produced by:
    cargo run --release -p skinki-harness -- longmemeval-eval \
        --path <longmemeval_m_cleaned.json> --pooled \
        --question-type multi-session --limit 41 --dump-texts <WORKDIR>/t6_dump
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_SERVER = "http://localhost:8081"
DEFAULT_WORKERS = 8
DEFAULT_MODEL_ID = "Qwen2.5-0.5B GGUF"
DEFAULT_MAX_TOKENS = 96

SYSTEM = (
    "You generate questions a user might later ask that ONE chat turn answers. "
    'Output ONLY compact JSON: {"questions": ["...", ...]} with AT MOST 3 short, '
    "self-contained questions in the same language as the turn. Questions must be "
    "answerable from THIS turn alone. If the turn is small talk, greetings, "
    'acknowledgements, or otherwise has no reusable fact, output '
    '{"questions": []}. No prose, no code fences, no commentary.'
)

ONESHOT_USER = "Caroline: I finally went to the LGBTQ support group downtown on Tuesday."
ONESHOT_ASSISTANT = (
    '{"questions":['
    '"What support group did Caroline attend?",'
    '"Where is the LGBTQ support group Caroline joined?",'
    '"When did Caroline go to the LGBTQ support group?"]}'
)

# Greedy non-greedy brace match — same shape as the graph template's _JSON_RE.
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dump-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--limit-instances", type=int, default=None)
    p.add_argument("--only-instance", default=None)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--max-questions",
        type=int,
        default=3,
        help="Cap questions per turn (defensive — the prompt already says ≤3).",
    )
    return p.parse_args()


def generate_and_write(
    server_url: str,
    text: str,
    max_tokens: int,
    max_questions: int,
    idx: int,
    model_id: str,
    out_path: Path,
    lock: threading.Lock,
    max_retries: int = 3,
) -> None:
    """Send one turn to llama-server, write result line to out_path immediately.
    Retries on failure; falls back to an empty question list after exhaustion."""
    payload = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": ONESHOT_USER},
                {"role": "assistant", "content": ONESHOT_ASSISTANT},
                {"role": "user", "content": text},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    questions: list[str] = []
    for attempt in range(max_retries):
        try:
            req = Request(
                f"{server_url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urlopen(req, timeout=120)
            data = json.loads(resp.read())
            completion = data["choices"][0]["message"]["content"].strip()

            m = _JSON_RE.search(completion)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    obj = {}
                raw = obj.get("questions", [])
                if isinstance(raw, list):
                    questions = [
                        str(q).strip()
                        for q in raw
                        if isinstance(q, (str, int, float)) and str(q).strip()
                    ][:max_questions]
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
    # else: all retries exhausted — write empty questions.

    line = json.dumps(
        {
            "entry_index": idx,
            "questions": questions,
            "model": model_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "v": 1,
        },
        ensure_ascii=False,
    )
    with lock:
        with out_path.open("a") as f:
            f.write(line + "\n")


def main() -> int:
    args = parse_args()

    manifest_path = args.dump_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())

    if args.limit_instances is not None:
        manifest = manifest[: args.limit_instances]
    if args.only_instance is not None:
        manifest = [m for m in manifest if m["safe_id"] == args.only_instance]
        if not manifest:
            print(f"error: no instance with safe_id={args.only_instance}", file=sys.stderr)
            return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build work queue, skipping already-done entries ---
    work = []  # (idx, text, out_path, lock, safe_id)
    total_turns = 0
    done_turns = 0
    locks: dict[str, threading.Lock] = {}

    for m in manifest:
        safe_id = m["safe_id"]
        entries_path = args.dump_dir / safe_id / "entries.json"
        if not entries_path.exists():
            continue
        entries = json.loads(entries_path.read_text())
        out_path = args.out_dir / safe_id / "doc2query.artifacts.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lock = locks.setdefault(safe_id, threading.Lock())

        # Find highest entry index already written (resumability).
        done_entries: set[int] = set()
        if out_path.exists():
            with out_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done_entries.add(int(json.loads(line)["entry_index"]))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass

        for i, text in enumerate(entries):
            if i in done_entries:
                done_turns += 1
                continue
            work.append((i, text, out_path, lock, safe_id))
            total_turns += 1

    if not work:
        print(f"Nothing to do ({done_turns} turns already on disk).", file=sys.stderr)
        return 0

    print(
        f"Generating doc2query for {total_turns} turns ({done_turns} already done) "
        f"from {len(manifest)} instances "
        f"via {args.server} ({args.workers} concurrent, incremental) ...",
        file=sys.stderr,
    )

    t0 = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                generate_and_write,
                args.server,
                text,
                args.max_tokens,
                args.max_questions,
                idx,
                args.model_id,
                out_path,
                lock,
            ): safe_id
            for idx, text, out_path, lock, safe_id in work
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            total_done = completed + failed + done_turns
            if total_done % 500 == 0:
                dt = time.time() - t0
                print(
                    f"  {total_done}/{total_turns + done_turns} turns "
                    f"({completed/dt:.0f} t/s)",
                    file=sys.stderr,
                )

    dt = time.time() - t0
    print(
        f"\nDone: {completed} turns in {dt/60:.1f} min "
        f"({completed/dt:.0f} t/s, "
        f"{done_turns} already done on disk)",
        file=sys.stderr,
    )
    print(
        "Next: cargo run --release -p skinki-harness -- longmemeval-eval \\\n"
        f"    --path <longmemeval_m_cleaned.json> --pooled \\\n"
        f"    --question-type multi-session --limit 41 \\\n"
        f"    --doc2query-artifacts {args.out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
