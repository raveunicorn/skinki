#!/usr/bin/env python3
"""Extract per-instance knowledge graphs from LongMemEval via llama-server.

Incremental version: each turn's result is written to its instance's artifact
log IMMEDIATELY (not accumulated in memory). Crash-safe and resumable — on
re-run, only incomplete instances are picked up.

Usage:
  1. Start llama-server:
       llama-server -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --port 8081 \
           -ngl 99 --ctx-size 4096 --batch-size 512

  2. Extract (resumable — safe to kill and re-run):
       python3 tools/extract-graph-llm-longmemeval.py \
           --dump-dir ./lme_multi_dump --out-dir ./lme_multi_dump \
           --server http://localhost:8081 --workers 8
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_SERVER = "http://localhost:8081"
DEFAULT_WORKERS = 8
DEFAULT_MODEL_ID = "Qwen2.5-0.5B GGUF"

SYSTEM = (
    "You extract a knowledge graph from a single chat message. "
    'Output ONLY compact JSON: {"entities":[...],"facts":[["s","r","o"]]}. '
    "Entities are specific people, places, organizations, objects, or events "
    "named or clearly referred to. Facts are concise triples explicitly stated. "
    "If the message is small talk with nothing substantive, "
    'output {"entities":[],"facts":[]}. No prose, no code fences.'
)

ONESHOT_USER = "Caroline: I finally went to the LGBTQ support group downtown on Tuesday."
ONESHOT_ASSISTANT = (
    '{"entities":["Caroline","LGBTQ support group"],'
    '"facts":[["Caroline","attended","LGBTQ support group"]]}'
)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


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
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return p.parse_args()


def extract_and_write(
    server_url: str,
    text: str,
    max_tokens: int,
    idx: int,
    out_path: Path,
    lock: threading.Lock,
    max_retries: int = 3,
) -> None:
    """Send one turn to llama-server, write result line to out_path immediately.
    Retries on failure; falls back to empty entities/facts after exhaustion."""
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
    ents, facts = [], []
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
                ents = obj.get("entities", [])
                ents = [
                    str(e).strip()
                    for e in ents
                    if isinstance(e, (str, int, float)) and str(e).strip()
                ]
                raw_facts = obj.get("facts", [])
                for f in raw_facts if isinstance(raw_facts, list) else []:
                    if isinstance(f, list) and len(f) == 3:
                        s, r, o = (str(x).strip() for x in f)
                        if s and o:
                            facts.append([s, r, o])
                    elif isinstance(f, dict):
                        s = str(f.get("subject", "")).strip()
                        o = str(f.get("object", "")).strip()
                        if s and o:
                            facts.append([s, str(f.get("relation", "")).strip(), o])
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
    # else: all retries exhausted — write empty.

    line = json.dumps(
        {"entry": idx, "entities": ents, "facts": facts, "model": "mlx", "v": 1},
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

    # --- Build work queue, skipping already-done turns ---
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
        out_path = args.out_dir / safe_id / "graph.artifacts.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lock = locks.setdefault(safe_id, threading.Lock())

        # Find highest entry already written (resumability).
        done_entries: set[int] = set()
        if out_path.exists():
            with out_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done_entries.add(int(json.loads(line)["entry"]))
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
        f"Extracting {total_turns} turns ({done_turns} already done) "
        f"from {len(manifest)} instances "
        f"via {args.server} ({args.workers} concurrent, incremental) ...",
        file=sys.stderr,
    )

    t0 = time.time()
    completed = total_done_turns = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_and_write,
                args.server,
                text,
                args.max_tokens,
                idx,
                out_path,
                lock,
            ): safe_id
            for idx, text, out_path, lock, safe_id in work
        }
        for future in as_completed(futures):
            safe_id = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as e:
                failed += 1
                # Already handled inside extract_and_write (writes empty on exhaustion).

            total_done = completed + failed + done_turns
            if total_done % 500 == 0:
                dt = time.time() - t0
                print(
                    f"  {total_done}/{total_turns + done_turns} turns "
                    f"({completed/dt:.0f} t/s, {failed} retry-exhausted)",
                    file=sys.stderr,
                )

    dt = time.time() - t0
    print(
        f"\nDone: {completed} turns in {dt/60:.1f} min "
        f"({completed/dt:.0f} t/s, {failed} retry-exhausted, "
        f"{done_turns} already done on disk)",
        file=sys.stderr,
    )
    print(
        "Next: cargo run --release -p skinki-harness -- longmemeval-eval \\\n"
        f"    --path <longmemeval_s_cleaned.json> \\\n"
        f"    --graph-artifacts-dir {args.out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
