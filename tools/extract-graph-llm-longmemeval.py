#!/usr/bin/env python3
"""Extract per-instance knowledge graphs from LongMemEval dumped texts.

Dev-only helper (NOT in the Cargo workspace). The *produce* side of the
Stage-3 replay seam for LongMemEval real text: where the synthetic corpus is
read by hand-written intro/rec/venue patterns, real dialogue needs a model.
This runs ONE local instruct model load over every turn of every instance in
the dump manifest, writing a per-instance artifact log (JSON-lines) of
extracted entities + (subject, relation, object) facts. skinki then rebuilds
a graph from each log deterministically and measures whether the extracted
structure beats plain BM25 on the `multi-session` category.

Flow
----
1. skinki dumps the per-instance turn texts:
       cargo run --release -p skinki-harness -- longmemeval-eval \
           --path longmemeval_s_cleaned.json --dump-texts ./lme_dump

2. This script extracts per-instance artifact logs (one model load, resumable):
       python3 tools/extract-graph-llm-longmemeval.py \
           --dump-dir ./lme_dump --out-dir ./lme_dump

   Writes ./lme_dump/<instance_id>/graph.artifacts.jsonl per instance.

3. skinki rebuilds graphs per instance and measures:
       cargo run --release -p skinki-harness -- longmemeval-eval \
           --path longmemeval_s_cleaned.json \
           --graph-artifacts-dir ./lme_dump

Artifact-log record (one JSON object per line, `entry` = index into that
instance's entries.json): identical to extract-graph-llm.py — the per-instance
logs are byte-compatible with the single-instance LoCoMo format.
    {"entry": 12, "entities": ["Caroline","Mel"],
     "facts": [["Caroline","met","Mel"]],
     "model": "Qwen/Qwen2.5-3B-Instruct", "v": 1}

Determinism: greedy decoding (do_sample=False). The model output is NOT
bit-reproducible across machines (AGENTS.md rule 3) — that is exactly why it
goes to a replay log that skinki rebuilds deterministically.
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM = (
    "You extract a knowledge graph from a single chat message. "
    'Output ONLY compact JSON of the form '
    '{"entities":[...],"facts":[["subject","relation","object"], ...]}. '
    "Entities are the specific people, places, organizations, objects, or "
    "events named or clearly referred to. Facts are concise (subject, "
    "relation, object) triples explicitly stated in the message. Use proper "
    "names where given. If the message is small talk with nothing substantive, "
    'output {"entities":[],"facts":[]}. No prose, no code fences.'
)

ONESHOT_USER = "Caroline: I finally went to the LGBTQ support group downtown on Tuesday."
ONESHOT_ASSISTANT = (
    '{"entities":["Caroline","LGBTQ support group"],'
    '"facts":[["Caroline","attended","LGBTQ support group"]]}'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dump-dir", required=True, type=Path, help="skinki longmemeval --dump-texts dir")
    p.add_argument("--out-dir", required=True, type=Path, help="where to write per-instance logs (often == dump-dir)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit-instances", type=int, default=None, help="only the first N instances (testing)")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--only-instance", default=None, help="only extract this one safe_id (testing)")
    p.add_argument(
        "--device",
        default="auto",
        help='"auto" (mps>cuda>cpu), or "mps"/"cuda"/"cpu"',
    )
    return p.parse_args()


def pick_device(choice: str) -> str:
    import torch

    if choice != "auto":
        return choice
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    m = _JSON_RE.search(text)
    if not m:
        return {"entities": [], "facts": []}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"entities": [], "facts": []}
    ents = obj.get("entities", [])
    facts = obj.get("facts", [])
    ents = [str(e).strip() for e in ents if isinstance(e, (str, int, float)) and str(e).strip()]
    norm_facts = []
    for f in facts if isinstance(facts, list) else []:
        if isinstance(f, list) and len(f) == 3:
            s, r, o = (str(x).strip() for x in f)
        elif isinstance(f, dict):
            s, r, o = (
                str(f.get("subject", "")).strip(),
                str(f.get("relation", "")).strip(),
                str(f.get("object", "")).strip(),
            )
        else:
            continue
        if s and o:
            norm_facts.append([s, r, o])
    return {"entities": ents, "facts": norm_facts}


def already_done(out_path: Path) -> int:
    """Resumability: highest `entry` already in the log + 1 (0 if none)."""
    if not out_path.exists():
        return 0
    done = -1
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done = max(done, int(json.loads(line)["entry"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return done + 1


def main() -> int:
    args = parse_args()

    manifest_path = args.dump_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found (run `skinki longmemeval-eval --dump-texts` first)", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())

    if args.limit_instances is not None:
        manifest = manifest[: args.limit_instances]
    if args.only_instance is not None:
        manifest = [m for m in manifest if m["safe_id"] == args.only_instance]
        if not manifest:
            print(f"error: no instance with safe_id={args.only_instance}", file=sys.stderr)
            return 1

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"error: missing dependency ({exc})", file=sys.stderr)
        print(
            "\nInstall in your venv:\n"
            "    pip install -U transformers torch\n"
            "(Qwen2.5 needs no HF license; google/gemma-* is gated -> "
            "huggingface-cli login.)",
            file=sys.stderr,
        )
        return 1

    device = pick_device(args.device)
    print(f"loading {args.model} on {device} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
    model.eval()

    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_done = 0
    for mi, m in enumerate(manifest):
        safe_id = m["safe_id"]
        entries_path = args.dump_dir / safe_id / "entries.json"
        if not entries_path.exists():
            print(f"  [{mi+1}/{len(manifest)}] {safe_id}: entries.json missing, skipping", file=sys.stderr)
            continue
        entries = json.loads(entries_path.read_text())

        out_path = args.out_dir / safe_id / "graph.artifacts.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        start = already_done(out_path)
        if start >= len(entries):
            continue  # fully done
        if start:
            print(f"  [{mi+1}/{len(manifest)}] {safe_id}: resuming at entry {start}", file=sys.stderr)

        written = 0
        with out_path.open("a") as out:
            for i in range(start, len(entries)):
                text = entries[i]
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": ONESHOT_USER},
                    {"role": "assistant", "content": ONESHOT_ASSISTANT},
                    {"role": "user", "content": text},
                ]
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tok(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                completion = tok.decode(
                    gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                rec = extract_json(completion)
                rec.update({"entry": i, "model": args.model, "v": 1})
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
        total_done += written
        print(f"  [{mi+1}/{len(manifest)}] {safe_id}: {written} turns -> {out_path}", file=sys.stderr)

    print(f"\nWrote {total_done} records across {len(manifest)} instances.", file=sys.stderr)
    print("Next: measure with", file=sys.stderr)
    print(f"  cargo run --release -p skinki-harness -- longmemeval-eval \\\n"
          f"      --path <longmemeval_s_cleaned.json> \\\n"
          f"      --graph-artifacts-dir {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
