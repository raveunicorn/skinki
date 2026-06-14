#!/usr/bin/env python3
"""Extract a per-turn knowledge graph from kortex-dumped texts with a local LLM.

Dev-only helper (NOT in the Cargo workspace). The *produce* side of the Stage-3
replay seam for REAL text: where the synthetic corpus is read by hand-written
intro/rec/venue patterns, real dialogue needs a model. This runs a local
instruct model over each dumped entry (one chat turn) and writes an append-only
**artifact log** (JSON-lines) of extracted entities + (subject, relation,
object) facts. kortex then rebuilds a graph from that log deterministically and
measures whether the extracted structure beats plain retrieval — proving (or
not) our unique graph layer on real data.

Flow
----
1. kortex dumps the canonical turn texts (same order kortex scores):
       cargo run --release -p kortex-harness -- locomo-eval \
           --path locomo10.json --sample 0 --dump-texts ./lc0
   (Start with ONE sample: `--sample 0` is ~600 turns / a quick run; `--sample
    all` is 5882 turns and can take hours.)

2. This script extracts to an artifact log (resumable — safe to Ctrl-C and rerun):
       python3 tools/extract-graph-llm.py \
           --entries ./lc0/entries.json --out ./lc0/graph.artifacts.jsonl

3. (next) kortex rebuilds a graph from graph.artifacts.jsonl and measures it.

Artifact-log record (one JSON object per line, `entry` = index into entries.json):
    {"entry": 12, "entities": ["Caroline","LGBTQ support group"],
     "facts": [["Caroline","attended","LGBTQ support group"]],
     "model": "Qwen/Qwen2.5-3B-Instruct", "v": 1}

Determinism: greedy decoding (do_sample=False). The model output is NOT
bit-reproducible across machines (AGENTS.md rule 3) — that is exactly why it
goes to a replay log that kortex rebuilds deterministically.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Model card target is Gemma; Qwen2.5-3B-Instruct is a strong, NON-gated default
# so the run works without Hugging Face license gymnastics. Swap with --model
# (e.g. google/gemma-2-2b-it) — the replay log format is identical.
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
    p.add_argument("--entries", required=True, type=Path, help="kortex dump entries.json")
    p.add_argument("--out", required=True, type=Path, help="artifact log (.jsonl) to append")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None, help="only the first N entries (testing)")
    p.add_argument("--max-new-tokens", type=int, default=192)
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
    """Best-effort: strip fences, grab the first {...}, normalize shapes."""
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
    # Normalize: entities -> list[str]; facts -> list[[s,r,o]].
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

    entries = json.loads(args.entries.read_text())
    if not isinstance(entries, list):
        print("error: entries.json must be a JSON array", file=sys.stderr)
        return 1
    if args.limit is not None:
        entries = entries[: args.limit]

    start = already_done(args.out)
    if start >= len(entries):
        print(f"nothing to do: {start} entries already in {args.out}")
        return 0
    if start:
        print(f"resuming at entry {start} ({start} already done)")

    device = pick_device(args.device)
    print(f"loading {args.model} on {device} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
    model.eval()

    import torch

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("a") as out:
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
            if written % 25 == 0:
                print(f"  {i + 1}/{len(entries)} turns extracted", file=sys.stderr)

    print(f"\nWrote {written} records -> {args.out}")
    print("Sample (first 3 lines):")
    with args.out.open() as f:
        for _, line in zip(range(3), f):
            print("  " + line.rstrip())
    print("\nSend me a few of these lines so I can build the rebuild/measure side.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
