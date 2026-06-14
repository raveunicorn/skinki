#!/usr/bin/env python3
"""Export EmbeddingGemma vectors for skinki's LoCoMo real-text validation.

Dev-only helper (NOT part of the Cargo workspace). Runs on a machine that can
load the model (e.g. an M1 Mac); the skinki sandbox cannot (no GPU/weights, HF
blocked). This is the *produce* side of the replay seam: it turns the canonical
texts skinki dumps into raw float32 vectors skinki loads back to measure the
real semantic lift over BM25.

End-to-end flow
---------------
1. skinki dumps the canonical texts (same order skinki scores them in):

       cargo run --release -p skinki-harness -- locomo-eval \
           --path locomo10.json --sample all --dump-texts ./lc

   -> ./lc/entries.json  (JSON array of entry texts, entry-id order)
      ./lc/queries.json  (JSON array of question texts, query order)

2. This script embeds them with EmbeddingGemma (asymmetric doc/query prompts,
   Matryoshka-truncated to --dim, L2-renormalized):

       python3 tools/export-embeddings-gemma.py \
           --entries ./lc/entries.json --queries ./lc/queries.json \
           --out-dir ./lc --dim 256

   -> ./lc/entries.f32  ./lc/queries.f32  (raw little-endian f32, dim*N row-major)

3. skinki loads them back and prints the semantic-real column:

       cargo run --release -p skinki-harness -- locomo-eval \
           --path locomo10.json --sample all --dim 256 \
           --embeddings-file ./lc/entries.f32 \
           --query-embeddings-file ./lc/queries.f32

Why asymmetric prompts: EmbeddingGemma is trained with task prefixes; documents
and queries use *different* prompts. Using them is what makes retrieval work.
Defaults below are the model-card retrieval prompts; override if needed.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

# EmbeddingGemma model-card retrieval prompts (the "{content}" is replaced).
DEFAULT_DOC_PROMPT = "title: none | text: {content}"
DEFAULT_QUERY_PROMPT = "task: search result | query: {content}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--entries",
        required=True,
        type=Path,
        help="JSON array of document texts (skinki dump: entries.json)",
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="JSON array of query texts (skinki dump: queries.json). Optional, "
        "but required to measure retrieval (docs+queries must share a space).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for entries.f32 (+ queries.f32) and meta.json",
    )
    p.add_argument(
        "--model",
        default="google/embeddinggemma-300m",
        help="sentence-transformers model name (default: %(default)s)",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Matryoshka truncation: keep the first DIM components, then "
        "re-L2-normalize. Must match the skinki --dim. Default: %(default)s.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--doc-prompt",
        default=DEFAULT_DOC_PROMPT,
        help='Document prompt template, must contain "{content}". '
        "Default: EmbeddingGemma retrieval document prompt.",
    )
    p.add_argument(
        "--query-prompt",
        default=DEFAULT_QUERY_PROMPT,
        help='Query prompt template, must contain "{content}". '
        "Default: EmbeddingGemma retrieval query prompt.",
    )
    return p.parse_args()


def _load_texts(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
        raise SystemExit(f"error: {path} must be a JSON array of strings")
    return data


def _embed(model, texts, prompt, batch_size, dim):
    import numpy as np

    if "{content}" not in prompt:
        raise SystemExit('error: prompt must contain "{content}"')
    prompted = [prompt.format(content=t) for t in texts]
    emb = model.encode(
        prompted,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    native = emb.shape[1]
    if dim > native:
        raise SystemExit(f"error: --dim {dim} exceeds native dim {native}")
    if dim < native:
        emb = emb[:, :dim]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
    return np.ascontiguousarray(emb, dtype="<f4")


def main() -> int:
    args = parse_args()

    try:
        import numpy as np  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"error: missing dependency ({exc})", file=sys.stderr)
        print(
            "\nInstall in a venv:\n"
            "    python3 -m venv .venv-embed && . .venv-embed/bin/activate "
            "&& pip install -U sentence-transformers\n"
            "\nEmbeddingGemma is gated on Hugging Face: accept the license on the "
            "model page, then `huggingface-cli login`.",
            file=sys.stderr,
        )
        return 1

    entries = _load_texts(args.entries)
    queries = _load_texts(args.queries) if args.queries else None
    if not entries:
        print("error: no entry texts to embed", file=sys.stderr)
        return 1

    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding {len(entries)} documents ...")
    doc_emb = _embed(model, entries, args.doc_prompt, args.batch_size, args.dim)
    (args.out_dir / "entries.f32").write_bytes(doc_emb.tobytes())

    q_count = 0
    if queries:
        print(f"Embedding {len(queries)} queries ...")
        q_emb = _embed(model, queries, args.query_prompt, args.batch_size, args.dim)
        (args.out_dir / "queries.f32").write_bytes(q_emb.tobytes())
        q_count = q_emb.shape[0]

    meta = {
        "model": args.model,
        "dim": args.dim,
        "entries": doc_emb.shape[0],
        "queries": q_count,
        "doc_prompt": args.doc_prompt,
        "query_prompt": args.query_prompt,
    }
    (args.out_dir / "embeddings.meta.json").write_text(json.dumps(meta, indent=2))

    mb = doc_emb.shape[0] * args.dim * struct.calcsize("f") / 1024 / 1024
    print(f"\nWrote {doc_emb.shape[0]} doc vectors x dim {args.dim} ({mb:.1f} MB)")
    if q_count:
        print(f"Wrote {q_count} query vectors -> {args.out_dir / 'queries.f32'}")
    print("\nNext (back in the skinki sandbox / repo):")
    print(
        "  cargo run --release -p skinki-harness -- locomo-eval --path locomo10.json "
        f"--sample all --dim {args.dim} "
        f"--embeddings-file {args.out_dir / 'entries.f32'} "
        f"--query-embeddings-file {args.out_dir / 'queries.f32'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
