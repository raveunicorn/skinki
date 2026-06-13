#!/usr/bin/env python3
"""Export real sentence-embeddings for a kortex corpus to a raw f32 file.

Dev-only helper, NOT part of the Cargo workspace. Takes a corpus produced by
`kortex generate --out ...` (JSON: {"meta":..., "entries":[{"id":.., "text":..}],
"ground_truth":...}), embeds the entry texts with a sentence-transformers
model, optionally Matryoshka-truncates + re-normalizes, and writes raw
little-endian float32 row-major data ready for:

    cargo run --release -p kortex-harness -- compress-bench --vectors-file <out> --dim <dim>
    cargo run --release -p kortex-harness -- scale-bench    --vectors-file <out> --dim <dim>
"""

import argparse
import json
import struct
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Path to corpus JSON produced by `kortex generate --out ...`",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path for raw little-endian f32 row-major vectors",
    )
    p.add_argument(
        "--model",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="sentence-transformers model name (default: %(default)s)",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Matryoshka truncation: keep only the first DIM components of "
        "each embedding, then re-L2-normalize. Default: keep the model's "
        "native dimensionality.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of corpus entries to embed (default: all)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: %(default)s)",
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="Document prefix prepended to each text before embedding. "
        'Default: "search_document: " for nomic models, empty otherwise '
        "(nomic models require task prefixes).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Imported lazily so --help works without the (heavy, optional) ML deps.
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"error: missing dependency ({exc})", file=sys.stderr)
        print(
            "\nInstall the embedding deps in a dedicated venv:\n"
            "    python3 -m venv .venv-embed && . .venv-embed/bin/activate "
            "&& pip install sentence-transformers einops\n"
            "\nNotes:\n"
            "  - nomic-ai/* models require trust_remote_code=True (this script "
            "passes it automatically).\n"
            "  - google/embeddinggemma-300m is gated on Hugging Face: accept "
            "the license on the model page, then run `huggingface-cli login`.",
            file=sys.stderr,
        )
        return 1

    corpus = json.loads(args.corpus.read_text())
    entries = corpus["entries"]
    if args.limit is not None:
        entries = entries[: args.limit]
    texts = [e["text"] for e in entries]
    if not texts:
        print("error: corpus has no entries to embed", file=sys.stderr)
        return 1

    prefix = args.prefix
    if prefix is None:
        prefix = "search_document: " if "nomic" in args.model.lower() else ""
    if prefix:
        texts = [prefix + t for t in texts]

    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model, trust_remote_code=True)

    print(f"Embedding {len(texts)} entries (batch size {args.batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    native_dim = embeddings.shape[1]
    if args.dim is not None and args.dim > native_dim:
        print(
            f"error: --dim {args.dim} exceeds the model's native dim {native_dim}",
            file=sys.stderr,
        )
        return 1
    dim = args.dim if args.dim is not None else native_dim
    if args.dim is not None:
        embeddings = embeddings[:, : args.dim]
        # Matryoshka truncation breaks unit-norm; re-normalize so downstream
        # dot products behave like cosine similarity again.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    embeddings = np.ascontiguousarray(embeddings, dtype="<f4")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    embeddings.tofile(args.out)

    meta = {
        "model": args.model,
        "dim": dim,
        "count": embeddings.shape[0],
        "corpus": str(args.corpus),
        "prefix": prefix,
    }
    meta_path = args.out.with_name(args.out.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    size_mb = (embeddings.shape[0] * dim * struct.calcsize("f")) / (1024 * 1024)
    print(
        f"\nWrote {embeddings.shape[0]} vectors x dim {dim} "
        f"({size_mb:.1f} MB) -> {args.out}"
    )
    print(f"Sidecar metadata -> {meta_path}")
    print("\nNext:")
    print(
        f"  cargo run --release -p kortex-harness -- compress-bench "
        f"--vectors-file {args.out} --dim {dim}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
