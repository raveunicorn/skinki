#!/usr/bin/env python3
"""Stage 1B follow-up: convert a ready-made Model2Vec static model (e.g.
minishlab/potion-base-8M) into the SKEMB001 artifact the Rust
`StaticEmbedder` serves.

Unlike `distill_static_embedder.py` (which distilled the teacher's *input*
word-embedding layer — falsified in D1 at recall@10 0.090), a Model2Vec
model is distilled by passing every vocab token through the FULL sentence
encoder and pooling the output, then PCA + Zipf re-weighting — the
attention/pooling knowledge D1 found missing is partially baked into the
static table. potion-base-8M: WordPiece tokenizer from bge-base-en-v1.5
(lowercase, `##` continuations, `[UNK]`), 29528 x 256 f32 — dim 256 native,
no PCA needed here; Zipf weighting already applied to the vectors, so the
SKEMB001 per-token weights are uniform 1.0 (0 for `[UNK]`/specials to keep
the Rust zero-vector OOV contract).

Dev tooling (rule-3 shape): runs offline, once, outside any gate; the
artifact is regenerable, not committed (fixtures/static_embed_*.skemb is
gitignored — model weights).

Usage:
  python3 scripts/convert_model2vec_to_skemb.py \
      --model minishlab/potion-base-8M \
      --out fixtures/static_embed_potion8m_256.skemb
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

MAGIC = b"SKEMB001"
FORMAT_VERSION = 1
FLAG_WORDPIECE = 1
UNK = "[UNK]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minishlab/potion-base-8M")
    ap.add_argument("--out", default="fixtures/static_embed_potion8m_256.skemb")
    args = ap.parse_args()

    snap = Path(
        snapshot_download(
            args.model,
            allow_patterns=["model.safetensors", "tokenizer.json", "config.json"],
        )
    )
    tok = json.loads((snap / "tokenizer.json").read_text())
    model = tok["model"]
    assert model["type"] == "WordPiece", f"need a WordPiece model2vec, got {model['type']}"
    assert model.get("unk_token") == UNK
    assert model.get("continuing_subword_prefix", "##") == "##"
    norm = tok.get("normalizer") or {}
    assert norm.get("lowercase", False), "Rust StaticEmbedder lowercases; tokenizer must too"

    vocab_to_id: dict[str, int] = model["vocab"]
    vocab_size = len(vocab_to_id)
    id_to_piece = [""] * vocab_size
    for piece, i in vocab_to_id.items():
        id_to_piece[i] = piece
    assert UNK in vocab_to_id, "artifact contract: [UNK] required"

    with safe_open(snap / "model.safetensors", framework="numpy") as f:
        table = f.get_tensor("embeddings").astype("<f4")
    assert table.shape[0] == vocab_size, (table.shape, vocab_size)
    dim = table.shape[1]

    # Zipf weighting is already baked into the Model2Vec vectors; plain mean
    # pooling is the model's own inference. Specials/[UNK] weight 0 keeps the
    # Rust empty/all-OOV -> zero-vector contract.
    weights = np.ones(vocab_size, dtype="<f4")
    zeroed = 0
    for i, piece in enumerate(id_to_piece):
        if piece.startswith("[") and piece.endswith("]"):
            weights[i] = 0.0
            zeroed += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", FORMAT_VERSION, dim, vocab_size, FLAG_WORDPIECE))
        for piece in id_to_piece:
            b = piece.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        f.write(table.tobytes())
        f.write(weights.tobytes())
    size = out.stat().st_size
    budget = 48 * 1024 * 1024
    print(
        f"[convert] {args.model}: vocab {vocab_size} x dim {dim}, "
        f"{zeroed} zero-weight special tokens -> {out} "
        f"({size} bytes; budget {budget}; {'OK' if size <= budget else 'OVER BUDGET'})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
