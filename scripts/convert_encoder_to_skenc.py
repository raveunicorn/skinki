#!/usr/bin/env python3
"""Stage 1C-B — T1: convert a BERT-class HF sentence encoder (bge-small-en-v1.5)
into the `SKENC001` weight artifact + dump the parity goldens for T2.

Dev tooling (rule-3 shape): runs offline, once, outside any gate. The ~130 MB
artifact is model weights and is NOT committed (`fixtures/*.skenc` is
gitignored except the toy); the committed parity contract is:

  - `fixtures/encoder_layer_golden.f32` — teacher activations after the
    embedding LayerNorm and after each encoder layer, for one fixed probe
    input. T2 compares layer-by-layer with a tolerance, so a numerical bug
    localizes to a single layer.
  - `fixtures/encoder_golden_embeddings.f32` — 32 fixed strings → pooled,
    L2-normalized teacher embeddings. T2 asserts cosine ≥ 0.999 per vector
    (byte-parity with a torch forward is impossible — the Rust side uses
    in-crate polynomial transcendentals and fixed-order f32 sums).

Tokenization for the goldens mirrors the *Rust* convention (lowercase, no
accent stripping, greedy WordPiece — the same mirror as
`distill_static_embedder.py`), then adds `[CLS]` / `[SEP]`, so the goldens
test the numerics, not HF tokenizer quirks the engine deliberately does not
reproduce.

Layout details the Rust reader depends on (see §4 of the spec + `format.rs`):
  - all W stored [in][out] row-major (torch Linear keeps [out][in] → transposed
    here), so the Rust side computes `out = x · W + b` with no transposes;
  - `pos_type_emb` = position embeddings + the token-type-0 row, pre-summed
    (the engine never uses segment B);
  - vocab strings in id order, NFC; `[UNK]`, `[CLS]`, `[SEP]` required.

Usage:
  python3 scripts/convert_encoder_to_skenc.py \
      --teacher BAAI/bge-small-en-v1.5 \
      --out fixtures/encoder_bge_small.skenc \
      --layer-golden-out fixtures/encoder_layer_golden.f32 \
      --golden-out fixtures/encoder_golden_embeddings.f32
"""

from __future__ import annotations

import argparse
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MAGIC = b"SKENC001"
FORMAT_VERSION = 1
ARCH_BERT = 1
POOLING_CLS = 0
POOLING_MEAN = 1
CONT = "##"
UNK, CLS, SEP = "[UNK]", "[CLS]", "[SEP]"

# The fixed probe input for the per-layer goldens: exercises whole words,
# `##` continuations, punctuation and an OOV/Cyrillic token.
PROBE = "the memory engine embeds rust vectors, deterministically. память"

# The 32 parity strings — same set as the 1B golden fixture so the two
# artifacts' parity suites stay comparable (kept in sync by hand; the list is
# the contract, the file is dev tooling).
GOLDENS = [
    "memory engine",
    "rust vector search",
    "the happy insight",
    "distributed systems latency budgets",
    "recall precision false insight apophenia",
    "sleep scheduler power idle thermal",
    "embed compress rabitq ivf mmap",
    "graph provenance derivation ledger staleness",
    "con",
    "ПАМЯТЬ",
    "memorys engine",
    "running tests",
    "",
    "   ",
    "memory, engine.",
    "the quick brown fox jumps over the lazy dog",
    "cafe",
    "naïve",
    "12345 numbers",
    "MIXED Case WORDS",
    "a the and of to in is it",
    "structural bridge detector benjamini hochberg",
    "raphæl",
    "Москва",
    "кофе по утрам",
    " tab\tspace",
    "punctuation!and?marks",
    "model2vec distillation static embedder table lookup pooling",
    "ffff zzzz qqqq",
    "edge case: a single character word a",
    "tokenizer wordpiece greedy longest match",
    "ABCdefGHI",
]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def pretokenize(text: str) -> list[str]:
    """Mirror of `skinki_vector::wordpiece::pretokenize` (Rust)."""
    text = nfc(text).lower()
    out: list[str] = []
    word_start: int | None = None
    for i, ch in enumerate(text):
        if ch.isalnum():
            if word_start is None:
                word_start = i
        else:
            if word_start is not None:
                out.append(text[word_start:i])
                word_start = None
            if not ch.isspace():
                out.append(ch)
    if word_start is not None:
        out.append(text[word_start:])
    return out


def wordpiece(word: str, piece_to_id: dict[str, int], unk_id: int) -> list[int]:
    """Mirror of `WordPieceTokenizer::wordpiece` (Rust)."""
    chars = list(word)
    if not chars:
        return []
    sub: list[int] = []
    start = 0
    while start < len(chars):
        end = len(chars)
        cur = None
        while start < end:
            sub_str = "".join(chars[start:end])
            key = f"{CONT}{sub_str}" if start > 0 else sub_str
            if key in piece_to_id:
                cur = piece_to_id[key]
                break
            end -= 1
        if cur is None:
            return [unk_id]
        sub.append(cur)
        start = end
    return sub


def encode_with_specials(
    text: str, piece_to_id: dict[str, int], unk_id: int, max_pos: int
) -> list[int]:
    """Rust-convention content ids, framed as [CLS] content [SEP], truncated
    to max_pos total (mirrors the Rust forward's framing exactly)."""
    ids: list[int] = []
    for word in pretokenize(text):
        ids.extend(wordpiece(word, piece_to_id, unk_id))
    ids = ids[: max_pos - 2]
    return [piece_to_id[CLS]] + ids + [piece_to_id[SEP]]


def tensor_le(t: torch.Tensor) -> bytes:
    return t.detach().to(torch.float32).numpy().astype("<f4").tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    ap.add_argument("--out", default="fixtures/encoder_bge_small.skenc")
    ap.add_argument("--layer-golden-out", default="fixtures/encoder_layer_golden.f32")
    ap.add_argument("--golden-out", default="fixtures/encoder_golden_embeddings.f32")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModel.from_pretrained(args.teacher).eval().to(torch.float32)
    cfg = model.config
    assert cfg.model_type == "bert", f"need a BERT-class teacher, got {cfg.model_type}"
    assert getattr(cfg, "position_embedding_type", "absolute") == "absolute"

    layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    ffn = cfg.intermediate_size
    heads = cfg.num_attention_heads
    max_pos = cfg.max_position_embeddings
    ln_eps = float(cfg.layer_norm_eps)

    # Vocab in id order, NFC-canonical.
    vocab_map = tokenizer.get_vocab()
    id_to_piece = [""] * len(vocab_map)
    for piece, i in vocab_map.items():
        id_to_piece[i] = nfc(piece)
    for required in (UNK, CLS, SEP):
        assert required in vocab_map, f"teacher vocab lacks {required}"
    piece_to_id = {p: i for i, p in enumerate(id_to_piece)}
    unk_id = piece_to_id[UNK]

    emb = model.embeddings
    # pos_type_emb: bake the token-type-0 row into every position row.
    pos_type = emb.position_embeddings.weight + emb.token_type_embeddings.weight[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        pooling = POOLING_CLS if args.pooling == "cls" else POOLING_MEAN
        f.write(
            struct.pack(
                "<9I",
                FORMAT_VERSION,
                ARCH_BERT,
                layers,
                hidden,
                ffn,
                heads,
                len(id_to_piece),
                max_pos,
                pooling,
            )
        )
        f.write(struct.pack("<f", ln_eps))
        for piece in id_to_piece:
            b = piece.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        while f.tell() % 4 != 0:
            f.write(b"\x00")

        f.write(tensor_le(emb.word_embeddings.weight))
        f.write(tensor_le(pos_type))
        f.write(tensor_le(emb.LayerNorm.weight))
        f.write(tensor_le(emb.LayerNorm.bias))
        for i in range(layers):
            lyr = model.encoder.layer[i]
            att, out = lyr.attention.self, lyr.attention.output
            # torch Linear weight is [out, in] — store [in, out].
            f.write(tensor_le(att.query.weight.T.contiguous()))
            f.write(tensor_le(att.query.bias))
            f.write(tensor_le(att.key.weight.T.contiguous()))
            f.write(tensor_le(att.key.bias))
            f.write(tensor_le(att.value.weight.T.contiguous()))
            f.write(tensor_le(att.value.bias))
            f.write(tensor_le(out.dense.weight.T.contiguous()))
            f.write(tensor_le(out.dense.bias))
            f.write(tensor_le(out.LayerNorm.weight))
            f.write(tensor_le(out.LayerNorm.bias))
            f.write(tensor_le(lyr.intermediate.dense.weight.T.contiguous()))
            f.write(tensor_le(lyr.intermediate.dense.bias))
            f.write(tensor_le(lyr.output.dense.weight.T.contiguous()))
            f.write(tensor_le(lyr.output.dense.bias))
            f.write(tensor_le(lyr.output.LayerNorm.weight))
            f.write(tensor_le(lyr.output.LayerNorm.bias))
    size = out_path.stat().st_size
    print(
        f"[convert] {args.teacher}: {layers}L x {hidden}H x {ffn}F, "
        f"vocab {len(id_to_piece)}, max_pos {max_pos} -> {out_path} ({size} bytes)",
        file=sys.stderr,
    )

    def run(ids: list[int]):
        t = torch.tensor([ids], dtype=torch.long)
        return model(
            input_ids=t,
            attention_mask=torch.ones_like(t),
            token_type_ids=torch.zeros_like(t),
            output_hidden_states=True,
        )

    # --- Layer goldens: probe input, hidden_states[0..layers] --------------
    # hidden_states[0] is the embedding output (post-LayerNorm); [1..layers]
    # are the outputs of each encoder layer. Format:
    #   u32 n_states | u32 seq | u32 hidden | u32 n_ids | n_ids×u32 |
    #   n_states × seq × hidden f32 LE
    probe_ids = encode_with_specials(PROBE, piece_to_id, unk_id, max_pos)
    states = run(probe_ids).hidden_states
    lg_path = Path(args.layer_golden_out)
    with open(lg_path, "wb") as f:
        f.write(struct.pack("<4I", len(states), len(probe_ids), hidden, len(probe_ids)))
        for i in probe_ids:
            f.write(struct.pack("<I", i))
        for st in states:
            f.write(st[0].to(torch.float32).numpy().astype("<f4").tobytes())
    print(
        f"[convert] wrote {lg_path} ({len(states)} states x {len(probe_ids)} tokens; "
        f"probe = {PROBE!r})",
        file=sys.stderr,
    )

    # --- E2E goldens: 32 strings -> pooled, L2-normalized ------------------
    g_path = Path(args.golden_out)
    with open(g_path, "wb") as f:
        f.write(struct.pack("<II", len(GOLDENS), hidden))
        for s in GOLDENS:
            b = s.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        for s in GOLDENS:
            ids = encode_with_specials(s, piece_to_id, unk_id, max_pos)
            last = run(ids).last_hidden_state[0]
            if args.pooling == "cls":
                v = last[0]
            else:
                v = last.mean(dim=0)
            v = v.to(torch.float32).numpy().astype(np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 1e-12:
                v = v / np.float32(norm)
            f.write(v.astype("<f4").tobytes())
    print(f"[convert] wrote {g_path} ({len(GOLDENS)} strings x {hidden} dims)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
