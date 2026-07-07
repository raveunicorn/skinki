#!/usr/bin/env python3
"""Stage 1C-B / 1D — T1: convert a BERT-class HF sentence encoder into the
`SKENC001` v2 weight artifact + dump the parity goldens for the Rust forward.

Two model families are supported (selected via `--tokenizer`, or inferred
from `--unigram-sku` if omitted — NOT from `cfg.model_type`, which does not
determine the tokenizer family: `multilingual-e5-small` is `model_type='bert'`
yet uses a SentencePiece Unigram vocab):

  * **WordPiece / BERT** (e.g. `BAAI/bge-small-en-v1.5`) — Stage 1C-B. The vocab
    is written inline (`vocab × (u32 len | UTF-8 bytes)`), id = order; `[UNK]`,
    `[CLS]`, `[SEP]` required. The v1 artifact layout the original T1 shipped is
    preserved verbatim under the new v2 header (which adds `tok_kind` and the
    two prefix strings). bge uses CLS pooling and carries its
    `"Represent this sentence for searching relevant passages: "` instruction
    in `query_prefix` (passage side empty) so the engine applies it
    automatically via `EmbedderSpec::Encoder` — the 1C-B D2 finding that a
    forgotten query prefix costs ~25% recall, made structural.
  * **Unigram / XLM-R** (e.g. `intfloat/multilingual-e5-small`) — Stage 1D T1.
    The tokenizer section is `<u32 sku_size><SKUNI001 bytes>` — the FULL
    SentencePiece Unigram artifact produced by `scripts/dump_unigram_fixtures.py`
    is inlined, so one file = one model (no external tokenizer path). e5 uses
    mean pooling and the model-contract prefixes `"query: "` / `"passage: "`,
    which the converter writes into the header so the engine never hardcodes
    them.

Dev tooling (rule-3 shape): runs offline, once, outside any gate. The model
weights are NOT committed (`fixtures/*.skenc` is gitignored except the toy);
the committed parity contract is:

  - `fixtures/encoder_layer_golden.f32` — teacher activations after the
    embedding LayerNorm and after each encoder layer, for one fixed probe
    input. T2 compares layer-by-layer with a tolerance, so a numerical bug
    localizes to a single layer.
  - `fixtures/encoder_golden_embeddings.f32` — 32 fixed strings → pooled,
    L2-normalized teacher embeddings. T2 asserts cosine ≥ 0.999 per vector
    (byte-parity with a torch forward is impossible — the Rust side uses
    in-crate polynomial transcendentals and fixed-order f32 sums).

**Stage 1D K0 landmine (the 1C-B "Rust-convention tokenization" lesson,
generalized to Unigram):** tokenization for the goldens mirrors the *Rust*
convention, then the model is run with `input_ids=torch.tensor([ids])`
directly. `AutoTokenizer.encode` is NEVER called inside the golden dump — for
WordPiece the Rust side lowercases + greedy-matches; for Unigram the HF Rust
`tokenizers` backend diverges from reference `sentencepiece` on adversarial
inputs (K0 finding), so the goldens test the *engine's* numerics against the
teacher over the engine's own id sequence, not HF tokenizer quirks. The
Unigram mirror below reimplements `UnigramTokenizer::encode_content` —
normalization (charsmap longest-match + SP whitespace/`▁` handling) and
Viterbi segmentation with the documented tie-break (longer piece on exact
score tie) — so the ids fed to torch are bit-stable across `transformers`
releases.

Layout details the Rust reader depends on (see `format.rs` and the spec §4):
  - header is v2: magic | version=2 | arch=1(BERT) | layers | hidden | ffn |
    heads | vocab | max_pos | pooling | tok_kind | query_prefix_len |
    passage_prefix_len | query_prefix | passage_prefix | ln_eps;
  - WordPiece section: vocab strings in id order, NFC; `[UNK]`, `[CLS]`,
    `[SEP]` required; Unigram section: `<u32 sku_size><SKUNI001 bytes>`;
  - all W stored [in][out] row-major (torch Linear keeps [out][in] → transposed
    here), so the Rust side computes `out = x · W + b` with no transposes;
  - `pos_type_emb` = position embeddings + the token-type-0 row, pre-summed
    (the engine never uses segment B).

Usage (bge-small, WordPiece / CLS):
  python3 scripts/convert_encoder_to_skenc.py \\
      --teacher BAAI/bge-small-en-v1.5 \\
      --pooling cls \\
      --out fixtures/encoder_bge_small.skenc \\
      --layer-golden-out fixtures/encoder_layer_golden.f32 \\
      --golden-out fixtures/encoder_golden_embeddings.f32

Usage (multilingual-e5-small, Unigram / mean / prefixes):
  python3 scripts/convert_encoder_to_skenc.py \\
      --teacher intfloat/multilingual-e5-small \\
      --tokenizer unigram \\
      --unigram-sku fixtures/unigram_e5_small.sku \\
      --pooling mean \\
      --query-prefix 'query: ' --passage-prefix 'passage: ' \\
      --out fixtures/encoder_e5_small.skenc \\
      --layer-golden-out fixtures/encoder_e5_layer_golden.f32 \\
      --golden-out fixtures/encoder_e5_golden_embeddings.f32
"""

from __future__ import annotations

import argparse
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

MAGIC = b"SKENC001"
FORMAT_VERSION = 2
ARCH_BERT = 1
POOLING_CLS = 0
POOLING_MEAN = 1
TOK_WORDPIECE = 0
TOK_UNIGRAM = 1
CONT = "##"
UNK, CLS, SEP = "[UNK]", "[CLS]", "[SEP]"

# The fixed probe input for the per-layer goldens: exercises whole words,
# `##` continuations, punctuation and an OOV/Cyrillic token. Same probe for
# every model so layer-drift comparisons are apples-to-apples.
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


# ---------------------------------------------------------------------------
# Rust-convention WordPiece mirror (BERT path) — unchanged from Stage 1C-B T1.
# ---------------------------------------------------------------------------


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def wordpiece_pretokenize(text: str) -> list[str]:
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


def wordpiece_encode_word(word: str, piece_to_id: dict[str, int], unk_id: int) -> list[int]:
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


def wordpiece_encode_content(text: str, piece_to_id: dict[str, int], unk_id: int) -> list[int]:
    ids: list[int] = []
    for word in wordpiece_pretokenize(text):
        ids.extend(wordpiece_encode_word(word, piece_to_id, unk_id))
    return ids


# ---------------------------------------------------------------------------
# Rust-convention Unigram mirror (XLM-R / e5 path). Reimplements
# `skinki_vector::unigram::{normalize, segment, encode_content}` — the same
# Python the K0 dump script validated, kept here so the goldens test the
# engine's tokenizer, not HF's Rust `tokenizers` backend.
# ---------------------------------------------------------------------------


SPACE_SYMBOL = "\u2581"  # ▁ — SentencePiece's whitespace escape

FLAG_ADD_DUMMY_PREFIX = 1 << 0
FLAG_REMOVE_EXTRA_WHITESPACES = 1 << 1
FLAG_ESCAPE_WHITESPACES = 1 << 2
FLAG_TREAT_WHITESPACE_AS_SUFFIX = 1 << 3


def _trie_longest_match(chars: list[str], pos: int, trie: dict) -> tuple[str, int] | None:
    """Char-keyed longest-prefix match, mirroring `longest_match` in Rust."""
    node = trie
    best: tuple[str, int] | None = None
    for i in range(pos, len(chars)):
        c = chars[i]
        if c not in node:
            break
        node = node[c]
        if "" in node:  # value marker at this node
            best = (node[""], i - pos + 1)
    return best


def _trie_insert(trie: dict, key: str, value: str) -> None:
    node = trie
    for c in key:
        node = node.setdefault(c, {})
    node[""] = value


def unigram_normalize(text: str, charsmap: dict, spec: tuple[bool, bool, bool, bool]) -> str:
    add_dummy, remove_extra, escape, as_suffix = spec
    if not text:
        return ""
    chars = list(text)
    n = len(chars)

    pos = 0
    if remove_extra:
        while pos < n:
            repl, consumed = _trie_longest_match(chars, pos, charsmap) or (chars[pos], 1)
            if repl != " ":
                break
            pos += consumed
    if pos >= n:
        return ""

    space_symbol = SPACE_SYMBOL if escape else " "

    out: list[str] = []
    if not as_suffix and add_dummy:
        out.append(space_symbol)

    is_prev_space = remove_extra
    while pos < n:
        repl, consumed = _trie_longest_match(chars, pos, charsmap) or (chars[pos], 1)
        sp = repl
        if is_prev_space:
            sp = sp.lstrip(" ")
        if sp:
            for ch in sp:
                out.append(space_symbol if ch == " " else ch)
            is_prev_space = sp.endswith(" ")
        pos += consumed
        if not remove_extra:
            is_prev_space = False

    if remove_extra:
        while out and out[-1] == space_symbol:
            out.pop()

    if as_suffix and add_dummy:
        out.append(space_symbol)
    return "".join(out)


def unigram_segment(
    normalized: str,
    vocab_trie: dict,
    unk_sp_id: int,
    unk_score: float,
) -> list[int]:
    """Viterbi DP mirroring `segment` in Rust; tie-break favors the longer
    piece (K0 design constraint)."""
    chars = list(normalized)
    n = len(chars)
    if n == 0:
        return []

    NEG_INF = float("-inf")
    dp = [NEG_INF] * (n + 1)
    dp[0] = 0.0
    back_start = [0] * (n + 1)
    back_id = [0] * (n + 1)

    for begin in range(n):
        if dp[begin] == NEG_INF:
            continue
        base = dp[begin]
        node = vocab_trie
        found_len1 = False
        for length in range(1, n - begin + 1):
            c = chars[begin + length - 1]
            if c not in node:
                break
            node = node[c]
            if "" in node:  # (id, score) stored under ""
                sp_id, score = node[""]
                if length == 1:
                    found_len1 = True
                end = begin + length
                total = base + score
                if total > dp[end]:
                    dp[end] = total
                    back_start[end] = begin
                    back_id[end] = sp_id
        if not found_len1:
            end = begin + 1
            total = base + unk_score
            if total > dp[end]:
                dp[end] = total
                back_start[end] = begin
                back_id[end] = unk_sp_id

    ids_rev: list[int] = []
    pos = n
    while pos > 0:
        ids_rev.append(back_id[pos])
        pos = back_start[pos]
    ids_rev.reverse()

    merged: list[int] = []
    for sp_id in ids_rev:
        if sp_id == unk_sp_id and merged and merged[-1] == unk_sp_id:
            continue
        merged.append(sp_id)
    return merged


def unigram_load_sku(path: Path) -> dict:
    """Parse a `SKUNI001` artifact enough to drive the Rust-convention mirror.
    Returns a dict with the charsmap trie, the vocab trie, the spec flags, and
    the sp_unk_id / fairseq_offset / unk_hf_id / unk_score fields."""
    data = path.read_bytes()
    assert data[:8] == b"SKUNI001", f"not a SKUNI001 artifact: {path}"
    r = 8
    version = struct.unpack_from("<I", data, r)[0]
    r += 4
    assert version == 1, f"unsupported sku version {version}"
    (vocab_size, sp_unk_id, fairseq_offset, unk_hf_id, bos_hf_id, eos_hf_id) = struct.unpack_from(
        "<6I", data, r
    )
    r += 24
    unk_score = struct.unpack_from("<f", data, r)[0]
    r += 4
    flags = struct.unpack_from("<I", data, r)[0]
    r += 4
    charsmap_entry_count = struct.unpack_from("<I", data, r)[0]
    r += 4

    vocab_trie: dict = {}
    pieces: list[tuple[str, float, int]] = []
    for _ in range(vocab_size):
        (plen,) = struct.unpack_from("<I", data, r)
        r += 4
        piece = data[r : r + plen].decode("utf-8")
        r += plen
        (score,) = struct.unpack_from("<f", data, r)
        r += 4
        (ty,) = struct.unpack_from("<I", data, r)
        r += 4
        pieces.append((piece, score, ty))
        if ty == 1:  # Normal
            _trie_insert(vocab_trie, piece, (len(pieces) - 1, score))

    charsmap: dict = {}
    for _ in range(charsmap_entry_count):
        (klen,) = struct.unpack_from("<I", data, r)
        r += 4
        key = data[r : r + klen].decode("utf-8")
        r += klen
        (vlen,) = struct.unpack_from("<I", data, r)
        r += 4
        val = data[r : r + vlen].decode("utf-8")
        r += vlen
        _trie_insert(charsmap, key, val)

    spec = (
        bool(flags & FLAG_ADD_DUMMY_PREFIX),
        bool(flags & FLAG_REMOVE_EXTRA_WHITESPACES),
        bool(flags & FLAG_ESCAPE_WHITESPACES),
        bool(flags & FLAG_TREAT_WHITESPACE_AS_SUFFIX),
    )
    return {
        "vocab_trie": vocab_trie,
        "charsmap": charsmap,
        "spec": spec,
        "sp_unk_id": sp_unk_id,
        "fairseq_offset": fairseq_offset,
        "unk_hf_id": unk_hf_id,
        "bos_hf_id": bos_hf_id,
        "eos_hf_id": eos_hf_id,
        "unk_score": unk_score,
    }


def unigram_encode_content(text: str, sku: dict) -> list[int]:
    normalized = unigram_normalize(text, sku["charsmap"], sku["spec"])
    sp_ids = unigram_segment(
        normalized, sku["vocab_trie"], sku["sp_unk_id"], sku["unk_score"]
    )
    out: list[int] = []
    for sp_id in sp_ids:
        if sp_id == sku["sp_unk_id"]:
            out.append(sku["unk_hf_id"])
        else:
            out.append(sp_id + sku["fairseq_offset"])
    return out


# ---------------------------------------------------------------------------
# Sequence framing (Rust-convention): [BOS] content [EOS], truncated to
# max_pos total. The BOS/EOS ids come from the same source the Rust side uses.
# ---------------------------------------------------------------------------


def frame_ids(content_ids: list[int], bos_id: int, eos_id: int, max_pos: int) -> list[int]:
    ids = content_ids[: max_pos - 2]
    return [bos_id] + ids + [eos_id]


def tensor_le(t: torch.Tensor) -> bytes:
    return t.detach().to(torch.float32).numpy().astype("<f4").tobytes()


def write_header(
    f,
    cfg,
    pooling: str,
    tok_kind: int,
    query_prefix: str,
    passage_prefix: str,
) -> None:
    f.write(MAGIC)
    f.write(struct.pack("<I", FORMAT_VERSION))
    f.write(struct.pack("<I", ARCH_BERT))
    f.write(
        struct.pack(
            "<8I",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_attention_heads,
            cfg.vocab_size,
            cfg.max_position_embeddings,
            POOLING_CLS if pooling == "cls" else POOLING_MEAN,
            tok_kind,
        )
    )
    qb = query_prefix.encode("utf-8")
    pb = passage_prefix.encode("utf-8")
    f.write(struct.pack("<I", len(qb)))
    f.write(struct.pack("<I", len(pb)))
    f.write(qb)
    f.write(pb)
    f.write(struct.pack("<f", float(cfg.layer_norm_eps)))


def write_wordpiece_section(f, tokenizer) -> tuple[dict[str, int], int]:
    vocab_map = tokenizer.get_vocab()
    id_to_piece = [""] * len(vocab_map)
    for piece, i in vocab_map.items():
        id_to_piece[i] = nfc(piece)
    for required in (UNK, CLS, SEP):
        assert required in vocab_map, f"teacher vocab lacks {required}"
    piece_to_id = {p: i for i, p in enumerate(id_to_piece)}
    unk_id = piece_to_id[UNK]
    for piece in id_to_piece:
        b = piece.encode("utf-8")
        f.write(struct.pack("<I", len(b)))
        f.write(b)
    return piece_to_id, unk_id


def write_unigram_section(f, sku_path: Path) -> None:
    sku_bytes = Path(sku_path).read_bytes()
    assert sku_bytes[:8] == b"SKUNI001", f"not a SKUNI001 artifact: {sku_path}"
    f.write(struct.pack("<I", len(sku_bytes)))
    f.write(sku_bytes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="BAAI/bge-small-en-v1.5")
    ap.add_argument(
        "--tokenizer",
        choices=["wordpiece", "unigram"],
        default=None,
        help="tokenizer family for the artifact. If omitted, inferred from "
        "the other flags: present --unigram-sku selects 'unigram', else "
        "'wordpiece'. (NOT auto-detected from model_type: multilingual-e5-small "
        "is model_type='bert' — a MiniLM student with an XLM-R vocab — yet "
        "needs the Unigram path; the model_type→tokenizer mapping is not "
        "reliable, so the caller states it.)",
    )
    ap.add_argument(
        "--unigram-sku",
        default=None,
        help="path to a SKUNI001 artifact from dump_unigram_fixtures.py "
        "(required for --tokenizer unigram)",
    )
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--passage-prefix", default="")
    ap.add_argument("--out", default="fixtures/encoder_bge_small.skenc")
    ap.add_argument("--layer-golden-out", default="fixtures/encoder_layer_golden.f32")
    ap.add_argument("--golden-out", default="fixtures/encoder_golden_embeddings.f32")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(args.teacher).eval().to(torch.float32)
    cfg = model.config
    assert cfg.model_type in ("bert", "xlm-roberta"), (
        f"need a BERT- or XLM-R-class teacher, got model_type={cfg.model_type!r}"
    )
    assert getattr(cfg, "position_embedding_type", "absolute") == "absolute"

    # Select the tokenizer family. Do NOT infer from model_type: the
    # tokenizer family and the encoder architecture are independent —
    # multilingual-e5-small is model_type='bert' (a MiniLM student) but uses
    # a SentencePiece Unigram vocab, while a real XLM-RoBERTa backbone uses
    # Unigram with model_type='xlm-roberta'. The caller states the family
    # explicitly (--tokenizer) or via the presence of --unigram-sku.
    if args.tokenizer is None:
        args.tokenizer = "unigram" if args.unigram_sku else "wordpiece"
    tok_kind = TOK_UNIGRAM if args.tokenizer == "unigram" else TOK_WORDPIECE
    if tok_kind == TOK_UNIGRAM:
        assert args.unigram_sku, (
            "--tokenizer unigram requires --unigram-sku PATH "
            "(run scripts/dump_unigram_fixtures.py first)"
        )

    # Real XLM-RoBERTa backbones (model_type='xlm-roberta', e.g. bge-m3) shift
    # position ids by padding_idx+1 when computing embeddings; the Rust
    # encoder indexes position rows 0..seq directly. This converter currently
    # handles only model_type='bert' (no shift) — the two MiniLM/BERT-shape
    # models we ship (bge-small-en, multilingual-e5-small). A genuine
    # XLM-RoBERTa teacher needs a separate ticket: reapply the padding_idx+1
    # shift to emb.position_embeddings before pre-summing token-type-0, then
    # verify layer parity. The check is loud (goldens would not converge) but
    # explicit so the failure points here, not at a mystery parity drop.
    assert cfg.model_type == "bert", (
        f"model_type={cfg.model_type!r} not supported by this converter: real "
        "XLM-RoBERTa backbones apply a padding_idx+1 position-id shift that "
        "is not yet implemented. Only model_type='bert' (bge-small-en, "
        "multilingual-e5-small) is handled. File a ticket for XLM-R backbone "
        "support."
    )

    layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    ffn = cfg.intermediate_size
    heads = cfg.num_attention_heads
    max_pos = cfg.max_position_embeddings

    # Per-family content encoder + framing ids, mirroring the Rust side. The
    # golden dump feeds these ids straight into the teacher — the K0 landmine:
    # NEVER call AutoTokenizer.encode here, the HF Rust `tokenizers` backend
    # diverges from reference sentencepiece on adversarial inputs.
    if tok_kind == TOK_WORDPIECE:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.teacher)
        piece_to_id, unk_id = {}, 0  # filled by write_wordpiece_section
        sku = None
    else:
        # --unigram-sku presence already asserted above.
        sku = unigram_load_sku(Path(args.unigram_sku))
        tokenizer = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        write_header(f, cfg, args.pooling, tok_kind, args.query_prefix, args.passage_prefix)

        if tok_kind == TOK_WORDPIECE:
            piece_to_id, unk_id = write_wordpiece_section(f, tokenizer)
            bos_id, eos_id = piece_to_id[CLS], piece_to_id[SEP]
        else:
            write_unigram_section(f, Path(args.unigram_sku))
            bos_id, eos_id = sku["bos_hf_id"], sku["eos_hf_id"]

        while f.tell() % 4 != 0:
            f.write(b"\x00")

        emb = model.embeddings
        pos_type = emb.position_embeddings.weight + emb.token_type_embeddings.weight[0]
        f.write(tensor_le(emb.word_embeddings.weight))
        f.write(tensor_le(pos_type))
        f.write(tensor_le(emb.LayerNorm.weight))
        f.write(tensor_le(emb.LayerNorm.bias))
        for i in range(layers):
            lyr = model.encoder.layer[i]
            att, out = lyr.attention.self, lyr.attention.output
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
    tok_label = "Unigram" if tok_kind == TOK_UNIGRAM else "WordPiece"
    print(
        f"[convert] {args.teacher}: {layers}L x {hidden}H x {ffn}F, "
        f"vocab {cfg.vocab_size}, max_pos {max_pos}, {tok_label}, "
        f"pooling={args.pooling}, q-prefix={args.query_prefix!r}, "
        f"p-prefix={args.passage_prefix!r} -> {out_path} ({size} bytes)",
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

    def encode_content(text: str) -> list[int]:
        if tok_kind == TOK_WORDPIECE:
            return wordpiece_encode_content(text, piece_to_id, unk_id)
        return unigram_encode_content(text, sku)

    def frame(text: str) -> list[int]:
        return frame_ids(encode_content(text), bos_id, eos_id, max_pos)

    # --- Layer goldens: probe input, hidden_states[0..layers] --------------
    probe_ids = frame(PROBE)
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
    # The Rust `RustEncoder::embed` applies the artifact's passage prefix at
    # index time, so the goldens must be dumped over the *passage-prefixed*
    # text — otherwise the parity test compares prefix-on (Rust) vs prefix-off
    # (torch) and fails on the first asymmetric model (e5). For symmetric
    # models (bge, empty prefixes) this is a no-op, so the bge goldens are
    # unaffected.
    g_path = Path(args.golden_out)
    with open(g_path, "wb") as f:
        f.write(struct.pack("<II", len(GOLDENS), hidden))
        for s in GOLDENS:
            b = s.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        for s in GOLDENS:
            passage_text = f"{args.passage_prefix}{s}"
            ids = frame(passage_text)
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
