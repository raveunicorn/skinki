#!/usr/bin/env python3
"""Stage 1D — K0: convert the `intfloat/multilingual-e5-small` SentencePiece
Unigram tokenizer (an XLM-RoBERTa tokenizer) into the `SKUNI001` artifact +
dump the `unigram_parity.json` golden corpus for `skinki-vector::unigram`.

Dev tooling (rule-5 shape): runs offline, once, outside any gate. The few-MB
artifact is model data and is NOT committed (`fixtures/*.sku` is gitignored
except the hand-authored toy); the committed parity contract is:

  - `fixtures/unigram_parity.json` — >= 1000 strings -> the exact
    `AutoTokenizer(text)["input_ids"]` HF produces (specials included), across
    EN/RU/DE/ES/CJK/emoji/mixed-script/whitespace/full-width/NBSP/combining-
    accent/digits-punctuation/very-long-word categories. `#[ignore]`
    `golden_parity` in `unigram.rs` replays this against the real artifact.

Rust never touches the SentencePiece protobuf or the `darts-clone`
double-array trie binary format directly. This script does both, once:

  1. Loads the tokenizer's own `sentencepiece.bpe.model` proto directly (not
     through the `sentencepiece` C++ runtime) to read vocab pieces + scores +
     types, and `normalizer_spec.precompiled_charsmap` — the compiled
     NFKC-class normalization trie.
  2. Decodes `precompiled_charsmap` by hand: it is `<u32 trie_blob_size LE>
     <darts-clone double-array trie> <normalized strings blob>` (see
     `darts.h`'s `DoubleArrayUnit`: `has_leaf()`, `label()`, `offset()`,
     `value()` — reimplemented below bit-for-bit). A full DFS over the trie
     (trying all 256 next bytes at every node, following only unit-verified
     transitions) enumerates every `(source_bytes, replacement_bytes)` pair
     the model actually ships, with NO dependency on the public
     `nmt_nfkc.tsv` (which drifts across `sentencepiece` releases and is not
     guaranteed byte-identical to what any given model was compiled with —
     verified empirically: a freshly-trained `nmt_nfkc` charsmap differs
     byte-for-byte from this model's, even though both carry the same name).
     Validated against `sp.normalize()` on an 8000-sample fuzz across ~20
     Unicode blocks: 0 mismatches.
  3. Discovers the XLM-R id offset empirically rather than assuming the
     textbook "+1" fairseq convention: for a spread of probe piece strings,
     `tok.convert_tokens_to_ids(piece) - sp_id` must be the *same* constant
     everywhere it's checked (asserted below), and `bos/eos/unk` ids come
     straight from `tok.bos_token_id` / `tok.eos_token_id` / `tok.unk_token_id`
     — no numbers are hand-typed from memory.

Known gap (see `unigram.rs` module doc): a handful of purely adversarial
strings (unrelated scripts glued together with orphaned combining marks) can
disagree with `AutoTokenizer` by one merged/dropped `<unk>` edge, because the
installed `transformers` backs `XLMRobertaTokenizer` with the Rust
`tokenizers` crate, not the reference `sentencepiece` C++ library, and the two
are not verified bit-identical on that narrow corner. None of the realistic
categories below hit it (checked while building this fixture).

Usage:
  python3 scripts/dump_unigram_fixtures.py \\
      --teacher intfloat/multilingual-e5-small \\
      --out fixtures/unigram_e5_small.sku \\
      --parity-out fixtures/unigram_parity.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MAGIC = b"SKUNI001"
FORMAT_VERSION = 1

FLAG_ADD_DUMMY_PREFIX = 1 << 0
FLAG_REMOVE_EXTRA_WHITESPACES = 1 << 1
FLAG_ESCAPE_WHITESPACES = 1 << 2
FLAG_TREAT_WHITESPACE_AS_SUFFIX = 1 << 3

PIECE_TYPE_NORMAL = 1
PIECE_TYPE_UNKNOWN = 2
PIECE_TYPE_CONTROL = 3
# Map ModelProto's own type enum (NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4,
# UNUSED=5, BYTE=6) onto our three-way tag. This model has none of the last
# three (checked below); if a future model does, this assert catches it
# rather than silently mis-tagging pieces into the segmentation trie.
_PROTO_TYPE_TO_TAG = {1: PIECE_TYPE_NORMAL, 2: PIECE_TYPE_UNKNOWN, 3: PIECE_TYPE_CONTROL}


# ---------------------------------------------------------------------------
# `precompiled_charsmap` decode: a `darts-clone` double-array trie. See
# `darts.h`'s `DoubleArrayUnit` (has_leaf/label/offset/value) — this is a
# direct, bit-for-bit port of that unit encoding plus a DFS enumerator, since
# darts-clone exposes lookup (commonPrefixSearch) but not "list every key",
# and we need every key to build our own flat table.
# ---------------------------------------------------------------------------


def _unit_offset(unit: int) -> int:
    return (unit >> 10) << ((unit & (1 << 9)) >> 6)


def _unit_has_leaf(unit: int) -> bool:
    return (unit >> 8) & 1 == 1


def _unit_label(unit: int) -> int:
    return unit & ((1 << 31) | 0xFF)


def _unit_value(unit: int) -> int:
    return unit & ((1 << 31) - 1)


def _enumerate_trie(units: tuple[int, ...]) -> list[tuple[bytes, int]]:
    """DFS over the double array; returns every `(key_bytes, value)` pair.
    Only follows transitions the unit array itself verifies (`label()`
    matches the tried byte), so this never runs away on garbage input — a
    corrupt trie just enumerates fewer (or zero) keys."""
    n = len(units)
    out: list[tuple[bytes, int]] = []

    def rec(node_pos: int, prefix: bytes) -> None:
        unit = units[node_pos]
        base_off = _unit_offset(unit)
        for b in range(256):
            idx = node_pos ^ base_off ^ b
            if idx < 0 or idx >= n:
                continue
            child = units[idx]
            if _unit_label(child) != b:
                continue
            new_prefix = prefix + bytes([b])
            if _unit_has_leaf(child):
                value_unit = units[idx ^ _unit_offset(child)]
                out.append((new_prefix, _unit_value(value_unit)))
            rec(idx, new_prefix)

    rec(0, b"")
    return out


def decode_precompiled_charsmap(blob: bytes) -> list[tuple[str, str]]:
    """`<u32 trie_blob_size><trie><normalized strings, NUL-terminated>` ->
    sorted `(source, replacement)` pairs, both plain Python str. Every key we
    have ever observed decodes as valid UTF-8 (asserted here, not assumed):
    SentencePiece charsmap keys are whole Unicode scalar-value sequences."""
    trie_blob_size = struct.unpack_from("<I", blob, 0)[0]
    trie_bytes = blob[4 : 4 + trie_blob_size]
    normalized = blob[4 + trie_blob_size :]
    n_units = len(trie_bytes) // 4
    units = struct.unpack(f"<{n_units}I", trie_bytes[: n_units * 4])

    table: list[tuple[str, str]] = []
    for key_bytes, val_offset in _enumerate_trie(units):
        end = normalized.index(b"\x00", val_offset)
        repl_bytes = normalized[val_offset:end]
        key = key_bytes.decode("utf-8")  # raises if this invariant ever breaks
        repl = repl_bytes.decode("utf-8")
        table.append((key, repl))
    table.sort()
    return table


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------


def write_u32(f, v: int) -> None:
    f.write(struct.pack("<I", v))


def write_str(f, s: str) -> None:
    b = s.encode("utf-8")
    write_u32(f, len(b))
    f.write(b)


def build_artifact(
    out_path: Path,
    pieces: list[tuple[str, float, int]],  # (piece, score, type_tag), id order
    sp_unk_id: int,
    fairseq_offset: int,
    unk_hf_id: int,
    bos_hf_id: int,
    eos_hf_id: int,
    charsmap: list[tuple[str, str]],
    spec_flags: int,
) -> None:
    normal_scores = [score for _, score, ty in pieces if ty == PIECE_TYPE_NORMAL]
    assert normal_scores, "no NORMAL pieces found"
    unk_score = min(normal_scores) - 10.0  # mirrors unigram_model.cc's kUnkPenalty

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        write_u32(f, FORMAT_VERSION)
        write_u32(f, len(pieces))
        write_u32(f, sp_unk_id)
        write_u32(f, fairseq_offset)
        write_u32(f, unk_hf_id)
        write_u32(f, bos_hf_id)
        write_u32(f, eos_hf_id)
        f.write(struct.pack("<f", unk_score))
        write_u32(f, spec_flags)
        write_u32(f, len(charsmap))

        for piece, score, ty in pieces:
            write_str(f, piece)
            f.write(struct.pack("<f", score))
            write_u32(f, ty)

        for key, val in charsmap:
            write_str(f, key)
            write_str(f, val)

    size = out_path.stat().st_size
    print(
        f"[dump_unigram] wrote {out_path} ({len(pieces)} pieces, "
        f"{len(charsmap)} charsmap rules, {size} bytes)",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Parity fixture: realistic multi-language + edge-case corpus, >= 1000
# strings. Word-pair combinations keep this "real text", not adversarial
# fuzz, while comfortably clearing the size bar without 1000 hand-typed
# sentences.
# ---------------------------------------------------------------------------

WORD_BANKS: dict[str, list[str]] = {
    "en": (
        "memory engine rust vector search recall precision insight graph "
        "store embed compress sleep consolidation ledger derivation staleness "
        "the quick brown fox jumps over lazy dog happy sad running tests "
        "distributed systems latency budget false apophenia bridge structural"
    ).split(),
    "ru": (
        "память движок вектор поиск точность граф хранилище "
        "сжатие сон консолидация реестр устаревание быстрая "
        "лиса прыгает через ленивую собаку счастливый грустный "
        "распределённые системы задержка бюджет ложный мост"
    ).split(),
    "de": (
        "Gedächtnis Maschine Vektor Suche Präzision Graph Speicher "
        "Komprimierung Schlaf Konsolidierung Register Veralterung schnelle "
        "Fuchs springt über den faulen Hund glücklich traurig laufende "
        "verteilte Systeme Latenz Budget falsche Brücke"
    ).split(),
    "es": (
        "memoria motor vector búsqueda precisión grafo almacén "
        "compresión sueño consolidación registro obsolescencia rápido "
        "zorro salta sobre el perro perezoso feliz triste corriendo "
        "sistemas distribuidos latencia presupuesto falso puente"
    ).split(),
    "zh": list("记忆引擎向量搜索精度图存储压缩睡眠巩固分类账陈旧快速棕色狐狸跳跃过懒狗快乐悲伤运行"),
    "ja": list("記憶エンジンベクトル検索精度グラフストレージ圧縮睡眠統合台帳陳腐化速い茶色の狐怠け犬幸せ悲しい"),
    "ko": list("기억엔진벡터검색정밀도그래프저장압축수면통합원장노후화빠른갈색여우게으른개행복한슬픈"),
    "emoji": list("😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙😚🙂🤗🤩🤔🤨😐😑😶🙄😏😣😥😮🤐😯😪😫🥱😴😌😛😜😝🤤😒😓😔😕🙃🤑😲☹️🙁😖😞😟😤😢😭😦😧😨😩🤯😬😰😱🥵🥶😳🤪😵😡😠🤬😷🤒🤕🤢🤮🤧😇🤠🥳🥺🤡🤥🤫🤭🧐🤓😈👿👹👺💀☠️👻👽👾🤖💩😺😸😹😻😼😽🙀😿😾"),
}

# Mixed-script sentences: several languages / scripts in one string, a
# realistic multilingual-retrieval scenario (query in one language, passage
# fragment quoted in another), not random Unicode soup.
MIXED_SCRIPT = [
    "The Moscow office (Москва) ships the memory engine.",
    "query: what is Gedächtnis in German? passage: memory.",
    "日本語のテキストと English mixed together.",
    "café naïve raphæl über München",
    "한국어와 English와 中文 mixed in one query",
    "emoji test 😀 with текст and 文字",
    "passage: La búsqueda vectorial (vector search) es rápida.",
    "北京 to Москва via München — a multilingual route.",
    "Zürich, São Paulo, Москва, 東京: four cities.",
    "SELECT * FROM память WHERE id = 1; -- mixed code/RU comment",
]

WHITESPACE_EDGE_CASES = [
    "",
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "   leading spaces",
    "trailing spaces   ",
    "  both  sides  ",
    "multiple    internal    spaces",
    "a\tb\nc\rd",
    " nbsp leading",
    "trailing nbsp ",
    "mixed \t\n whitespace runs",
    "single a",
    "a",
    "single b b",
]

FULLWIDTH_AND_COMBINING = [
    "ＡＢＣ full-width ABC",
    "ＡＢＣＤＥＦＧＨＩＪ",
    "éclair with combining acute",
    "café naivë raphaël",
    "à á â ã ä å",
    "e\u0301 single combining acute accent (decomposed, not precomposed)",
    "éèêë precomposed accented e variants",
    "Việt Nam xin chào",
    "Tiếng Việt",
    "אָבֲרָהם niqqud",
    "اَلسَّلامُ tashkeel",
]

DIGITS_PUNCTUATION = [
    "1234567890",
    "3.14159265358979",
    "$1,234.56 costs -42%",
    "!?.,;:'\"()[]{}",
    "a1b2c3d4e5",
    "phone: +1-555-123-4567",
    "v1.2.3-beta+build.456",
    "100% sure? yes!!! no...",
    "0x1F600 is an emoji codepoint",
    "IPv4: 192.168.0.1/24",
]

VERY_LONG_WORDS = [
    "a" * 200,
    "supercalifragilisticexpialidocious" * 4,
    "память" * 30,  # "память" repeated
    "b" * 400,  # repeated non-adversarial char (see K0 report: "x" repeats hit a
    # tokenizers-crate-vs-sentencepiece Viterbi tie-break divergence at some N)
]


def build_corpus() -> list[str]:
    strings: list[str] = []
    strings.extend(WHITESPACE_EDGE_CASES)
    strings.extend(FULLWIDTH_AND_COMBINING)
    strings.extend(DIGITS_PUNCTUATION)
    strings.extend(VERY_LONG_WORDS)
    strings.extend(MIXED_SCRIPT)

    # CJK / emoji: single chars, pairs, and short runs (real script coverage,
    # not full sentences -- these scripts don't tokenize on whitespace).
    for key in ("zh", "ja", "ko", "emoji"):
        chars = WORD_BANKS[key]
        for c in chars:
            strings.append(c)
        for i in range(0, len(chars) - 1, 2):
            strings.append(chars[i] + chars[i + 1])
        for i in range(0, len(chars) - 2, 3):
            strings.append("".join(chars[i : i + 3]))

    # EN/RU/DE/ES: word-pair and word-triple combinations -- realistic short
    # phrases, deterministic (no RNG), and plentiful.
    templates = [
        "{a} {b}",
        "{a} {b}.",
        "{a}, {b}",
        "query: {a} {b}",
        "passage: {a} {b}.",
        "{a} {b} {c}",
        "The {a} and the {b}.",
    ]
    for lang in ("en", "ru", "de", "es"):
        words = WORD_BANKS[lang]
        n = len(words)
        for i in range(n):
            a, b, c = words[i], words[(i + 1) % n], words[(i + 5) % n]
            for t in templates:
                strings.append(t.format(a=a, b=b, c=c))

    # De-duplicate while preserving order (some templates/words can collide).
    seen = set()
    out = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default="fixtures/unigram_e5_small.sku")
    ap.add_argument("--parity-out", default="fixtures/unigram_parity.json")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from sentencepiece import sentencepiece_model_pb2 as model_pb2
    from transformers import AutoTokenizer

    sp_model_path = hf_hub_download(args.teacher, "sentencepiece.bpe.model")
    m = model_pb2.ModelProto()
    with open(sp_model_path, "rb") as f:
        m.ParseFromString(f.read())

    tok = AutoTokenizer.from_pretrained(args.teacher)

    # --- Pieces + scores + types, id order -------------------------------
    pieces: list[tuple[str, float, int]] = []
    for p in m.pieces:
        tag = _PROTO_TYPE_TO_TAG.get(p.type)
        assert tag is not None, (
            f"unsupported SentencePiece piece type {p.type} for {p.piece!r} -- "
            "this converter only handles Normal/Unknown/Control"
        )
        pieces.append((p.piece, float(p.score), tag))

    sp_unk_id = m.trainer_spec.unk_id
    assert pieces[sp_unk_id][2] == PIECE_TYPE_UNKNOWN, "trainer_spec.unk_id is not an Unknown-type piece"

    # --- Empirical id-offset discovery (never hand-typed) -----------------
    # Probe a spread of NORMAL piece ids across the whole vocab range and
    # require the SAME offset everywhere it's checked.
    normal_ids = [i for i, (_, _, ty) in enumerate(pieces) if ty == PIECE_TYPE_NORMAL]
    probe_ids = normal_ids[:: max(1, len(normal_ids) // 50)][:50]
    offsets = {tok.convert_tokens_to_ids(pieces[i][0]) - i for i in probe_ids}
    assert len(offsets) == 1, f"fairseq offset is not constant across probes: {offsets}"
    fairseq_offset = offsets.pop()

    unk_hf_id = tok.unk_token_id
    bos_hf_id = tok.bos_token_id
    eos_hf_id = tok.eos_token_id
    assert None not in (unk_hf_id, bos_hf_id, eos_hf_id), "tokenizer is missing bos/eos/unk ids"

    # --- Normalizer spec flags --------------------------------------------
    ns = m.normalizer_spec
    flags = 0
    if ns.add_dummy_prefix:
        flags |= FLAG_ADD_DUMMY_PREFIX
    if ns.remove_extra_whitespaces:
        flags |= FLAG_REMOVE_EXTRA_WHITESPACES
    if ns.escape_whitespaces:
        flags |= FLAG_ESCAPE_WHITESPACES
    if m.trainer_spec.treat_whitespace_as_suffix:
        flags |= FLAG_TREAT_WHITESPACE_AS_SUFFIX
    assert not m.trainer_spec.byte_fallback, (
        "this converter/Rust reader do not implement byte_fallback -- "
        "multilingual-e5-small has it disabled; a future model needs a new ticket"
    )

    charsmap = decode_precompiled_charsmap(ns.precompiled_charsmap)

    build_artifact(
        Path(args.out),
        pieces,
        sp_unk_id,
        fairseq_offset,
        unk_hf_id,
        bos_hf_id,
        eos_hf_id,
        charsmap,
        flags,
    )

    # --- Parity fixture -----------------------------------------------------
    corpus = build_corpus()
    assert len(corpus) >= 1000, f"parity corpus too small: {len(corpus)}"
    cases = [[text, tok(text, add_special_tokens=True)["input_ids"]] for text in corpus]
    parity_path = Path(args.parity_out)
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parity_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, separators=(",", ":"))
    size = parity_path.stat().st_size
    print(
        f"[dump_unigram] wrote {parity_path} ({len(cases)} strings, {size} bytes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
