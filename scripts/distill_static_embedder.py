#!/usr/bin/env python3
"""Stage 1B — T1: distill a static WordPiece embedder into the SKEMB001
artifact + dump the cross-impl parity golden fixture.

The Model2Vec recipe (Min et al. 2023, "Model2Vec: Distill a Small Fast Model
from a Sentence Transformer") applied to a WordPiece teacher:

  1. Take the teacher's input token embedding matrix E (vocab × D).
  2. (Optional) PCA E to `dim` (we distill to 256 per the §2 artifact-size
     budget; bge-small is 384, so PCA-256 keeps ~67% dims).
  3. L2-normalize each row of the distilled table.
  4. Per-token weights: uniform 1.0 for content pieces (the Model2Vec
     "no-weight" baseline — see the D1 note at the weights block below for why
     the rank-based Zipf/SIF first cut was wrong for BERT vocab order);
     [UNK] / special tokens weight 0 so OOV contributes nothing to the pooled
     mean — matches the Rust reader's zero-vector contract.
  5. Write the SKEMB001 artifact: magic | version | dim | vocab | flags |
     len-prefixed vocab strings | table | weights.
  6. Dump `golden_embeddings.f32`: 32 fixed strings → the embedding bytes the
     Rust reader must reproduce bit-for-bit. The reference here is a faithful
     re-implementation of the Rust pooling/WordPiece (NOT HF's tokenizer — the
     Rust embedder reimplements WordPiece and lowercase-only normalization, so
     this script mirrors that exact algorithm to be the parity oracle).

This script is **dev tooling**: it runs offline, once, outside any gate (rule-3
shape: the artifact is the replay). The ~30 MB artifact is model weights and is
NOT committed (see .gitignore); the committed parity contract is the golden
fixture + the `#[ignore]` `golden_parity` test in skinki-vector, run manually
after regeneration. 0 network in CI. Adding `sentence-transformers`/`torch` to
a gate would violate the minimal-deps law; no runtime Rust dependency.

Usage:
  python3 scripts/distill_static_embedder.py \
      --teacher BAAI/bge-small-en-v1.5 --dim 256 \
      --out fixtures/static_embed_bge_small_256.skemb \
      --golden-out fixtures/golden_embeddings.f32
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MAGIC = b"SKEMB001"
FORMAT_VERSION = 1
FLAG_WORDPIECE = 1
UNK = "[UNK]"
CONT = "##"
# Zipf/SIF offset: w(rank) = 1 / (rank + ZIPF_OFFSET). a=2.0 matches the
# Model2Vec default weighting family (a small constant keeps high-frequency
# tokens from dominating the mean); the exact value is replayed from this file.
ZIPF_OFFSET = 2.0


def nfc(text: str) -> str:
    """NFC normalize — matches the Rust reader's expectation (the artifact is
    the contract; the script applies NFC so the dumped vocab is canonical)."""
    return unicodedata.normalize("NFC", text)


def pretokenize(text: str) -> list[str]:
    """Mirror of `static_embed::pretokenize` (Rust): lowercase, alphanumeric
    runs extended in place, each punctuation char its own token, whitespace
    dropped. We split on the same `is_alphanumeric` semantics Rust uses
    (Unicode-aware)."""
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
    """Mirror of `StaticEmbedder::wordpiece` (Rust): greedy longest-match with
    `##` continuation prefix and `[UNK]` fallback on a no-match start."""
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


def encode(text: str, piece_to_id: dict[str, int], unk_id: int) -> list[int]:
    ids: list[int] = []
    for word in pretokenize(text):
        ids.extend(wordpiece(word, piece_to_id, unk_id))
    return ids


def pool(
    text: str,
    table: np.ndarray,
    weights: np.ndarray,
    piece_to_id: dict[str, int],
    unk_id: int,
) -> np.ndarray:
    """Reference pooling — a faithful Python port of `StaticEmbedder::embed`
    (Rust). Critical details for bit-exact parity:

      * summation order: token order from `encode`, left-to-right
      * f32 accumulation: scalar Python float ops (NOT numpy array ops) so no
        SIMD/FMA contraction — Rust's naive `*x += w * v` loop is reproduced
        instruction-for-instruction; numpy vectorized `w * row` can emit a
        fused multiply-add that differs in the last ULP (invariant §4).
      * divide by sum-of-weights, then L2-normalize
      * empty / all-[UNK] (weight sum ~ 0) -> zero vector
    """
    ids = encode(text, piece_to_id, unk_id)
    dim = table.shape[1]
    acc = [0.0] * dim  # pure-Python f32 accumulator (cast per op below)
    wsum = 0.0
    for tid in ids:
        w = float(np.float32(weights[tid]))
        row = table[tid]
        for i in range(dim):
            # Rust: `*x += w * v` — one f32 multiply, one f32 add, in that
            # order (no FMA). Python floats are f64; we truncate to f32 after
            # each op to mirror Rust's f32 arithmetic exactly.
            v = float(np.float32(row[i]))
            acc[i] = float(np.float32(acc[i] + np.float32(w * v)))
        wsum = float(np.float32(wsum + w))
    if wsum > 1e-12:
        for i in range(dim):
            acc[i] = float(np.float32(acc[i] / np.float32(wsum)))
    # L2-normalize. Rust computes the norm via f32 sum-of-squares then sqrt.
    sq = 0.0
    for i in range(dim):
        sq = float(np.float32(sq + np.float32(acc[i] * acc[i])))
    n = float(np.float32(np.float32(sq) ** np.float32(0.5)))
    if n > 1e-12:
        for i in range(dim):
            acc[i] = float(np.float32(acc[i] / np.float32(n)))
    return np.array(acc, dtype=np.float32)


def l2_normalize_rows(table: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (table / norms).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--out", required=True, help="output SKEMB001 artifact path")
    ap.add_argument(
        "--golden-out",
        required=True,
        help="output golden embeddings (raw f32 LE) path",
    )
    ap.add_argument(
        "--max-vocab",
        type=int,
        default=0,
        help="if >0, prune the vocab to the first N tokens (debug)",
    )
    args = ap.parse_args()

    print(f"[distill] teacher = {args.teacher}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModel.from_pretrained(args.teacher)
    emb = model.embeddings.word_embeddings.weight.detach().to(torch.float32).cpu()
    vocab_size_full, dim_full = emb.shape
    print(
        f"[distill] teacher word embeddings: {vocab_size_full} × {dim_full}",
        file=sys.stderr,
    )

    # Vocabulary from HF, in id order. Keep the [UNK] id from the teacher.
    inv = {v: k for k, v in tok.get_vocab().items()}
    vocab = [inv[i] for i in range(vocab_size_full)]
    unk_id = tok.unk_token_id
    assert vocab[unk_id] == UNK, f"vocab[{unk_id}] = {vocab[unk_id]!r}, expected {UNK}"

    if args.max_vocab and args.max_vocab < vocab_size_full:
        vocab_size = args.max_vocab
        vocab = vocab[:vocab_size]
        emb = emb[:vocab_size]
        print(f"[distill] pruned to first {vocab_size} tokens", file=sys.stderr)
    else:
        vocab_size = vocab_size_full

    # PCA to `args.dim` (Model2Vec recipe). Sklearn is a sentence-transformers
    # transitive dep; we do PCA in numpy to avoid adding a sklearn runtime
    # requirement: center, SVD, project, truncate.
    target = args.dim
    print(f"[distill] PCA {dim_full} -> {target}", file=sys.stderr)
    X = emb.numpy().astype(np.float64)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # SVD: Xc = U S Vt; principal components are columns of V (rows of Vt).
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:target]  # (target, dim_full)
    Y = Xc @ comps.T  # (vocab, target)
    # Sign convention: fix each PC's sign by the larger-magnitude element so the
    # table is deterministic across SVD implementations (LAPACK sign ambiguity).
    for j in range(target):
        col = Y[:, j]
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0:
            Y[:, j] = -col
    table = Y.astype(np.float32)
    table = l2_normalize_rows(table)
    print(
        f"[distill] table {table.shape}, "
        f"|row| range [{np.linalg.norm(table,axis=1).min():.4f}, "
        f"{np.linalg.norm(table, axis=1).max():.4f}]",
        file=sys.stderr,
    )

    # SIF/Zipf down-weighting. The T1 first cut used `w = 1/(vocab_id + a)` as
    # a "serviceable proxy for frequency rank" — but BERT vocab is NOT strictly
    # frequency-ordered beyond the first ~1k tokens: punctuation (`!` `"` `#`...)
    # sit near the start and got the HIGHEST weights, while rare content words
    # (`restaurant`, `birthday`) got the LOWEST. This inverted SIF and
    # collapsed retrieval (D1 measured static recall@10 = 0.004 vs hash 0.019).
    #
    # Fix (D1): uniform nonzero weighting (w=1 for every non-special token) is
    # the Model2Vec "no-weight" baseline; it recovers recall@10 to 0.090 —
    # still below BM25 (0.134) and the §2 bar (0.22); the D1 verdict in
    # specs/STAGE_1B_STATIC_EMBEDDER.md records the hypothesis as falsified.
    # Frequency-derived SIF is a T7 follow-up (needs a corpus frequency dump);
    # the §4 invariant only requires the weighting be bit-reproducible from
    # this script, which uniform trivially is.
    weights = np.ones(vocab_size, dtype=np.float32)
    for i, piece in enumerate(vocab):
        if piece == UNK:
            weights[i] = 0.0  # OOV contributes nothing — Rust zero-vector contract.
        elif piece.startswith("[") and piece.endswith("]"):
            weights[i] = 0.0  # special tokens: [CLS]/[SEP]/[PAD]/`/`/[unused*]
        # else: uniform weight 1.0 (mean pooling)
    print(
        f"[distill] weights: nonzero={int((weights>0).sum())}, "
        f"max={weights.max():.4f}, min(nonzero)={weights[weights>0].min():.6f}",
        file=sys.stderr,
    )

    # --- Write SKEMB001 artifact ---------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    piece_to_id = {p: i for i, p in enumerate(vocab)}
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", FORMAT_VERSION, target, vocab_size, FLAG_WORDPIECE))
        for piece in vocab:
            b = piece.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        # table: vocab × dim f32 LE, row-major.
        f.write(table.astype("<f4").tobytes())
        # weights: vocab f32 LE.
        f.write(weights.astype("<f4").tobytes())
    size = out_path.stat().st_size
    budget = 48 * 1024 * 1024
    print(
        f"[distill] wrote {out_path} ({size} bytes; budget {budget}; "
        f"{'OK' if size <= budget else 'OVER BUDGET'})",
        file=sys.stderr,
    )

    # --- Dump golden_embeddings.f32 ------------------------------------------
    # 32 fixed strings (English + Russian + mixed + edge cases) → the parity
    # reference. The #[ignore] `golden_parity` test in skinki-vector reads
    # this file and asserts embed(s) == reference bytes exactly; run it
    # manually whenever the artifact is regenerated:
    #   cargo test -p skinki-vector golden_parity -- --ignored
    # Format: u32 count, u32 dim, then `count` × (u32 len, str bytes) header
    # for each string, then count × dim f32 LE embedding rows. The header
    # precedes the rows so the Rust test can re-iterate strings in order.
    goldens = [
        "memory engine",
        "rust vector search",
        "the happy insight",
        "distributed systems latency budgets",
        "recall precision false insight apophenia",
        "sleep scheduler power idle thermal",
        "embed compress rabitq ivf mmap",
        "graph provenance derivation ledger staleness",
        "con",
        "ПАМЯТЬ",  # Cyrillic, lowercase-folded to "память"
        "memorys engine",  # WordPiece split: memory + ##s
        "running tests",
        "",
        "   ",
        "memory, engine.",
        "the quick brown fox jumps over the lazy dog",
        "cafe",  # NFC normalisation; accents are NOT stripped (Rust lowercases only)
        "naïve",
        "12345 numbers",
        "MIXED Case WORDS",
        "a the and of to in is it",
        "structural bridge detector benjamini hochberg",
        "raphæl",  # ligature -> NFC decomposition; kept as-is by Rust lowercase
        "Москва",  # Cyrillic, the roadmap STT target language
        "кофе по утрам",
        " tab\tspace",
        "punctuation!and?marks",
        "model2vec distillation static embedder table lookup pooling",
        "ffff zzzz qqqq",  # likely all-[UNK]
        "edge case: a single character word a",
        "tokenizer wordpiece greedy longest match",
        "ABCdefGHI",
    ]
    assert len(goldens) == 32, len(goldens)
    golden_path = Path(args.golden_out)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "wb") as f:
        f.write(struct.pack("<II", len(goldens), target))
        for s in goldens:
            b = s.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        for s in goldens:
            v = pool(s, table, weights, piece_to_id, unk_id)
            f.write(v.astype("<f4").tobytes())
    print(
        f"[distill] wrote {golden_path} ({len(goldens)} strings × {target} dims)",
        file=sys.stderr,
    )

    # --- Self-check: report norms for sanity --------------------------------
    for s in ["memory engine", "", "ffff zzzz qqqq"]:
        v = pool(s, table, weights, piece_to_id, unk_id)
        n = float(np.sqrt(np.dot(v, v)))
        print(f"[distill]   ||embed({s!r})|| = {n:.6f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
