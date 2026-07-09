# Stage 1E - Full-D1 encoder bakeoff and base-class port decision (SPEC)

- **Status:** proposed next Stage-1 substage after 1D.
- **Owner of the design (frontier/human):** frontier. This stage is a
  measurement and architecture decision, not a bulk implementation stage.
- **Delegatable to (cheaper model):** partially. Export plumbing, metadata
  logs, and converter extensions are delegatable after the candidate order and
  kill criteria are frozen. The model/serving decision stays frontier-owned.

> Read [`../AGENTS.md`](../AGENTS.md). The gate is law. Do not weaken the
> inherited `rrf(bm25+encoder) recall@10 >= 0.30` full-row bar. Do not add
> runtime dependencies. Do not add `unsafe`. Dev-tooling may run external model
> stacks offline to produce replay artifacts, but every asserted measurement
> rebuilds from logged artifacts and never runs inference inside the gate.

## 0. Verdict on Stage 1D / e5-small

`multilingual-e5-small` failed to transfer because the 41q/201k trend row was
not a reliable proxy for the full 121q/594k haystack. The trend row had a much
stronger lexical baseline (`bm25` 0.341), fewer distractors, and enough local
semantic signal for a 384-dim mean-pooled small tower to look healthy. On the
full D1 row, nearest-neighbor margins collapsed: `semantic-real(e5-small)`
landed at 0.152 and `rrf(bm25+e5)` at 0.160 against the `>= 0.30` bar.
Coarse-to-fine also collapsed (0.017 recall), which confirms the failure is not
a missing acceleration ticket; the small tower does not separate the relevant
turns in a 594k-entry pool.

The lesson for 1E: **quality must be killed or kept on the full D1 row before
any port, int8, or serving work.** Trend rows are useful only for smoke and
debugging.

## 1. Recommended substage

**Stage 1E - Full-D1 encoder bakeoff and base-class port decision.**

Hypothesis: the remaining Stage-1 retrieval gap is mostly model capacity /
training, not the Rust forward path. A stronger encoder can clear
`rrf(bm25+encoder) recall@10 >= 0.30` on the full LongMemEval D1 row, and if it
does, only then is it worth paying the port/latency/compression cost needed to
make it self-contained in the Rust engine.

This stage is intentionally quality-first:

1. Export embeddings for a short ranked list of candidates using offline
   dev-tooling into replayable f32 artifacts.
2. Score the **full D1 row** with the existing `longmemeval-eval --pooled`
   replay path.
3. Port only the first candidate that clears the full-row gate and passes
   license / architecture review.

No candidate becomes a served default from a trend-row result, and no
candidate gets SDOT/int8 work before its f32 replay clears the quality gate.

## 2. Ranked strategies

| Rank | Strategy | Expected value | Why |
| --- | --- | --- | --- |
| 1 | **Full-D1 bakeoff of larger/base-class encoders, then SKENC001 port of the first winner** | Highest | It tests the biggest remaining uncertainty with the lowest new-engine surface. It preserves the self-contained Rust target if the winner is BERT/XLM-R-shaped, reuses `SKENC001`, Unigram/WordPiece, prefix handling, `encoder-embed`, and the `rrf(bm25+real)` column. |
| 2 | **EmbeddingGemma-class bridge / port / license decision** | High quality upside, higher product risk | EmbeddingGemma-class models are the closest known quality reference, but they are not a small `SKENC001` extension: architecture, license, artifact redistribution, and query latency all need a separate human decision. Measure as an upper-bound replay, but do not port first unless base-class candidates miss. |
| 3 | **Late interaction / MaxSim / multi-vector retrieval** | Medium-high if dense retrieval lands near the bar | This attacks mean-pooling loss directly, especially multi-hop evidence scattered across turns. It also multiplies storage and index complexity, so it is best as Stage 1F if a strong encoder reaches roughly 0.24-0.29 but cannot cross 0.30 with single-vector RRF. |
| 4 | **Learned query decomposition / reranking** | Medium, but not first | It may be necessary for the last multi-hop gap, but it introduces LLM/logging complexity, query-time cost, and a broader eval surface. The 1D doc2query spike already gave a negative early signal with a 0.5B generator. |
| 5 | **BM25/static/RRF parameter tuning or e5-small acceleration** | Low | 1B static and 1D e5-small are closed negative. Tweaking fusion depth or making the same weak embeddings faster will not close a 0.140 absolute recall gap. |

### Candidate order for Strategy 1

Run quality replay in this order, with a model-card/license snapshot recorded in
the artifact log:

1. [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)
   - first candidate. Same E5 prefix discipline as 1D, XLM-R-like shape already
   close to `SKENC001` v2, MIT license, 12 layers with 768-dim output. It is the
   cleanest test of "small failed because it was too small."
2. [`Snowflake/snowflake-arctic-embed-m-v1.5`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)
   - Apache-2.0, 109M BERT-class retrieval model, 768-dim output, explicitly
   trained for MRL / compressed embeddings. English-first, so it cannot alone
   settle the multilingual product question, but it is a strong quality and
   compression candidate.
3. [`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
   - Apache-2.0 and multilingual with strong retrieval claims, but higher port
   risk because the model card requires `trust_remote_code` in the standard path
   and exposes long-context / sparse behavior outside current `SKENC001`.
4. [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) - MIT,
   BERT-base-shaped, simple English baseline. Use it to separate "E5 family
   capacity" from "multilingual XLM-R vocabulary" effects if the first candidate
   is ambiguous.
5. [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m)
   or successor - quality reference / escape hatch. The current model-card
   metadata says `License: gemma`, and Hugging Face access requires accepting
   Google's usage conditions. It needs a license and architecture decision before
   porting; a replay result can still decide whether it is worth that discussion.

## 3. Budgets / fitness function

| Metric | Budget | How measured |
| --- | --- | --- |
| Full D1 `rrf(bm25+encoder)` recall@10 | **>= 0.30** hard gate | `longmemeval-eval --pooled --question-type multi-session` with candidate entry/query f32 files |
| Full D1 encoder solo recall@10 | reported, no hard bar | same run, `semantic-real` column |
| Full D1 answer@10 / ndcg@10 | reported; answer regression vs BM25 is a serving blocker if severe | same run |
| Replay determinism | byte-identical score inputs from frozen text dump + embedding files | artifact log SHA-256s, row counts, dims |
| Port parity, if candidate clears quality | min cosine >= 0.999 vs teacher goldens; thread-count byte equality | `skinki-encoder` ignored parity tests + CI toy tests |
| Query embed latency, if candidate clears quality | p95 <= 50 ms target; <= 150 ms hard retrieval-budget cap before bridge/int8 | local telemetry over D1 queries |
| Backfill throughput, if candidate clears quality | 5M <= 10 days sleep-time or explicit bridge/offline-only decision | `encoder-embed` throughput telemetry |
| Deps / unsafe / network | no new runtime deps, no new `unsafe`, no network in gates | review + CI |

The hard Stage-1 success condition is the first row. A candidate that misses
`0.30` is not ported, optimized, quantized, or served.

## 4. Artifact requirements

Every candidate replay must produce an append-only artifact record. Recommended
path shape:

```
artifacts/stage1e/<model_slug>/
  stage1e.embeddings.jsonl
  entries.f32
  queries.f32
  scores.txt
```

The JSONL record must include:

- candidate id, upstream model id, upstream revision/commit, license string,
  and model-card URL;
- text dump path plus SHA-256 of `entries.json` and `queries.json`;
- exact export command, tool revision, pooling mode, query/passage prefixes,
  normalization flag, truncation/max length, output dim, row counts, file sizes,
  and SHA-256 of both f32 files;
- model artifact hashes used by the exporter (`safetensors`, tokenizer model,
  vocab, config, and any custom code snapshot);
- machine/threads as telemetry only;
- replay command and raw score table.

These artifacts are not necessarily committed because the dataset and model
weights are large/private. The spec records the numbers; the log is the local
source of truth. Gates consume the replayed f32 files and metadata, not live
model inference.

## 5. Tickets

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| **D0 candidate freeze + license snapshot** | design | frontier/human | Candidate order above confirmed; each model has a recorded license, architecture notes, expected tokenizer/pooling/prefix contract, and port-risk note. Any model with unclear redistribution rights can be measured locally but cannot be shipped or committed as weights. |
| **T1 canonical full-D1 dump** | impl | cheaper | One canonical dump generated from the same inputs as 1D D2: full LongMemEval `multi-session`, 594,708 entries / 121 queries. SHA-256s of `entries.json` and `queries.json` recorded. |
| **T2 replay exporter wrapper** | impl | cheaper | A dev-only script or runbook exports entry/query embeddings for a candidate into flat f32 LE files plus `stage1e.embeddings.jsonl`. No Rust runtime dependency is added. Prefix and pooling are model-specific and recorded. |
| **T3 full-row replay score** | impl | cheaper | Existing harness scores `semantic-real`, `coarse2fine(3)`, and `rrf(bm25+real)`. `scores.txt` records the raw table. If `rrf < 0.30`, the candidate stops here. |
| **D1 quality verdict** | design | frontier/human | First candidate with `rrf >= 0.30` is selected for port/serving work. If all compatible base-class candidates miss, Stage 1E closes negative and the next substage is EmbeddingGemma-class or late-interaction, depending on the best replay row. |
| **T4 SKENC001 port of the winner** | impl | cheaper with frontier review | Only after D1 passes. Extend converter/format as needed for the winner, add toy fixture/goldens, and keep `#![forbid(unsafe_code)]`. No new crates. |
| **T5 in-engine replay parity** | impl | frontier-reviewed | `encoder-embed` with the Rust artifact reproduces the teacher replay closely enough that `rrf` remains `>= 0.30`; parity goldens min cosine `>= 0.999`; outputs byte-identical across `--threads 1/4/8`. |
| **D2 serving decision** | design | frontier/human | If quality passes but latency/backfill fails, choose one of: query bridge, sleep-only/offline use, or defer served default. Do not approve SDOT/int8 until this point. |

## 6. Commands

Use the exact dataset path from the 1D D2 run. The placeholders are deliberate:
the private dataset and artifact dirs are not committed.

```bash
# Repo hygiene
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check

# Stage 1 compression/index CI gate remains unchanged
cargo run --release -p skinki-harness -- compress-bench \
    --source synthetic --dim 256 --vectors 4000 --queries 100 --assert-gate

# T1: canonical full-D1 text dump
cargo run --release -p skinki-harness -- longmemeval-eval \
    --path "$LME_M" \
    --pooled \
    --question-type multi-session \
    --dump-texts "$STAGE1E_DUMP"

# T2: candidate export, dev-tooling only.
# The concrete script may be model-specific; it must write entries.f32,
# queries.f32, and append stage1e.embeddings.jsonl.
python3 tools/export-stage1e-embeddings.py \
    --model intfloat/multilingual-e5-base \
    --texts "$STAGE1E_DUMP" \
    --out "$STAGE1E_OUT/e5-base"

# T3: replay-only full-row score
cargo run --release -p skinki-harness -- longmemeval-eval \
    --path "$LME_M" \
    --pooled \
    --question-type multi-session \
    --embeddings-file "$STAGE1E_OUT/e5-base/entries.f32" \
    --query-embeddings-file "$STAGE1E_OUT/e5-base/queries.f32" \
    | tee "$STAGE1E_OUT/e5-base/scores.txt"
```

After a candidate clears D1 and is ported:

```bash
cargo run --release -p skinki-harness -- encoder-embed \
    --artifact fixtures/encoder_<winner>.skenc \
    --texts "$STAGE1E_DUMP" \
    --out "$STAGE1E_OUT/<winner>-rust" \
    --threads 8

cargo run --release -p skinki-harness -- longmemeval-eval \
    --path "$LME_M" \
    --pooled \
    --question-type multi-session \
    --embeddings-file "$STAGE1E_OUT/<winner>-rust/entries.f32" \
    --query-embeddings-file "$STAGE1E_OUT/<winner>-rust/queries.f32"
```

## 7. Kill criteria

- Kill a candidate immediately if its full-row `rrf(bm25+encoder)` recall@10
  is below 0.30. Record the number; do not port it.
- Kill the base-class path if at least two compatible base-class families miss
  by a wide margin (`rrf < 0.24`) and no candidate shows a monotonic path to
  0.30. Move to EmbeddingGemma-class or late interaction.
- If a candidate lands in the near-miss band (`0.24 <= rrf < 0.30`) with strong
  solo recall and good answer@10, do **not** port it as the default; write the
  next spec as late-interaction / multi-vector retrieval using that candidate as
  the tower.
- If a candidate clears quality but fails license review, it can remain a local
  reference/teacher only. It cannot become a shipped default or committed model
  artifact.
- If a candidate clears quality but the Rust port cannot meet parity
  (`min cosine < 0.999`) or thread-count determinism, the port is killed even if
  the Python replay looked good.
- If quality clears but latency/backfill misses, do not lower the quality bar.
  Decide between query bridge, offline/sleep-only embedding, or a new
  human-approved acceleration ticket.

## 8. What not to do next

- **Do not spend SDOT/int8 on e5-small.** e5-small is short by 0.140 absolute
  recall on the full-row `rrf` gate and regresses answer@10 vs BM25. SDOT would
  make a weak retriever faster and would introduce a new `unsafe` quarantine
  before the quality problem is solved.
- Do not tune e5-small prefixes, pooling, fusion depth, coarse-to-fine, or
  static artifacts. 1B and 1D already closed those families for the served
  default.
- Do not make another trend-row served decision. The 41q row predicted the wrong
  outcome for e5-small; Stage 1E decisions require full D1 replay.
- Do not add ONNX Runtime, tokenizers, candle, safetensors, llama.cpp, or a
  sidecar as a Rust runtime dependency. They are allowed only as offline
  dev-tooling that produces replay artifacts.
- Do not start late interaction, learned reranking, or query decomposition
  until the base encoder quality ladder says whether a single-vector stronger
  tower is enough.
- Do not lower the `>= 0.30` bar because the local-first constraint is hard.
  The honest result can be "self-contained base-class encoders miss"; the gate
  still stays true.

## 9. Definition of done

- [ ] Full D1 canonical text dump hashes recorded.
- [ ] At least the first two ranked candidates replay-scored on the full D1 row,
      or an earlier candidate clears `rrf >= 0.30`.
- [ ] D1 verdict recorded in this spec: selected winner, near-miss, or negative
      close-out.
- [ ] If there is a winner, `SKENC001` port and in-engine replay preserve the
      full-row gate and determinism.
- [ ] Served-default decision recorded without weakening AGENTS.md constraints.
