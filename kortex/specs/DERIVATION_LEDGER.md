# Derivation Ledger — staleness-aware memory via hash-linked reasoning (design note)

- **Status:** **v0 built + gated** (`crates/kortex-ledger` + `kortex
  ledger-bench --assert-gate`). The algorithm core (content-addressed DAG +
  deterministic `stale_closure` + the §6 metric), JSON persistence, property/
  golden tests, and a corpus-wired benchmark over the planted contradictions are
  in and green in CI. On the V2 corpus the ledger reaches **invalidation-recall
  1.000 at 0 over-invalidation** versus a provenance-free baseline's **0.000** —
  the silent-staleness gap made into a number. Durable append-only persistence
  (on `kortex-store`) and Stage-3 integration are the remaining steps; full
  adoption into the graph/insight path still wants a human go.
- **Where it touches the stack:** L0 provenance (Stage 2, exists), L2b graph and
  two-tier extraction (Stage 3), the Insight Engine (Stage 5). It is the
  connective tissue that makes "cite or be silent" survive *time*.
- **Origin:** a Reddit thread on agent memory + a user observation. The kernel
  idea, paraphrased: *don't store raw facts, store the **reasoning chain** behind
  them; when an input changes, the chain breaks and you know to re-evaluate.*
  The mental model that crystallized it: this is the **tamper-evidence property
  of a blockchain** — alter one link and the mismatch is immediately visible.

## 1. The problem this targets: silent staleness, and the missing "why"

Every memory system stores facts. Almost none track **when a fact stopped being
true** or **why it was believed in the first place.** Two failure modes follow:

- **Silent staleness.** A premise shifts — a library's behavior changes in a
  minor version, a config is updated, an assumption is revised — and every
  conclusion derived from it is now quietly wrong. Pure fact storage has no
  signal that anything broke.
- **The lost "why".** For a coding/agentic memory the expensive thing to lose is
  not *what* was decided but *why*. The "why" is what stops the agent from
  repeating a mistake or undoing a deliberate decision.

Our corpus already plants exactly this phenomenon as ground truth:
**`Contradiction`** ("a belief stated, then reversed over time"). Stage 5 is
meant to *detect* a contradiction. But detection alone is half the job — nothing
in the current design **propagates** a contradiction forward to everything that
was derived from the now-false belief. That propagation is what this note adds.

## 2. What actually transfers from "blockchain" (and what must not)

The intuition is right, but the buzzword would be wrong to import literally.
Being precise about the borrow keeps us honest (Law 2: earn the right to invent;
don't cargo-cult sophistication).

**Transfers — the real primitive:**

- **Hash-linking for break-detection.** Each derived item stores the *content
  hash* of every premise it consumed. To check if it is stale, recompute the
  current hash of each premise; any mismatch = a broken link = re-evaluate.
  Detection is **O(1) per edge** — a hash compare. This is the tamper-evidence
  property, and it is exactly what the user described.
- **Append-only history.** A ledger of derivations is an append-only log — the
  same shape as L0 and the Stage 3 artifact log. We already own this machinery
  (`kortex-store`: segmented append-only files, rotation, fsync, recovery,
  128-bit content hashing for dedup).

**Does NOT transfer — leave it at the door:**

- **Consensus / proof-of-work / mining / tokens.** This is a single-user, local,
  *trusted* store. There are no Byzantine actors and no global agreement
  problem. None of that machinery applies; importing it would be pure overhead.
- **A single linear chain.** Reasoning is **many-to-one**: a conclusion depends
  on several premises; those depend on others. The correct structure is a
  **Merkle DAG** (directed acyclic graph), not a linear blockchain. Git's
  commit/tree objects, Nix derivations, and Bazel's action graph are the right
  cousins — content-addressed dependency DAGs, not currencies.
- **Immutability as the goal.** A blockchain never revises history. We want the
  *opposite use* of the same primitive: cheap detection of change so we can
  re-derive. Same hash-linking, inverted intent.

So the honest one-liner: **this is "Git/Nix for an agent's beliefs" — a
content-addressed Merkle DAG of derivations whose hash links make staleness
detectable and propagatable deterministically.** "Blockchain" is the mental
model for the tamper-evidence property; it is not the implementation.

## 3. The data model (sketch)

Every memory item already has a content hash (Law: provenance to source bytes).
A *derived* item additionally records **how** it came to be:

```
DerivationRecord {
    output:    ContentHash,        // hash of the produced claim/edge/summary/answer
    inputs:    [ContentHash],      // hash-pinned snapshot of every premise consumed
    method:    MethodId,           // which deterministic op OR which LLM prompt
    method_v:  Hash,               // params + model/version/prompt hash (the "why" of HOW)
    produced:  LogicalTime,        // append-only position; no wall clock in logic
}
```

- `inputs` are **pinned by hash**, not by id. That is the whole trick.
- `method` + `method_v` make the **"why" first-class**: not just *which* premises
  but *which reasoning operation*, versioned. A minor-version bump of an
  extractor changes `method_v`, which flags everything it produced — this is the
  Redditor's "library changed in a minor version" case, caught mechanically.

### Staleness, defined

An output is **stale** iff, for any input id, the *current* content hash of that
id ≠ the pinned hash stored in the record (or its `method_v` no longer matches
the current method). Staleness is **transitive**: mark an item stale → mark its
dependents stale (a topological sweep over the DAG). A superseding belief (a
`Contradiction` resolving to a new value) changes a leaf hash, and the wave
propagates to exactly the conclusions that rested on the old value — and no
others.

This gives the project something it does not have today: **memory that knows
when it has become wrong**, deterministically, without re-running any inference.

## 4. Why it fits the two laws and the hard rules

- **Law 1 (intelligence in the substrate).** Break-detection and propagation are
  pure substrate machinery — hash compares and a topological sweep. The model is
  never consulted to *find* staleness; it is only asked to re-narrate the small
  set that got flagged.
- **Law 2 (earn invention with a benchmark).** We do **not** invent a new
  storage engine: the ledger is `kortex-store` reused. The only new thing is the
  derivation record + the propagation algorithm, and §6 makes it falsifiable on
  ground truth before we commit to it.
- **AGENTS Rule 2 (determinism).** `stale_set(ledger)` is a pure function of the
  ledger — same ledger → byte-identical stale set. No wall clock, no `rand`.
- **AGENTS Rule 3 (LLM outputs replayable).** The ledger *is* the replay log.
  `method` records which inference produced an edge; rebuilding the graph from
  the ledger is deterministic even though producing it was not. This note
  *strengthens* Rule 3 rather than bending it: the artifact log it already
  mandates becomes a dependency-typed log.

## 5. Where it slots into the roadmap

It is **cross-cutting**, not a single new stage. Concretely:

- **Foundation (small, buildable early, behind Stage 2 machinery):** a
  `kortex-ledger` concept — `DerivationRecord`, content-hash pinning, an
  append-only store (reuse `kortex-store`), and `stale_set` / propagation as
  pure functions with property + golden tests. No LLM needed; testable on
  synthetic derivations.
- **Stage 3 (GraphRAG):** every extracted entity/relation edge and every RAPTOR
  summary is written *with* a `DerivationRecord` (its input unit hashes + the
  extractor's `method_v`). Re-extraction on changed inputs becomes incremental
  by construction — you only re-run the flagged subset.
- **Stage 5 (Insight Engine):** the keystone. An insight is a derivation over
  evidence; when a `Contradiction` lands, propagation retracts exactly the
  insights it undermines. "Cite or be silent" gains a time axis: an insight
  whose chain is broken is *demoted to needs-revalidation*, never silently
  served. This is the anti-hallucination guarantee surviving change over time.

## 6. The benchmark that decides whether this is real (Law 2)

We do not adopt this because it sounds elegant; we adopt it iff it measurably
improves decisions over time — which the harness can already test, because
contradictions are planted ground truth. Proposed metric, addable at Stage 0
level over synthetic derivations and properly at Stage 5:

> **Staleness propagation accuracy.** Plant a contradiction at logical time *T*
> that flips premise *p*. Let `D(p)` be the set of derived conclusions that
> actually depended on *p*. After propagation, measure
> **invalidation-recall** = |flagged ∩ D(p)| / |D(p)| (did we catch everything
> that went stale?) and **over-invalidation rate** = |flagged \ D(p)| / |flagged|
> (did we needlessly flag conclusions that did *not* depend on *p*?).

Target shape: invalidation-recall → 1.0 (a broken premise must never leave a
dependent silently valid) at a low over-invalidation rate (don't cry wolf on the
whole graph). This is the honest "does it help the agent decide better over
time?" test, made into a number the gate can check.

## 7. Honest risks / open questions

- **Granularity / fan-out.** If every unit is a premise of a huge community
  summary, one change can flag a large subgraph. Mitigation: derive over
  *salient* premises (the few that actually moved the conclusion), and treat
  propagation as "needs revalidation," not "delete" — re-derivation is cheap for
  the deterministic tier and budgeted for the LLM tier (Stage 3 arithmetic).
- **What counts as "changed".** Semantic vs byte change: a paraphrase of a
  premise is a new hash but maybe not a new meaning. v0 is conservative
  (hash-exact = re-validate); a later refinement can gate re-derivation on an
  embedding-distance threshold before spending LLM tokens. Measure before adding.
- **Cost of the ledger itself.** One record per derived item. At 5M units with a
  graph this is bounded and append-only — model it under the Stage 2 byte
  budgets before building, same discipline as `STAGE_3_BUDGET.md`.
- **Is the foundation worth building before Stage 3 exists?** Probably yes as a
  *thin* trait + pure functions + property tests (cheap, de-risks Stage 3's
  incrementality), but full value only lands with the graph and insights. Decide
  with the human.

## 8. One-line summary

Store the *reasoning*, hash-pin its premises, and a changed premise breaks the
link — turning "memory that stores facts" into **memory that knows when its facts
went stale and why** — the user's blockchain intuition, implemented as the
Git/Nix-shaped Merkle DAG it actually wants to be, and benchmarked on the
contradictions our corpus already plants.
