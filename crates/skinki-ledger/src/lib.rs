#![forbid(unsafe_code)]
//! # Derivation ledger — staleness-aware memory
//!
//! Every memory system stores *facts*. Almost none track *when a fact stopped
//! being true* or *why it was believed*. This crate is the thin foundation for
//! the project's answer (see `skinki/specs/DERIVATION_LEDGER.md`): store the
//! **reasoning chain**, not just the conclusion, and **hash-pin every premise**.
//! When a premise changes its content hash no longer matches the pinned one —
//! the link breaks — and every conclusion that rested on it is flagged stale.
//!
//! The mental model is the tamper-evidence property of a blockchain (alter one
//! link and the mismatch is visible at once). The *implementation* is the
//! Git/Nix shape that property actually wants: a content-addressed **Merkle DAG**
//! of derivations — many-to-one (a conclusion depends on several premises), no
//! consensus, no proof-of-work, no chain-as-currency. We borrow the hash link,
//! nothing else.
//!
//! ## What this v0 delivers
//!
//! - [`ContentHash`] — a 128-bit content address (dual-seed FNV-1a, mirroring
//!   `skinki-store`'s `content_hash_128` for conceptual consistency).
//! - [`Derivation`] — one output produced by a [`MethodStamp`] from hash-pinned
//!   [`inputs`](Derivation::inputs).
//! - [`Ledger`] — an append-only log of derivations and the **deterministic**
//!   [`stale_closure`](Ledger::stale_closure): given the premises whose content
//!   changed (and/or the methods whose version moved), the exact transitive set
//!   of conclusions that must be re-evaluated.
//! - [`score_staleness`] — the honest benchmark from the design note:
//!   *invalidation-recall* (did we catch everything that went stale?) and
//!   *over-invalidation* (did we needlessly flag the independent?).
//!
//! Determinism is law here (`AGENTS.md` rule 2): every public result is a sorted
//! [`BTreeSet`], independent of insertion or traversal order, so the same ledger
//! and the same change set always produce a byte-identical stale set.
//!
//! Persistence is intentionally **out of scope for v0**: the ledger is an
//! append-only log, exactly the shape `skinki-store` already durably persists,
//! so wiring it to disk is a later, mechanical step. v0 proves the *algorithm*.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Content hashing — 128-bit, dual-seed FNV-1a (mirrors skinki-store)
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01B3;

fn fnv1a_64(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = FNV_OFFSET ^ seed;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A 128-bit content address. Two facts with the same bytes hash equal; a
/// single changed byte (a paraphrase, a corrected value, a superseding belief)
/// produces a different hash — which is exactly the signal staleness rides on.
///
/// `Ord` is derived so every set/map keyed by a hash iterates in a fixed order,
/// keeping [`Ledger::stale_closure`] deterministic.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct ContentHash(pub u128);

impl ContentHash {
    /// Content-address arbitrary bytes (the premise/fact/conclusion payload).
    pub fn of(bytes: &[u8]) -> Self {
        let hi = fnv1a_64(bytes, 0x012B_9B0A_BE15_D09D);
        let lo = fnv1a_64(bytes, FNV_OFFSET);
        ContentHash(((hi as u128) << 64) | (lo as u128))
    }
}

// ---------------------------------------------------------------------------
// The "why / how": method identity + version
// ---------------------------------------------------------------------------

/// Which reasoning operation produced a derivation, plus a version stamp.
///
/// The version is the design note's handle on the Redditor's "a library changed
/// in a minor version" case: bump the version of an extractor / prompt / rule
/// and every conclusion it produced is flagged for re-derivation, without any
/// premise content having changed. `id` identifies the operation; `version` is
/// any monotonic stamp (a build hash, a prompt revision, a semver-encoded int).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct MethodStamp {
    pub id: u32,
    pub version: u64,
}

impl MethodStamp {
    pub fn new(id: u32, version: u64) -> Self {
        MethodStamp { id, version }
    }
}

// ---------------------------------------------------------------------------
// A single derivation
// ---------------------------------------------------------------------------

/// One node of the derivation DAG: `output` was produced by `method` from the
/// hash-pinned `inputs`. Inputs may be raw facts *or* the outputs of earlier
/// derivations — that is what makes the ledger a DAG rather than a flat list.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Derivation {
    pub output: ContentHash,
    pub inputs: Vec<ContentHash>,
    pub method: MethodStamp,
}

impl Derivation {
    pub fn new(output: ContentHash, inputs: Vec<ContentHash>, method: MethodStamp) -> Self {
        Derivation {
            output,
            inputs,
            method,
        }
    }
}

// ---------------------------------------------------------------------------
// The ledger
// ---------------------------------------------------------------------------

/// An append-only log of [`Derivation`]s, plus the staleness algorithm over it.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Ledger {
    records: Vec<Derivation>,
}

impl Ledger {
    pub fn new() -> Self {
        Ledger::default()
    }

    /// Append one derivation. Append-only by construction: there is no remove or
    /// mutate — a superseding belief is a *new* record with a new output hash,
    /// and the old conclusions go stale through [`stale_closure`].
    pub fn record(&mut self, d: Derivation) {
        self.records.push(d);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[Derivation] {
        &self.records
    }

    /// The deterministic transitive set of **stale output hashes**.
    ///
    /// A derivation's output is stale iff:
    /// 1. one of its `inputs` is in `changed_inputs` (a premise's content moved),
    ///    or
    /// 2. its method's current version (from `current_method_versions`) differs
    ///    from the version it was produced with (the "how" changed), or
    /// 3. one of its `inputs` is itself a stale output (transitive break).
    ///
    /// `current_method_versions` maps a method `id` to its *current* version; an
    /// `id` absent from the map is treated as unchanged. Methods not consulted
    /// at all (empty map) reduce this to pure premise-change propagation, which
    /// is the common case the corpus's planted contradictions exercise.
    ///
    /// The result is a sorted [`BTreeSet`]: the same `(ledger, changed_inputs,
    /// versions)` always yields a byte-identical set, regardless of record or
    /// traversal order.
    pub fn stale_closure(
        &self,
        changed_inputs: &BTreeSet<ContentHash>,
        current_method_versions: &BTreeMap<u32, u64>,
    ) -> BTreeSet<ContentHash> {
        // Reverse adjacency: which records consume a given hash as an input.
        // Built once; keyed by hash so iteration is deterministic.
        let mut consumers: BTreeMap<ContentHash, Vec<usize>> = BTreeMap::new();
        for (i, d) in self.records.iter().enumerate() {
            for &input in &d.inputs {
                consumers.entry(input).or_default().push(i);
            }
        }

        let directly_stale = |d: &Derivation| -> bool {
            d.inputs.iter().any(|h| changed_inputs.contains(h))
                || matches!(
                    current_method_versions.get(&d.method.id),
                    Some(&v) if v != d.method.version
                )
        };

        // Worklist fixpoint. `stale` is the answer; pushing a record index that
        // is already covered is harmless — the set insert dedups it. The final
        // set is order-independent, so pop order never affects the result.
        let mut stale: BTreeSet<ContentHash> = BTreeSet::new();
        let mut worklist: Vec<usize> = (0..self.records.len())
            .filter(|&i| directly_stale(&self.records[i]))
            .collect();

        while let Some(i) = worklist.pop() {
            let out = self.records[i].output;
            if stale.insert(out) {
                // `out` is newly stale: everything consuming it breaks too.
                if let Some(cs) = consumers.get(&out) {
                    worklist.extend(cs.iter().copied());
                }
            }
        }
        stale
    }

    /// Persist the ledger as JSON. v0 persistence: the in-memory `Vec` *is* the
    /// append-only log, so a serialized snapshot is faithful. Durable
    /// append/rotation/torn-tail recovery is a later step on `skinki-store`'s
    /// machinery (per the design note); this is enough to carry a ledger across
    /// sessions and to pin a golden snapshot in tests.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(self).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(std::io::Error::other)
    }
}

// ---------------------------------------------------------------------------
// The benchmark (design note §6): does staleness detection actually work?
// ---------------------------------------------------------------------------

/// Accuracy of a stale set against the ground-truth dependents of a change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StalenessScore {
    /// |flagged ∩ truth| / |truth| — fraction of genuinely-stale conclusions
    /// caught. Must reach 1.0: a broken premise must never leave a dependent
    /// silently valid. Defined as 1.0 when there is nothing to catch.
    pub invalidation_recall: f64,
    /// |flagged \ truth| / |flagged| — fraction of flags that were needless
    /// (conclusions that did *not* depend on the change). Should stay near 0:
    /// don't cry wolf on the whole graph. Defined as 0.0 when nothing is flagged.
    pub over_invalidation: f64,
}

/// Score a computed stale set against the true dependents of a change — the
/// honest "does it help the agent decide better over time?" measurement made
/// into two numbers a gate can check.
pub fn score_staleness(
    flagged: &BTreeSet<ContentHash>,
    true_dependents: &BTreeSet<ContentHash>,
) -> StalenessScore {
    let hits = flagged.intersection(true_dependents).count();
    let invalidation_recall = if true_dependents.is_empty() {
        1.0
    } else {
        hits as f64 / true_dependents.len() as f64
    };
    let over_invalidation = if flagged.is_empty() {
        0.0
    } else {
        (flagged.len() - hits) as f64 / flagged.len() as f64
    };
    StalenessScore {
        invalidation_recall,
        over_invalidation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(s: &str) -> ContentHash {
        ContentHash::of(s.as_bytes())
    }

    /// Default method stamp for tests that don't exercise versioning.
    fn m() -> MethodStamp {
        MethodStamp::new(1, 1)
    }

    fn set(items: &[ContentHash]) -> BTreeSet<ContentHash> {
        items.iter().copied().collect()
    }

    #[test]
    fn content_hash_is_stable_and_discriminating() {
        // Stable across calls (no salt drift), and a one-character change moves
        // the hash — the whole premise of break-detection.
        assert_eq!(h("coffee = anxiety"), h("coffee = anxiety"));
        assert_ne!(h("coffee = anxiety"), h("coffee = fine"));
    }

    #[test]
    fn direct_premise_change_flags_the_conclusion() {
        // fact ──▶ conclusion. Change the fact; the conclusion goes stale.
        let fact = h("coffee gives me anxiety");
        let concl = h("avoid caffeine after noon");
        let mut l = Ledger::new();
        l.record(Derivation::new(concl, vec![fact], m()));

        let stale = l.stale_closure(&set(&[fact]), &BTreeMap::new());
        assert_eq!(stale, set(&[concl]));
    }

    #[test]
    fn staleness_propagates_transitively_through_the_dag() {
        //   p ──▶ a ──▶ c
        //   q ──▶ b ──▶ c
        // Changing p must invalidate a and c, but NOT b (q is untouched).
        let (p, q) = (h("p"), h("q"));
        let (a, b, c) = (h("a"), h("b"), h("c"));
        let mut l = Ledger::new();
        l.record(Derivation::new(a, vec![p], m()));
        l.record(Derivation::new(b, vec![q], m()));
        l.record(Derivation::new(c, vec![a, b], m()));

        let stale = l.stale_closure(&set(&[p]), &BTreeMap::new());
        assert_eq!(stale, set(&[a, c]), "must flag a and c, never b");
    }

    #[test]
    fn independent_branch_is_never_flagged() {
        // Two disjoint chains; touching one must leave the other pristine.
        let (p, q) = (h("p"), h("q"));
        let (x, y) = (h("x"), h("y"));
        let mut l = Ledger::new();
        l.record(Derivation::new(x, vec![p], m()));
        l.record(Derivation::new(y, vec![q], m()));

        let stale = l.stale_closure(&set(&[p]), &BTreeMap::new());
        assert_eq!(stale, set(&[x]));
    }

    #[test]
    fn method_version_bump_invalidates_its_outputs() {
        // No premise changed, but the extractor (method 7) was upgraded from
        // v3 to v4 — everything it produced must be re-derived.
        let fact = h("some unit text");
        let edge = h("entity->relation edge");
        let mut l = Ledger::new();
        l.record(Derivation::new(edge, vec![fact], MethodStamp::new(7, 3)));

        let current = BTreeMap::from([(7u32, 4u64)]);
        let stale = l.stale_closure(&BTreeSet::new(), &current);
        assert_eq!(stale, set(&[edge]));

        // Same version => nothing stale.
        let same = BTreeMap::from([(7u32, 3u64)]);
        assert!(l.stale_closure(&BTreeSet::new(), &same).is_empty());
    }

    #[test]
    fn stale_closure_is_deterministic_regardless_of_record_order() {
        // Build the same DAG with records appended in two different orders; the
        // stale set must be byte-identical (BTreeSet, order-independent).
        let (p, a, b, c) = (h("p"), h("a"), h("b"), h("c"));
        let da = Derivation::new(a, vec![p], m());
        let db = Derivation::new(b, vec![a], m());
        let dc = Derivation::new(c, vec![b], m());

        let mut l1 = Ledger::new();
        l1.record(da.clone());
        l1.record(db.clone());
        l1.record(dc.clone());

        let mut l2 = Ledger::new();
        l2.record(dc);
        l2.record(db);
        l2.record(da);

        let s1 = l1.stale_closure(&set(&[p]), &BTreeMap::new());
        let s2 = l2.stale_closure(&set(&[p]), &BTreeMap::new());
        assert_eq!(s1, s2);
        assert_eq!(s1, set(&[a, b, c]));
    }

    #[test]
    fn benchmark_perfect_propagation_on_a_planted_contradiction() {
        // The design-note §6 metric. Plant premise p with a known dependent set
        // D(p) = {a, c} (a derived from p; c from a). A contradiction flips p.
        // Exact-hash propagation should achieve recall 1.0 at 0 over-flagging.
        let (p, q) = (h("belief: X is true"), h("unrelated fact"));
        let (a, b, c) = (h("a from p"), h("b from q"), h("c from a"));
        let mut l = Ledger::new();
        l.record(Derivation::new(a, vec![p], m()));
        l.record(Derivation::new(b, vec![q], m()));
        l.record(Derivation::new(c, vec![a], m()));

        let flagged = l.stale_closure(&set(&[p]), &BTreeMap::new());
        let truth = set(&[a, c]);
        let score = score_staleness(&flagged, &truth);

        assert_eq!(score.invalidation_recall, 1.0, "must catch every dependent");
        assert_eq!(
            score.over_invalidation, 0.0,
            "must not flag the independent b"
        );
    }

    #[test]
    fn save_load_roundtrip_and_serialization_is_deterministic() {
        let (p, q, a, b, c) = (h("p"), h("q"), h("a"), h("b"), h("c"));
        let mut l = Ledger::new();
        l.record(Derivation::new(a, vec![p], m()));
        l.record(Derivation::new(b, vec![q], MethodStamp::new(2, 5)));
        l.record(Derivation::new(c, vec![a, b], m()));

        // Serializing the same ledger twice is byte-identical (no map/set order
        // leaking into the snapshot).
        let s1 = serde_json::to_vec(&l).unwrap();
        let s2 = serde_json::to_vec(&l).unwrap();
        assert_eq!(s1, s2);

        let dir = std::env::temp_dir().join(format!("skinki_ledger_io_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ledger.json");
        l.save(&path).unwrap();
        let loaded = Ledger::load(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(l.records(), loaded.records());
        // A reloaded ledger answers staleness identically.
        assert_eq!(
            l.stale_closure(&set(&[p]), &BTreeMap::new()),
            loaded.stale_closure(&set(&[p]), &BTreeMap::new())
        );
    }

    #[test]
    fn score_edge_cases_are_well_defined() {
        let empty = BTreeSet::new();
        // Nothing to catch and nothing flagged: vacuously perfect.
        let s = score_staleness(&empty, &empty);
        assert_eq!(s.invalidation_recall, 1.0);
        assert_eq!(s.over_invalidation, 0.0);

        // Flagged something with no true dependents: all of it is over-flagging.
        let flagged = set(&[h("x")]);
        let s = score_staleness(&flagged, &empty);
        assert_eq!(s.over_invalidation, 1.0);
    }
}
