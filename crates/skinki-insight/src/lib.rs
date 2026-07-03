//! Stage 5 — the Insight Engine (anti-hallucination keystone).
//!
//! Law 1 in its purest form: the intelligence is in the *memory*, not the
//! model. Discovery and statistical validation are **deterministic** (AGENTS
//! rule 2); the LLM only *narrates*, and only with citations ("cite-or-silence"),
//! so the count of uncited claims is **zero by construction**.
//!
//! Pipeline: a [`Detector`] proposes [`InsightCandidate`]s from an
//! [`InsightInput`] (the *only* view of the corpus a detector may see — never
//! the planted answers); [`validate`] gates them with Benjamini–Hochberg FDR +
//! surprise/support floors (this is the apophenia discriminator, frontier-owned
//! and provably correct); a [`Narrator`] verbalizes each survivor under the
//! cite-or-silence contract; [`InsightEngine::discover`] emits only what is
//! validated, narrated, **and** cited.
//!
//! ## Division of ownership (see `specs/STAGE_5.md`)
//! - **Frontier-owned (this file, the keystone):** the fairness boundary
//!   [`InsightInput`], the FDR core [`validate`], the cite-or-silence enforcement
//!   in [`InsightEngine::discover`], and the reference [`StructuralBridgeDetector`]
//!   (the worked example that *passes* the gate) + the naive [`CoMentionDetector`]
//!   contrast (the one that *fails* apophenia — proof the gate has teeth).
//! - **Delegated impl tickets:** [`InsightKind::TemporalLead`] /
//!   [`InsightKind::Contradiction`] detectors and the live (replayed) LLM
//!   narrator — they feed the same [`validate`] + cite-or-silence machinery.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use skinki_corpus::{Corpus, Entity, EntityId, Entry, EntryId};
use skinki_eval::DiscoveredInsight;
use skinki_ledger::{ContentHash, Derivation, Ledger, MethodStamp};
use skinki_sleep::{Job, StepBudget, StepOutcome};

pub type CandidateId = u64;

/// The family of structural/statistical pattern a detector proposes. Each maps
/// to a planted ground-truth type the gate scores against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightKind {
    /// A rare entity bridging two otherwise-disconnected clusters
    /// (`skinki_corpus::InsightBridge`). The reference detector targets this.
    StructuralBridge,
    /// Entity A's mentions lead event B by a fixed lag
    /// (`skinki_corpus::TemporalPattern`). Impl ticket.
    TemporalLead,
    /// A belief stated then reversed (`skinki_corpus::Contradiction`), surfaced
    /// via the derivation ledger's staleness flag. Impl ticket.
    Contradiction,
}

fn kind_tag(k: InsightKind) -> u8 {
    match k {
        InsightKind::StructuralBridge => 1,
        InsightKind::TemporalLead => 2,
        InsightKind::Contradiction => 3,
    }
}

/// Candidate ids are namespaced by detector family (kind tag in the top byte),
/// so pooling candidates from several detectors can never collide — before
/// this, the structural detector used raw entity ids while the others counted
/// from 0, and a full multi-detector engine silently shadowed candidates in
/// its id-keyed maps (specs/STAGE_5C.md T1).
#[inline]
pub fn candidate_id(kind: InsightKind, seq: u64) -> CandidateId {
    debug_assert!(seq < (1 << 56), "candidate seq overflows the id namespace");
    ((kind_tag(kind) as u64) << 56) | seq
}

/// The per-candidate test result, **before** multiple-hypothesis correction.
/// `surprise` is the apophenia discriminator (a hub spreads thin → low
/// surprise); `support` the minimum-evidence guard; `p_value` the input to
/// BH-FDR; `effect` the ranking key.
#[derive(Debug, Clone, Copy)]
pub struct Statistic {
    pub effect: f64,
    pub p_value: f64,
    pub support: u32,
    pub surprise: f64,
}

/// A raw, pre-validation candidate connection. `evidence` is provenance and
/// **must be non-empty** — a candidate with no citable support never enters the
/// pipeline (it could never satisfy cite-or-silence).
#[derive(Debug, Clone)]
pub struct InsightCandidate {
    pub id: CandidateId,
    pub kind: InsightKind,
    /// The entities this connection is about; `entities[0]` is the bridge entity
    /// for [`InsightKind::StructuralBridge`].
    pub entities: Vec<EntityId>,
    pub evidence: Vec<EntryId>,
    pub stat: Statistic,
    /// A short, factual claim *derived from the data* (not model prose). The
    /// narrator may rewrite it, but its citations must stay ⊆ `evidence`.
    pub claim: String,
}

/// The **only** view of the corpus a detector is allowed to see: observable
/// signal, never the planted answer key. [`Self::from_corpus`] is the single
/// audited seam — it drops `ground_truth.{insights, negative_bridges, multi_hop,
/// recall, temporal, contradictions}`, keeping only the entries and the entity
/// *vocabulary* (names + kinds + cluster labels, the established "fair
/// vocabulary" convention — see `STAGE_3.md`). This makes "no peeking at the
/// answer key" a **type guarantee**, not a reviewer's hope.
pub struct InsightInput<'a> {
    pub entries: &'a [Entry],
    pub vocab: &'a [Entity],
}

impl<'a> InsightInput<'a> {
    pub fn from_corpus(c: &'a Corpus) -> Self {
        InsightInput {
            entries: &c.entries,
            vocab: &c.ground_truth.entities,
        }
    }
}

/// A detector proposes candidates **deterministically** from the substrate:
/// pure function of `input`, no wall clock, no `HashMap` iteration order.
pub trait Detector {
    fn name(&self) -> &str;
    fn propose(&self, input: &InsightInput) -> Vec<InsightCandidate>;
}

/// Cite-or-silence narration. `narrate` returns `None` when the narrator chooses
/// **silence** (cannot ground the claim / low confidence). A returned
/// [`NarratedInsight`] *must* carry non-empty citations ⊆ the candidate's
/// evidence; [`InsightEngine::discover`] drops any record that violates this, so
/// the `= 0` uncited budget cannot regress.
///
/// The reference [`ExtractiveNarrator`] is deterministic (no model). The live
/// LLM narrator (Stage 6/7) implements this same trait and is **replayed** from
/// an artifact log in the gate — never inferred in CI (rule 3).
pub trait Narrator {
    fn narrate(&self, c: &InsightCandidate, input: &InsightInput) -> Option<NarratedInsight>;
}

/// A narrated insight. `citations` MUST be non-empty and ⊆ the candidate's
/// `evidence`.
#[derive(Debug, Clone)]
pub struct NarratedInsight {
    pub text: String,
    pub citations: Vec<EntryId>,
}

// ---------------------------------------------------------------------------
// Statistical validation — the apophenia discriminator (frontier-owned)
// ---------------------------------------------------------------------------

/// Validation thresholds. `fdr_q` is the Benjamini–Hochberg false-discovery
/// rate; `min_surprise` rejects hubs (which spread thin → low surprise);
/// `min_support` rejects coincidences.
#[derive(Debug, Clone, Copy)]
pub struct ValidationCfg {
    pub fdr_q: f64,
    pub min_surprise: f64,
    pub min_support: u32,
}

impl Default for ValidationCfg {
    /// Calibrated on the V2 synthetic corpus (the D1 design ticket). Raising
    /// these never lowers a budget; do not weaken without sign-off.
    fn default() -> Self {
        ValidationCfg {
            fdr_q: 0.05,
            min_surprise: 0.60,
            min_support: 2,
        }
    }
}

/// Gate candidates: apply the support + surprise floors, then **Benjamini–
/// Hochberg FDR** over the survivors' p-values at level `cfg.fdr_q`. Pure and
/// deterministic; returns accepted ids in stable `(effect desc, id asc)` order.
///
/// This is where apophenia hubs die: a hub co-occurs with everything, so its
/// per-pair concentration is low (low `surprise`) and, under the null, its
/// dominant-cluster count is unremarkable (high `p_value`). BH-FDR controls the
/// expected fraction of false discoveries among *all* surfaced insights — the
/// principled answer to "we tested thousands of pairs, some will look linked by
/// chance."
pub fn validate(cands: &[InsightCandidate], cfg: &ValidationCfg) -> Vec<CandidateId> {
    // Pre-filter: support + surprise floors.
    let mut survivors: Vec<&InsightCandidate> = cands
        .iter()
        .filter(|c| c.stat.support >= cfg.min_support && c.stat.surprise >= cfg.min_surprise)
        .collect();
    if survivors.is_empty() {
        return Vec::new();
    }

    // BH step-up. Sort ascending by p (id-tiebroken for a stable threshold).
    survivors.sort_by(|a, b| {
        a.stat
            .p_value
            .total_cmp(&b.stat.p_value)
            .then(a.id.cmp(&b.id))
    });
    let m = survivors.len() as f64;
    // Largest rank i (1-based) with p_(i) <= (i/m)*q defines the cutoff p-value;
    // all survivors with p <= cutoff are accepted.
    let mut cutoff: Option<f64> = None;
    for (i, c) in survivors.iter().enumerate() {
        let rank = (i + 1) as f64;
        if c.stat.p_value <= (rank / m) * cfg.fdr_q {
            cutoff = Some(c.stat.p_value);
        }
    }
    let Some(cutoff) = cutoff else {
        return Vec::new();
    };

    let mut accepted: Vec<&InsightCandidate> = survivors
        .into_iter()
        .filter(|c| c.stat.p_value <= cutoff)
        .collect();
    accepted.sort_by(|a, b| {
        b.stat
            .effect
            .total_cmp(&a.stat.effect)
            .then(a.id.cmp(&b.id))
    });
    accepted.into_iter().map(|c| c.id).collect()
}

/// [`validate`] applied **per detector family** (specs/STAGE_5C.md T3): BH's
/// false-discovery guarantee is about one family of hypotheses, and pooling
/// families breaks it in both directions — a contradiction candidate's exact
/// `p = 0` occupies a top BH rank and *raises* the cutoff for every structural/
/// temporal candidate, while a large noisy family dilutes a small precise one.
/// Accepted ids are returned family-by-family in a fixed kind order, each
/// family in `validate`'s stable (effect desc, id asc) order.
pub fn validate_per_kind(cands: &[InsightCandidate], cfg: &ValidationCfg) -> Vec<CandidateId> {
    let mut accepted: Vec<CandidateId> = Vec::new();
    for kind in [
        InsightKind::StructuralBridge,
        InsightKind::TemporalLead,
        InsightKind::Contradiction,
    ] {
        let family: Vec<InsightCandidate> =
            cands.iter().filter(|c| c.kind == kind).cloned().collect();
        if !family.is_empty() {
            accepted.extend(validate(&family, cfg));
        }
    }
    accepted
}

// ---------------------------------------------------------------------------
// Shared topic-cluster profiling (deterministic, over the fair vocabulary)
// ---------------------------------------------------------------------------

/// Minimum entity-name length to match as a mention — guards against short
/// surface forms colliding with unrelated substrings.
const MIN_NAME_LEN: usize = 3;

/// `(lowercased topic phrase, cluster)` pairs from the corpus topic lexicon —
/// the observable vocabulary that maps an entry's text to topic clusters.
fn topic_index() -> Vec<(String, String)> {
    let mut v = Vec::new();
    for (cluster, topics) in skinki_corpus::topic_lexicon() {
        for t in *topics {
            v.push((t.to_lowercase(), cluster.to_string()));
        }
    }
    v
}

/// How one entity's mention-entries distribute over **topic clusters** — the
/// observable signal that separates a 2-cluster bridge from a 4-cluster hub.
struct EntityProfile {
    id: EntityId,
    name: String,
    /// cluster -> mention entries (sorted, dedup) carrying that cluster's topic.
    by_cluster: BTreeMap<String, Vec<EntryId>>,
}

/// Build a [`EntityProfile`] per vocab entity. Deterministic: vocab order is the
/// corpus's stable entity order; clusters/entries are `BTreeMap`/sorted.
fn profile_entities(input: &InsightInput) -> Vec<EntityProfile> {
    let topics = topic_index();
    let lower: Vec<(EntryId, String)> = input
        .entries
        .iter()
        .map(|e| (e.id, e.text.to_lowercase()))
        .collect();

    // Each entry's topic clusters, computed once.
    let mut entry_clusters: BTreeMap<EntryId, Vec<String>> = BTreeMap::new();
    for (id, text) in &lower {
        let mut cs: Vec<String> = Vec::new();
        for (phrase, cluster) in &topics {
            if text.contains(phrase.as_str()) && !cs.contains(cluster) {
                cs.push(cluster.clone());
            }
        }
        if !cs.is_empty() {
            entry_clusters.insert(*id, cs);
        }
    }

    let mut out = Vec::with_capacity(input.vocab.len());
    for ent in input.vocab {
        let name = ent.name.to_lowercase();
        let mut by_cluster: BTreeMap<String, Vec<EntryId>> = BTreeMap::new();
        if name.len() >= MIN_NAME_LEN {
            for (id, text) in &lower {
                if !text.contains(&name) {
                    continue;
                }
                if let Some(cs) = entry_clusters.get(id) {
                    for c in cs {
                        by_cluster.entry(c.clone()).or_default().push(*id);
                    }
                }
            }
            for v in by_cluster.values_mut() {
                v.sort_unstable();
                v.dedup();
            }
        }
        out.push(EntityProfile {
            id: ent.id,
            name: ent.name.clone(),
            by_cluster,
        });
    }
    out
}

/// Per-cluster share of all entity↔cluster mentions — the null model: under H0,
/// a topic-neutral entity's mentions land in clusters proportional to this.
fn cluster_prevalence(profiles: &[EntityProfile]) -> (BTreeMap<String, u64>, f64) {
    let mut total: BTreeMap<String, u64> = BTreeMap::new();
    let mut grand: u64 = 0;
    for p in profiles {
        for (c, es) in &p.by_cluster {
            *total.entry(c.clone()).or_insert(0) += es.len() as u64;
            grand += es.len() as u64;
        }
    }
    (total, grand.max(1) as f64)
}

// ---------------------------------------------------------------------------
// Binomial upper tail (the structural detector's null model)
// ---------------------------------------------------------------------------

/// `ln(n!)` by direct summation — `n` is a small co-mention count here.
fn ln_factorial(n: u32) -> f64 {
    (2..=n).map(|x| (x as f64).ln()).sum()
}

fn ln_choose(n: u32, k: u32) -> f64 {
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// `P(X >= k)` for `X ~ Binomial(n, q)`. Computed in log-space for stability.
fn binom_upper_tail(n: u32, k: u32, q: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let q = q.clamp(1e-9, 1.0 - 1e-9);
    let (lq, l1q) = (q.ln(), (1.0 - q).ln());
    let mut p = 0.0;
    for i in k..=n {
        let ln_pmf = ln_choose(n, i) + (i as f64) * lq + ((n - i) as f64) * l1q;
        p += ln_pmf.exp();
    }
    p.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Reference detector — StructuralBridge (frontier-owned, PASSES the gate)
// ---------------------------------------------------------------------------

/// The worked-example detector: surfaces rare entities that bridge **two**
/// clusters, and stays silent on hubs that span many. For each entity it
/// computes, over the entries that mention it:
/// - `support` = cross-cluster co-mentions,
/// - `spread` = distinct partner clusters,
/// - `concentration` = share of co-mentions in the dominant partner cluster,
/// - `surprise = concentration / spread` (a focused 2-cluster bridge → ~1; a
///   hub spreading across many → small),
/// - `p_value` = binomial upper tail of the dominant-cluster count under a null
///   that co-mentions distribute by cluster prevalence (a hub's dominant cluster
///   is *expected* to be large → unremarkable → high p).
///
/// [`validate`] then applies the surprise floor (kills hubs) and BH-FDR (kills
/// chance). This is the D1 design ticket, resolved here.
#[derive(Default)]
pub struct StructuralBridgeDetector;

impl Detector for StructuralBridgeDetector {
    fn name(&self) -> &str {
        "structural-bridge"
    }

    fn propose(&self, input: &InsightInput) -> Vec<InsightCandidate> {
        let profiles = profile_entities(input);
        let (cluster_total, grand) = cluster_prevalence(&profiles);
        let mut out: Vec<InsightCandidate> = Vec::new();

        for p in &profiles {
            let spread = p.by_cluster.len() as u32;
            if spread < 2 {
                continue; // a bridge must touch at least two clusters
            }
            // Cluster counts, sorted by (count desc, cluster asc) — deterministic.
            let mut counts: Vec<(&String, u32)> = p
                .by_cluster
                .iter()
                .map(|(c, es)| (c, es.len() as u32))
                .collect();
            counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

            let total: u32 = counts.iter().map(|(_, n)| n).sum();
            let (c1, n1) = (counts[0].0.clone(), counts[0].1);
            let (c2, n2) = (counts[1].0.clone(), counts[1].1);
            let top2 = n1 + n2;

            // A focused 2-cluster bridge → concentration ~1, spread 2 →
            // surprise ~1. A hub spread thin over many clusters → small surprise.
            let concentration = top2 as f64 / total as f64;
            let surprise = concentration / (spread as f64 - 1.0);

            // Null: top-2 clusters' combined prevalence; a hub's dominant pair is
            // *expected* to be large (high p), a rare focused bridge is not.
            let q2 = (*cluster_total.get(&c1).unwrap_or(&0) + *cluster_total.get(&c2).unwrap_or(&0))
                as f64
                / grand;
            let p_value = binom_upper_tail(total, top2, q2);

            let mut evidence: Vec<EntryId> = Vec::new();
            evidence.extend(p.by_cluster.get(&c1).into_iter().flatten().copied());
            evidence.extend(p.by_cluster.get(&c2).into_iter().flatten().copied());
            evidence.sort_unstable();
            evidence.dedup();
            if evidence.is_empty() {
                continue;
            }

            out.push(InsightCandidate {
                id: candidate_id(InsightKind::StructuralBridge, p.id),
                kind: InsightKind::StructuralBridge,
                entities: vec![p.id],
                evidence,
                stat: Statistic {
                    effect: surprise,
                    p_value,
                    support: total,
                    surprise,
                },
                claim: format!(
                    "{} links the otherwise-separate topics '{}' and '{}'",
                    p.name, c1, c2
                ),
            });
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Naive baseline — the Law-2 contrast (FIRES on apophenia → fails the gate)
// ---------------------------------------------------------------------------

/// The detector to beat: it proposes *every* entity that touches another
/// cluster, with no surprise discrimination (`surprise = 1`, `p = 0`). It is
/// guaranteed to fire on apophenia hubs, so it makes the gate's
/// `negative_hits = 0` budget *fail* — exactly the proof that the structural
/// detector's statistics are earned, not decorative.
#[derive(Default)]
pub struct CoMentionDetector;

impl Detector for CoMentionDetector {
    fn name(&self) -> &str {
        "co-mention-naive"
    }

    fn propose(&self, input: &InsightInput) -> Vec<InsightCandidate> {
        let profiles = profile_entities(input);
        let mut out: Vec<InsightCandidate> = Vec::new();
        for p in &profiles {
            if p.by_cluster.len() < 2 {
                continue; // touches >= 2 clusters -> "a bridge!", no discrimination
            }
            let mut evidence: Vec<EntryId> = p.by_cluster.values().flatten().copied().collect();
            evidence.sort_unstable();
            evidence.dedup();
            let support = evidence.len() as u32;
            out.push(InsightCandidate {
                id: candidate_id(InsightKind::StructuralBridge, p.id),
                kind: InsightKind::StructuralBridge,
                entities: vec![p.id],
                evidence,
                // No discrimination: everything is "surprising" and "significant".
                stat: Statistic {
                    effect: 1.0,
                    p_value: 0.0,
                    support,
                    surprise: 1.0,
                },
                claim: format!("{} co-occurs across topics", p.name),
            });
        }
        out
    }
}

// ---------------------------------------------------------------------------
// TemporalLeadDetector — split-half lag selection + exact circular-shift null
// ---------------------------------------------------------------------------
//
// For every ordered entity pair (A, B) with enough mentions the detector asks:
// do B's mentions follow A's at a fixed lag? Two moves keep the statistics
// honest (specs/STAGE_5C.md T4 — this replaces the analytic binomial null,
// whose Bonferroni×lags correction sat behind a hard p<1e-6 pre-filter that
// voided the FDR semantics, and whose uniform-days assumption would
// hallucinate leads out of bursty real data):
//
// 1. **Selection and testing are split.** The best lag δ* is chosen on A's
//    odd-indexed mention days only; the test statistic is the alignment count
//    at that *fixed* lag on the held-out even-indexed days. The "searched 91
//    lags and kept the max" optimism never reaches the p-value, so no
//    multiple-testing correction is needed — there is exactly one test.
// 2. **The null is an exact circular-shift enumeration.** Under H0 (no phase
//    relation between the two series) every relative offset of B's days is
//    equally likely, so with period D = max_day + 1:
//        p = #{ δ in 0..D : c_test(δ) >= c_test(δ*) } / D,
//    where c_test(δ) counts held-out A-days with a B-day within ±tol of A+δ
//    (circularly). Enumerated in full — deterministic, no RNG, no analytic
//    approximation, resolution 1/D — and it inherits B's real day
//    distribution, so shared burstiness/seasonality is *in* the null instead
//    of being mistaken for signal.
//
// Survivors feed the per-family BH-FDR in `validate` like every detector.
// Targets the planted [`skinki_corpus::TemporalPattern`] ground truth on V2.

/// Deduped, sorted mention days for one entity.
struct MentionSeries {
    entity: EntityId,
    days: Vec<u32>,
    entry_ids: Vec<EntryId>,
}

/// True if `needle` (lowercase) occurs in `haystack` (lowercase) as a whole
/// token — bounded by non-alphanumeric *characters* on both ends. Stops a short
/// entity name like "rust" from matching "trust" / "frustrated" (the bug that
/// inflated a tool to 1763 phantom mentions and flooded the temporal detector
/// with ubiquitous-entity false leads).
///
/// Boundaries are checked per `char`, not per byte (specs/STAGE_5C.md T2): with
/// byte-level `is_ascii_alphanumeric`, every non-ASCII letter looks like a
/// boundary, so Cyrillic text would match needles inside words ("раст" inside
/// "контраст") — the exact phantom-mention bug, reintroduced for the product's
/// own target language.
fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let mut from = 0;
    while let Some(pos) = haystack[from..].find(needle) {
        let i = from + pos;
        let before_ok = haystack[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());
        let after = i + needle.len();
        let after_ok = haystack[after..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        // Advance past the first char of this match (a char boundary by
        // construction — the needle matched here).
        from = i + needle.chars().next().map_or(1, |c| c.len_utf8());
    }
    false
}

fn profile_entity_days(input: &InsightInput) -> Vec<MentionSeries> {
    // Lowercase every entry once, not once per entity: O(N) allocations
    // instead of O(V·N) (specs/STAGE_5C.md T5).
    let lowered: Vec<(u32, EntryId, String)> = input
        .entries
        .iter()
        .map(|e| (e.day, e.id, e.text.to_lowercase()))
        .collect();
    let mut out: Vec<MentionSeries> = Vec::with_capacity(input.vocab.len());
    for e in input.vocab {
        let name_lower = e.name.to_lowercase();
        let mut mentions: Vec<(u32, EntryId)> = lowered
            .iter()
            .filter(|(_, _, text)| contains_word(text, &name_lower))
            .map(|(day, id, _)| (*day, *id))
            .collect();
        mentions.sort_unstable_by_key(|(d, id)| (*d, *id));
        mentions.dedup();
        out.push(MentionSeries {
            entity: e.id,
            days: mentions.iter().map(|(d, _)| *d).collect(),
            entry_ids: mentions.iter().map(|(_, id)| *id).collect(),
        });
    }
    out
}

pub struct TemporalLeadDetector;

impl Detector for TemporalLeadDetector {
    fn name(&self) -> &str {
        "temporal-lead"
    }

    fn propose(&self, input: &InsightInput) -> Vec<InsightCandidate> {
        let series = profile_entity_days(input);
        const MIN_MENTIONS: usize = 5;
        const MAX_LAG: u32 = 90;
        const LAG_TOLERANCE: u32 = 1;
        const MIN_COUNT: u32 = 4;
        const MIN_RATIO: f64 = 0.35;
        const MIN_TEST_COUNT: u32 = 2;

        let period = input.entries.iter().map(|e| e.day).max().unwrap_or(1825) + 1;

        let mut out: Vec<InsightCandidate> = Vec::new();
        let mut next_seq: u64 = 0;

        // Helper: count how many A mentions have a B mention at roughly `lag`
        // days. `b_days` is sorted, so each probe is a binary search.
        fn count_at_lag(a_days: &[u32], b_days: &[u32], lag: u32, tol: u32) -> u32 {
            let mut c: u32 = 0;
            for &da in a_days {
                let lo = da + lag.saturating_sub(tol);
                let hi = da + lag + tol;
                let i = b_days.partition_point(|&db| db < lo);
                if i < b_days.len() && b_days[i] <= hi {
                    c += 1;
                }
            }
            c
        }

        // Helper: collect (a_idx, b_day) pairs where B is within tolerance of A+lag.
        fn evidence_pairs(
            a_days: &[u32],
            a_eids: &[EntryId],
            b_days: &[u32],
            b_eids: &[EntryId],
            lag: u32,
            tol: u32,
        ) -> Vec<(EntryId, u32)> {
            let mut pairs: Vec<(EntryId, u32)> = Vec::new();
            for (&da, &eid_a) in a_days.iter().zip(a_eids.iter()) {
                let lo = da + lag.saturating_sub(tol);
                let hi = da + lag + tol;
                for (&db, &_eid_b) in b_days.iter().zip(b_eids.iter()) {
                    if db >= lo && db <= hi {
                        pairs.push((eid_a, db));
                        break; // one pair per A mention
                    }
                }
            }
            pairs
        }

        for a in series.iter().filter(|s| s.days.len() >= MIN_MENTIONS) {
            for b in series.iter().filter(|s| s.days.len() >= MIN_MENTIONS) {
                if a.entity == b.entity {
                    continue;
                }
                // Split A's (sorted) mention days: odd indices select the lag,
                // even indices test it. Interleaving keeps both halves spread
                // over the whole time range, so drift can't put all the signal
                // in one half.
                let select: Vec<u32> = a.days.iter().copied().skip(1).step_by(2).collect();
                let held_out: Vec<u32> = a.days.iter().copied().step_by(2).collect();

                // Choose δ* on the selection half only (ties → smallest lag).
                let mut best_lag: u32 = 0;
                let mut best_sel: u32 = 0;
                for lag in 0..=MAX_LAG {
                    let c = count_at_lag(&select, &b.days, lag, LAG_TOLERANCE);
                    if c > best_sel {
                        best_sel = c;
                        best_lag = lag;
                    }
                }
                if best_sel == 0 {
                    continue;
                }

                // Effect-size floors on the full data (guards, not tests: they
                // never look at the null distribution).
                let full_count = count_at_lag(&a.days, &b.days, best_lag, LAG_TOLERANCE);
                if full_count < MIN_COUNT || (full_count as f64) < (a.days.len() as f64 * MIN_RATIO)
                {
                    continue;
                }

                // The one test: alignment of the HELD-OUT days at the fixed δ*.
                let observed = count_at_lag(&held_out, &b.days, best_lag, LAG_TOLERANCE);
                if observed < MIN_TEST_COUNT {
                    continue;
                }

                // Exact circular-shift null: smear B's days ±tol on the circle
                // of length `period`, then enumerate every offset δ and count
                // how often the held-out days align at least as well as at δ*.
                let mut smear = vec![false; period as usize];
                for &d in &b.days {
                    for off in -(LAG_TOLERANCE as i64)..=(LAG_TOLERANCE as i64) {
                        let idx = (d as i64 + off).rem_euclid(period as i64) as usize;
                        smear[idx] = true;
                    }
                }
                let mut ge: u32 = 0;
                for delta in 0..period {
                    let mut c: u32 = 0;
                    for &ad in &held_out {
                        if smear[((ad + delta) % period) as usize] {
                            c += 1;
                        }
                    }
                    if c >= observed {
                        ge += 1;
                    }
                }
                // δ = δ* itself is in the enumeration, so p >= 1/period —
                // a proper, never-zero permutation p-value.
                let p_value = ge as f64 / period as f64;

                let effect = full_count as f64 / ((a.days.len() * b.days.len()) as f64).sqrt();
                let support = full_count;

                let pairs = evidence_pairs(
                    &a.days,
                    &a.entry_ids,
                    &b.days,
                    &b.entry_ids,
                    best_lag,
                    LAG_TOLERANCE,
                );
                let mut evidence: Vec<EntryId> = Vec::new();
                let mut seen_evidence: BTreeSet<EntryId> = BTreeSet::new();
                for (eid_a, target_day) in pairs {
                    if seen_evidence.insert(eid_a) {
                        evidence.push(eid_a);
                    }
                    for (&db, &eid_b) in b.days.iter().zip(b.entry_ids.iter()) {
                        if db == target_day {
                            if seen_evidence.insert(eid_b) {
                                evidence.push(eid_b);
                            }
                            break;
                        }
                    }
                }
                evidence.sort_unstable();
                if evidence.is_empty() {
                    continue;
                }

                out.push(InsightCandidate {
                    id: candidate_id(InsightKind::TemporalLead, next_seq),
                    kind: InsightKind::TemporalLead,
                    entities: vec![a.entity, b.entity],
                    evidence,
                    stat: Statistic {
                        effect,
                        p_value,
                        support,
                        surprise: effect,
                    },
                    claim: format!(
                        "{} leads {} by {} days ({} co-occurrences, {} / {} mentions)",
                        input.vocab[a.entity as usize].name,
                        input.vocab[b.entity as usize].name,
                        best_lag,
                        full_count,
                        a.days.len(),
                        b.days.len(),
                    ),
                });
                next_seq += 1;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// ContradictionDetector — sentiment reversal over entity mentions (T3)
// ---------------------------------------------------------------------------
//
// The detector for [`InsightKind::Contradiction`]: finds entities that appear
// in two entries with opposing sentiment — one entry asserts a belief about X
// (marked by positive cues like "is the best choice", "wins"), a later entry
// reverses it (marked by negative cues like "was a mistake", "nothing but pain").
// This targets the planted [`skinki_corpus::Contradiction`] ground truth.
//
// The ledger aspect (specs/STAGE_5.md T3/T5) is deferred: embedding these
// candidates as [`skinki_ledger::Derivation`] records so a changed premise
// flags the contradiction as stale is a plumbing task. The detection logic
// itself is self-contained and deterministic.

/// Positive-stance cue fragments where the entity is the **grammatical subject**
/// — anchored to the name so the stance is attributed to *that* entity, not to
/// every name in the sentence.
const POSITIVE_CUES: &[&str] = &["is the best", "wins", "coming back to"];
/// Negative-stance cues where the entity is the subject of the regret. Crucially
/// these do NOT include sentence-level fragments like "clearly better" or
/// "changed my mind" — those refer to the *replacement* (Y), and attributing
/// them to every name in the entry was the chief false-positive source.
const NEGATIVE_CUES: &[&str] = &[
    "was a mistake",
    "has been nothing but pain",
    "regret picking",
];

/// True if any `cue` is **name-anchored** to `name` in `text` (all lowercase):
/// either "`name` `cue`" (the name is the subject — "rust was a mistake") or
/// "`cue` `name`" (the name is the object — "regret picking rust"). This pins
/// the stance to the right entity, so "rust was a mistake. go is clearly better"
/// marks only `rust` as reversed, never `go`.
fn cue_anchored(text: &str, name: &str, cues: &[&str]) -> bool {
    cues.iter()
        .any(|c| text.contains(&format!("{name} {c}")) || text.contains(&format!("{c} {name}")))
}

pub struct ContradictionDetector;

impl Detector for ContradictionDetector {
    fn name(&self) -> &str {
        "contradiction"
    }

    fn propose(&self, input: &InsightInput) -> Vec<InsightCandidate> {
        let mut out: Vec<InsightCandidate> = Vec::new();
        let mut next_seq: u64 = 0;

        // Lowercase every entry once, not once per entity (specs/STAGE_5C.md T5).
        let lowered: Vec<(&Entry, String)> = input
            .entries
            .iter()
            .map(|e| (e, e.text.to_lowercase()))
            .collect();

        for e in input.vocab {
            let name_lower = e.name.to_lowercase();
            // Name-anchored attribution (not "name appears AND a cue appears
            // anywhere in the entry") is what keeps a replacement entity Y, named
            // in the same reversal sentence, from being falsely flagged.
            let mut praises: Vec<&Entry> = Vec::new();
            let mut regrets: Vec<&Entry> = Vec::new();
            for (entry, t) in &lowered {
                if cue_anchored(t, &name_lower, POSITIVE_CUES) {
                    praises.push(entry);
                }
                if cue_anchored(t, &name_lower, NEGATIVE_CUES) {
                    regrets.push(entry);
                }
            }
            // Need an actual reversal: a regret dated after some endorsement.
            let earliest_praise = praises.iter().map(|p| p.day).min();
            let latest_regret = regrets.iter().map(|r| r.day).max();
            let reversal = matches!((earliest_praise, latest_regret), (Some(p), Some(r)) if r > p);
            if !reversal {
                continue;
            }
            // ONE candidate per entity, citing ALL of its endorsement and regret
            // entries. Recovering which endorsement pairs with which regret from
            // text alone is ambiguous when a tool is reused across interleaved
            // contradictions; emitting the entity-level reversal with the full
            // evidence set sidesteps that mis-pairing (which produced cross-
            // contradiction false positives) while citing every relevant entry —
            // so every planted before/after pair for X is covered.
            let mut evidence: Vec<EntryId> =
                praises.iter().chain(regrets.iter()).map(|e| e.id).collect();
            evidence.sort_unstable();
            evidence.dedup();
            let pos = praises.iter().min_by_key(|p| (p.day, p.id)).unwrap();
            let neg = regrets.iter().max_by_key(|r| (r.day, r.id)).unwrap();
            out.push(InsightCandidate {
                id: candidate_id(InsightKind::Contradiction, next_seq),
                kind: InsightKind::Contradiction,
                entities: vec![e.id],
                evidence,
                // Deterministic reversal; passes the validate() floors. p=0 is
                // safe here because name-anchored attribution (not raw sentiment)
                // gates a candidate — there is no multiple-testing search.
                stat: Statistic {
                    effect: 1.0,
                    p_value: 0.0,
                    support: 2,
                    surprise: 1.0,
                },
                claim: format!(
                    "{}: initially endorsed ('{}'), later reversed ('{}')",
                    e.name,
                    &pos.text[..pos.text.len().min(60)],
                    &neg.text[..neg.text.len().min(60)],
                ),
            });
            next_seq += 1;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Cite-or-silence reference narrator (deterministic; no model)
// ---------------------------------------------------------------------------

/// Deterministic narrator: emits the candidate's data-derived `claim`, citing
/// its evidence verbatim. Never fabricates; cites only what the detector
/// grounded. The live LLM narrator (Stage 6/7) replaces this behind the same
/// trait and is replayed in the gate.
#[derive(Default)]
pub struct ExtractiveNarrator;

impl Narrator for ExtractiveNarrator {
    fn narrate(&self, c: &InsightCandidate, _input: &InsightInput) -> Option<NarratedInsight> {
        if c.evidence.is_empty() {
            return None; // nothing to cite -> silence
        }
        Some(NarratedInsight {
            text: c.claim.clone(),
            citations: c.evidence.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// LLM-narration artifact log — the T4 replay contract
// ---------------------------------------------------------------------------
//
// Every LLM-narrated insight is recorded to an append-only JSON-lines log
// before being returned, so the narration is **replayable** (AGENTS rule 3):
// the live model (`produce`) is not bit-reproducible, but `rebuild(log)` is
// fully deterministic. The gate replays a checked-in fixture log — never
// infers.
//
// The produce side uses the [`ExtractiveNarrator`] as a deterministic LLM
// stand-in (the real model plugs in at Stage 6/7 behind the same trait). The
// replay side loads the log and returns the recorded narration byte-identically.

/// One record in the narration artifact log.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct NarrationRecord {
    pub candidate_id: CandidateId,
    pub text: String,
    pub citations: Vec<EntryId>,
    pub model: String,
    pub v: u64,
}

/// Thin namespace for narration artifact-log operations.
pub struct NarrationLog;

impl NarrationLog {
    /// Append one record as a JSON line to `path` (create-or-append).
    pub fn append(path: &std::path::Path, rec: &NarrationRecord) -> std::io::Result<()> {
        use std::io::Write;
        let line = serde_json::to_string(rec).map_err(std::io::Error::other)?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    /// Replay all records from a JSON-lines file. Deterministic: records
    /// appear in file order, and the returned map preserves insertion order.
    pub fn replay(path: &std::path::Path) -> std::io::Result<Vec<NarrationRecord>> {
        let text = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let rec: NarrationRecord = serde_json::from_str(line).map_err(std::io::Error::other)?;
            out.push(rec);
        }
        Ok(out)
    }
}

/// An LLM narrator that records to / replays from an artifact log.
///
/// In **produce** mode (`Self::produce(log_path)`), every narration is
/// appended to the log as a [`NarrationRecord`]. The narration itself comes
/// from the inner narrator (the extractive narrator as an LLM stand-in; a
/// real model at Stage 6/7).
///
/// In **replay** mode (`Self::replay(log_path)`), the log is loaded into a
/// lookup table; [`Narrator::narrate`] returns the pre-recorded narration,
/// making the gate's `rebuild(log)` byte-deterministic.
pub struct LlmNarrator {
    mode: LlmMode,
    model_name: String,
    version: u64,
}

enum LlmMode {
    /// Append narrations to this log file.
    Produce(std::path::PathBuf),
    /// Look up narrations from this pre-loaded map: candidate_id → record.
    Replay(Vec<NarrationRecord>),
}

impl LlmNarrator {
    /// Produce: narrate candidates via `inner` and append to `log_path`.
    /// The inner narrator provides the actual text; the log is the replay
    /// contract.
    pub fn produce(log_path: &std::path::Path) -> Self {
        LlmNarrator {
            mode: LlmMode::Produce(log_path.to_path_buf()),
            model_name: "extractive-standin".into(),
            version: 1,
        }
    }

    /// Replay: load the log and return pre-recorded narrations.
    /// Deterministic — same log → same output every call.
    pub fn replay(log_path: &std::path::Path) -> std::io::Result<Self> {
        let records = NarrationLog::replay(log_path)?;
        Ok(LlmNarrator {
            mode: LlmMode::Replay(records),
            model_name: "replay".into(),
            version: 1,
        })
    }
}

impl Narrator for LlmNarrator {
    fn narrate(&self, c: &InsightCandidate, input: &InsightInput) -> Option<NarratedInsight> {
        match &self.mode {
            LlmMode::Replay(records) => {
                let rec = records.iter().find(|r| r.candidate_id == c.id)?;
                let citations: Vec<EntryId> = rec.citations.clone();
                if citations.is_empty() || !citations.iter().all(|e| c.evidence.contains(e)) {
                    return None;
                }
                Some(NarratedInsight {
                    text: rec.text.clone(),
                    citations,
                })
            }
            LlmMode::Produce(log_path) => {
                // Use the extractive narrator as the LLM stand-in.
                let inner = ExtractiveNarrator;
                let ni = inner.narrate(c, input)?;
                let rec = NarrationRecord {
                    candidate_id: c.id,
                    text: ni.text.clone(),
                    citations: ni.citations.clone(),
                    model: self.model_name.clone(),
                    v: self.version,
                };
                // Best-effort append; a failed write is not a narration failure.
                let _ = NarrationLog::append(log_path, &rec);
                Some(ni)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// The full Stage-5 pipeline. Holds detectors, the validation config, and a
/// narrator. [`Self::discover`] is deterministic given the input and a fixed
/// narrator.
pub struct InsightEngine {
    detectors: Vec<Box<dyn Detector>>,
    cfg: ValidationCfg,
    narrator: Box<dyn Narrator>,
}

impl InsightEngine {
    /// The reference engine that **passes** the gate: the structural-bridge
    /// detector + the extractive narrator + the calibrated default config.
    pub fn structural() -> Self {
        InsightEngine {
            detectors: vec![Box::new(StructuralBridgeDetector)],
            cfg: ValidationCfg::default(),
            narrator: Box::new(ExtractiveNarrator),
        }
    }

    /// The temporal-lead engine — **only** the temporal detector, so
    /// `score_temporal` measures it cleanly (bundling other detectors counts
    /// their insights as temporal false positives). Production uses
    /// [`Self::full_produce`] to run every detector together.
    pub fn temporal() -> Self {
        InsightEngine {
            detectors: vec![Box::new(TemporalLeadDetector)],
            cfg: ValidationCfg {
                fdr_q: 0.01,
                ..ValidationCfg::default()
            },
            narrator: Box::new(ExtractiveNarrator),
        }
    }

    /// The contradiction engine — **only** the contradiction detector, so
    /// `score_contradiction` measures it cleanly.
    pub fn contradiction() -> Self {
        InsightEngine {
            detectors: vec![Box::new(ContradictionDetector)],
            cfg: ValidationCfg {
                fdr_q: 0.01,
                ..ValidationCfg::default()
            },
            narrator: Box::new(ExtractiveNarrator),
        }
    }

    /// The naive contrast that **fails** apophenia. Used only by the gate's
    /// "has teeth" column / test.
    pub fn naive() -> Self {
        InsightEngine {
            detectors: vec![Box::new(CoMentionDetector)],
            cfg: ValidationCfg::default(),
            narrator: Box::new(ExtractiveNarrator),
        }
    }

    pub fn with_parts(
        detectors: Vec<Box<dyn Detector>>,
        cfg: ValidationCfg,
        narrator: Box<dyn Narrator>,
    ) -> Self {
        InsightEngine {
            detectors,
            cfg,
            narrator,
        }
    }

    /// Produce: run the structural engine, recording narrations to `log_path`.
    /// Uses [`LlmNarrator::produce`] so every narrated insight is persisted
    /// for replay. The narration text comes from the extractive narrator (LLM
    /// stand-in; a real model goes here at Stage 6/7).
    pub fn structural_produce(log_path: &std::path::Path) -> Self {
        InsightEngine {
            detectors: vec![Box::new(StructuralBridgeDetector)],
            cfg: ValidationCfg::default(),
            narrator: Box::new(LlmNarrator::produce(log_path)),
        }
    }

    /// Produce: full engine (structural + temporal + contradiction) recording
    /// narrations to `log_path`.
    pub fn full_produce(log_path: &std::path::Path) -> Self {
        InsightEngine {
            detectors: vec![
                Box::new(StructuralBridgeDetector),
                Box::new(TemporalLeadDetector),
                Box::new(ContradictionDetector),
            ],
            cfg: ValidationCfg {
                fdr_q: 0.01,
                ..ValidationCfg::default()
            },
            narrator: Box::new(LlmNarrator::produce(log_path)),
        }
    }

    /// propose → validate (FDR) → narrate (cite-or-silence) → emit. Every
    /// emitted insight is validated, narrated, and **cited** (citations
    /// non-empty ∧ ⊆ the candidate's evidence); anything else is dropped, so
    /// the uncited-claims count is zero.
    pub fn discover(&self, input: &InsightInput) -> Vec<DiscoveredInsight> {
        let mut cands: Vec<InsightCandidate> = Vec::new();
        for d in &self.detectors {
            cands.extend(d.propose(input));
        }
        // Stable handle for narration order + evidence lookup.
        let by_id: BTreeMap<CandidateId, &InsightCandidate> =
            cands.iter().map(|c| (c.id, c)).collect();

        let accepted = validate_per_kind(&cands, &self.cfg);
        let mut out: Vec<DiscoveredInsight> = Vec::new();
        for id in accepted {
            let Some(&cand) = by_id.get(&id) else {
                continue;
            };
            let Some(n) = self.narrator.narrate(cand, input) else {
                continue; // silence
            };
            // Cite-or-silence enforcement: drop uncited / hallucinated citations.
            if n.citations.is_empty() || !n.citations.iter().all(|e| cand.evidence.contains(e)) {
                continue;
            }
            out.push(DiscoveredInsight {
                description: n.text,
                supporting_entries: n.citations,
                bridge_entity: cand.entities.first().copied(),
            });
        }
        out
    }
}

// ---------------------------------------------------------------------------
// T5 — ledger wiring
// ---------------------------------------------------------------------------

/// Record a [`skinki_ledger::Derivation`] for each surfaced insight, pinning
/// the insight to its evidence. When any evidence entry's text changes, its
/// content hash changes → [`skinki_ledger::Ledger::stale_closure`] flags the
/// insight as stale. This is the anti-hallucination keystone.
pub fn record_insight_derivations(
    insights: &[DiscoveredInsight],
    corpus: &Corpus,
    ledger: &mut Ledger,
    detector_id: u32,
    detector_version: u64,
) {
    for d in insights {
        let mut input_hashes: Vec<ContentHash> = d
            .supporting_entries
            .iter()
            .filter_map(|&eid| corpus.entries.get(eid as usize))
            .map(|e| ContentHash::of(e.text.as_bytes()))
            .collect();
        input_hashes.sort_by_key(|h| h.0);
        input_hashes.dedup_by(|a, b| a == b);
        if input_hashes.is_empty() {
            continue;
        }
        let output = ContentHash::of(
            format!(
                "insight|{}|{}|{:?}",
                d.description,
                d.bridge_entity.map_or(0, |e| e),
                d.supporting_entries
            )
            .as_bytes(),
        );
        ledger.record(Derivation::new(
            output,
            input_hashes,
            MethodStamp::new(detector_id, detector_version),
        ));
    }
}

// ---------------------------------------------------------------------------
// T6 — telemetry + Stage-4 Job
// ---------------------------------------------------------------------------

/// Rough resident-byte estimate, projected to 5M entries.
pub fn resident_bytes(discovered: &[DiscoveredInsight], n_entries: usize) -> usize {
    const HDR: usize = 24;
    let mut bytes = 0usize;
    for d in discovered {
        bytes += d.description.len() + HDR;
        bytes += d.supporting_entries.len() * std::mem::size_of::<EntryId>() + HDR;
        bytes += 1; // bridge_entity: Option<EntityId>
    }
    let scale = 5_000_000_f64 / (n_entries.max(1) as f64);
    (bytes as f64 * scale) as usize
}

/// Wraps insight discovery as a Stage-4 sleep [`Job`]: runs only on idle+power.
#[allow(dead_code)]
pub struct InsightJob {
    name: String,
    priority: u8,
    discovered: Vec<DiscoveredInsight>,
    done: bool,
}

#[allow(dead_code)]
impl InsightJob {
    pub fn new(name: &str, priority: u8, engine: &InsightEngine, input: &InsightInput) -> Self {
        let discovered = engine.discover(input);
        InsightJob {
            name: name.to_string(),
            priority,
            discovered,
            done: false,
        }
    }

    pub fn discovered(&self) -> &[DiscoveredInsight] {
        &self.discovered
    }
}

impl Job for InsightJob {
    fn id(&self) -> &str {
        &self.name
    }
    fn priority(&self) -> u8 {
        self.priority
    }
    fn step(&mut self, budget: StepBudget) -> StepOutcome {
        let total = self.discovered.len() as u64;
        if budget.max_items == 0 {
            return StepOutcome::Progress {
                done: (if self.done { total } else { 0 }),
                total,
            };
        }
        self.done = true;
        StepOutcome::Progress { done: total, total }
    }
    fn checkpoint(&self) -> Vec<u8> {
        vec![self.done as u8]
    }
    fn restore(&mut self, state: &[u8]) {
        self.done = state.first().copied() == Some(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skinki_corpus::{generate, Difficulty, GenConfig};
    use skinki_eval::score_insights;

    fn v2() -> Corpus {
        generate(&GenConfig {
            seed: 42,
            years: 5,
            entries_per_day: 6,
            difficulty: Difficulty::V2,
        })
    }

    #[test]
    fn bh_fdr_is_monotone_in_q() {
        let mk = |id: u64, p: f64| InsightCandidate {
            id,
            kind: InsightKind::StructuralBridge,
            entities: vec![id],
            evidence: vec![0],
            stat: Statistic {
                effect: 1.0,
                p_value: p,
                support: 5,
                surprise: 1.0,
            },
            claim: String::new(),
        };
        let cands: Vec<_> = (0..10).map(|i| mk(i, i as f64 * 0.01)).collect();
        let loose = ValidationCfg {
            fdr_q: 0.20,
            min_surprise: 0.0,
            min_support: 0,
        };
        let tight = ValidationCfg {
            fdr_q: 0.01,
            min_surprise: 0.0,
            min_support: 0,
        };
        let a = validate(&cands, &loose);
        let b = validate(&cands, &tight);
        // Looser q accepts a superset of tighter q.
        assert!(
            b.iter().all(|id| a.contains(id)),
            "BH must be monotone in q"
        );
        assert!(a.len() >= b.len());
    }

    #[test]
    fn binomial_upper_tail_basics() {
        // P(X>=0) = 1; P(X>=n) = q^n; monotonically non-increasing in k.
        assert!((binom_upper_tail(5, 0, 0.3) - 1.0).abs() < 1e-9);
        assert!((binom_upper_tail(4, 4, 0.5) - 0.5f64.powi(4)).abs() < 1e-9);
        let a = binom_upper_tail(10, 3, 0.2);
        let b = binom_upper_tail(10, 6, 0.2);
        assert!(a > b, "tail must shrink as k grows");
    }

    #[test]
    fn discover_is_deterministic() {
        let c = v2();
        let input = InsightInput::from_corpus(&c);
        let eng = InsightEngine::structural();
        let a = eng.discover(&input);
        let b = eng.discover(&input);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.bridge_entity, y.bridge_entity);
            assert_eq!(x.supporting_entries, y.supporting_entries);
            assert_eq!(x.description, y.description);
        }
    }

    #[test]
    fn every_emitted_insight_is_cited() {
        // The = 0 uncited budget, structurally: nothing the engine emits ever
        // lacks supporting_entries. Post-D0 the reference also surfaces > 0.
        let c = v2();
        let input = InsightInput::from_corpus(&c);
        let out = InsightEngine::structural().discover(&input);
        assert!(!out.is_empty(), "reference should surface the rare bridges");
        assert!(out.iter().all(|d| !d.supporting_entries.is_empty()));
    }

    #[test]
    fn reference_earns_recall_and_is_apophenia_safe_unlike_naive() {
        // The keystone, post-D0: the statistically-validated reference engine
        // recovers the planted bridges (recall) with NO apophenia hits, while the
        // naive co-mention baseline fires on every trap (precision collapses) —
        // proof the FDR/surprise validation is earned, not decorative.
        let c = v2();
        let input = InsightInput::from_corpus(&c);
        let planted = &c.ground_truth.insights;
        let neg = &c.ground_truth.negative_bridges;

        let r = score_insights(&InsightEngine::structural().discover(&input), planted, neg);
        let n = score_insights(&InsightEngine::naive().discover(&input), planted, neg);

        assert_eq!(r.negative_hits, 0, "reference must not certify apophenia");
        assert!(r.recall >= 0.50, "reference recall {} < 0.50", r.recall);
        assert!(
            r.precision.is_some_and(|p| p >= 0.70),
            "reference precision {:?} < 0.70",
            r.precision
        );
        assert!(
            n.negative_hits > 0,
            "naive baseline should fire on apophenia (teeth check)"
        );
    }

    #[test]
    fn uncited_narrations_are_dropped() {
        // A narrator that returns empty citations must never produce output.
        struct BadNarrator;
        impl Narrator for BadNarrator {
            fn narrate(&self, _c: &InsightCandidate, _i: &InsightInput) -> Option<NarratedInsight> {
                Some(NarratedInsight {
                    text: "uncited claim".into(),
                    citations: vec![],
                })
            }
        }
        let c = v2();
        let input = InsightInput::from_corpus(&c);
        let eng = InsightEngine::with_parts(
            vec![Box::new(StructuralBridgeDetector)],
            ValidationCfg::default(),
            Box::new(BadNarrator),
        );
        assert!(eng.discover(&input).is_empty(), "uncited must be dropped");
    }

    #[test]
    fn narration_log_round_trip() {
        let tmp = std::env::temp_dir().join("skinki_narr_log_t4.jsonl");
        let _ = std::fs::remove_file(&tmp);
        let rec = NarrationRecord {
            candidate_id: 42,
            text: "test narration".into(),
            citations: vec![1, 2],
            model: "test".into(),
            v: 1,
        };
        NarrationLog::append(&tmp, &rec).unwrap();
        let replayed = NarrationLog::replay(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0], rec);
    }

    #[test]
    fn llm_narrator_replay_is_deterministic() {
        let tmp = std::env::temp_dir().join("skinki_narr_replay_t4.jsonl");
        let _ = std::fs::remove_file(&tmp);

        let rec = NarrationRecord {
            candidate_id: 1,
            text: "replayed text".into(),
            citations: vec![10, 11],
            model: "test".into(),
            v: 1,
        };
        NarrationLog::append(&tmp, &rec).unwrap();

        let narrator = LlmNarrator::replay(&tmp).unwrap();
        let candidate = InsightCandidate {
            id: 1,
            kind: InsightKind::StructuralBridge,
            entities: vec![0],
            evidence: vec![10, 11, 12],
            stat: Statistic {
                effect: 1.0,
                p_value: 0.0,
                support: 3,
                surprise: 1.0,
            },
            claim: "original claim".into(),
        };
        let dummy = InsightInput {
            entries: &[],
            vocab: &[],
        };
        let n1 = narrator.narrate(&candidate, &dummy);
        let n2 = narrator.narrate(&candidate, &dummy);
        let _ = std::fs::remove_file(&tmp);
        let (n1, n2) = (n1.unwrap(), n2.unwrap());
        assert_eq!(n1.text, "replayed text");
        assert_eq!(n1.text, n2.text);
        assert_eq!(n1.citations, n2.citations);
    }
}

#[cfg(test)]
mod hardening_5c {
    use super::*;
    use skinki_corpus::{generate, Difficulty, GenConfig};

    fn v2(seed: u64) -> Corpus {
        generate(&GenConfig {
            seed,
            years: 5,
            entries_per_day: 6,
            difficulty: Difficulty::V2,
        })
    }

    #[test]
    fn contains_word_is_unicode_aware() {
        // ASCII behavior preserved: substrings inside words never match.
        assert!(!contains_word("i distrust this", "rust"));
        assert!(!contains_word("so frustrated today", "rust"));
        assert!(contains_word("rewrote it in rust!", "rust"));
        // Cyrillic: with byte-level boundaries every non-ASCII char looked
        // like a boundary, so "раст" would match inside "контраст".
        assert!(!contains_word("сплошной контраст", "раст"));
        assert!(!contains_word("новое растение", "раст"));
        assert!(contains_word("посадил раст в саду", "раст"));
        assert!(contains_word("раст — хороший выбор", "раст"));
    }

    #[test]
    fn candidate_ids_are_namespaced_by_kind() {
        // The same seq in different families must never collide (the bug that
        // let a full multi-detector engine shadow candidates in id-keyed maps).
        let a = candidate_id(InsightKind::StructuralBridge, 0);
        let b = candidate_id(InsightKind::TemporalLead, 0);
        let c = candidate_id(InsightKind::Contradiction, 0);
        assert!(a != b && b != c && a != c);
    }

    #[test]
    fn full_engine_surfaces_union_of_isolated_engines() {
        // A candidate must be judged only against its own family: running all
        // three detectors together yields exactly the union of running each
        // alone (same cfg), as (description, citations) multisets.
        let c = v2(42);
        let input = InsightInput::from_corpus(&c);
        let cfg = ValidationCfg {
            fdr_q: 0.01,
            ..ValidationCfg::default()
        };
        let single = |d: Box<dyn Detector>| {
            InsightEngine::with_parts(vec![d], cfg, Box::new(ExtractiveNarrator)).discover(&input)
        };
        let mut union: Vec<(String, Vec<EntryId>)> = Vec::new();
        for out in [
            single(Box::new(StructuralBridgeDetector)),
            single(Box::new(TemporalLeadDetector)),
            single(Box::new(ContradictionDetector)),
        ] {
            union.extend(
                out.into_iter()
                    .map(|d| (d.description, d.supporting_entries)),
            );
        }
        let full = InsightEngine::with_parts(
            vec![
                Box::new(StructuralBridgeDetector),
                Box::new(TemporalLeadDetector),
                Box::new(ContradictionDetector),
            ],
            cfg,
            Box::new(ExtractiveNarrator),
        )
        .discover(&input);
        let mut got: Vec<(String, Vec<EntryId>)> = full
            .into_iter()
            .map(|d| (d.description, d.supporting_entries))
            .collect();
        union.sort();
        got.sort();
        assert_eq!(got, union, "bundling detectors must not change the result");
    }

    #[test]
    fn per_family_validation_is_isolated() {
        // Contradiction candidates carry exact p = 0; pooled into one BH pass
        // they occupy the top ranks and RAISE the cutoff for the structural
        // family (accepting what per-family BH rejects). The structural
        // acceptance set must be identical with and without them present.
        let mk = |kind: InsightKind, seq: u64, p: f64| InsightCandidate {
            id: candidate_id(kind, seq),
            kind,
            entities: vec![seq],
            evidence: vec![0],
            stat: Statistic {
                effect: 1.0,
                p_value: p,
                support: 5,
                surprise: 1.0,
            },
            claim: String::new(),
        };
        let structural: Vec<InsightCandidate> = (0..4)
            .map(|i| mk(InsightKind::StructuralBridge, i, 0.02 + i as f64 * 0.05))
            .collect();
        let mut mixed = structural.clone();
        mixed.extend((0..4).map(|i| mk(InsightKind::Contradiction, i, 0.0)));

        let cfg = ValidationCfg {
            fdr_q: 0.05,
            min_surprise: 0.0,
            min_support: 0,
        };
        let alone: Vec<CandidateId> = validate_per_kind(&structural, &cfg);
        let with_zeros: Vec<CandidateId> = validate_per_kind(&mixed, &cfg)
            .into_iter()
            .filter(|id| structural.iter().any(|c| c.id == *id))
            .collect();
        assert_eq!(
            alone, with_zeros,
            "another family's p=0 must not move this family's BH cutoff"
        );
        // And the pooled validate() really does differ here — the reason
        // validate_per_kind exists.
        let pooled_structural: Vec<CandidateId> = validate(&mixed, &cfg)
            .into_iter()
            .filter(|id| structural.iter().any(|c| c.id == *id))
            .collect();
        assert_ne!(
            alone, pooled_structural,
            "test vector should demonstrate the pooling distortion"
        );
    }

    #[test]
    fn temporal_is_silent_on_day_shuffled_corpus() {
        // Null-corpus property: destroy every phase relation by shuffling the
        // day fields (a seeded permutation of the same multiset of days) and
        // the temporal detector must surface nothing.
        fn splitmix(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        for (corpus_seed, shuffle_seed) in [(42u64, 1u64), (7, 2), (42, 3)] {
            let c = v2(corpus_seed);
            let mut days: Vec<u32> = c.entries.iter().map(|e| e.day).collect();
            let mut s = 0xD1CE ^ shuffle_seed;
            for i in (1..days.len()).rev() {
                let j = (splitmix(&mut s) % (i as u64 + 1)) as usize;
                days.swap(i, j);
            }
            let shuffled: Vec<Entry> = c
                .entries
                .iter()
                .cloned()
                .zip(days)
                .map(|(mut e, d)| {
                    e.day = d;
                    e
                })
                .collect();
            let input = InsightInput {
                entries: &shuffled,
                vocab: &c.ground_truth.entities,
            };
            let out = InsightEngine::temporal().discover(&input);
            assert!(
                out.is_empty(),
                "corpus seed {corpus_seed} / shuffle {shuffle_seed}: temporal \
                 detector surfaced {} insights on a phase-destroyed corpus",
                out.len()
            );
        }
    }
}

#[cfg(test)]
mod t2_t3_precision {
    use super::*;
    use skinki_corpus::{generate, Difficulty, GenConfig};
    use skinki_eval::{score_contradiction, score_temporal};

    fn corpus(seed: u64) -> Corpus {
        generate(&GenConfig {
            seed,
            years: 5,
            entries_per_day: 6,
            difficulty: Difficulty::V2,
        })
    }

    #[test]
    fn temporal_meets_recall_without_false_insights() {
        // T2 keystone: recover planted lead→trail patterns (recall >= 0.50) with
        // the hard false-insight bar (< 0.05). The density-corrected binomial
        // null + word-boundary matching are what keep ubiquitous-entity and
        // rare-lead→common-trail coincidences out.
        for seed in [42, 7] {
            let c = corpus(seed);
            let input = InsightInput::from_corpus(&c);
            let out = InsightEngine::temporal().discover(&input);
            let s = score_temporal(&out, &c.ground_truth.temporal);
            assert!(
                s.recall >= 0.50,
                "seed {seed}: temporal recall {}",
                s.recall
            );
            assert!(
                s.false_insight_rate.is_some_and(|f| f < 0.05),
                "seed {seed}: temporal false-insight {:?}",
                s.false_insight_rate
            );
        }
    }

    #[test]
    fn contradiction_meets_recall_without_false_insights() {
        // T3 keystone: recover planted reversals (recall >= 0.80) at < 0.05
        // false-insight. Name-anchored stance attribution (not "name + any cue")
        // is what stops the replacement entity Y being flagged.
        for seed in [42, 7] {
            let c = corpus(seed);
            let input = InsightInput::from_corpus(&c);
            let out = InsightEngine::contradiction().discover(&input);
            let s = score_contradiction(&out, &c.ground_truth.contradictions);
            assert!(
                s.recall >= 0.80,
                "seed {seed}: contradiction recall {}",
                s.recall
            );
            assert!(
                s.false_insight_rate.is_some_and(|f| f < 0.05),
                "seed {seed}: contradiction false-insight {:?}",
                s.false_insight_rate
            );
        }
    }
}
