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

use std::collections::BTreeMap;

use skinki_corpus::{Corpus, Entity, EntityId, Entry, EntryId};
use skinki_eval::DiscoveredInsight;

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
                id: p.id,
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
                id: p.id,
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

        let accepted = validate(&cands, &self.cfg);
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
}
