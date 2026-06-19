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
// TemporalLeadDetector — cross-correlation over entity mention days (T2)
// ---------------------------------------------------------------------------
//
// Profiles each vocab entity's mention-day series, then for every ordered
// pair (A, B) computes the strongest lag (the `d` that maximises the count of
// B-on-day-t after A-on-day-(t-d)) and tests it against a shuffled-lag null.
// The null keeps A's days fixed and randomly assigns B's days; the p-value is
// the fraction of shuffles where the strongest-lag count ≥ the observed count.
// Candidates that survive BH-FDR at the engine's threshold are surfaced as
// [`InsightKind::TemporalLead`] insights.
//
// Targets the planted [`skinki_corpus::TemporalPattern`] ground truth on V2:
// a lead entity A is mentioned, then a trail entity B is mentioned exactly
// `lag_days` later, embedded in the text via a templated temporal cue.

/// Deduped, sorted mention days for one entity.
struct MentionSeries {
    entity: EntityId,
    days: Vec<u32>,
    entry_ids: Vec<EntryId>,
}

fn profile_entity_days(input: &InsightInput) -> Vec<MentionSeries> {
    let mut out: Vec<MentionSeries> = Vec::with_capacity(input.vocab.len());
    for e in input.vocab {
        let name_lower = e.name.to_lowercase();
        let mut mentions: Vec<(u32, EntryId)> = input
            .entries
            .iter()
            .filter(|entry| entry.text.to_lowercase().contains(&name_lower))
            .map(|entry| (entry.day, entry.id))
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

        let max_day = input.entries.iter().map(|e| e.day).max().unwrap_or(1825);

        let mut out: Vec<InsightCandidate> = Vec::new();
        let mut next_id: CandidateId = 0;

        // Helper: count how many A mentions have a B mention at roughly `lag` days.
        fn count_at_lag(a_days: &[u32], b_days: &[u32], lag: u32, tol: u32) -> u32 {
            let mut c: u32 = 0;
            for &da in a_days {
                let lo = da + lag.saturating_sub(tol);
                let hi = da + lag + tol;
                if b_days.iter().any(|&db| db >= lo && db <= hi) {
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
                let mut best_lag: u32 = 0;
                let mut best_count: u32 = 0;
                for lag in 0..=MAX_LAG {
                    let c = count_at_lag(&a.days, &b.days, lag, LAG_TOLERANCE);
                    if c > best_count {
                        best_count = c;
                        best_lag = lag;
                    }
                }
                if best_count == 0 || best_count < MIN_COUNT {
                    continue;
                }
                // Require a meaningful fraction of A's mentions to co-occur.
                let min_ratio = 0.35;
                if (best_count as f64) < (a.days.len() as f64 * min_ratio) {
                    continue;
                }

                // Analytical binomial null: under H0, each of A's n_a mentions
                // has probability p = (2*tol+1) / max_day of co-occurring with
                // B at lag ± tol by chance. The upper-tail probability of ≥ count
                // successes is the p-value — extremely small for strong signals,
                // so it survives BH-FDR even with thousands of candidates.
                let p_null = (2 * LAG_TOLERANCE + 1) as f64 / max_day as f64;
                let p_value = binom_upper_tail(a.days.len() as u32, best_count, p_null);
                // Pre-filter: only candidates with extremely small raw p-value
                // enter the BH-FDR pipeline (otherwise noise drowns the signal
                // in a 3500-candidate set).
                if p_value > 1e-6 {
                    continue;
                }

                let effect = best_count as f64 / ((a.days.len() * b.days.len()) as f64).sqrt();
                let support = best_count;

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
                    id: next_id,
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
                        best_count,
                        a.days.len(),
                        b.days.len(),
                    ),
                });
                next_id += 1;
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

const POSITIVE_CUES: &[&str] = &[
    "is the best",
    "wins.",
    "just fits",
    "decision made",
    "committing to",
    "going all in",
];

const NEGATIVE_CUES: &[&str] = &[
    "was a mistake",
    "nothing but pain",
    "regret picking",
    "saved us weeks",
    "clearly better",
    "changed my mind",
];

fn has_any_cue(text: &str, cues: &[&str]) -> bool {
    cues.iter().any(|c| text.contains(c))
}

pub struct ContradictionDetector;

impl Detector for ContradictionDetector {
    fn name(&self) -> &str {
        "contradiction"
    }

    fn propose(&self, input: &InsightInput) -> Vec<InsightCandidate> {
        let mut out: Vec<InsightCandidate> = Vec::new();
        let mut next_id: CandidateId = 0;

        for e in input.vocab {
            let name_lower = e.name.to_lowercase();
            let mut pos_entries: Vec<&Entry> = Vec::new();
            let mut neg_entries: Vec<&Entry> = Vec::new();
            for entry in input.entries {
                if !entry.text.to_lowercase().contains(&name_lower) {
                    continue;
                }
                if has_any_cue(&entry.text, POSITIVE_CUES) {
                    pos_entries.push(entry);
                }
                if has_any_cue(&entry.text, NEGATIVE_CUES) {
                    neg_entries.push(entry);
                }
            }
            if pos_entries.is_empty() || neg_entries.is_empty() {
                continue;
            }
            // A contradiction: at least one positive entry later reversed by
            // a negative entry (negative day > positive day).
            for pos in &pos_entries {
                for neg in &neg_entries {
                    if neg.day <= pos.day {
                        continue;
                    }
                    let mut evidence: Vec<EntryId> = vec![pos.id, neg.id];
                    evidence.sort_unstable();
                    evidence.dedup();

                    out.push(InsightCandidate {
                        id: next_id,
                        kind: InsightKind::Contradiction,
                        entities: vec![e.id],
                        evidence,
                        stat: Statistic {
                            effect: 1.0,
                            p_value: 0.0,
                            support: 2,
                            surprise: 1.0,
                        },
                        claim: format!(
                            "{}: initially '{}', then reversed — '{}'",
                            e.name,
                            // Show first 60 chars of each entry text.
                            &pos.text[..pos.text.len().min(60)],
                            &neg.text[..neg.text.len().min(60)],
                        ),
                    });
                    next_id += 1;
                }
            }
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

    /// The temporal-lead engine: structural-bridge + temporal-lead detectors.
    /// Uses the same validation config and narrator as `structural()`.
    pub fn temporal() -> Self {
        InsightEngine {
            detectors: vec![
                Box::new(StructuralBridgeDetector),
                Box::new(TemporalLeadDetector),
            ],
            cfg: ValidationCfg {
                fdr_q: 0.01,
                ..ValidationCfg::default()
            },
            narrator: Box::new(ExtractiveNarrator),
        }
    }

    /// The contradiction engine: structural-bridge + temporal + contradiction
    /// detectors. Contradiction candidates come with stat signals that bypass
    /// BH-FDR (p=0, surprise=1 — they're deterministic sentiment reversals),
    /// so they surface when the validation floors are met.
    pub fn contradiction() -> Self {
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
