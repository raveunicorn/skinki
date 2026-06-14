#![forbid(unsafe_code)]
//! Evaluation harness for the Kortex memory engine.
//!
//! Defines the [`RetrievalSystem`] interface that every "system under test"
//! implements, the metric primitives (recall@k, precision@k, nDCG@k,
//! answer-in-top-k, and insight precision / false-insight rate), and the
//! [`Report`] type that ties scores together with latency/memory telemetry.
//!
//! Metric primitives are pure functions; orchestration (timing, iterating
//! queries) lives in the harness so this crate stays dependency-light.

use kortex_corpus::{Corpus, EntityId, EntryId, InsightBridge, NegativeBridge};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// System under test
// ---------------------------------------------------------------------------

/// A candidate connection the system believes is non-obvious and grounded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredInsight {
    pub description: String,
    /// Entries the system cites as evidence (provenance).
    pub supporting_entries: Vec<EntryId>,
    /// Optional guess at the bridging entity.
    pub bridge_entity: Option<EntityId>,
}

/// The interface every memory backend implements so the harness can score it.
///
/// `search` is mandatory; `answer` and `discover_insights` are optional and
/// default to "not supported" so a plain lexical baseline can opt out.
pub trait RetrievalSystem {
    fn name(&self) -> &str;

    /// Build whatever indexes the system needs from the corpus.
    fn index(&mut self, corpus: &Corpus);

    /// Return up to `k` entry ids most relevant to `query`, best first.
    fn search(&self, query: &str, k: usize) -> Vec<EntryId>;

    /// Optional: a direct answer string (QA). Defaults to none.
    fn answer(&self, _question: &str) -> Option<String> {
        None
    }

    /// Optional: proactively surfaced non-obvious connections. Defaults to none.
    fn discover_insights(&self) -> Vec<DiscoveredInsight> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Metric primitives
// ---------------------------------------------------------------------------

fn hits(retrieved: &[EntryId], relevant: &[EntryId], k: usize) -> usize {
    retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count()
}

/// Fraction of relevant items found within the top `k`.
pub fn recall_at_k(retrieved: &[EntryId], relevant: &[EntryId], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    hits(retrieved, relevant, k) as f64 / relevant.len() as f64
}

/// Fraction of the top `k` that are relevant.
pub fn precision_at_k(retrieved: &[EntryId], relevant: &[EntryId], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    hits(retrieved, relevant, k) as f64 / k as f64
}

/// Normalized discounted cumulative gain at `k` (binary relevance).
pub fn ndcg_at_k(retrieved: &[EntryId], relevant: &[EntryId], k: usize) -> f64 {
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            if relevant.contains(id) {
                1.0 / ((i + 2) as f64).log2()
            } else {
                0.0
            }
        })
        .sum();
    let ideal_hits = relevant.len().min(k);
    let idcg: f64 = (0..ideal_hits).map(|i| 1.0 / ((i + 2) as f64).log2()).sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Whether the gold answer text appears in any of the retrieved entries
/// (a retrieval-grounded proxy for "could the model answer from this context").
pub fn answer_in_entries(corpus: &Corpus, retrieved: &[EntryId], answer: &str) -> bool {
    let needle = answer.to_lowercase();
    retrieved.iter().any(|&id| {
        corpus
            .entry_text(id)
            .map(|t| t.to_lowercase().contains(&needle))
            .unwrap_or(false)
    })
}

fn jaccard(a: &[EntryId], b: &[EntryId]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.iter().filter(|x| b.contains(x)).count();
    let union = a.len() + b.iter().filter(|x| !a.contains(x)).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Does a discovered insight match a planted one? Either the bridge entity
/// matches, or the cited evidence overlaps the planted support enough.
fn insight_matches(d: &DiscoveredInsight, planted: &InsightBridge) -> bool {
    if let Some(be) = d.bridge_entity {
        if be == planted.bridge_entity {
            return true;
        }
    }
    jaccard(&d.supporting_entries, &planted.supporting_entries) >= 0.3
}

/// Does a discovered insight fall into a planted apophenia trap? Same matching
/// rule as for positives, against the trap's entity/entries.
fn negative_matches(d: &DiscoveredInsight, neg: &NegativeBridge) -> bool {
    if let Some(be) = d.bridge_entity {
        if be == neg.entity {
            return true;
        }
    }
    jaccard(&d.supporting_entries, &neg.entries) >= 0.3
}

/// Score surfaced insights against the planted ground truth. This is the
/// keystone anti-hallucination metric: we want high recall of real insights
/// AND a low false-insight rate (no apophenia).
///
/// `negatives` are V2 apophenia traps (hub entities spanning many clusters).
/// A surfaced insight that matches a trap — and no positive — is a *certified*
/// false insight (`negative_hits`); unlike generic false positives, these are
/// links a naive co-occurrence detector is guaranteed to fire on.
pub fn score_insights(
    discovered: &[DiscoveredInsight],
    planted: &[InsightBridge],
    negatives: &[NegativeBridge],
) -> InsightScores {
    let surfaced = discovered.len();
    let true_positives = discovered
        .iter()
        .filter(|d| planted.iter().any(|p| insight_matches(d, p)))
        .count();
    let matched_planted = planted
        .iter()
        .filter(|p| discovered.iter().any(|d| insight_matches(d, p)))
        .count();
    let negative_hits = discovered
        .iter()
        .filter(|d| {
            !planted.iter().any(|p| insight_matches(d, p))
                && negatives.iter().any(|n| negative_matches(d, n))
        })
        .count();

    let precision = if surfaced == 0 {
        None
    } else {
        Some(true_positives as f64 / surfaced as f64)
    };
    let false_insight_rate = if surfaced == 0 {
        None
    } else {
        Some((surfaced - true_positives) as f64 / surfaced as f64)
    };
    let recall = if planted.is_empty() {
        0.0
    } else {
        matched_planted as f64 / planted.len() as f64
    };

    InsightScores {
        planted: planted.len(),
        surfaced,
        matched: matched_planted,
        precision,
        recall,
        false_insight_rate,
        negative_hits,
    }
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalScores {
    pub queries: usize,
    pub k: usize,
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub ndcg_at_k: f64,
    pub answer_in_topk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightScores {
    pub planted: usize,
    pub surfaced: usize,
    pub matched: usize,
    pub precision: Option<f64>,
    pub recall: f64,
    pub false_insight_rate: Option<f64>,
    /// Surfaced insights that hit a planted apophenia trap (V2 corpora).
    #[serde(default)]
    pub negative_hits: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Latency {
    pub count: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub system: String,
    pub corpus_entries: usize,
    pub recall: RetrievalScores,
    pub multi_hop: RetrievalScores,
    pub insight: InsightScores,
    pub latency: Option<Latency>,
    pub peak_rss_bytes: Option<u64>,
}

fn opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{:.3}", x),
        None => "n/a".to_string(),
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Kortex Stage 0 — Eval Report ===")?;
        writeln!(f, "system           : {}", self.system)?;
        writeln!(f, "corpus entries   : {}", self.corpus_entries)?;
        writeln!(
            f,
            "recall@{:<3}       : R={:.3}  P={:.3}  nDCG={:.3}  answer-in-topk={:.3}  (n={})",
            self.recall.k,
            self.recall.recall_at_k,
            self.recall.precision_at_k,
            self.recall.ndcg_at_k,
            self.recall.answer_in_topk,
            self.recall.queries
        )?;
        writeln!(
            f,
            "multi-hop@{:<3}    : R={:.3}  P={:.3}  nDCG={:.3}  answer-in-topk={:.3}  (n={})",
            self.multi_hop.k,
            self.multi_hop.recall_at_k,
            self.multi_hop.precision_at_k,
            self.multi_hop.ndcg_at_k,
            self.multi_hop.answer_in_topk,
            self.multi_hop.queries
        )?;
        writeln!(
            f,
            "insight          : recall={:.3}  precision={}  false-rate={}  neg-hits={}  (planted={}, surfaced={}, matched={})",
            self.insight.recall,
            opt(self.insight.precision),
            opt(self.insight.false_insight_rate),
            self.insight.negative_hits,
            self.insight.planted,
            self.insight.surfaced,
            self.insight.matched
        )?;
        if let Some(l) = &self.latency {
            writeln!(
                f,
                "query latency    : p50={:.3}ms  p95={:.3}ms  mean={:.3}ms  max={:.3}ms  (n={})",
                l.p50_ms, l.p95_ms, l.mean_ms, l.max_ms, l.count
            )?;
        }
        if let Some(rss) = self.peak_rss_bytes {
            writeln!(f, "peak RSS         : {:.1} MB", rss as f64 / 1_048_576.0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_and_precision_basics() {
        let retrieved = vec![1u64, 2, 3, 4, 5];
        let relevant = vec![2u64, 4];
        assert_eq!(recall_at_k(&retrieved, &relevant, 5), 1.0);
        assert_eq!(recall_at_k(&retrieved, &relevant, 1), 0.0);
        assert_eq!(precision_at_k(&retrieved, &relevant, 2), 0.5);
    }

    #[test]
    fn ndcg_rewards_higher_ranks() {
        let top = vec![2u64, 9, 9];
        let low = vec![9u64, 9, 2];
        let relevant = vec![2u64];
        assert!(ndcg_at_k(&top, &relevant, 3) > ndcg_at_k(&low, &relevant, 3));
        assert!((ndcg_at_k(&top, &relevant, 3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn insight_scoring_handles_empty() {
        let scores = score_insights(&[], &[], &[]);
        assert_eq!(scores.surfaced, 0);
        assert_eq!(scores.recall, 0.0);
        assert!(scores.precision.is_none());
        assert_eq!(scores.negative_hits, 0);
    }

    #[test]
    fn insight_scoring_counts_apophenia_traps() {
        let planted = vec![InsightBridge {
            id: 0,
            bridge_entity: 1,
            cluster_a: "work".into(),
            cluster_b: "music".into(),
            description: String::new(),
            supporting_entries: vec![10, 11, 12],
            surprise: 0.8,
        }];
        let negatives = vec![NegativeBridge {
            id: 0,
            entity: 7,
            clusters: vec![
                "work".into(),
                "health".into(),
                "travel".into(),
                "music".into(),
            ],
            entries: vec![20, 21, 22],
        }];
        let discovered = vec![
            // A real hit: matches the planted bridge entity.
            DiscoveredInsight {
                description: String::new(),
                supporting_entries: vec![10, 11],
                bridge_entity: Some(1),
            },
            // Apophenia: fires on the hub entity — certified false.
            DiscoveredInsight {
                description: String::new(),
                supporting_entries: vec![20, 21],
                bridge_entity: Some(7),
            },
            // Generic false positive: matches neither positives nor traps.
            DiscoveredInsight {
                description: String::new(),
                supporting_entries: vec![900],
                bridge_entity: None,
            },
        ];
        let s = score_insights(&discovered, &planted, &negatives);
        assert_eq!(s.matched, 1);
        assert_eq!(s.negative_hits, 1);
        assert_eq!(s.surfaced, 3);
        assert!((s.false_insight_rate.unwrap() - 2.0 / 3.0).abs() < 1e-9);
    }
}
