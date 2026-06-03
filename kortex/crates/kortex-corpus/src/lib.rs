#![forbid(unsafe_code)]
//! Deterministic synthetic "years of thoughts" generator.
//!
//! The whole point of Stage 0 is a *measuring stick*. We cannot benchmark a
//! memory engine without ground truth, and real journals don't come labeled.
//! So we synthesize a multi-year stream of journal/voice entries and, while
//! doing so, deliberately *plant* five phenomena with machine-checkable answers:
//!
//! 1. Recall facts        — a single entry states a fact; can we retrieve it?
//! 2. Multi-hop chains     — the answer requires joining two distant entries.
//! 3. Temporal patterns    — entity A's mentions lead event B by a fixed lag.
//! 4. Contradictions       — a belief stated, then reversed later in time.
//! 5. Insight bridges      — a rare entity secretly links two unrelated clusters.
//!
//! Generation is driven by a hand-rolled SplitMix64 PRNG so the corpus (and thus
//! the ground truth) is byte-reproducible across platforms and CI — no `rand`
//! version drift, no hidden nondeterminism.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type EntryId = u64;
pub type EntityId = u64;

// ---------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64) — tiny, no_std-friendly, fully reproducible.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish integer in `[0, n)`. Modulo bias is irrelevant for synthetic data.
    fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        debug_assert!(hi > lo);
        lo + (self.next_u64() % (hi - lo) as u64) as u32
    }

    fn chance(&mut self, p: f64) -> bool {
        (self.next_u64() as f64) / (u64::MAX as f64) < p
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

// ---------------------------------------------------------------------------
// Civil date (Howard Hinnant's algorithm) — avoids a chrono dependency.
// ---------------------------------------------------------------------------

/// Days from 1970-01-01 to 2018-01-01 (corpus epoch).
const BASE_DAYS: i64 = 17_532;

fn date_string(day_offset: u32) -> String {
    let (y, m, d) = civil_from_days(BASE_DAYS + day_offset as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    Text,
    Voice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub id: EntryId,
    /// Day offset from the corpus epoch.
    pub day: u32,
    pub date: String,
    pub kind: EntryKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    Person,
    Book,
    Project,
    Topic,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub kind: EntityKind,
    pub cluster: String,
}

/// A factual claim stated in a single entry. Drives recall QA.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecallQuery {
    pub id: usize,
    pub question: String,
    pub answer: String,
    /// Entries that contain the answer (relevance judgments for recall@k / nDCG).
    pub relevant_entries: Vec<EntryId>,
}

/// A question whose answer requires joining two distant entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiHopQuery {
    pub id: usize,
    pub question: String,
    pub answer: String,
    /// The chain of entries that must be combined to answer.
    pub hop_entries: Vec<EntryId>,
}

/// Entity `leading` tends to be mentioned ~`lag_days` before event `trailing`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub id: usize,
    pub leading: EntityId,
    pub trailing: EntityId,
    pub lag_days: u32,
    pub description: String,
    pub lead_entries: Vec<EntryId>,
    pub trail_entries: Vec<EntryId>,
}

/// A belief stated, then reversed later.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contradiction {
    pub id: usize,
    pub topic: String,
    pub entry_before: EntryId,
    pub entry_after: EntryId,
}

/// A rare entity that bridges two otherwise-disconnected clusters. This is the
/// "fourth-dimension" needle the Insight Engine must surface (Stage 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InsightBridge {
    pub id: usize,
    pub bridge_entity: EntityId,
    pub cluster_a: String,
    pub cluster_b: String,
    pub description: String,
    pub supporting_entries: Vec<EntryId>,
    /// How surprising the link is (0..1); higher = more disconnected clusters.
    pub surprise: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GroundTruth {
    pub entities: Vec<Entity>,
    pub recall: Vec<RecallQuery>,
    pub multi_hop: Vec<MultiHopQuery>,
    pub temporal: Vec<TemporalPattern>,
    pub contradictions: Vec<Contradiction>,
    pub insights: Vec<InsightBridge>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusMeta {
    pub seed: u64,
    pub years: u32,
    pub num_entries: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Corpus {
    pub meta: CorpusMeta,
    pub entries: Vec<Entry>,
    pub ground_truth: GroundTruth,
}

impl Corpus {
    pub fn entry_text(&self, id: EntryId) -> Option<&str> {
        // Entries are stored in id order, so this is O(1).
        self.entries.get(id as usize).map(|e| e.text.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct GenConfig {
    pub seed: u64,
    pub years: u32,
    /// Average routine entries per day. Crank this to stress-test "years of
    /// thoughts" scale (e.g. ~270/day over 10 years approximates ~1M entries).
    pub entries_per_day: u32,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            seed: 42,
            years: 5,
            entries_per_day: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

const PEOPLE: &[&str] = &[
    "Anna", "Marcus", "Lena", "Pavel", "Sofia", "Dmitry", "Clara", "Yuki", "Omar", "Nina",
    "Viktor", "Mara", "Ivan", "Stella", "Hugo",
];
const BOOKS: &[&str] = &[
    "Deep Work",
    "Sapiens",
    "The Pragmatic Programmer",
    "Dune",
    "Meditations",
    "Thinking Fast and Slow",
    "Antifragile",
    "The Beginning of Infinity",
    "Gödel Escher Bach",
    "The Mythical Man-Month",
];
const PROJECTS: &[&str] = &[
    "Project Aurora",
    "the billing refactor",
    "the mobile rewrite",
    "Project Helix",
    "the search migration",
];
const TOOLS: &[&str] = &[
    "Postgres", "Redis", "SwiftUI", "Rust", "Kafka", "Sqlite", "Datalog",
];
const FEELINGS: &[&str] = &[
    "optimistic",
    "frustrated",
    "calm",
    "tired",
    "curious",
    "restless",
];

const CLUSTERS: &[&str] = &["work", "health", "music", "travel", "reading"];

const TOPICS_BY_CLUSTER: &[(&str, &[&str])] = &[
    (
        "work",
        &[
            "distributed systems",
            "code review",
            "on-call",
            "latency budgets",
        ],
    ),
    (
        "health",
        &["trail running", "sleep", "strength training", "nutrition"],
    ),
    (
        "music",
        &[
            "jazz harmony",
            "the band rehearsal",
            "a new synth",
            "songwriting",
        ],
    ),
    (
        "travel",
        &[
            "the Lisbon trip",
            "train schedules",
            "a hike in the Alps",
            "visa paperwork",
        ],
    ),
    (
        "reading",
        &[
            "habit formation",
            "memory consolidation",
            "stoicism",
            "complexity theory",
        ],
    ),
];

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Tag {
    Recall(usize),
    MultiHopA(usize),
    MultiHopB(usize),
    TemporalLead(usize),
    TemporalTrail(usize),
    ContradictionBefore(usize),
    ContradictionAfter(usize),
    Insight(usize),
}

struct RawEntry {
    day: u32,
    kind: EntryKind,
    text: String,
    tag: Option<Tag>,
}

struct RecallParams {
    person: EntityId,
    book: EntityId,
}
struct MultiHopParams {
    p: EntityId,
    book: EntityId,
}
struct TemporalParams {
    a: EntityId,
    b: EntityId,
    lag: u32,
}
struct ContradictionParams {
    topic: String,
}
struct InsightParams {
    e: EntityId,
    cluster_a: String,
    cluster_b: String,
}

struct Generator {
    rng: Rng,
    total_days: u32,
    entries_per_day: u32,
    raw: Vec<RawEntry>,
    entities: Vec<Entity>,
    interner: HashMap<String, EntityId>,
    recall_params: Vec<RecallParams>,
    multihop_params: Vec<MultiHopParams>,
    temporal_params: Vec<TemporalParams>,
    contradiction_params: Vec<ContradictionParams>,
    insight_params: Vec<InsightParams>,
}

impl Generator {
    fn intern(&mut self, name: &str, kind: EntityKind, cluster: &str) -> EntityId {
        if let Some(&id) = self.interner.get(name) {
            return id;
        }
        let id = self.entities.len() as EntityId;
        self.entities.push(Entity {
            id,
            name: name.to_string(),
            kind,
            cluster: cluster.to_string(),
        });
        self.interner.insert(name.to_string(), id);
        id
    }

    fn name_of(&self, id: EntityId) -> &str {
        &self.entities[id as usize].name
    }

    fn push(&mut self, day: u32, kind: EntryKind, text: String, tag: Option<Tag>) {
        self.raw.push(RawEntry {
            day,
            kind,
            text,
            tag,
        });
    }

    fn kind(&mut self) -> EntryKind {
        if self.rng.chance(0.30) {
            EntryKind::Voice
        } else {
            EntryKind::Text
        }
    }

    fn routine_day(&mut self, day: u32) {
        // 0..=2*epd entries, averaging `entries_per_day`.
        let n = self.rng.below((2 * self.entries_per_day + 1) as usize);
        for _ in 0..n {
            let cluster = *self.rng.pick(CLUSTERS);
            let topics = TOPICS_BY_CLUSTER
                .iter()
                .find(|(c, _)| *c == cluster)
                .map(|(_, t)| *t)
                .unwrap_or(&[]);
            let topic = (*self.rng.pick(topics)).to_string();
            let feeling = (*self.rng.pick(FEELINGS)).to_string();
            let person = (*self.rng.pick(PEOPLE)).to_string();
            let text = match self.rng.below(5) {
                0 => format!(
                    "Spent the morning on {}. Feeling {feeling} about {topic}.",
                    self.rng.pick(PROJECTS)
                ),
                1 => format!("Talked with {person} about {topic} today."),
                2 => format!("Reading more on {topic}. {feeling} about where it leads."),
                3 => format!(
                    "Tried {} for {topic}. Mixed results, a bit {feeling}.",
                    self.rng.pick(TOOLS)
                ),
                _ => format!("Quiet day. Thought about {topic} on a walk, felt {feeling}."),
            };
            let kind = self.kind();
            self.push(day, kind, text, None);
        }
    }

    fn plan_recall(&mut self, count: usize) {
        for _ in 0..count {
            let person_name = (*self.rng.pick(PEOPLE)).to_string();
            let book_name = (*self.rng.pick(BOOKS)).to_string();
            let person = self.intern(&person_name, EntityKind::Person, "reading");
            let book = self.intern(&book_name, EntityKind::Book, "reading");
            let qid = self.recall_params.len();
            self.recall_params.push(RecallParams { person, book });
            let day = self.rng.range(0, self.total_days);
            let text = format!(
                "{person_name} recommended the book {book_name} to me today. Want to remember that."
            );
            let kind = self.kind();
            self.push(day, kind, text, Some(Tag::Recall(qid)));
        }
    }

    fn plan_multihop(&mut self, count: usize) {
        for _ in 0..count {
            let p_name = (*self.rng.pick(PEOPLE)).to_string();
            let mut q_name = (*self.rng.pick(PEOPLE)).to_string();
            while q_name == p_name {
                q_name = (*self.rng.pick(PEOPLE)).to_string();
            }
            let book_name = (*self.rng.pick(BOOKS)).to_string();
            let p = self.intern(&p_name, EntityKind::Person, "social");
            self.intern(&q_name, EntityKind::Person, "social"); // register Q
            let book = self.intern(&book_name, EntityKind::Book, "reading");
            let plan = self.multihop_params.len();
            self.multihop_params.push(MultiHopParams { p, book });

            let day1 = self.rng.range(0, self.total_days.saturating_sub(2).max(1));
            let day2 = self
                .rng
                .range(day1 + 1, (day1 + 60).min(self.total_days).max(day1 + 2));
            let t1 = format!("{p_name} introduced me to {q_name} at the meetup.");
            let t2 = format!("{q_name} recommended the book {book_name}. Noted.");
            let k1 = self.kind();
            let k2 = self.kind();
            self.push(day1, k1, t1, Some(Tag::MultiHopA(plan)));
            self.push(day2, k2, t2, Some(Tag::MultiHopB(plan)));
        }
    }

    fn plan_temporal(&mut self, count: usize) {
        let leads = ["caffeine", "late-night coding", "skipped workouts"];
        let trails = ["a migraine", "bad sleep", "an anxious morning"];
        for _ in 0..count {
            let lead_name = (*self.rng.pick(&leads)).to_string();
            let trail_name = (*self.rng.pick(&trails)).to_string();
            let a = self.intern(&lead_name, EntityKind::Topic, "health");
            let b = self.intern(&trail_name, EntityKind::Topic, "health");
            let lag = self.rng.range(2, 7);
            let plan = self.temporal_params.len();
            self.temporal_params.push(TemporalParams { a, b, lag });

            // Several lead mentions, each followed `lag` days later by a trail event.
            let occurrences = self.rng.range(5, 9);
            for _ in 0..occurrences {
                let start = self
                    .rng
                    .range(0, self.total_days.saturating_sub(lag + 1).max(1));
                let lead_text = format!("Lots of {lead_name} again today.");
                let kl = self.kind();
                self.push(start, kl, lead_text, Some(Tag::TemporalLead(plan)));
                let trail_text = format!("Woke up with {trail_name}. Rough.");
                let kt = self.kind();
                self.push(start + lag, kt, trail_text, Some(Tag::TemporalTrail(plan)));
            }
        }
    }

    fn plan_contradictions(&mut self, count: usize) {
        for _ in 0..count {
            let x_name = (*self.rng.pick(TOOLS)).to_string();
            let mut y_name = (*self.rng.pick(TOOLS)).to_string();
            while y_name == x_name {
                y_name = (*self.rng.pick(TOOLS)).to_string();
            }
            self.intern(&x_name, EntityKind::Tool, "work"); // register X
            self.intern(&y_name, EntityKind::Tool, "work"); // register Y
            let plan = self.contradiction_params.len();
            self.contradiction_params.push(ContradictionParams {
                topic: x_name.clone(),
            });
            let day1 = self.rng.range(0, self.total_days.saturating_sub(2).max(1));
            let day2 = self.rng.range(day1 + 1, self.total_days.max(day1 + 2));
            let t1 = format!("Convinced {x_name} is the best choice. Going all in on it.");
            let t2 =
                format!("Changed my mind: {x_name} was a mistake. {y_name} is clearly better.");
            let k1 = self.kind();
            let k2 = self.kind();
            self.push(day1, k1, t1, Some(Tag::ContradictionBefore(plan)));
            self.push(day2, k2, t2, Some(Tag::ContradictionAfter(plan)));
        }
    }

    fn plan_insights(&mut self, count: usize) {
        for _ in 0..count {
            let e_name = (*self.rng.pick(PEOPLE)).to_string();
            // Pick two distinct clusters to bridge.
            let ca = (*self.rng.pick(CLUSTERS)).to_string();
            let mut cb = (*self.rng.pick(CLUSTERS)).to_string();
            while cb == ca {
                cb = (*self.rng.pick(CLUSTERS)).to_string();
            }
            let e = self.intern(&e_name, EntityKind::Person, "bridge");
            let plan = self.insight_params.len();
            self.insight_params.push(InsightParams {
                e,
                cluster_a: ca.clone(),
                cluster_b: cb.clone(),
            });
            // A few entries place the bridge entity inside each cluster's context.
            let per_side = self.rng.range(2, 4);
            for _ in 0..per_side {
                let day = self.rng.range(0, self.total_days);
                let topic = self.cluster_topic(&ca);
                let text = format!("{e_name} showed up while I was deep in {topic}.");
                let k = self.kind();
                self.push(day, k, text, Some(Tag::Insight(plan)));
            }
            for _ in 0..per_side {
                let day = self.rng.range(0, self.total_days);
                let topic = self.cluster_topic(&cb);
                let text = format!("Ran into {e_name} again, this time around {topic}.");
                let k = self.kind();
                self.push(day, k, text, Some(Tag::Insight(plan)));
            }
        }
    }

    fn cluster_topic(&mut self, cluster: &str) -> String {
        let topics = TOPICS_BY_CLUSTER
            .iter()
            .find(|(c, _)| *c == cluster)
            .map(|(_, t)| *t)
            .unwrap_or(&[]);
        if topics.is_empty() {
            cluster.to_string()
        } else {
            (*self.rng.pick(topics)).to_string()
        }
    }

    fn finalize(mut self) -> Corpus {
        // Stable sort by day; entries keep insertion order within a day. Then ids
        // are assigned in chronological order.
        self.raw.sort_by_key(|r| r.day);

        let mut entries = Vec::with_capacity(self.raw.len());
        // Accumulators keyed by plan id.
        let mut recall_entry: Vec<Option<EntryId>> = vec![None; self.recall_params.len()];
        let mut mh_a: Vec<Option<EntryId>> = vec![None; self.multihop_params.len()];
        let mut mh_b: Vec<Option<EntryId>> = vec![None; self.multihop_params.len()];
        let mut temp_lead: Vec<Vec<EntryId>> = vec![Vec::new(); self.temporal_params.len()];
        let mut temp_trail: Vec<Vec<EntryId>> = vec![Vec::new(); self.temporal_params.len()];
        let mut con_before: Vec<Option<EntryId>> = vec![None; self.contradiction_params.len()];
        let mut con_after: Vec<Option<EntryId>> = vec![None; self.contradiction_params.len()];
        let mut ins_entries: Vec<Vec<EntryId>> = vec![Vec::new(); self.insight_params.len()];

        // Consume `raw` so entry text is moved, not cloned (matters at scale).
        let raw = std::mem::take(&mut self.raw);
        for (idx, r) in raw.into_iter().enumerate() {
            let id = idx as EntryId;
            if let Some(tag) = r.tag {
                match tag {
                    Tag::Recall(q) => recall_entry[q] = Some(id),
                    Tag::MultiHopA(p) => mh_a[p] = Some(id),
                    Tag::MultiHopB(p) => mh_b[p] = Some(id),
                    Tag::TemporalLead(p) => temp_lead[p].push(id),
                    Tag::TemporalTrail(p) => temp_trail[p].push(id),
                    Tag::ContradictionBefore(p) => con_before[p] = Some(id),
                    Tag::ContradictionAfter(p) => con_after[p] = Some(id),
                    Tag::Insight(p) => ins_entries[p].push(id),
                }
            }
            entries.push(Entry {
                id,
                day: r.day,
                date: date_string(r.day),
                kind: r.kind,
                text: r.text,
            });
        }

        let mut gt = GroundTruth {
            entities: self.entities.clone(),
            ..Default::default()
        };

        for (q, params) in self.recall_params.iter().enumerate() {
            if let Some(entry) = recall_entry[q] {
                gt.recall.push(RecallQuery {
                    id: q,
                    question: format!("What book did {} recommend?", self.name_of(params.person)),
                    answer: self.name_of(params.book).to_string(),
                    relevant_entries: vec![entry],
                });
            }
        }

        for (p, params) in self.multihop_params.iter().enumerate() {
            if let (Some(a), Some(b)) = (mh_a[p], mh_b[p]) {
                gt.multi_hop.push(MultiHopQuery {
                    id: p,
                    question: format!(
                        "What book was recommended by the person {} introduced me to?",
                        self.name_of(params.p)
                    ),
                    answer: self.name_of(params.book).to_string(),
                    hop_entries: vec![a, b],
                });
            }
        }

        for (p, params) in self.temporal_params.iter().enumerate() {
            if !temp_lead[p].is_empty() && !temp_trail[p].is_empty() {
                gt.temporal.push(TemporalPattern {
                    id: p,
                    leading: params.a,
                    trailing: params.b,
                    lag_days: params.lag,
                    description: format!(
                        "{} tends to precede {} by ~{} days",
                        self.name_of(params.a),
                        self.name_of(params.b),
                        params.lag
                    ),
                    lead_entries: temp_lead[p].clone(),
                    trail_entries: temp_trail[p].clone(),
                });
            }
        }

        for (p, params) in self.contradiction_params.iter().enumerate() {
            if let (Some(before), Some(after)) = (con_before[p], con_after[p]) {
                gt.contradictions.push(Contradiction {
                    id: p,
                    topic: params.topic.clone(),
                    entry_before: before,
                    entry_after: after,
                });
            }
        }

        for (p, params) in self.insight_params.iter().enumerate() {
            if ins_entries[p].len() >= 2 {
                gt.insights.push(InsightBridge {
                    id: p,
                    bridge_entity: params.e,
                    cluster_a: params.cluster_a.clone(),
                    cluster_b: params.cluster_b.clone(),
                    description: format!(
                        "{} secretly links '{}' and '{}'",
                        self.name_of(params.e),
                        params.cluster_a,
                        params.cluster_b
                    ),
                    supporting_entries: ins_entries[p].clone(),
                    surprise: 0.8,
                });
            }
        }

        let num_entries = entries.len();
        Corpus {
            meta: CorpusMeta {
                seed: 0, // filled by caller
                years: self.total_days / 365,
                num_entries,
            },
            entries,
            ground_truth: gt,
        }
    }
}

/// Generate a deterministic synthetic corpus from the given config.
pub fn generate(config: &GenConfig) -> Corpus {
    let total_days = config.years.max(1) * 365;
    let mut g = Generator {
        rng: Rng::new(config.seed),
        total_days,
        entries_per_day: config.entries_per_day,
        raw: Vec::new(),
        entities: Vec::new(),
        interner: HashMap::new(),
        recall_params: Vec::new(),
        multihop_params: Vec::new(),
        temporal_params: Vec::new(),
        contradiction_params: Vec::new(),
        insight_params: Vec::new(),
    };

    // Plant phenomena first (counts scale with corpus length).
    let years = config.years.max(1) as usize;
    g.plan_recall(years * 8);
    g.plan_multihop(years * 4);
    g.plan_temporal(years.max(1));
    g.plan_contradictions(years * 2);
    g.plan_insights(years.max(1));

    // Fill in routine background entries day by day.
    for day in 0..total_days {
        g.routine_day(day);
    }

    let mut corpus = g.finalize();
    corpus.meta.seed = config.seed;
    corpus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_runs() {
        let cfg = GenConfig {
            seed: 7,
            years: 2,
            ..Default::default()
        };
        let a = serde_json::to_string(&generate(&cfg)).unwrap();
        let b = serde_json::to_string(&generate(&cfg)).unwrap();
        assert_eq!(a, b, "same seed must produce byte-identical corpus");
    }

    #[test]
    fn different_seeds_differ() {
        let a = generate(&GenConfig {
            seed: 1,
            years: 2,
            ..Default::default()
        });
        let b = generate(&GenConfig {
            seed: 2,
            years: 2,
            ..Default::default()
        });
        assert_ne!(a.entries.len(), 0);
        assert_ne!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }

    #[test]
    fn ground_truth_references_are_valid() {
        let c = generate(&GenConfig {
            seed: 42,
            years: 3,
            ..Default::default()
        });
        let n = c.entries.len() as u64;
        let check = |id: EntryId| assert!(id < n, "entry id {id} out of range {n}");

        for q in &c.ground_truth.recall {
            for &e in &q.relevant_entries {
                check(e);
            }
            // The answer must actually appear in the relevant entry.
            let txt = c.entry_text(q.relevant_entries[0]).unwrap();
            assert!(
                txt.contains(&q.answer),
                "recall answer '{}' missing from entry '{}'",
                q.answer,
                txt
            );
        }
        for q in &c.ground_truth.multi_hop {
            for &e in &q.hop_entries {
                check(e);
            }
        }
        for t in &c.ground_truth.temporal {
            for &e in t.lead_entries.iter().chain(&t.trail_entries) {
                check(e);
            }
        }
        for con in &c.ground_truth.contradictions {
            check(con.entry_before);
            check(con.entry_after);
            assert!(
                con.entry_before < con.entry_after,
                "before must precede after"
            );
        }
        for ins in &c.ground_truth.insights {
            assert!(ins.supporting_entries.len() >= 2);
            for &e in &ins.supporting_entries {
                check(e);
            }
        }
    }

    #[test]
    fn plants_something_of_each_kind() {
        let c = generate(&GenConfig {
            seed: 3,
            years: 3,
            ..Default::default()
        });
        assert!(!c.ground_truth.recall.is_empty());
        assert!(!c.ground_truth.multi_hop.is_empty());
        assert!(!c.ground_truth.temporal.is_empty());
        assert!(!c.ground_truth.contradictions.is_empty());
        assert!(!c.ground_truth.insights.is_empty());
    }
}
