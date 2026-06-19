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

/// Generation hardness. `V1` reproduces the original Stage 0 generator
/// byte-for-byte (single-template phenomena; lexically easy). `V2` is the
/// hardened corpus: paraphrase banks, coreference in multi-hop chains,
/// planted *negative* bridges (apophenia traps), lexical distractors near the
/// needles, and non-stationary topic drift. V2 exists because a measuring
/// stick that a regex can max out cannot measure Stages 3-5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Difficulty {
    V1,
    #[default]
    V2,
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

/// An apophenia trap (V2 only): an entity that casually spans *many* clusters.
/// A naive "name appears in two clusters" detector fires on every pair of its
/// clusters, but none of those links is an insight — the entity is a hub, so
/// co-occurrence carries no surprise. A real Insight Engine must rank by
/// rarity/surprise and stay silent here; matches against these entries are
/// certified false insights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NegativeBridge {
    pub id: usize,
    pub entity: EntityId,
    /// The clusters the entity casually appears in (>= 3).
    pub clusters: Vec<String>,
    pub entries: Vec<EntryId>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GroundTruth {
    pub entities: Vec<Entity>,
    pub recall: Vec<RecallQuery>,
    pub multi_hop: Vec<MultiHopQuery>,
    pub temporal: Vec<TemporalPattern>,
    pub contradictions: Vec<Contradiction>,
    pub insights: Vec<InsightBridge>,
    /// V2 apophenia traps; empty for V1 corpora.
    #[serde(default)]
    pub negative_bridges: Vec<NegativeBridge>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusMeta {
    pub seed: u64,
    pub years: u32,
    pub num_entries: usize,
    #[serde(default)]
    pub difficulty: Difficulty,
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
    pub difficulty: Difficulty,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            seed: 42,
            years: 5,
            entries_per_day: 2,
            difficulty: Difficulty::V2,
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

/// Dedicated, RARE names for V2 insight-bridge entities — disjoint from [`PEOPLE`]
/// so a planted bridge does NOT recur as a background distractor. This is the D0
/// fix that makes the insight signal detectable: a real bridge appears only in
/// its two planted clusters (a rare 2-cluster entity), cleanly separable from a
/// common multi-cluster apophenia hub. See `specs/STAGE_5.md`.
const BRIDGE_PEOPLE: &[&str] = &[
    "Quillon",
    "Zephyrine",
    "Caradoc",
    "Isolde",
    "Thessaly",
    "Oberon",
    "Vespera",
    "Lysander",
    "Calixto",
    "Marisol",
    "Evander",
    "Ondine",
    "Peregrine",
    "Saffron",
    "Tiberius",
    "Wrenna",
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

/// Where multi-hop introductions happen (V2). The venue is the coreference
/// anchor: when the second hop drops the person's name, the venue is the only
/// thread linking the two entries.
const VENUES: &[&str] = &[
    "the meetup",
    "the conference",
    "the climbing gym",
    "the workshop",
    "the book club",
];

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

/// The topic lexicon: each cluster and the surface topic phrases that appear in
/// its entries. Exposed as **observable vocabulary** (the analogue of the
/// entity-name gazetteer the Stage-3 graph already uses) so a downstream engine
/// can map an entry's text to a topic cluster *without* reading the planted
/// answer key (`ground_truth.{insights, negative_bridges, ...}`). It is NOT
/// ground truth — a real deployment would obtain this from topic modelling /
/// clustering; here it stands in for that observable signal. The Stage-5 Insight
/// Engine uses it to measure how many distinct clusters an entity spans (a
/// 2-cluster bridge is surprising; a 4-cluster hub is apophenia).
pub fn topic_lexicon() -> &'static [(&'static str, &'static [&'static str])] {
    TOPICS_BY_CLUSTER
}

/// The cluster labels (observable vocabulary; see [`topic_lexicon`]).
pub fn cluster_labels() -> &'static [&'static str] {
    CLUSTERS
}

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
    NegativeBridge(usize),
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

struct NegativeParams {
    e: EntityId,
    clusters: Vec<String>,
}

struct Generator {
    rng: Rng,
    total_days: u32,
    entries_per_day: u32,
    difficulty: Difficulty,
    raw: Vec<RawEntry>,
    entities: Vec<Entity>,
    interner: HashMap<String, EntityId>,
    recall_params: Vec<RecallParams>,
    multihop_params: Vec<MultiHopParams>,
    temporal_params: Vec<TemporalParams>,
    contradiction_params: Vec<ContradictionParams>,
    insight_params: Vec<InsightParams>,
    negative_params: Vec<NegativeParams>,
    /// Names already used as positive bridge entities (so V2 negatives avoid them).
    bridge_names: Vec<String>,
    /// V2 only: per-year cluster weights for non-stationary topic drift.
    year_weights: Vec<Vec<u32>>,
}

impl Generator {
    /// Template index for a paraphrase bank. V1 always renders template 0 (the
    /// legacy string) and — crucially — consumes no RNG draw, so the V1 stream
    /// stays byte-identical to the original generator.
    fn tpl(&mut self, n: usize) -> usize {
        match self.difficulty {
            Difficulty::V1 => 0,
            Difficulty::V2 => self.rng.below(n),
        }
    }

    /// A probability gate that exists only in V2 (V1 consumes no draw).
    fn v2_chance(&mut self, p: f64) -> bool {
        match self.difficulty {
            Difficulty::V1 => false,
            Difficulty::V2 => self.rng.chance(p),
        }
    }
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

    /// Pick the day's cluster. V1: uniform (legacy). V2: weighted by the
    /// current year's drifted weights, so the background distribution is
    /// non-stationary — patterns learned on year 1 don't trivially hold in
    /// year 5.
    fn routine_cluster(&mut self, day: u32) -> &'static str {
        match self.difficulty {
            Difficulty::V1 => CLUSTERS[self.rng.below(CLUSTERS.len())],
            Difficulty::V2 => {
                let year = (day / 365) as usize;
                let w = &self.year_weights[year.min(self.year_weights.len() - 1)];
                let total: u32 = w.iter().sum();
                let mut r = self.rng.below(total.max(1) as usize) as u32;
                for (i, &wi) in w.iter().enumerate() {
                    if r < wi {
                        return CLUSTERS[i];
                    }
                    r -= wi;
                }
                CLUSTERS[CLUSTERS.len() - 1]
            }
        }
    }

    fn routine_day(&mut self, day: u32) {
        // 0..=2*epd entries, averaging `entries_per_day`.
        let n = self.rng.below((2 * self.entries_per_day + 1) as usize);
        for _ in 0..n {
            let cluster = self.routine_cluster(day);
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

    /// Render the recall fact through a paraphrase bank. Every template must
    /// contain the book title verbatim (the answer-in-entry invariant) and the
    /// person's name (so the fact is findable in principle), but only some
    /// share vocabulary with the question — partial lexical overlap is the
    /// whole point of V2.
    fn recall_text(&mut self, p: &str, b: &str) -> String {
        match self.tpl(8) {
            0 => format!("{p} recommended the book {b} to me today. Want to remember that."),
            1 => format!(
                "Coffee with {p} today — they kept coming back to {b}, said I have to read it."
            ),
            2 => format!("{p} couldn't stop praising {b}. Added it to my list."),
            3 => format!("Note to self: {b} — a tip from {p}."),
            4 => format!("{p} swears {b} changed how they think about everything."),
            5 => format!("Got a reading suggestion from {p}: {b}. Sounds promising."),
            6 => format!("{p} told me to read {b}; apparently it's exactly my kind of thing."),
            _ => format!("Long chat with {p} about books — the clear favorite was {b}."),
        }
    }

    /// A lexical distractor near a recall needle: same person, *different*
    /// book, no recommendation semantics. Pulls BM25 toward the wrong entry.
    fn recall_distractor_text(&mut self, p: &str, other: &str) -> String {
        match self.tpl(3) {
            0 => format!("Saw {p} reading {other} on the train this morning."),
            1 => format!("{p} mentioned {other} in passing; didn't sound convinced."),
            _ => format!("Almost bought {other} at the shop; ran into {p} there too."),
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
            let text = self.recall_text(&person_name, &book_name);
            let kind = self.kind();
            self.push(day, kind, text, Some(Tag::Recall(qid)));

            // V2: 1-2 distractor entries (person + a different book).
            if self.difficulty == Difficulty::V2 {
                let n = self.rng.range(1, 3);
                for _ in 0..n {
                    let mut other = (*self.rng.pick(BOOKS)).to_string();
                    while other == book_name {
                        other = (*self.rng.pick(BOOKS)).to_string();
                    }
                    self.intern(&other, EntityKind::Book, "reading");
                    let dday = self.rng.range(0, self.total_days);
                    let dtext = self.recall_distractor_text(&person_name, &other);
                    let dkind = self.kind();
                    self.push(dday, dkind, dtext, None);
                }
            }
        }
    }

    /// Hop A: the introduction. Always names both people and the venue (the
    /// venue is the coreference anchor for hop B).
    fn multihop_a_text(&mut self, p: &str, q: &str, venue: &str) -> String {
        match self.tpl(3) {
            0 => format!("{p} introduced me to {q} at {venue}."),
            1 => format!("Met {q} today through {p}, at {venue} of all places."),
            _ => format!("{p} brought {q} along to {venue}; we hit it off."),
        }
    }

    /// Hop B with the person named (no coreference).
    fn multihop_b_text(&mut self, q: &str, b: &str) -> String {
        match self.tpl(3) {
            0 => format!("{q} recommended the book {b}. Noted."),
            1 => format!("{q} kept insisting I read {b}."),
            _ => format!("Turns out {q} is a huge fan of {b}; told me to start it this week."),
        }
    }

    /// Hop B with coreference: the person's name is *absent*; only the venue
    /// links back to hop A. Joining now requires entity linking, not string
    /// matching.
    fn multihop_b_coref_text(&mut self, venue: &str, b: &str) -> String {
        match self.tpl(2) {
            0 => format!("The person I met at {venue} recommended the book {b}. Noted."),
            _ => format!("That new acquaintance from {venue} told me to read {b}."),
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
            // V1 always uses the legacy venue; V2 varies it (the coref anchor).
            let venue = match self.difficulty {
                Difficulty::V1 => "the meetup".to_string(),
                Difficulty::V2 => (*self.rng.pick(VENUES)).to_string(),
            };
            let t1 = self.multihop_a_text(&p_name, &q_name, &venue);
            let t2 = if self.v2_chance(0.4) {
                self.multihop_b_coref_text(&venue, &book_name)
            } else {
                self.multihop_b_text(&q_name, &book_name)
            };
            let k1 = self.kind();
            let k2 = self.kind();
            self.push(day1, k1, t1, Some(Tag::MultiHopA(plan)));
            self.push(day2, k2, t2, Some(Tag::MultiHopB(plan)));

            // V2: a distractor — Q talking about a *different* book.
            if self.difficulty == Difficulty::V2 && self.rng.chance(0.5) {
                let mut other = (*self.rng.pick(BOOKS)).to_string();
                while other == book_name {
                    other = (*self.rng.pick(BOOKS)).to_string();
                }
                self.intern(&other, EntityKind::Book, "reading");
                let dday = self.rng.range(0, self.total_days);
                let dtext =
                    format!("{q_name} and I argued about {other} for an hour; not convinced.");
                let dkind = self.kind();
                self.push(dday, dkind, dtext, None);
            }
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
            let t1 = match self.tpl(3) {
                0 => format!("Convinced {x_name} is the best choice. Going all in on it."),
                1 => format!("After a week of digging, {x_name} wins. Committing to it."),
                _ => format!("I keep coming back to {x_name}; it just fits. Decision made."),
            };
            let t2 = match self.tpl(3) {
                0 => format!("Changed my mind: {x_name} was a mistake. {y_name} is clearly better."),
                1 => format!("Six months in, {x_name} has been nothing but pain. Moving everything to {y_name}."),
                _ => format!("Regret picking {x_name}. {y_name} would have saved us weeks."),
            };
            let k1 = self.kind();
            let k2 = self.kind();
            self.push(day1, k1, t1, Some(Tag::ContradictionBefore(plan)));
            self.push(day2, k2, t2, Some(Tag::ContradictionAfter(plan)));
        }
    }

    fn plan_insights(&mut self, count: usize) {
        for _ in 0..count {
            // Consume the PEOPLE pick UNCONDITIONALLY so the V2 RNG stream stays
            // byte-identical to the legacy generator; `picked` still seeds
            // `bridge_names` below so negative-bridge avoidance draws the same RNG.
            let picked = (*self.rng.pick(PEOPLE)).to_string();
            let plan = self.insight_params.len();
            // D0 (V2 only): bridge entities get UNIQUE, RARE names from a
            // dedicated pool so a planted 2-cluster bridge is statistically
            // separable from a common multi-cluster apophenia hub. V1 keeps the
            // legacy PEOPLE name so its byte-frozen golden holds.
            let e_name = if self.difficulty == Difficulty::V2 {
                BRIDGE_PEOPLE[plan % BRIDGE_PEOPLE.len()].to_string()
            } else {
                picked.clone()
            };
            // Pick two distinct clusters to bridge.
            let ca = (*self.rng.pick(CLUSTERS)).to_string();
            let mut cb = (*self.rng.pick(CLUSTERS)).to_string();
            while cb == ca {
                cb = (*self.rng.pick(CLUSTERS)).to_string();
            }
            let e = self.intern(&e_name, EntityKind::Person, "bridge");
            self.bridge_names.push(picked.clone());
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
                let text = match self.tpl(3) {
                    0 => format!("{e_name} showed up while I was deep in {topic}."),
                    1 => format!("Unexpectedly, {e_name} had sharp opinions on {topic}."),
                    _ => {
                        format!("Spent the afternoon on {topic}; {e_name} joined halfway through.")
                    }
                };
                let k = self.kind();
                self.push(day, k, text, Some(Tag::Insight(plan)));
            }
            for _ in 0..per_side {
                let day = self.rng.range(0, self.total_days);
                let topic = self.cluster_topic(&cb);
                let text = match self.tpl(3) {
                    0 => format!("Ran into {e_name} again, this time around {topic}."),
                    1 => format!("{e_name} again — this time in the middle of {topic}."),
                    _ => format!("Funny how {e_name} keeps appearing whenever {topic} comes up."),
                };
                let k = self.kind();
                self.push(day, k, text, Some(Tag::Insight(plan)));
            }
        }
    }

    /// V2 only: plant apophenia traps — hub entities casually spanning many
    /// clusters. See [`NegativeBridge`].
    fn plan_negative_bridges(&mut self, count: usize) {
        debug_assert_eq!(self.difficulty, Difficulty::V2);
        for _ in 0..count {
            // Avoid names already used as positive bridges; bounded retries
            // keep this deterministic even if the pool is nearly exhausted.
            let mut e_name = (*self.rng.pick(PEOPLE)).to_string();
            for _ in 0..50 {
                if !self.bridge_names.contains(&e_name) {
                    break;
                }
                e_name = (*self.rng.pick(PEOPLE)).to_string();
            }
            let e = self.intern(&e_name, EntityKind::Person, "hub");
            // The hub spans 4 distinct clusters: any pair co-occurs, so no
            // single pair is surprising.
            let mut clusters: Vec<String> = Vec::new();
            while clusters.len() < 4 {
                let c = (*self.rng.pick(CLUSTERS)).to_string();
                if !clusters.contains(&c) {
                    clusters.push(c);
                }
            }
            let plan = self.negative_params.len();
            self.negative_params.push(NegativeParams {
                e,
                clusters: clusters.clone(),
            });
            for cluster in &clusters.clone() {
                let n = self.rng.range(1, 3);
                for _ in 0..n {
                    let day = self.rng.range(0, self.total_days);
                    let topic = self.cluster_topic(cluster);
                    let text = match self.tpl(3) {
                        0 => format!("{e_name} texted while I was busy with {topic}. Small world."),
                        1 => format!("Mentioned {topic} to {e_name} in passing; we moved on quickly."),
                        _ => format!("{e_name} was around again today; mostly small talk while I dealt with {topic}."),
                    };
                    let k = self.kind();
                    self.push(day, k, text, Some(Tag::NegativeBridge(plan)));
                }
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
        let mut neg_entries: Vec<Vec<EntryId>> = vec![Vec::new(); self.negative_params.len()];

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
                    Tag::NegativeBridge(p) => neg_entries[p].push(id),
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

        for (p, params) in self.negative_params.iter().enumerate() {
            if neg_entries[p].len() >= 2 {
                gt.negative_bridges.push(NegativeBridge {
                    id: p,
                    entity: params.e,
                    clusters: params.clusters.clone(),
                    entries: neg_entries[p].clone(),
                });
            }
        }

        let num_entries = entries.len();
        Corpus {
            meta: CorpusMeta {
                seed: 0, // filled by caller
                years: self.total_days / 365,
                num_entries,
                difficulty: self.difficulty,
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
        difficulty: config.difficulty,
        raw: Vec::new(),
        entities: Vec::new(),
        interner: HashMap::new(),
        recall_params: Vec::new(),
        multihop_params: Vec::new(),
        temporal_params: Vec::new(),
        contradiction_params: Vec::new(),
        insight_params: Vec::new(),
        negative_params: Vec::new(),
        bridge_names: Vec::new(),
        year_weights: Vec::new(),
    };

    // Plant phenomena first (counts scale with corpus length).
    let years = config.years.max(1) as usize;
    g.plan_recall(years * 8);
    g.plan_multihop(years * 4);
    g.plan_temporal(years.max(1));
    g.plan_contradictions(years * 2);
    g.plan_insights(years.max(1));
    if config.difficulty == Difficulty::V2 {
        g.plan_negative_bridges(years.max(1));

        // Non-stationary topic drift: each year mutates the previous year's
        // cluster weights. Year 0 is uniform.
        let mut w = vec![4u32; CLUSTERS.len()];
        g.year_weights.push(w.clone());
        for _ in 1..config.years.max(1) {
            for wi in w.iter_mut() {
                if g.rng.chance(0.5) {
                    let delta = g.rng.range(1, 4) as i64;
                    let sign = if g.rng.chance(0.5) { 1 } else { -1 };
                    *wi = (*wi as i64 + sign * delta).clamp(1, 9) as u32;
                }
            }
            g.year_weights.push(w.clone());
        }
    }

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

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01B3);
        }
        h
    }

    /// V1 must reproduce the original (pre-V2) generator byte-for-byte. The
    /// hashes below were captured from the legacy code before the V2 changes;
    /// if this test breaks, a V1 code path consumed an RNG draw it must not.
    #[test]
    fn v1_entries_match_legacy_golden() {
        for (seed, years, want_count, want_hash) in [
            (42u64, 2u32, 1538usize, 0x3e08_004b_4da7_57d5u64),
            (7, 3, 2314, 0x9aae_f0c5_b6e2_7c8cu64),
        ] {
            let c = generate(&GenConfig {
                seed,
                years,
                entries_per_day: 2,
                difficulty: Difficulty::V1,
            });
            assert_eq!(c.entries.len(), want_count, "seed {seed}: entry count");
            let json = serde_json::to_string(&c.entries).unwrap();
            assert_eq!(
                fnv1a64(json.as_bytes()),
                want_hash,
                "seed {seed}: V1 entries diverged from the legacy generator"
            );
            assert!(c.ground_truth.negative_bridges.is_empty());
        }
    }

    #[test]
    fn v2_plants_negative_bridges_v1_does_not() {
        let v2 = generate(&GenConfig {
            seed: 11,
            years: 3,
            ..Default::default()
        });
        assert!(!v2.ground_truth.negative_bridges.is_empty());
        for nb in &v2.ground_truth.negative_bridges {
            assert!(nb.clusters.len() >= 3, "hub must span many clusters");
            assert!(nb.entries.len() >= 2);
            let n = v2.entries.len() as u64;
            for &e in &nb.entries {
                assert!(e < n);
            }
            // A negative entity must not double as a positive bridge entity.
            for ins in &v2.ground_truth.insights {
                assert_ne!(
                    nb.entity, ins.bridge_entity,
                    "negative bridge entity collides with a planted insight"
                );
            }
        }
    }

    #[test]
    fn v2_recall_uses_paraphrases() {
        let c = generate(&GenConfig {
            seed: 5,
            years: 4,
            ..Default::default()
        });
        let legacy_form = c
            .ground_truth
            .recall
            .iter()
            .filter(|q| {
                c.entry_text(q.relevant_entries[0])
                    .unwrap()
                    .contains("recommended the book")
            })
            .count();
        let total = c.ground_truth.recall.len();
        assert!(total >= 8);
        assert!(
            legacy_form < total,
            "all {total} recall entries use the legacy template — paraphrasing inactive"
        );
    }

    #[test]
    fn v2_multihop_has_coreference_hops() {
        let c = generate(&GenConfig {
            seed: 9,
            years: 5,
            ..Default::default()
        });
        // At least one hop-B entry must not name any person — the join must go
        // through the venue anchor instead of string matching.
        let coref_hops = c
            .ground_truth
            .multi_hop
            .iter()
            .filter(|q| {
                let txt = c.entry_text(q.hop_entries[1]).unwrap();
                !PEOPLE.iter().any(|p| txt.contains(p))
            })
            .count();
        assert!(
            coref_hops > 0,
            "expected some coreference hops among {} multi-hop chains",
            c.ground_truth.multi_hop.len()
        );
    }

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
        for nb in &c.ground_truth.negative_bridges {
            assert!(nb.entries.len() >= 2);
            for &e in &nb.entries {
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
        assert!(!c.ground_truth.negative_bridges.is_empty());
    }
}
