#![forbid(unsafe_code)]
//! Stage 4 — "Sleep" engine: interruptible background consolidation scheduler.
//!
//! All consolidation runs as interruptible, resumable, checkpointed background
//! jobs that execute only while the machine is idle, on power, and thermally OK.
//! A deterministic simulator proves the scheduling policy correct before any
//! real job exists.
//!
//! # Architecture
//!
//! - [`PowerSignals`] trait — environmental gating signals (power, idle, thermal).
//!   Real macOS impl behind `cfg(target_os = "macos")`; [`FakeSignals`] for tests.
//! - [`Job`] trait — a unit of consolidation work, chunked so it can stop at
//!   boundaries. Real jobs plug in at Stage 3/5.
//! - [`Scheduler`] — priority queue with signal-gated [`Scheduler::tick`] and
//!   on-disk persistence for crash/restart safety.
//! - [`run_sim`] — deterministic simulator that drives a scripted timeline over
//!   stub jobs, checking all six gate metrics.

use std::collections::BinaryHeap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// The per-step work cap the scheduler enforces on every [`Scheduler::tick`].
/// The gate check ([`check_gate`]) reuses this same constant, so the policy
/// and its verification can never silently diverge.
const STEP_MAX_ITEMS: u32 = 1000;

/// Soft per-step time bound handed to each job (best-effort, not enforced).
const STEP_SOFT_DEADLINE_MS: u32 = 100;

// ---------------------------------------------------------------------------
// T1: Core types — StepBudget, StepOutcome, PowerSignals, Job
// ---------------------------------------------------------------------------

/// Per-step resource cap. A `step()` call must not exceed these bounds.
#[derive(Debug, Clone, Copy)]
pub struct StepBudget {
    pub max_items: u32,
    /// Soft time bound in milliseconds — best-effort, not enforced by the
    /// scheduler itself, but the job should self-limit.
    pub soft_deadline_ms: u32,
}

/// Outcome of one `step()` call on a [`Job`].
#[derive(Debug, Clone)]
pub enum StepOutcome {
    Progress {
        /// Cumulative work done so far (including this step).
        done: u64,
        /// Total work units for this job (constant across steps).
        total: u64,
    },
    Finished,
}

/// Environmental gating signals.
///
/// The scheduler permits work only when all three signals are true.
pub trait PowerSignals: Send {
    /// True when the machine is on external power (not battery).
    fn on_external_power(&self) -> bool;
    /// True when the user has been idle long enough.
    fn user_idle(&self) -> bool;
    /// True when the system is not thermally throttling.
    fn thermal_ok(&self) -> bool;
}

/// A unit of consolidation work that executes in bounded steps.
///
/// Implementations must be chunked so they can stop at step boundaries
/// (the scheduler re-checks signals between steps). `checkpoint`/`restore`
/// must be lossless: remaining work after restore == before.
pub trait Job: Send {
    /// Stable identifier for this job (used in persistence).
    fn id(&self) -> &str;
    /// Priority — higher runs first. Ties are broken by submission order.
    fn priority(&self) -> u8;
    /// Advance work by at most `budget.max_items` items, respecting
    /// `budget.soft_deadline_ms` best-effort.
    ///
    /// A call with `budget.max_items == 0` must be a side-effect-free *probe*:
    /// it does no work and reports `Progress { done, total }` (or `Finished`
    /// if already complete) without consuming any. [`Scheduler::submit`] relies
    /// on this to learn a job's total work at enqueue time.
    fn step(&mut self, budget: StepBudget) -> StepOutcome;
    /// Serialise remaining work so the job can be restored later.
    fn checkpoint(&self) -> Vec<u8>;
    /// Restore from a previously-saved checkpoint.
    fn restore(&mut self, state: &[u8]);
}

// ---------------------------------------------------------------------------
// T2+T3: Scheduler with priority queue + on-disk persistence
// ---------------------------------------------------------------------------

/// A job entry stored in the scheduler's priority queue.
struct QueueEntry {
    job: Box<dyn Job>,
    /// Higher runs first; ties broken by `seq` (lower = earlier submission).
    priority: u8,
    seq: u64,
    /// Remaining work units, updated after each `step()`. Used by `pending_work()`.
    remaining: u64,
}

impl QueueEntry {
    fn sort_key(&self) -> impl Ord {
        (self.priority, std::cmp::Reverse(self.seq))
    }
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key() == other.sort_key()
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

/// A checkpoint record, written to the on-disk queue file after each `tick()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntry {
    pub id: String,
    pub priority: u8,
    pub seq: u64,
    /// Hex-encoded checkpoint blob from `Job::checkpoint()`.
    pub state: String,
    pub remaining: u64,
}

/// The sleep scheduler: priority queue + signal gating + on-disk persistence.
pub struct Scheduler<S: PowerSignals> {
    signals: S,
    pub(crate) heap: BinaryHeap<QueueEntry>,
    dir: PathBuf,
    next_seq: u64,
    tick_count: u64,
    /// True when `open` found a non-empty checkpoint on disk that has not yet
    /// been loaded via `restore_from_checkpoint`. Guards `submit` from
    /// overwriting (and thereby destroying) the persisted queue.
    pending_restore: bool,
}

impl<S: PowerSignals> Scheduler<S> {
    /// Open (or create) a scheduler state directory.
    ///
    /// If a checkpoint file exists from a previous run, its entries are loaded
    /// as raw records; call [`restore_from_checkpoint`](Scheduler::restore_from_checkpoint)
    /// to reconstruct the `Job` objects and re-submit them.
    pub fn open(dir: &Path, signals: S) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)
            .with_context(|| format!("creating scheduler dir {}", dir.display()))?;
        let mut sched = Self {
            signals,
            heap: BinaryHeap::new(),
            dir: dir.to_path_buf(),
            next_seq: 0,
            tick_count: 0,
            pending_restore: false,
        };
        // A non-empty checkpoint must be restored before new jobs are submitted,
        // otherwise the first `submit` would overwrite the persisted queue.
        sched.pending_restore = !sched.load_checkpoint_entries()?.is_empty();
        Ok(sched)
    }

    /// Submit a new job to the queue. Persists the checkpoint immediately.
    ///
    /// A zero-budget probe step is called to determine the initial remaining
    /// work; the job state is preserved (the probe does not consume work).
    pub fn submit(&mut self, mut job: Box<dyn Job>) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.pending_restore,
            "scheduler has an unrestored on-disk checkpoint; call \
             restore_from_checkpoint() before submitting (submitting now would \
             overwrite and lose the persisted queue)"
        );
        let remaining = match job.step(StepBudget {
            max_items: 0,
            soft_deadline_ms: 0,
        }) {
            StepOutcome::Progress { total, .. } => total,
            StepOutcome::Finished => 0,
        };
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(QueueEntry {
            priority: job.priority(),
            job,
            seq,
            remaining,
        });
        self.save_checkpoint()?;
        Ok(())
    }

    /// Advance work by one step if signals permit. Returns `true` if work ran.
    ///
    /// # Policy
    /// A step executes iff `on_external_power && user_idle && thermal_ok`.
    /// Otherwise this is a no-op (returns `false`). Between steps, the scheduler
    /// persists progress so a crash loses at most one step of work.
    ///
    /// The three signals are queried in a fixed order each tick —
    /// `on_external_power`, then `user_idle`, then `thermal_ok`. Scripted
    /// sources such as [`FakeSignals`] depend on this order to advance their
    /// timeline exactly once per tick.
    pub fn tick(&mut self) -> anyhow::Result<bool> {
        self.tick_count += 1;
        let power = self.signals.on_external_power();
        let idle = self.signals.user_idle();
        let thermal = self.signals.thermal_ok();
        if !power || !idle || !thermal {
            return Ok(false);
        }
        let Some(mut entry) = self.heap.pop() else {
            return Ok(false);
        };
        let outcome = entry.job.step(StepBudget {
            max_items: STEP_MAX_ITEMS,
            soft_deadline_ms: STEP_SOFT_DEADLINE_MS,
        });
        match &outcome {
            StepOutcome::Progress { done, total } => {
                entry.remaining = total.saturating_sub(*done);
                self.heap.push(entry);
            }
            StepOutcome::Finished => {
                entry.remaining = 0;
            }
        }
        self.save_checkpoint()?;
        Ok(true)
    }

    /// Return the sum of remaining work over all currently queued jobs.
    pub fn pending_work(&self) -> u64 {
        self.heap.iter().map(|e| e.remaining).sum()
    }

    /// Number of ticks processed (including blocked ones).
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Read the raw checkpoint entries from disk.
    pub fn load_checkpoint_entries(&self) -> anyhow::Result<Vec<CheckpointEntry>> {
        let path = self.queue_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let data =
            fs::read(&path).with_context(|| format!("reading checkpoint {}", path.display()))?;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let entries: Vec<CheckpointEntry> =
            serde_json::from_slice(&data).context("parsing checkpoint")?;
        Ok(entries)
    }

    /// Restore jobs from the checkpoint file: deserialise entries,
    /// reconstruct `Job` objects using the provided factory, and re-submit them.
    ///
    /// The factory receives `(id, priority, &checkpoint_bytes)` and must
    /// return a fully-restored `Box<dyn Job>`.
    pub fn restore_from_checkpoint<F>(&mut self, factory: F) -> anyhow::Result<()>
    where
        F: Fn(&str, u8, &[u8]) -> Box<dyn Job>,
    {
        let entries = self.load_checkpoint_entries()?;
        for e in entries {
            let state = hex::decode(&e.state).map_err(|err| {
                anyhow::anyhow!("decoding checkpoint state for {}: {}", e.id, err)
            })?;
            let job = factory(&e.id, e.priority, &state);
            // Preserve the original submission order (`seq`) so priority ties
            // resolve identically before and after a restart — resume is then
            // lossless in ordering, not just in remaining work.
            self.next_seq = self.next_seq.max(e.seq + 1);
            self.heap.push(QueueEntry {
                priority: e.priority,
                job,
                seq: e.seq,
                remaining: e.remaining,
            });
        }
        self.pending_restore = false;
        self.save_checkpoint()?;
        Ok(())
    }

    pub(crate) fn queue_path(&self) -> PathBuf {
        self.dir.join("sleep_queue.json")
    }

    fn save_checkpoint(&self) -> anyhow::Result<()> {
        let mut entries: Vec<CheckpointEntry> = Vec::new();
        let mut sorted: Vec<&QueueEntry> = self.heap.iter().collect();
        sorted.sort_by(|a, b| b.cmp(a)); // priority desc
        for e in &sorted {
            let state = e.job.checkpoint();
            entries.push(CheckpointEntry {
                id: e.job.id().to_string(),
                priority: e.priority,
                seq: e.seq,
                state: hex::encode(&state),
                remaining: e.remaining,
            });
        }
        let json = serde_json::to_vec(&entries).context("serialising checkpoint")?;

        let tmp_path = self.dir.join("sleep_queue.tmp");
        {
            let mut f = fs::File::create(&tmp_path)
                .with_context(|| format!("creating {}", tmp_path.display()))?;
            f.write_all(&json)
                .with_context(|| format!("writing {}", tmp_path.display()))?;
            f.flush().context("flushing tmp checkpoint")?;
            f.sync_all().context("fsync tmp checkpoint")?;
        }
        let target = self.queue_path();
        fs::rename(&tmp_path, &target)
            .with_context(|| format!("renaming checkpoint to {}", target.display()))?;
        let dir = fs::File::open(&self.dir).context("opening dir for fsync")?;
        dir.sync_all().context("fsync dir")?;
        Ok(())
    }
}

// Tiny hex encode/decode — no external deps.
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push(NYBBLE[(*b >> 4) as usize]);
            s.push(NYBBLE[(*b & 0x0F) as usize]);
        }
        s
    }

    pub fn decode(s: &str) -> Result<Vec<u8>, String> {
        if !s.len().is_multiple_of(2) {
            return Err("odd hex length".into());
        }
        let mut v = Vec::with_capacity(s.len() / 2);
        let bytes = s.as_bytes();
        for chunk in bytes.chunks(2) {
            let hi = hex_val(chunk[0])?;
            let lo = hex_val(chunk[1])?;
            v.push(hi << 4 | lo);
        }
        Ok(v)
    }

    const NYBBLE: [char; 16] = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    ];

    fn hex_val(b: u8) -> Result<u8, String> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err(format!("invalid hex byte {b}")),
        }
    }
}

// ---------------------------------------------------------------------------
// T4: FakeSignals — deterministic scripted timeline
// ---------------------------------------------------------------------------

/// One segment of a scripted signal timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineSegment {
    /// First tick (inclusive) this segment applies to.
    pub tick_start: u64,
    /// Last tick (inclusive) this segment applies to.
    pub tick_end: u64,
    pub on_power: bool,
    pub user_idle: bool,
    pub thermal_ok: bool,
}

/// A deterministic signal source driven by a scripted timeline.
///
/// The scheduler calls the three signal methods once per `tick()`. On the first
/// call (`on_external_power`) the internal tick counter advances; subsequent
/// calls in the same tick (`user_idle`, `thermal_ok`) read the same counter.
pub struct FakeSignals {
    timeline: Vec<TimelineSegment>,
    tick: std::cell::Cell<u64>,
}

impl FakeSignals {
    /// Create a new FakeSignals from a scripted timeline.
    ///
    /// The timeline must be non-empty. Segments must cover consecutive,
    /// non-overlapping tick ranges starting from 1. After the last segment's
    /// `tick_end`, the last segment repeats indefinitely.
    pub fn new(timeline: Vec<TimelineSegment>) -> Self {
        assert!(!timeline.is_empty(), "timeline must be non-empty");
        Self {
            timeline,
            tick: std::cell::Cell::new(0),
        }
    }

    fn segment_for(&self, tick: u64) -> &TimelineSegment {
        for seg in &self.timeline {
            if tick >= seg.tick_start && tick <= seg.tick_end {
                return seg;
            }
        }
        self.timeline.last().unwrap()
    }
}

impl PowerSignals for FakeSignals {
    fn on_external_power(&self) -> bool {
        let t = self.tick.get() + 1;
        self.tick.set(t);
        self.segment_for(t).on_power
    }

    fn user_idle(&self) -> bool {
        self.segment_for(self.tick.get()).user_idle
    }

    fn thermal_ok(&self) -> bool {
        self.segment_for(self.tick.get()).thermal_ok
    }
}

// ---------------------------------------------------------------------------
// Stub job for simulation
// ---------------------------------------------------------------------------

/// A minimal [`Job`] implementation that advances `items_per_step` work units
/// per call. Used by the deterministic simulator; real consolidation algorithms
/// (Leiden communities, RAPTOR summaries, PPR) plug in later at Stage 3/5.
pub struct StubJob {
    id: String,
    priority: u8,
    remaining: u64,
    total: u64,
    items_per_step: u64,
}

impl StubJob {
    pub fn new(id: impl Into<String>, priority: u8, total_work: u64, items_per_step: u64) -> Self {
        Self {
            id: id.into(),
            priority,
            remaining: total_work,
            total: total_work,
            items_per_step,
        }
    }

    /// Factory: reconstruct from a checkpoint blob.
    ///
    /// Format: `[remaining: u64 LE][total: u64 LE][items_per_step: u64 LE]` (24 bytes).
    pub fn from_checkpoint(id: &str, priority: u8, state: &[u8]) -> Self {
        assert!(state.len() >= 24, "StubJob checkpoint too short");
        let remaining = u64::from_le_bytes(state[0..8].try_into().unwrap());
        let total = u64::from_le_bytes(state[8..16].try_into().unwrap());
        let items_per_step = u64::from_le_bytes(state[16..24].try_into().unwrap());
        Self {
            id: id.to_string(),
            priority,
            remaining,
            total,
            items_per_step,
        }
    }
}

impl Job for StubJob {
    fn id(&self) -> &str {
        &self.id
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn step(&mut self, budget: StepBudget) -> StepOutcome {
        let to_do = (budget.max_items as u64)
            .min(self.items_per_step)
            .min(self.remaining);
        self.remaining -= to_do;
        let done = self.total - self.remaining;
        if self.remaining == 0 {
            StepOutcome::Finished
        } else {
            StepOutcome::Progress {
                done,
                total: self.total,
            }
        }
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(24);
        v.extend_from_slice(&self.remaining.to_le_bytes());
        v.extend_from_slice(&self.total.to_le_bytes());
        v.extend_from_slice(&self.items_per_step.to_le_bytes());
        v
    }

    fn restore(&mut self, state: &[u8]) {
        assert!(state.len() >= 24, "StubJob checkpoint too short");
        // Lossless: restore every field the checkpoint carries, not just
        // `remaining`. `total` and `items_per_step` are needed for the job to
        // keep stepping correctly after a restore.
        self.remaining = u64::from_le_bytes(state[0..8].try_into().unwrap());
        self.total = u64::from_le_bytes(state[8..16].try_into().unwrap());
        self.items_per_step = u64::from_le_bytes(state[16..24].try_into().unwrap());
    }
}

// ---------------------------------------------------------------------------
// T6: Deterministic simulator
// ---------------------------------------------------------------------------

/// Configuration for [`run_sim`].
pub struct SimConfig {
    /// The scripted signal timeline (see [`FakeSignals`]).
    pub timeline: Vec<TimelineSegment>,
    /// Stub jobs to submit before the simulation starts.
    pub jobs: Vec<StubJob>,
}

/// One entry in the deterministic execution trace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SimTraceEntry {
    /// 1-indexed tick number (same as scheduler tick_count after the tick).
    pub tick: u64,
    /// "ran" (step executed), "blocked" (signals false), "drained" (queue empty).
    pub action: String,
    /// Job id that ran, if any.
    pub job_id: Option<String>,
    /// Work done this step (for "ran") or null.
    pub work_done: Option<u64>,
    /// Cumulative remaining work after this tick.
    pub pending: u64,
}

/// Result of a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimResult {
    /// The timeline that was used (for gate verification).
    pub timeline: Vec<TimelineSegment>,
    pub total_work: u64,
    pub completed_work: u64,
    /// Work items processed during blocked windows (should be 0).
    pub work_during_blocked: u64,
    pub total_ticks: u64,
    pub trace: Vec<SimTraceEntry>,
}

/// Gate check result after verifying a SimResult against its timeline.
#[derive(Debug)]
pub struct GateVerdict {
    pub passed: bool,
    pub failures: Vec<String>,
}

/// Run a fully deterministic simulation over a scripted timeline.
///
/// Given a fixed `SimConfig`, produces a byte-identical trace every time
/// (same timeline + same jobs → same trace).
pub fn run_sim(config: SimConfig) -> SimResult {
    let signals = FakeSignals::new(config.timeline.clone());
    let dir = temp_dir_for_sim();
    let _ = fs::remove_dir_all(&dir);
    let mut sched = Scheduler::open(&dir, signals).expect("open scheduler");

    let total_work: u64 = config.jobs.iter().map(|j| j.total).sum();
    for job in config.jobs {
        sched.submit(Box::new(job)).expect("submit job");
    }

    let mut trace: Vec<SimTraceEntry> = Vec::new();
    let max_ticks = 10_000_000u64;

    for _ in 0..max_ticks {
        let pending_before = sched.pending_work();
        let top_id = peek_top_id(&sched.heap);

        let ran = sched.tick().expect("tick");
        let pending_after = sched.pending_work();

        if ran {
            let work_done = pending_before.saturating_sub(pending_after);
            trace.push(SimTraceEntry {
                tick: sched.tick_count(),
                action: "ran".into(),
                job_id: top_id,
                work_done: Some(work_done),
                pending: pending_after,
            });
        } else if pending_before == 0 {
            trace.push(SimTraceEntry {
                tick: sched.tick_count(),
                action: "drained".into(),
                job_id: None,
                work_done: None,
                pending: 0,
            });
            break;
        } else {
            trace.push(SimTraceEntry {
                tick: sched.tick_count(),
                action: "blocked".into(),
                job_id: None,
                work_done: None,
                pending: pending_after,
            });
        }
    }

    // Verify gate: replay the timeline against the trace to compute
    // work_during_blocked.
    let work_during_blocked =
        verify_gate_internal(&config.timeline, &trace, total_work, sched.pending_work());

    let completed = total_work.saturating_sub(sched.pending_work());

    let _ = fs::remove_dir_all(&dir);

    SimResult {
        timeline: config.timeline,
        total_work,
        completed_work: completed,
        work_during_blocked,
        total_ticks: sched.tick_count(),
        trace,
    }
}

fn peek_top_id(heap: &BinaryHeap<QueueEntry>) -> Option<String> {
    // BinaryHeap only exposes peek(), which gives a reference to the max element.
    // But iter() is arbitrary order. Since peek() gives the max, and we want the
    // top job id, we use peek().
    heap.peek().map(|e| e.job.id().to_string())
}

/// Verify gate metrics from a trace + timeline.
///
/// Returns the count of work items processed during blocked windows (should be 0).
fn verify_gate_internal(
    timeline: &[TimelineSegment],
    trace: &[SimTraceEntry],
    _total_work: u64,
    _remaining: u64,
) -> u64 {
    let mut blocked_work: u64 = 0;

    for entry in trace {
        if entry.action != "ran" {
            continue;
        }
        // Determine if this tick was in a blocked window according to the timeline.
        let seg = segment_at_tick(timeline, entry.tick);
        let signals_ok = seg.on_power && seg.user_idle && seg.thermal_ok;
        if !signals_ok {
            blocked_work += entry.work_done.unwrap_or(0);
        }
    }

    blocked_work
}

fn segment_at_tick(timeline: &[TimelineSegment], tick: u64) -> &TimelineSegment {
    for seg in timeline {
        if tick >= seg.tick_start && tick <= seg.tick_end {
            return seg;
        }
    }
    timeline.last().unwrap()
}

/// Verify all six gate metrics against a SimResult.
///
/// Returns a [`GateVerdict`] with pass/fail and a list of failures.
pub fn check_gate(result: &SimResult) -> GateVerdict {
    let mut failures: Vec<String> = Vec::new();

    // 1. Work during blocked windows = 0
    let blocked_work = verify_gate_internal(&result.timeline, &result.trace, result.total_work, 0);
    if blocked_work > 0 {
        failures.push(format!("work_during_blocked={blocked_work} (budget: 0)"));
    }

    // 2. Backlog drained: 100%
    if result.completed_work != result.total_work {
        failures.push(format!(
            "backlog drain: {}/{} (budget: 100%)",
            result.completed_work, result.total_work
        ));
    }

    // 3. Resume correctness: a mid-run crash+restore drains the same total work
    //    with nothing lost or double-counted. Proven on the real persistence
    //    path by `resume_is_lossless` (round-trip) and `scheduler_checkpoint_persistence`.

    // 4. Per-step budget respected: each "ran" entry must have work_done <= STEP_MAX_ITEMS.
    for entry in &result.trace {
        if let Some(done) = entry.work_done {
            if done > STEP_MAX_ITEMS as u64 {
                failures.push(format!(
                    "tick {}: work_done={done} exceeds step budget ({STEP_MAX_ITEMS})",
                    entry.tick
                ));
            }
        }
    }

    // 5. Determinism: the caller verifies by running twice (test and golden test do this)

    // 6. Pause latency: work stops at next step boundary (verified by trace having no
    //    "ran" entries in blocked windows — same as check 1)

    GateVerdict {
        passed: failures.is_empty(),
        failures,
    }
}

fn temp_dir_for_sim() -> PathBuf {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    static CNT: AtomicU64 = AtomicU64::new(0);
    let n = CNT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("skinki_sleep_sim_{}_{n}", std::process::id()))
}

// ---------------------------------------------------------------------------
// Default (non-macOS) PowerSignals: always blocked (conservative)
// ---------------------------------------------------------------------------

/// Default (non-macOS) PowerSignals: always blocked.
///
/// On non-macOS platforms, the sleep engine won't run background work until
/// a platform-specific implementation is provided.
pub struct DefaultSignals;

impl PowerSignals for DefaultSignals {
    fn on_external_power(&self) -> bool {
        false
    }
    fn user_idle(&self) -> bool {
        false
    }
    fn thermal_ok(&self) -> bool {
        false
    }
}

// macOS implementation lives in macos.rs (behind #[cfg(target_os = "macos")])
#[cfg(target_os = "macos")]
pub mod macos;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- T1: StubJob basics ---

    #[test]
    fn stub_job_advances_work() {
        let mut job = StubJob::new("test", 5, 100, 10);
        assert_eq!(job.id(), "test");
        assert_eq!(job.priority(), 5);

        let out = job.step(StepBudget {
            max_items: 10,
            soft_deadline_ms: 100,
        });
        match out {
            StepOutcome::Progress { done, total } => {
                assert_eq!(done, 10);
                assert_eq!(total, 100);
            }
            StepOutcome::Finished => panic!("expected Progress"),
        }

        for _ in 0..8 {
            job.step(StepBudget {
                max_items: 10,
                soft_deadline_ms: 100,
            });
        }
        let out = job.step(StepBudget {
            max_items: 10,
            soft_deadline_ms: 100,
        });
        assert!(matches!(out, StepOutcome::Finished));
    }

    #[test]
    fn stub_job_respects_budget_cap() {
        let mut job = StubJob::new("cap", 3, 50, 10);
        let out = job.step(StepBudget {
            max_items: 3,
            soft_deadline_ms: 100,
        });
        match out {
            StepOutcome::Progress { done, .. } => {
                assert_eq!(done, 3, "budget max_items=3 caps work per step");
            }
            StepOutcome::Finished => panic!("expected Progress"),
        }
    }

    // --- T2: Policy gate — no work while blocked ---

    #[test]
    fn no_work_during_blocked_signals() {
        let timeline = vec![TimelineSegment {
            tick_start: 1,
            tick_end: 1000,
            on_power: false,
            user_idle: false,
            thermal_ok: false,
        }];
        let signals = FakeSignals::new(timeline);
        let dir = temp_dir_for_sim();
        let _ = fs::remove_dir_all(&dir);
        let mut sched = Scheduler::open(&dir, signals).unwrap();

        let job = StubJob::new("blocked_test", 10, 100, 10);
        sched.submit(Box::new(job)).unwrap();

        for _ in 0..100 {
            let ran = sched.tick().unwrap();
            assert!(!ran, "work ran while signals were blocked");
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // --- T2: Priority ordering ---

    #[test]
    fn priority_ordering() {
        let timeline = vec![TimelineSegment {
            tick_start: 1,
            tick_end: 1000,
            on_power: true,
            user_idle: true,
            thermal_ok: true,
        }];
        let signals = FakeSignals::new(timeline);
        let dir = temp_dir_for_sim();
        let _ = fs::remove_dir_all(&dir);
        let mut sched = Scheduler::open(&dir, signals).unwrap();

        let low = StubJob::new("low", 1, 10, 10);
        let high = StubJob::new("high", 10, 10, 10);
        sched.submit(Box::new(low)).unwrap();
        sched.submit(Box::new(high)).unwrap();

        // First tick should process high-priority job to completion.
        let _ = sched.tick().unwrap();
        assert_eq!(sched.pending_work(), 10); // only low (10) remains

        let _ = fs::remove_dir_all(&dir);
    }

    // --- T3: Checkpoint → restore round-trip property ---

    #[test]
    fn checkpoint_restore_roundtrip() {
        let mut job = StubJob::new("rt", 5, 100, 7);
        job.step(StepBudget {
            max_items: 10,
            soft_deadline_ms: 100,
        }); // -7 → 93
        job.step(StepBudget {
            max_items: 10,
            soft_deadline_ms: 100,
        }); // -7 → 86

        let ck = job.checkpoint();
        assert_eq!(ck.len(), 24);

        let restored = StubJob::from_checkpoint("rt", 5, &ck);
        assert_eq!(restored.remaining, 86);

        let ck2 = restored.checkpoint();
        let restored2 = StubJob::from_checkpoint("rt", 5, &ck2);
        assert_eq!(restored2.remaining, 86);
    }

    #[test]
    fn checkpoint_restore_random_sizes() {
        for seed in 0..50u64 {
            let total = 100 + (seed % 900);
            let items_per_step = 1 + (seed % 50);
            let steps_to_run = (seed % 20) as usize;

            let mut job = StubJob::new("prop", 5, total, items_per_step);
            for _ in 0..steps_to_run {
                job.step(StepBudget {
                    max_items: items_per_step as u32,
                    soft_deadline_ms: 100,
                });
            }
            let remaining_before = job.remaining;
            let ck = job.checkpoint();
            let restored = StubJob::from_checkpoint("prop", 5, &ck);
            assert_eq!(
                restored.remaining, remaining_before,
                "seed {seed}: remaining mismatch after restore"
            );
        }
    }

    // --- T4: Deterministic golden trace ---

    #[test]
    fn deterministic_trace() {
        let timeline = vec![
            TimelineSegment {
                tick_start: 1,
                tick_end: 5,
                on_power: true,
                user_idle: true,
                thermal_ok: true,
            },
            TimelineSegment {
                tick_start: 6,
                tick_end: 10,
                on_power: false,
                user_idle: false,
                thermal_ok: false,
            },
            TimelineSegment {
                tick_start: 11,
                tick_end: 20,
                on_power: true,
                user_idle: true,
                thermal_ok: true,
            },
        ];

        let make = || -> Vec<SimTraceEntry> {
            let jobs = vec![StubJob::new("a", 5, 30, 10), StubJob::new("b", 3, 20, 10)];
            run_sim(SimConfig {
                timeline: timeline.clone(),
                jobs,
            })
            .trace
        };

        let t1 = make();
        let t2 = make();
        assert_eq!(t1, t2, "trace must be deterministic");
    }

    // --- Full simulation golden: all 6 gate metrics ---

    #[test]
    fn full_sim_all_metrics_pass() {
        let timeline = vec![
            TimelineSegment {
                tick_start: 1,
                tick_end: 20,
                on_power: true,
                user_idle: true,
                thermal_ok: true,
            },
            TimelineSegment {
                tick_start: 21,
                tick_end: 30,
                on_power: false,
                user_idle: false,
                thermal_ok: false,
            },
            TimelineSegment {
                tick_start: 31,
                tick_end: 50,
                on_power: true,
                user_idle: true,
                thermal_ok: true,
            },
            TimelineSegment {
                tick_start: 51,
                tick_end: 60,
                on_power: true,
                user_idle: false,
                thermal_ok: true,
            },
            TimelineSegment {
                tick_start: 61,
                tick_end: 200,
                on_power: true,
                user_idle: true,
                thermal_ok: true,
            },
        ];

        let jobs = vec![
            StubJob::new("heavy", 10, 200, 15),
            StubJob::new("light", 5, 100, 20),
        ];

        let result = run_sim(SimConfig {
            timeline: timeline.clone(),
            jobs,
        });

        // 1. Backlog drained: 100%
        assert_eq!(
            result.completed_work, result.total_work,
            "backlog must be 100% drained"
        );
        assert_eq!(result.total_work, 300);

        // 2. No work during blocked windows
        for entry in &result.trace {
            if entry.action == "blocked" {
                assert!(
                    entry.work_done.is_none(),
                    "blocked tick produced work: {entry:?}"
                );
            }
        }

        // 3. Gate check passes
        let verdict = check_gate(&result);
        assert!(
            verdict.passed,
            "gate failed: {}",
            verdict.failures.join("; ")
        );

        // 4. All entries respect step budget
        for entry in &result.trace {
            if let Some(done) = entry.work_done {
                assert!(
                    done <= STEP_MAX_ITEMS as u64,
                    "step budget exceeded at tick {}",
                    entry.tick
                );
            }
        }

        // 5. Work during blocked = 0 (verified by gate)
        assert_eq!(result.work_during_blocked, 0);
    }

    // --- Pause latency: work stops at step boundary when signals flip ---

    #[test]
    fn pause_latency_at_step_boundary() {
        // Active for 2 ticks, then blocked — any in-progress work must stop
        // after the current step completes (i.e., no partial step).
        let timeline = vec![
            TimelineSegment {
                tick_start: 1,
                tick_end: 2,
                on_power: true,
                user_idle: true,
                thermal_ok: true,
            },
            TimelineSegment {
                tick_start: 3,
                tick_end: 100,
                on_power: false,
                user_idle: false,
                thermal_ok: false,
            },
        ];

        let jobs = vec![StubJob::new("j", 10, 50, 15)];
        let result = run_sim(SimConfig { timeline, jobs });

        // At most 2 "ran" entries (ticks 1 and 2).
        let ran_count = result.trace.iter().filter(|e| e.action == "ran").count();
        assert!(ran_count <= 2);

        // After tick 2, all subsequent ticks should be "blocked" until we stop.
        for entry in result.trace.iter().skip(ran_count) {
            assert!(
                entry.action == "blocked" || entry.action == "drained",
                "expected blocked/drained after signal flip, got {:?}",
                entry.action
            );
        }
    }

    // --- Scheduler restore from checkpoint file ---

    #[test]
    fn scheduler_checkpoint_persistence() {
        let timeline = vec![TimelineSegment {
            tick_start: 1,
            tick_end: 1000,
            on_power: true,
            user_idle: true,
            thermal_ok: true,
        }];
        let dir = std::env::temp_dir().join(format!("skinki_sleep_ckpt_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        // First session
        {
            let signals = FakeSignals::new(timeline.clone());
            let mut sched = Scheduler::open(&dir, signals).unwrap();
            let job = StubJob::new("persist", 7, 50, 12);
            sched.submit(Box::new(job)).unwrap();
            // Run a few ticks
            for _ in 0..3 {
                sched.tick().unwrap();
            }
            // Drop scheduler (simulates crash)
        }

        // Second session: read checkpoint, verify entries exist
        {
            let signals = FakeSignals::new(timeline.clone());
            let sched = Scheduler::open(&dir, signals).unwrap();
            let _entries = sched.load_checkpoint_entries().unwrap();
            assert!(
                !_entries.is_empty(),
                "checkpoint file should persist entries"
            );
            assert_eq!(_entries[0].id, "persist");
            assert_eq!(_entries[0].priority, 7);
        }

        // Third session: restore and continue
        {
            let signals = FakeSignals::new(timeline);
            let mut sched = Scheduler::open(&dir, signals).unwrap();
            let factory = |id: &str, pri: u8, state: &[u8]| -> Box<dyn Job> {
                Box::new(StubJob::from_checkpoint(id, pri, state))
            };
            sched.restore_from_checkpoint(factory).unwrap();
            // Drain remaining work
            while sched.pending_work() > 0 {
                sched.tick().unwrap();
            }
            assert_eq!(sched.pending_work(), 0);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // --- T3: StubJob::restore is lossless (all fields, not just remaining) ---

    #[test]
    fn stub_job_restore_is_lossless() {
        let mut job = StubJob::new("orig", 5, 300, 17);
        job.step(StepBudget {
            max_items: 17,
            soft_deadline_ms: 100,
        });
        let ck = job.checkpoint();

        // A blank job restored from the checkpoint must match field-for-field.
        let mut blank = StubJob::new("orig", 5, 1, 1);
        blank.restore(&ck);
        assert_eq!(blank.remaining, job.remaining);
        assert_eq!(blank.total, job.total);
        assert_eq!(blank.items_per_step, job.items_per_step);
        // And it must keep stepping correctly (needs `total`/`items_per_step`).
        assert_eq!(blank.checkpoint(), ck);
    }

    // --- T3: submit before restore is rejected (no silent queue destruction) ---

    #[test]
    fn submit_before_restore_is_rejected() {
        let timeline = vec![TimelineSegment {
            tick_start: 1,
            tick_end: 100,
            on_power: true,
            user_idle: true,
            thermal_ok: true,
        }];
        let dir = std::env::temp_dir().join(format!("skinki_sleep_guard_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        // Session 1: persist a job.
        {
            let mut sched = Scheduler::open(&dir, FakeSignals::new(timeline.clone())).unwrap();
            sched
                .submit(Box::new(StubJob::new("persisted", 5, 100, 10)))
                .unwrap();
        }
        // Session 2: opening over the checkpoint must refuse a naive submit
        // (which would overwrite and destroy the persisted queue).
        {
            let mut sched = Scheduler::open(&dir, FakeSignals::new(timeline.clone())).unwrap();
            let err = sched.submit(Box::new(StubJob::new("intruder", 9, 50, 10)));
            assert!(err.is_err(), "submit before restore must be rejected");
            // After restoring, submit is allowed again and the queue is intact.
            let factory = |id: &str, pri: u8, state: &[u8]| -> Box<dyn Job> {
                Box::new(StubJob::from_checkpoint(id, pri, state))
            };
            sched.restore_from_checkpoint(factory).unwrap();
            sched
                .submit(Box::new(StubJob::new("ok_now", 1, 10, 10)))
                .unwrap();
            let ids: Vec<String> = sched
                .load_checkpoint_entries()
                .unwrap()
                .into_iter()
                .map(|e| e.id)
                .collect();
            assert!(
                ids.iter().any(|i| i == "persisted"),
                "persisted job survived"
            );
            assert!(ids.iter().any(|i| i == "ok_now"));
        }
        let _ = fs::remove_dir_all(&dir);
    }

    // --- T6: golden locked-hash trace (pins the scheduling policy byte-for-byte) ---

    fn fnv1a64(data: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in data {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    #[test]
    fn golden_trace_hash_is_locked() {
        let timeline = vec![
            TimelineSegment {
                tick_start: 1,
                tick_end: 5,
                on_power: true,
                user_idle: true,
                thermal_ok: true,
            },
            TimelineSegment {
                tick_start: 6,
                tick_end: 10,
                on_power: false,
                user_idle: false,
                thermal_ok: false,
            },
            TimelineSegment {
                tick_start: 11,
                tick_end: 20,
                on_power: true,
                user_idle: true,
                thermal_ok: true,
            },
        ];
        let jobs = vec![StubJob::new("a", 5, 30, 10), StubJob::new("b", 3, 20, 10)];
        let result = run_sim(SimConfig { timeline, jobs });
        let bytes = serde_json::to_vec(&result.trace).expect("serialise trace");
        let hash = fnv1a64(&bytes);
        // Locked golden — any change to the scheduling policy moves this hash.
        assert_eq!(
            hash, 0xe345_a3f1_e1f2_0148,
            "sleep scheduling trace changed (hash {hash:#018x}); if intentional, update the golden"
        );
    }

    // --- T3/T6: a mid-run crash + restore drains the identical total work ---

    #[test]
    fn resume_is_lossless() {
        // All-active timeline: every tick runs, so a restart's tick-count reset
        // is irrelevant and we isolate the persistence/restore path itself.
        let timeline = vec![TimelineSegment {
            tick_start: 1,
            tick_end: 100_000,
            on_power: true,
            user_idle: true,
            thermal_ok: true,
        }];
        let make_jobs = || {
            vec![
                StubJob::new("j1", 7, 230, 13),
                StubJob::new("j2", 7, 170, 9), // same priority as j1 — exercises seq tie-break
                StubJob::new("j3", 3, 90, 25),
            ]
        };
        let total: u64 = make_jobs().iter().map(|j| j.total).sum();
        let factory = |id: &str, pri: u8, state: &[u8]| -> Box<dyn Job> {
            Box::new(StubJob::from_checkpoint(id, pri, state))
        };

        // Run A: uninterrupted.
        let dir_a = std::env::temp_dir().join(format!("skinki_sleep_resA_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir_a);
        let ticks_a = {
            let mut s = Scheduler::open(&dir_a, FakeSignals::new(timeline.clone())).unwrap();
            for j in make_jobs() {
                s.submit(Box::new(j)).unwrap();
            }
            let mut n = 0u64;
            while s.pending_work() > 0 {
                s.tick().unwrap();
                n += 1;
            }
            assert_eq!(s.pending_work(), 0);
            n
        };

        // Run B: crash + restore after 5 ticks, then drain.
        let dir_b = std::env::temp_dir().join(format!("skinki_sleep_resB_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir_b);
        {
            let mut s = Scheduler::open(&dir_b, FakeSignals::new(timeline.clone())).unwrap();
            for j in make_jobs() {
                s.submit(Box::new(j)).unwrap();
            }
            for _ in 0..5 {
                s.tick().unwrap();
            }
            // drop `s` — simulated crash; checkpoint is on disk.
        }
        let ticks_b = {
            let mut s = Scheduler::open(&dir_b, FakeSignals::new(timeline.clone())).unwrap();
            s.restore_from_checkpoint(factory).unwrap();
            let mut n = 5u64;
            while s.pending_work() > 0 {
                s.tick().unwrap();
                n += 1;
            }
            assert_eq!(s.pending_work(), 0, "resumed run must fully drain");
            n
        };

        // Losslessness: same number of work-steps to drain — no step lost or
        // repeated across the crash, and tie-break order survived the restore.
        assert_eq!(ticks_a, ticks_b, "resume took a different number of steps");
        assert!(total > 0);

        let _ = fs::remove_dir_all(&dir_a);
        let _ = fs::remove_dir_all(&dir_b);
    }
}
