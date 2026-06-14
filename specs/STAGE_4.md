# Stage 4 — "Sleep" engine: interruptible background consolidation scheduler (SPEC)

- **Status:** **done** — scheduler + deterministic simulator shipped; `sleep-sim
  --assert-gate` green (all six metrics). Real consolidation jobs are Stage 3/5.
- **Owner of the design (frontier/human):** done below — the scheduler is a
  platform-agnostic, trait-driven skeleton. The *real* consolidation jobs (Leiden
  communities, RAPTOR summaries, PPR precompute) are Stage 3/5 and are **not** in
  this hand-off.
- **Delegatable to (cheaper model):** **yes** — all impl tickets. The macOS power
  binding is mechanical and behind a trait with a deterministic fake for tests.

> Read [`../../AGENTS.md`](../../AGENTS.md). Gate is law; determinism mandatory;
> no new deps (pure Rust + allowed: `serde`, `serde_json`, `libc`, `clap`,
> `anyhow`).

## 1. Hypothesis

All expensive consolidation can run as **interruptible, resumable, checkpointed
background jobs that execute only while the machine is idle, on power, and
thermally OK** — draining a backlog over days without any perceptible realtime or
battery impact. A deterministic simulator can prove the scheduling policy is
correct *before* any real job exists.

## 2. Budgets / fitness function (the gate)

Measured by `skinki sleep-sim` over a scripted signal timeline + synthetic backlog.

| Metric | Budget | How measured |
| --- | --- | --- |
| Work during "blocked" windows | **0** | sim asserts no `step()` runs while !(power & idle & thermal) |
| Backlog drained | 100% of total work | sim runs the scripted week to completion |
| Resume correctness | exact | checkpoint→restart→restore reproduces remaining work |
| Per-step budget respected | always | no `step()` exceeds its `StepBudget` (item/time cap) |
| Determinism | byte-identical trace | same (timeline, seed) → identical execution log |
| Pause latency | <= 1 step boundary | work stops at the next `step()` boundary when signals flip |

`sleep-sim --assert-gate` exits non-zero on any violation.

## 3. Public interface

New crate `skinki-sleep`. Contract:

```rust
/// Environmental gating signals. macOS impl behind cfg; FakeSignals for tests/sim.
pub trait PowerSignals {
    fn on_external_power(&self) -> bool;
    fn user_idle(&self) -> bool;     // no input for >= idle_threshold
    fn thermal_ok(&self) -> bool;    // not throttling
}

pub struct StepBudget { pub max_items: u32, pub soft_deadline_ms: u32 }

pub enum StepOutcome { Progress { done: u64, total: u64 }, Finished }

/// A unit of consolidation work. Must be chunked so it can stop at boundaries.
pub trait Job {
    fn id(&self) -> &str;
    fn priority(&self) -> u8;                 // higher runs first
    fn step(&mut self, budget: StepBudget) -> StepOutcome;
    fn checkpoint(&self) -> Vec<u8>;          // serialize remaining work
    fn restore(&mut self, state: &[u8]);      // resume after restart
}

pub struct Scheduler<S: PowerSignals> { /* persistent queue + signals */ }

impl<S: PowerSignals> Scheduler<S> {
    pub fn open(dir: &std::path::Path, signals: S) -> anyhow::Result<Self>;
    pub fn submit(&mut self, job: Box<dyn Job>) -> anyhow::Result<()>;
    /// Run one scheduler tick: if signals allow, advance the top job by one
    /// step; else do nothing. Persists progress. Returns whether work ran.
    pub fn tick(&mut self) -> anyhow::Result<bool>;
    pub fn pending_work(&self) -> u64;        // sum of remaining over all jobs
}

/// Deterministic test/sim signal source driven by a scripted timeline.
pub struct FakeSignals { /* schedule of (tick_range -> signal state) */ }
```

### Design, locked

- **Policy:** a job runs (one `step`) on a `tick` **iff** `on_external_power() &&
  user_idle() && thermal_ok()`. Otherwise `tick` is a no-op (returns `false`).
- **Interruptibility:** work happens only inside `step()`, which is bounded by
  `StepBudget`. The scheduler re-checks signals *between* steps, so pause latency
  is at most one step.
- **Persistence:** after each `step`, write the job's `checkpoint()` to a small
  on-disk queue file (reuse `serde_json` or manual encoding). On `open`, restore
  in-flight jobs. This gives crash/restart safety for free.
- **macOS signals (impl ticket, behind `cfg(target_os = "macos")`):** power via
  `IOPSCopyPowerSourcesInfo` / `pmset -g batt`; idle via
  `CGEventSourceSecondsSinceLastEventType`; thermal via
  `NSProcessInfo.thermalState`. The simplest robust route is allowed (e.g. shell
  out to `pmset` for power) as long as it's wrapped behind `PowerSignals`. On
  non-macOS, a conservative default impl (returns "blocked") is fine.

## 4. Invariants

- A `step()` **never** runs unless all three signals are true at the tick.
- Submitting/draining is deterministic for a fixed `(timeline, seed, jobs)`.
- `checkpoint` → `restore` is lossless: remaining work after restore == before.
- No `unsafe` (the macOS binding uses safe FFI/`Command`, or `libc` only if
  unavoidable and quarantined). No new deps.

## 5. Test plan

- **Unit:** policy gate (no work while blocked); priority ordering; per-step cap.
- **Property:** checkpoint→restore round-trip preserves remaining work for random
  job sizes and interruption points.
- **Golden:** a scripted week timeline + a fixed backlog → assert a locked
  execution trace hash (determinism) and 100% drain.
- **Sim:** `sleep-sim` builds N stub jobs with known total work over a scripted
  timeline with alternating active/idle and battery/power windows; asserts the
  six gate metrics.
- **Gate command:** `cargo run --release -p skinki-harness -- sleep-sim --assert-gate`

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| D1: which real jobs + their chunking | design | frontier | Stage 3/5; **not in this hand-off** |
| T1: `skinki-sleep` crate + `Job`/`StepBudget`/`StepOutcome` traits | impl | cheaper | builds; trait objects work |
| T2: `Scheduler` with priority queue + signal-gated `tick` | impl | cheaper | policy unit tests pass |
| T3: on-disk checkpoint/restore queue (crash-safe) | impl | cheaper | restore property test passes |
| T4: `FakeSignals` scripted timeline + deterministic sim driver | impl | cheaper | golden trace stable |
| T5: macOS `PowerSignals` impl behind cfg + non-macOS default | impl | cheaper | compiles on macOS + linux; behind trait |
| T6: `sleep-sim` CLI + `--assert-gate` + tests | impl | cheaper | gate exits non-zero on violation |

## 7. Definition of done

- [x] `sleep-sim --assert-gate` green; all six gate metrics hold.
- [x] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [x] CI: add a `sleep gate` step.
- [x] ROADMAP Stage 4 row → done; this spec Status → done.

## 8. Out of scope

- The actual consolidation algorithms (communities/summaries/PPR) — **Stage 3/5**
  plug in as `Job` implementations later.
- Real battery-draw measurement on hardware — validated at **Stage 7** on a live
  M1 Air; this stage proves the *policy* in simulation.
