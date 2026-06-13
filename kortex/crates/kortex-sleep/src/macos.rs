//! macOS `PowerSignals` implementation using system command-line tools.
//!
//! All three signals are queried via `std::process::Command` with a short
//! time-based cache to avoid excessive shelling out.
//!
//! - Power: `pmset -g batt` → parse "AC Power" / "Battery Power"
//! - Idle: `ioreg -c IOHIDSystem -r -d 1 -n root` → parse HIDIdleTime (ns)
//! - Thermal: `pmset -g therm` → check "CPU_Scheduler_Limit" (100 = normal)

use std::process::Command;
use std::time::{Duration, Instant};

use super::PowerSignals;

/// Cached value with a TTL.
struct Cached<T> {
    value: T,
    at: Instant,
    ttl: Duration,
}

impl<T: Copy> Cached<T> {
    fn new(value: T, ttl: Duration) -> Self {
        Self {
            value,
            at: Instant::now(),
            ttl,
        }
    }

    fn get_or_update(&mut self, fetch: impl Fn() -> T) -> T {
        if self.at.elapsed() < self.ttl {
            return self.value;
        }
        self.value = fetch();
        self.at = Instant::now();
        self.value
    }
}

/// macOS `PowerSignals` using system utilities.
///
/// Each signal is cached independently with a 2-second TTL so the scheduler
/// can call `tick()` frequently without spawning a process every time.
pub struct MacOSSignals {
    idle_threshold_secs: u64,
    power_cache: std::cell::RefCell<Cached<bool>>,
    idle_cache: std::cell::RefCell<Cached<bool>>,
    thermal_cache: std::cell::RefCell<Cached<bool>>,
}

impl MacOSSignals {
    /// Create a new macOS signal source.
    ///
    /// `idle_threshold_secs` is the minimum idle time (seconds) before
    /// `user_idle()` returns true.
    pub fn new(idle_threshold_secs: u64) -> Self {
        let ttl = Duration::from_secs(2);
        Self {
            idle_threshold_secs,
            power_cache: std::cell::RefCell::new(Cached::new(false, ttl)),
            idle_cache: std::cell::RefCell::new(Cached::new(false, ttl)),
            thermal_cache: std::cell::RefCell::new(Cached::new(true, ttl)),
        }
    }
}

impl PowerSignals for MacOSSignals {
    fn on_external_power(&self) -> bool {
        self.power_cache.borrow_mut().get_or_update(check_ac_power)
    }

    fn user_idle(&self) -> bool {
        let threshold = self.idle_threshold_secs;
        self.idle_cache
            .borrow_mut()
            .get_or_update(|| check_user_idle(threshold))
    }

    fn thermal_ok(&self) -> bool {
        self.thermal_cache
            .borrow_mut()
            .get_or_update(check_thermal_ok)
    }
}

fn check_ac_power() -> bool {
    match Command::new("pmset").args(["-g", "batt"]).output() {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            // "Now drawing from 'AC Power'" vs "'Battery Power'". Require a
            // positive AC reading; anything ambiguous is treated as battery so
            // we never run background work on an unconfirmed power source.
            s.contains("AC Power") && !s.contains("Battery Power")
        }
        _ => false, // conservative: assume battery (block work) on any failure
    }
}

fn check_user_idle(threshold_secs: u64) -> bool {
    let idle_ns = match get_hid_idle_ns() {
        Some(ns) => ns,
        None => return false,
    };
    let idle_secs = idle_ns / 1_000_000_000;
    idle_secs >= threshold_secs
}

fn get_hid_idle_ns() -> Option<u64> {
    let out = Command::new("ioreg")
        .args(["-c", "IOHIDSystem", "-r", "-d", "1", "-n", "root"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    // Parse only the HIDIdleTime line (`"HIDIdleTime" = 123456789`). Matching on
    // any quoted field would grab an unrelated numeric property and report a
    // bogus idle time — which could let work run while the user is active.
    for line in s.lines() {
        if !line.contains("HIDIdleTime") {
            continue;
        }
        if let Some(eq) = line.find('=') {
            if let Ok(ns) = line[eq + 1..].trim().parse::<u64>() {
                return Some(ns);
            }
        }
    }
    None
}

fn check_thermal_ok() -> bool {
    match Command::new("pmset").args(["-g", "therm"]).output() {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            // "CPU_Scheduler_Limit = 100" means no throttling.
            // Values < 100 indicate thermal throttling.
            for line in s.lines() {
                if line.contains("CPU_Scheduler_Limit") {
                    if let Some(eq) = line.find('=') {
                        let val = line[eq + 1..].trim();
                        if let Ok(pct) = val.parse::<u32>() {
                            return pct >= 100;
                        }
                    }
                }
            }
            // If we can't parse, assume OK (the command succeeded but format
            // changed — don't block work on parse failure).
            true
        }
        // A missing/failed therm query shouldn't halt all consolidation.
        _ => true,
    }
}
