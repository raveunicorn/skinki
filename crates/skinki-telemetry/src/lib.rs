//! Telemetry for the hard fitness budgets (latency, RAM, and — later — battery).
//!
//! This is the only crate permitted to use `unsafe`, and only to read process
//! resource usage via `getrusage`. Everything else stays safe by construction.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Summary statistics over a set of measured query latencies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencySummary {
    pub count: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
}

impl LatencySummary {
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return LatencySummary::default();
        }
        let mut ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = ms.len();
        let pct = |p: f64| {
            let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
            ms[idx]
        };
        let sum: f64 = ms.iter().sum();
        LatencySummary {
            count: n,
            p50_ms: pct(0.50),
            p95_ms: pct(0.95),
            mean_ms: sum / n as f64,
            max_ms: *ms.last().unwrap(),
        }
    }
}

/// Peak resident set size of the current process, in bytes.
///
/// `getrusage(RUSAGE_SELF).ru_maxrss` reports bytes on macOS and kilobytes on
/// Linux, so we normalize. On non-unix targets this returns `None`.
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: `getrusage` only writes into the zero-initialized `usage`
        // struct and returns a status code; no aliasing or lifetime concerns.
        unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
                return None;
            }
            let maxrss = usage.ru_maxrss as u64;
            #[cfg(target_os = "macos")]
            {
                Some(maxrss)
            }
            #[cfg(not(target_os = "macos"))]
            {
                Some(maxrss * 1024)
            }
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Battery-drain measurement. Stubbed for Stage 0.
///
/// Stage 4 ("sleep" consolidation) wires this to macOS power telemetry
/// (IOKit `IOPSCopyPowerSourcesInfo` or parsing `pmset -g batt`) so background
/// work can be held to a hard battery budget.
pub fn battery_drain_percent_per_hour() -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_summary_orders_percentiles() {
        let ds: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let s = LatencySummary::from_durations(&ds);
        assert_eq!(s.count, 100);
        assert!(s.p50_ms <= s.p95_ms);
        assert!(s.p95_ms <= s.max_ms);
        assert!((s.max_ms - 100.0).abs() < 1e-9);
    }

    #[test]
    fn empty_is_zero() {
        let s = LatencySummary::from_durations(&[]);
        assert_eq!(s.count, 0);
        assert_eq!(s.mean_ms, 0.0);
    }

    #[test]
    fn rss_is_positive_on_unix() {
        if cfg!(unix) {
            assert!(peak_rss_bytes().unwrap() > 0);
        }
    }
}
