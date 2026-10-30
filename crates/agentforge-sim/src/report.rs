//! Simulation results and their rendering.

use serde::{Deserialize, Serialize};

/// Metrics of one simulation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimReport {
    pub policy: String,
    pub sessions: u32,
    pub sessions_done: u32,
    pub steps: u64,
    pub makespan_s: f64,
    pub ttft_mean_s: f64,
    pub ttft_p50_s: f64,
    pub ttft_p90_s: f64,
    pub ttft_p99_s: f64,
    pub step_p50_s: f64,
    pub step_p90_s: f64,
    pub step_p99_s: f64,
    pub recompute_events: u64,
    pub recompute_tokens: u64,
    pub foreground_kv_wait_s: f64,
    pub fetched_gb: f64,
    pub prefetched_gb: f64,
    pub spill_gb: f64,
    pub hidden_spill_gb: f64,
    pub evictions: u64,
    pub evicted_drops: u64,
    pub planned_drops: u64,
    pub capacity_waits: u64,
    pub prefetch_skipped: u64,
    pub device_busy_s: f64,
    pub prefill_tokens: u64,
}

/// Nearest-rank percentile of `samples` (0 for an empty slice).
pub fn percentile(samples: &[f64], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    }
}

/// A markdown table with one column per report.
pub fn render_table(reports: &[SimReport]) -> String {
    let rows: Vec<(&str, Vec<String>)> = vec![
        (
            "sessions done / total",
            reports
                .iter()
                .map(|r| format!("{} / {}", r.sessions_done, r.sessions))
                .collect(),
        ),
        (
            "steps",
            reports.iter().map(|r| r.steps.to_string()).collect(),
        ),
        (
            "makespan (s)",
            reports
                .iter()
                .map(|r| format!("{:.1}", r.makespan_s))
                .collect(),
        ),
        (
            "TTFT mean (s)",
            reports
                .iter()
                .map(|r| format!("{:.3}", r.ttft_mean_s))
                .collect(),
        ),
        (
            "TTFT p50 (s)",
            reports
                .iter()
                .map(|r| format!("{:.3}", r.ttft_p50_s))
                .collect(),
        ),
        (
            "TTFT p90 (s)",
            reports
                .iter()
                .map(|r| format!("{:.3}", r.ttft_p90_s))
                .collect(),
        ),
        (
            "TTFT p99 (s)",
            reports
                .iter()
                .map(|r| format!("{:.3}", r.ttft_p99_s))
                .collect(),
        ),
        (
            "step latency p50 (s)",
            reports
                .iter()
                .map(|r| format!("{:.2}", r.step_p50_s))
                .collect(),
        ),
        (
            "step latency p90 (s)",
            reports
                .iter()
                .map(|r| format!("{:.2}", r.step_p90_s))
                .collect(),
        ),
        (
            "step latency p99 (s)",
            reports
                .iter()
                .map(|r| format!("{:.2}", r.step_p99_s))
                .collect(),
        ),
        (
            "prefix recomputes",
            reports
                .iter()
                .map(|r| r.recompute_events.to_string())
                .collect(),
        ),
        (
            "recomputed tokens",
            reports
                .iter()
                .map(|r| r.recompute_tokens.to_string())
                .collect(),
        ),
        (
            "foreground KV wait (s)",
            reports
                .iter()
                .map(|r| format!("{:.1}", r.foreground_kv_wait_s))
                .collect(),
        ),
        (
            "fetched in foreground (GB)",
            reports
                .iter()
                .map(|r| format!("{:.0}", r.fetched_gb))
                .collect(),
        ),
        (
            "prefetched (GB)",
            reports
                .iter()
                .map(|r| format!("{:.0}", r.prefetched_gb))
                .collect(),
        ),
        (
            "spilled during waits (GB)",
            reports
                .iter()
                .map(|r| format!("{:.0}", r.spill_gb))
                .collect(),
        ),
        (
            "spills hidden by the wait (GB)",
            reports
                .iter()
                .map(|r| format!("{:.0}", r.hidden_spill_gb))
                .collect(),
        ),
        (
            "pressure evictions",
            reports.iter().map(|r| r.evictions.to_string()).collect(),
        ),
        (
            "evictions that dropped KV",
            reports
                .iter()
                .map(|r| r.evicted_drops.to_string())
                .collect(),
        ),
        (
            "planned drops",
            reports
                .iter()
                .map(|r| r.planned_drops.to_string())
                .collect(),
        ),
        (
            "capacity waits",
            reports
                .iter()
                .map(|r| r.capacity_waits.to_string())
                .collect(),
        ),
        (
            "prefetches skipped",
            reports
                .iter()
                .map(|r| r.prefetch_skipped.to_string())
                .collect(),
        ),
        (
            "device busy (s)",
            reports
                .iter()
                .map(|r| format!("{:.1}", r.device_busy_s))
                .collect(),
        ),
        (
            "prefill tokens",
            reports
                .iter()
                .map(|r| r.prefill_tokens.to_string())
                .collect(),
        ),
    ];
    let mut out = String::new();
    out.push_str("| metric |");
    for r in reports {
        out.push_str(&format!(" {} |", r.policy));
    }
    out.push('\n');
    out.push_str("|---|");
    for _ in reports {
        out.push_str("---:|");
    }
    out.push('\n');
    for (name, values) in rows {
        out.push_str(&format!("| {name} |"));
        for v in values {
            out.push_str(&format!(" {v} |"));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 0.5), 51.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 1.0), 100.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert!((mean(&v) - 50.5).abs() < 1e-12);
    }
}
