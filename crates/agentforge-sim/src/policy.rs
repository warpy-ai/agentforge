//! KV residency policies compared by the simulator.

use agentforge_compile::{plan_residency, ResidencyAction, TierBudget};
use agentforge_ir::LatencyModel;
use agentforge_machine::MemoryTier;

/// What a policy may look at when deciding.
#[derive(Debug, Clone, Copy)]
pub struct PolicyContext<'a> {
    pub now: f64,
    pub tiers: &'a [MemoryTier],
    pub tier_free: &'a [f64],
    pub tp: u32,
    pub device_free: f64,
    pub device_capacity: f64,
    pub recompute_s: f64,
    /// Typical prefill-plus-decode duration of a step on this device.
    pub step_hint_s: f64,
}

/// An idle session whose KV could be evicted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Victim {
    pub session: usize,
    pub kv_bytes: f64,
    pub last_use: f64,
    pub predicted_return: f64,
}

/// A KV residency policy.
pub trait Policy {
    fn name(&self) -> &'static str;

    /// Decide what to do with a session's KV when it starts waiting on a tool. `latency` is the
    /// latency model of the tool the agent program says it is calling; `mirrored_bytes` of the
    /// `kv_bytes` already sit in a tier from earlier write-through copies.
    fn on_tool_call(
        &self,
        cx: &PolicyContext<'_>,
        kv_bytes: f64,
        mirrored_bytes: f64,
        latency: &LatencyModel,
    ) -> ResidencyAction;

    /// Order candidate victims; the first entries are evicted first.
    fn rank_victims(&self, cx: &PolicyContext<'_>, victims: &mut [Victim]);
}

/// Reactive baseline: keep everything on the device and evict least recently used under
/// pressure. This is what a serving engine does without any knowledge of the agent program.
#[derive(Debug, Default, Clone, Copy)]
pub struct Lru;

impl Policy for Lru {
    fn name(&self) -> &'static str {
        "lru"
    }

    fn on_tool_call(
        &self,
        _: &PolicyContext<'_>,
        _: f64,
        _: f64,
        _: &LatencyModel,
    ) -> ResidencyAction {
        ResidencyAction::Keep
    }

    fn rank_victims(&self, _: &PolicyContext<'_>, victims: &mut [Victim]) {
        victims.sort_by(|a, b| a.last_use.total_cmp(&b.last_use));
    }
}

/// Reactive write-through baseline: while a session waits on a tool, copy the part of its KV
/// that no tier holds yet to the shallowest tier with room and keep the device copy, then evict
/// least recently used under pressure. An eviction writes only what the copies have not covered,
/// and nothing when they cover all of it. Like LRU it knows nothing about the agent program.
#[derive(Debug, Default, Clone, Copy)]
pub struct WriteThrough;

impl Policy for WriteThrough {
    fn name(&self) -> &'static str {
        "write-through"
    }

    fn on_tool_call(
        &self,
        cx: &PolicyContext<'_>,
        kv_bytes: f64,
        mirrored_bytes: f64,
        _: &LatencyModel,
    ) -> ResidencyAction {
        let delta = kv_bytes - mirrored_bytes;
        if delta <= 0.0 {
            return ResidencyAction::Keep;
        }
        match cx.tier_free.iter().position(|free| *free >= delta) {
            Some(tier_index) => ResidencyAction::WriteThrough { tier_index },
            None => ResidencyAction::Keep,
        }
    }

    fn rank_victims(&self, cx: &PolicyContext<'_>, victims: &mut [Victim]) {
        Lru.rank_victims(cx, victims);
    }
}

/// Program-aware plan: spill during predictable waits, prefetch before the predicted return,
/// and evict the session whose return is farthest away first.
#[derive(Debug, Default, Clone, Copy)]
pub struct Plan;

impl Policy for Plan {
    fn name(&self) -> &'static str {
        "plan"
    }

    fn on_tool_call(
        &self,
        cx: &PolicyContext<'_>,
        kv_bytes: f64,
        _: f64,
        latency: &LatencyModel,
    ) -> ResidencyAction {
        let budgets: Vec<TierBudget<'_>> = cx
            .tiers
            .iter()
            .zip(cx.tier_free.iter())
            .map(|(tier, free)| TierBudget {
                tier,
                free_bytes: *free,
            })
            .collect();
        let pressure = if cx.device_capacity > 0.0 {
            1.0 - cx.device_free / cx.device_capacity
        } else {
            1.0
        };
        plan_residency(
            kv_bytes,
            latency,
            &budgets,
            cx.tp,
            cx.recompute_s,
            pressure,
            cx.step_hint_s,
        )
        .action
    }

    fn rank_victims(&self, _: &PolicyContext<'_>, victims: &mut [Victim]) {
        victims.sort_by(|a, b| b.predicted_return.total_cmp(&a.predicted_return));
    }
}

/// Policies by name.
pub fn by_name(name: &str) -> Option<Box<dyn Policy>> {
    match name {
        "lru" => Some(Box::new(Lru)),
        "write-through" => Some(Box::new(WriteThrough)),
        "plan" => Some(Box::new(Plan)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_machine::Machine;

    #[test]
    fn write_through_copies_only_what_no_tier_holds() {
        let m = Machine::builtin("h200").unwrap();
        let wait = LatencyModel::Fixed { seconds: 30.0 };
        let decide = |tier_free: &[f64], kv: f64, mirrored: f64| {
            let cx = PolicyContext {
                now: 0.0,
                tiers: &m.tiers,
                tier_free,
                tp: 8,
                device_free: 0.0,
                device_capacity: 945e9,
                recompute_s: 7.0,
                step_hint_s: 1.5,
            };
            WriteThrough.on_tool_call(&cx, kv, mirrored, &wait)
        };
        assert_eq!(
            decide(&[2e12, 30e12], 20e9, 0.0),
            ResidencyAction::WriteThrough { tier_index: 0 }
        );
        assert_eq!(
            decide(&[1e9, 30e12], 20e9, 19.5e9),
            ResidencyAction::WriteThrough { tier_index: 0 }
        );
        assert_eq!(
            decide(&[1e9, 30e12], 20e9, 0.0),
            ResidencyAction::WriteThrough { tier_index: 1 }
        );
        assert_eq!(decide(&[2e12, 30e12], 20e9, 20e9), ResidencyAction::Keep);
        assert_eq!(decide(&[1e9, 1e9], 20e9, 0.0), ResidencyAction::Keep);
        assert_eq!(by_name("write-through").unwrap().name(), "write-through");
    }
}
