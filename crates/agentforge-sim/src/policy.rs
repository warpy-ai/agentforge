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
    /// latency model of the tool the agent program says it is calling.
    fn on_tool_call(
        &self,
        cx: &PolicyContext<'_>,
        kv_bytes: f64,
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

    fn on_tool_call(&self, _: &PolicyContext<'_>, _: f64, _: &LatencyModel) -> ResidencyAction {
        ResidencyAction::Keep
    }

    fn rank_victims(&self, _: &PolicyContext<'_>, victims: &mut [Victim]) {
        victims.sort_by(|a, b| a.last_use.total_cmp(&b.last_use));
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
        "plan" => Some(Box::new(Plan)),
        _ => None,
    }
}
