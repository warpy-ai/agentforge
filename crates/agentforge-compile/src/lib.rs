//! Compiler passes that consume the Agent Program IR and a machine description.
//!
//! - [`kv_liveness`] derives, for every tool a step may call, how long the step's KV cache is
//!   expected to lie idle before the next use;
//! - [`plan_residency`] decides, Belady-style, whether a session's KV should stay on the device,
//!   be spilled to a tier during the wait and prefetched before the predicted return, or be
//!   dropped and recomputed;
//! - [`place`] assigns prefill and decode to device classes of a machine.

use agentforge_cost::transfer_seconds;
use agentforge_ir::{AgentProgram, LatencyModel, ModelSpec, StepId};
use agentforge_machine::{Device, DeviceClass, Machine, MemoryTier};

/// Expected idle gap of a step's KV cache after one of its possible tool calls.
#[derive(Debug, Clone, PartialEq)]
pub struct StepLiveness {
    pub step: StepId,
    pub tool: String,
    pub probability: f64,
    pub expected_gap_s: f64,
    pub p10_s: f64,
    pub p50_s: f64,
    pub p90_s: f64,
}

/// KV liveness for every tool call a step may make.
pub fn kv_liveness(program: &AgentProgram) -> Vec<StepLiveness> {
    program
        .steps
        .iter()
        .flat_map(|step| {
            step.transition
                .tool_calls()
                .into_iter()
                .map(move |(probability, call)| StepLiveness {
                    step: step.id,
                    tool: call.name.clone(),
                    probability,
                    expected_gap_s: call.latency.mean(),
                    p10_s: call.latency.quantile(0.1),
                    p50_s: call.latency.quantile(0.5),
                    p90_s: call.latency.quantile(0.9),
                })
        })
        .collect()
}

/// What to do with a session's KV cache while it waits on a tool.
#[derive(Debug, Clone, PartialEq)]
pub enum ResidencyAction {
    Keep,
    Spill {
        tier_index: usize,
        tier: String,
        prefetch_lead_s: f64,
    },
    Drop,
}

/// A residency decision with the numbers behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidencyDecision {
    pub action: ResidencyAction,
    pub fetch_s: Option<f64>,
    pub write_s: Option<f64>,
    pub recompute_s: f64,
    pub rationale: String,
}

/// A tier and how much of it is free right now.
#[derive(Debug, Clone, Copy)]
pub struct TierBudget<'a> {
    pub tier: &'a MemoryTier,
    pub free_bytes: f64,
}

struct Candidate {
    index: usize,
    fetch_s: f64,
    write_s: f64,
    fits: bool,
}

/// Plan the residency of `kv_bytes` of KV cache across a tool wait described by `wait`.
///
/// `tiers` are ordered shallow to deep; `tp` scales tier bandwidth (one link per device);
/// `recompute_s` is the cost of rebuilding the KV from scratch; `pressure` in `0..=1` is how
/// full the device KV budget is; `step_hint_s` is the typical duration of a step on this
/// device, the shortest time for which freed memory can be of use to anyone else.
pub fn plan_residency(
    kv_bytes: f64,
    wait: &LatencyModel,
    tiers: &[TierBudget<'_>],
    tp: u32,
    recompute_s: f64,
    pressure: f64,
    step_hint_s: f64,
) -> ResidencyDecision {
    let tp = f64::from(tp.max(1));
    let pressure = pressure.clamp(0.0, 1.0);
    let p20 = wait.quantile(0.2);
    let p50 = wait.quantile(0.5);
    let candidates: Vec<Candidate> = tiers
        .iter()
        .enumerate()
        .map(|(index, b)| Candidate {
            index,
            fetch_s: transfer_seconds(kv_bytes, b.tier.read_bw_gbps * tp, b.tier.latency_us),
            write_s: transfer_seconds(kv_bytes, b.tier.write_bw_gbps * tp, b.tier.latency_us),
            fits: b.free_bytes >= kv_bytes,
        })
        .collect();
    let shallow_fetch = candidates
        .iter()
        .filter(|c| c.fits)
        .map(|c| c.fetch_s)
        .fold(f64::INFINITY, f64::min);
    let spill = |c: &Candidate, why: String| ResidencyDecision {
        action: ResidencyAction::Spill {
            tier_index: c.index,
            tier: tiers[c.index].tier.name.clone(),
            prefetch_lead_s: 1.5 * c.fetch_s,
        },
        fetch_s: Some(c.fetch_s),
        write_s: Some(c.write_s),
        recompute_s,
        rationale: why,
    };
    let keep = |why: String| ResidencyDecision {
        action: ResidencyAction::Keep,
        fetch_s: shallow_fetch.is_finite().then_some(shallow_fetch),
        write_s: None,
        recompute_s,
        rationale: why,
    };

    // 1. No contention: idle KV on an uncontended device costs nothing, and moving it would
    //    only add a fetch on the way back.
    if pressure < 0.8 {
        return keep(format!(
            "device pressure {pressure:.2} is below 0.8; nothing is waiting for this memory"
        ));
    }
    // 2. Coming back soon: memory freed for less than a step helps nobody, and a round trip
    //    through any tier costs more than it frees.
    let horizon = (3.0 * shallow_fetch.min(recompute_s)).max(2.0 * step_hint_s.max(0.0));
    if p50 < horizon {
        return keep(format!(
            "median wait {p50:.2}s is shorter than the useful horizon {horizon:.2}s (fetch {:.2}s, step {step_hint_s:.2}s)",
            shallow_fetch.min(recompute_s)
        ));
    }
    // 3. A tier whose write hides inside the median wait and whose fetch hides inside the p20
    //    wait: spill now, prefetch before the early return.
    if let Some(c) = candidates
        .iter()
        .find(|c| c.fits && c.fetch_s <= p20 && c.write_s <= p50)
    {
        return spill(
            c,
            format!(
                "pressure {pressure:.2}; write {:.2}s fits in the median wait {p50:.2}s and fetch {:.2}s fits in the p20 wait {p20:.2}s",
                c.write_s, c.fetch_s
            ),
        );
    }
    // 4. Otherwise the shallowest tier with room, even if the fetch may be exposed.
    if let Some(c) = candidates.iter().find(|c| c.fits) {
        return spill(
            c,
            format!("pressure {pressure:.2}; shallowest tier with room"),
        );
    }
    // 5. No room anywhere: drop when recompute is cheaper than holding memory for the wait.
    if recompute_s < p50 {
        return ResidencyDecision {
            action: ResidencyAction::Drop,
            fetch_s: None,
            write_s: None,
            recompute_s,
            rationale: format!(
                "no tier has room; recompute {recompute_s:.2}s is cheaper than holding memory for the median wait {p50:.2}s"
            ),
        };
    }
    keep("no tier has room and recompute is not cheaper than waiting".to_string())
}

/// Where prefill and decode should run on a machine.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementPlan {
    pub prefill_device: String,
    pub decode_device: String,
    pub kv_link: Option<String>,
    pub rationale: String,
}

/// Assign prefill and decode to device classes.
///
/// Prefill is compute-bound, so it goes to the bulk-memory class (GPU, NPU, CPU) with the most
/// aggregate compute for the model's weight precision; SRAM decode tiers are skipped when a bulk
/// class exists, because they stream weights through SRAM and are built for decode. Decode is
/// bandwidth-bound, so an SRAM tier that holds the weights wins; otherwise the highest-bandwidth
/// class whose aggregate memory holds the weights. This mirrors the 2026 production splits
/// (Vera Rubin with Groq 3 LPX, Trainium with Cerebras CS-3).
pub fn place(model: &ModelSpec, machine: &Machine) -> PlacementPlan {
    let weight_bytes = model.weight_bytes() as f64;
    let aggregate_tflops =
        |d: &Device| d.tflops(model.weight_precision).unwrap_or(0.0) * f64::from(d.count);
    let holds_weights = |d: &Device| d.memory_bytes() * f64::from(d.count) >= weight_bytes;
    let by_bandwidth = |a: &&Device, b: &&Device| a.memory_bw_tbps.total_cmp(&b.memory_bw_tbps);
    let bulk: Vec<&Device> = machine
        .devices
        .iter()
        .filter(|d| d.class != DeviceClass::SramLpu)
        .collect();
    let prefill_pool: Vec<&Device> = if bulk.is_empty() {
        machine.devices.iter().collect()
    } else {
        bulk
    };
    let prefill = prefill_pool
        .iter()
        .copied()
        .max_by(|a, b| aggregate_tflops(a).total_cmp(&aggregate_tflops(b)))
        .expect("validated machines have at least one device");
    let decode = machine
        .devices
        .iter()
        .filter(|d| d.class == DeviceClass::SramLpu && holds_weights(d))
        .max_by(by_bandwidth)
        .or_else(|| {
            machine
                .devices
                .iter()
                .filter(|d| holds_weights(d))
                .max_by(by_bandwidth)
        })
        .unwrap_or(prefill);
    let kv_link = if prefill.name != decode.name {
        machine
            .links
            .iter()
            .max_by(|a, b| a.bw_gbps.total_cmp(&b.bw_gbps))
            .map(|l| l.name.clone())
    } else {
        None
    };
    PlacementPlan {
        prefill_device: prefill.name.clone(),
        decode_device: decode.name.clone(),
        kv_link,
        rationale: format!(
            "prefill on '{}' ({} TFLOPS at {} x{}); decode on '{}' ({} TB/s, {} GB x{} holds {:.0} GB of weights)",
            prefill.name,
            prefill.tflops(model.weight_precision).unwrap_or(0.0),
            model.weight_precision,
            prefill.count,
            decode.name,
            decode.memory_bw_tbps,
            decode.memory_gb,
            decode.count,
            weight_bytes / 1e9
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_machine::Machine;

    fn host_and_nvme(machine: &Machine) -> Vec<TierBudget<'_>> {
        machine
            .tiers
            .iter()
            .map(|t| TierBudget {
                tier: t,
                free_bytes: t.capacity_bytes(),
            })
            .collect()
    }

    #[test]
    fn short_waits_keep_kv_resident() {
        let m = Machine::builtin("h200").unwrap();
        let tiers = host_and_nvme(&m);
        let wait = LatencyModel::LogNormal {
            median_s: 1.5,
            sigma: 0.8,
        };
        for pressure in [0.1, 0.9] {
            let d = plan_residency(38e9, &wait, &tiers, 8, 7.0, pressure, 1.2);
            assert_eq!(d.action, ResidencyAction::Keep, "{}", d.rationale);
        }
    }

    #[test]
    fn uncontended_memory_is_never_spilled() {
        let m = Machine::builtin("h200").unwrap();
        let tiers = host_and_nvme(&m);
        let wait = LatencyModel::LogNormal {
            median_s: 600.0,
            sigma: 0.7,
        };
        let d = plan_residency(38e9, &wait, &tiers, 8, 7.0, 0.2, 1.2);
        assert_eq!(d.action, ResidencyAction::Keep, "{}", d.rationale);
    }

    #[test]
    fn long_waits_spill_to_host_with_prefetch() {
        let m = Machine::builtin("h200").unwrap();
        let tiers = host_and_nvme(&m);
        let wait = LatencyModel::LogNormal {
            median_s: 45.0,
            sigma: 0.7,
        };
        let d = plan_residency(38e9, &wait, &tiers, 8, 7.0, 0.9, 1.2);
        match d.action {
            ResidencyAction::Spill {
                tier_index,
                ref tier,
                prefetch_lead_s,
            } => {
                assert_eq!(tier_index, 0);
                assert_eq!(tier, "host_dram");
                assert!(
                    prefetch_lead_s > 0.0 && prefetch_lead_s < 1.0,
                    "{prefetch_lead_s}"
                );
            }
            other => panic!("expected a spill, got {other:?}: {}", d.rationale),
        }
    }

    #[test]
    fn full_host_falls_through_to_nvme() {
        let m = Machine::builtin("h200").unwrap();
        let tiers: Vec<TierBudget<'_>> = m
            .tiers
            .iter()
            .map(|t| TierBudget {
                tier: t,
                free_bytes: if t.name == "host_dram" {
                    0.0
                } else {
                    t.capacity_bytes()
                },
            })
            .collect();
        let wait = LatencyModel::LogNormal {
            median_s: 180.0,
            sigma: 1.0,
        };
        let d = plan_residency(38e9, &wait, &tiers, 8, 7.0, 0.9, 1.2);
        match d.action {
            ResidencyAction::Spill { ref tier, .. } => assert_eq!(tier, "nvme"),
            other => panic!("expected an nvme spill, got {other:?}: {}", d.rationale),
        }
    }

    #[test]
    fn no_room_under_pressure_drops_when_recompute_is_cheap() {
        let m = Machine::builtin("h200").unwrap();
        let tiers: Vec<TierBudget<'_>> = m
            .tiers
            .iter()
            .map(|t| TierBudget {
                tier: t,
                free_bytes: 0.0,
            })
            .collect();
        let wait = LatencyModel::LogNormal {
            median_s: 120.0,
            sigma: 0.5,
        };
        let d = plan_residency(38e9, &wait, &tiers, 8, 7.0, 0.95, 1.2);
        assert_eq!(d.action, ResidencyAction::Drop, "{}", d.rationale);
    }

    #[test]
    fn liveness_lists_every_tool_arm() {
        let p = AgentProgram::coding_agent_default();
        let l = kv_liveness(&p);
        assert_eq!(l.len(), 3);
        assert!(l.iter().all(|x| x.p10_s < x.p50_s && x.p50_s < x.p90_s));
        let total: f64 = l.iter().map(|x| x.probability).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn placement_prefers_bulk_prefill_and_sram_decode() {
        let model = ModelSpec::preset("llama-3.1-8b").unwrap();
        let mut machine = Machine::builtin("b200").unwrap();
        let lpx = Machine::builtin("lpx-tier").unwrap();
        machine.devices.extend(lpx.devices);
        machine.links.extend(lpx.links);
        let plan = place(&model, &machine);
        assert_eq!(plan.prefill_device, "b200-sxm", "{}", plan.rationale);
        assert_eq!(plan.decode_device, "lpu", "{}", plan.rationale);
        assert!(plan.kv_link.is_some());
        let single = place(&model, &Machine::builtin("b200").unwrap());
        assert_eq!(single.prefill_device, single.decode_device);
        assert!(single.kv_link.is_none());
    }
}
