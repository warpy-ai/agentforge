//! Analytical cost model.
//!
//! Prefill and decode phases are modelled as the larger of a compute time and a memory time
//! plus a small per-layer overhead, in the style of GenZ and LLMCompass (single-digit percent
//! error on GPUs in their validations). Tensor parallelism scales compute and bandwidth linearly
//! and adds two ring all-reduces of the activations per layer over the scale-up link.

use agentforge_ir::{ModelSpec, Precision};
use agentforge_machine::{Device, Link};

/// Achieved fractions of peak and fixed overheads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Efficiency {
    /// Fraction of peak compute reached by large prefill GEMMs.
    pub mfu_prefill: f64,
    /// Fraction of peak compute reached by decode GEMVs and small-batch GEMMs.
    pub mfu_decode: f64,
    /// Fraction of peak memory bandwidth reached by streaming kernels.
    pub bw_util: f64,
    /// Fixed overhead per layer per phase (kernel launches, synchronization).
    pub per_layer_overhead_s: f64,
    /// Fraction of peak link bandwidth reached by tensor-parallel all-reduces. An assumption:
    /// achieved NCCL bandwidth is typically 70 to 80 percent of the link peak, and the
    /// calibration rows (TP 1 and TP 2) cannot pin it down.
    pub link_util: f64,
}

impl Default for Efficiency {
    fn default() -> Self {
        Efficiency {
            mfu_prefill: 0.6,
            mfu_decode: 0.3,
            bw_util: 0.8,
            per_layer_overhead_s: 2e-5,
            link_util: 0.75,
        }
    }
}

/// What a phase runs on: one device class scaled by tensor parallelism, with the link the
/// tensor-parallel group all-reduces over.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    pub device: &'a Device,
    pub tp: u32,
    /// The scale-up link between the `tp` devices; required when `tp > 1`.
    pub link: Option<&'a Link>,
    pub eff: Efficiency,
}

/// The cost breakdown of one phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Phase {
    pub flops: f64,
    pub bytes: f64,
    pub compute_s: f64,
    pub memory_s: f64,
    pub overhead_s: f64,
    /// Tensor-parallel all-reduce time, on the critical path: Megatron-style tensor parallelism
    /// does not overlap its all-reduces with compute in the engines this model calibrates against.
    pub comm_s: f64,
    pub seconds: f64,
}

impl Phase {
    fn new(
        flops: f64,
        bytes: f64,
        flops_per_s: f64,
        bytes_per_s: f64,
        overhead_s: f64,
        comm_s: f64,
    ) -> Phase {
        let compute_s = flops / flops_per_s;
        let memory_s = bytes / bytes_per_s;
        Phase {
            flops,
            bytes,
            compute_s,
            memory_s,
            overhead_s,
            comm_s,
            seconds: compute_s.max(memory_s) + overhead_s + comm_s,
        }
    }

    /// Which resource bounds the phase.
    pub fn bound(&self) -> &'static str {
        if self.compute_s >= self.memory_s {
            "compute"
        } else {
            "memory"
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CostError {
    #[error("device '{device}' has no compute entry usable for precision {precision}")]
    NoCompute {
        device: String,
        precision: Precision,
    },
    #[error("device '{device}' at tp = {tp} needs a link for the tensor-parallel all-reduces")]
    NoLink { device: String, tp: u32 },
}

fn device_flops_per_s(device: &Device, precision: Precision) -> Result<f64, CostError> {
    device
        .tflops(precision)
        .map(|t| t * 1e12)
        .ok_or_else(|| CostError::NoCompute {
            device: device.name.clone(),
            precision,
        })
}

fn all_reduce_seconds(
    model: &ModelSpec,
    target: &Target<'_>,
    tokens: f64,
) -> Result<f64, CostError> {
    if target.tp <= 1 {
        return Ok(0.0);
    }
    let link = target.link.ok_or_else(|| CostError::NoLink {
        device: target.device.name.clone(),
        tp: target.tp,
    })?;
    let tp = f64::from(target.tp);
    let activation_bytes = 2.0 * tokens * f64::from(model.d_model);
    let moved = 2.0 * (tp - 1.0) / tp * activation_bytes;
    let one = transfer_seconds(moved, link.bw_gbps * target.eff.link_util, link.latency_us);
    Ok(2.0 * f64::from(model.n_layers) * one)
}

/// KV bytes stored for a context of `tokens` tokens (sliding windows and compression applied).
pub fn kv_bytes(model: &ModelSpec, tokens: u64) -> u64 {
    model.kv_bytes_per_token() * model.attention.stored_kv_tokens(tokens)
}

fn attention_flops(model: &ModelSpec, new_tokens: u64, cached_prefix: u64) -> f64 {
    // QK^T and PV: 4 FLOPs per attended key per query per head, averaged over the new tokens.
    let span_start = model.attention.attended_tokens(cached_prefix) as f64;
    let span_end = model.attention.attended_tokens(cached_prefix + new_tokens) as f64;
    let avg_span = 0.5 * (span_start + span_end);
    4.0 * f64::from(model.n_layers)
        * f64::from(model.n_heads)
        * f64::from(model.head_dim)
        * new_tokens as f64
        * avg_span
}

/// Cost of prefilling `new_tokens` behind `cached_prefix` already-resident tokens.
pub fn prefill(
    model: &ModelSpec,
    target: &Target<'_>,
    new_tokens: u64,
    cached_prefix: u64,
) -> Result<Phase, CostError> {
    let (device, eff) = (target.device, &target.eff);
    let tp = f64::from(target.tp.max(1));
    let flops_per_s = device_flops_per_s(device, model.weight_precision)? * tp * eff.mfu_prefill;
    let bytes_per_s = device.memory_bw_bytes_per_s() * tp * eff.bw_util;
    let n = new_tokens as f64;
    let linear_flops = 2.0 * model.active_params_per_token() as f64 * n;
    let flops = linear_flops + attention_flops(model, new_tokens, cached_prefix);
    let bytes = model.weight_bytes() as f64
        + kv_bytes(model, cached_prefix) as f64
        + model.kv_bytes_per_token() as f64 * n;
    let overhead = eff.per_layer_overhead_s * f64::from(model.n_layers);
    let comm = all_reduce_seconds(model, target, n)?;
    Ok(Phase::new(
        flops,
        bytes,
        flops_per_s,
        bytes_per_s,
        overhead,
        comm,
    ))
}

/// Cost of one decode step (one token per sequence) for `batch` sequences at `context_tokens`.
pub fn decode_step(
    model: &ModelSpec,
    target: &Target<'_>,
    context_tokens: u64,
    batch: u64,
) -> Result<Phase, CostError> {
    let (device, eff) = (target.device, &target.eff);
    let tp = f64::from(target.tp.max(1));
    let batch = batch.max(1) as f64;
    let flops_per_s = device_flops_per_s(device, model.weight_precision)? * tp * eff.mfu_decode;
    let bytes_per_s = device.memory_bw_bytes_per_s() * tp * eff.bw_util;
    let per_seq_flops =
        2.0 * model.active_params_per_token() as f64 + attention_flops(model, 1, context_tokens);
    let flops = per_seq_flops * batch;
    let bytes = model.weight_bytes() as f64 + batch * kv_bytes(model, context_tokens) as f64;
    let overhead = eff.per_layer_overhead_s * f64::from(model.n_layers);
    let comm = all_reduce_seconds(model, target, batch)?;
    Ok(Phase::new(
        flops,
        bytes,
        flops_per_s,
        bytes_per_s,
        overhead,
        comm,
    ))
}

/// Time to move `bytes` over a channel of `bw_gbps` GB/s with `latency_us` of latency.
pub fn transfer_seconds(bytes: f64, bw_gbps: f64, latency_us: f64) -> f64 {
    if bw_gbps <= 0.0 {
        return f64::INFINITY;
    }
    latency_us * 1e-6 + bytes / (bw_gbps * 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_machine::Machine;

    fn h200() -> Machine {
        Machine::builtin("h200").unwrap()
    }

    fn on_nvlink(m: &Machine, tp: u32) -> Target<'_> {
        Target {
            device: &m.devices[0],
            tp,
            link: m.links.iter().find(|l| l.name == "nvlink4"),
            eff: Efficiency::default(),
        }
    }

    #[test]
    fn prefill_grows_with_tokens_and_is_compute_bound() {
        let model = ModelSpec::preset("llama-3.1-70b").unwrap();
        let m = h200();
        let t = on_nvlink(&m, 8);
        let small = prefill(&model, &t, 875, 119_000).unwrap();
        let large = prefill(&model, &t, 119_000, 0).unwrap();
        assert!(large.seconds > 10.0 * small.seconds);
        assert_eq!(large.bound(), "compute");
        // A full 119k-token recompute on eight H200s lands in the single-digit seconds.
        assert!(
            large.seconds > 2.0 && large.seconds < 30.0,
            "{}",
            large.seconds
        );
    }

    #[test]
    fn decode_at_batch_one_is_memory_bound() {
        let model = ModelSpec::preset("llama-3.1-70b").unwrap();
        let m = h200();
        let t = on_nvlink(&m, 8);
        let step = decode_step(&model, &t, 119_000, 1).unwrap();
        assert_eq!(step.bound(), "memory");
        assert!(
            step.seconds > 1e-3 && step.seconds < 0.1,
            "{}",
            step.seconds
        );
        let batched = decode_step(&model, &t, 119_000, 32).unwrap();
        assert!(batched.seconds > step.seconds);
        assert!(batched.seconds < 32.0 * step.seconds);
    }

    #[test]
    fn tp_communication_is_on_the_critical_path() {
        let model = ModelSpec::preset("llama-3.1-70b").unwrap();
        let m = h200();
        let step = decode_step(&model, &on_nvlink(&m, 8), 119_000, 1).unwrap();
        let share = step.comm_s / step.seconds;
        assert!(
            step.comm_s > 0.0 && share > 0.01 && share < 0.25,
            "comm {} s of {} s",
            step.comm_s,
            step.seconds
        );
        let critical = step.compute_s.max(step.memory_s) + step.overhead_s + step.comm_s;
        assert!((step.seconds - critical).abs() < 1e-12);
        let single = decode_step(&model, &on_nvlink(&m, 1), 119_000, 1).unwrap();
        assert_eq!(single.comm_s, 0.0);
    }

    #[test]
    fn tp_without_a_link_is_an_error() {
        let model = ModelSpec::preset("llama-3.1-70b").unwrap();
        let m = h200();
        let unlinked = |tp| Target {
            link: None,
            ..on_nvlink(&m, tp)
        };
        assert!(matches!(
            prefill(&model, &unlinked(2), 875, 119_000),
            Err(CostError::NoLink { tp: 2, .. })
        ));
        assert!(matches!(
            decode_step(&model, &unlinked(2), 119_000, 1),
            Err(CostError::NoLink { tp: 2, .. })
        ));
        assert!(decode_step(&model, &unlinked(1), 119_000, 1).is_ok());
    }

    #[test]
    fn kv_bytes_follow_attention_kind() {
        let model = ModelSpec::preset("llama-3.1-70b").unwrap();
        assert_eq!(kv_bytes(&model, 119_000), 119_000 * 327_680);
        let mut swa = model.clone();
        swa.attention = agentforge_ir::AttentionKind::SlidingWindow { window: 128 };
        assert_eq!(kv_bytes(&swa, 119_000), 128 * 327_680);
    }

    #[test]
    fn transfer_math() {
        let s = transfer_seconds(38e9, 400.0, 0.0);
        assert!((s - 0.095).abs() < 1e-9);
        assert!(transfer_seconds(1.0, 0.0, 0.0).is_infinite());
    }

    #[test]
    fn missing_precision_is_an_error() {
        let mut m = h200();
        let d = &mut m.devices[0];
        d.compute_tflops.clear();
        d.compute_tflops.insert("int8".to_string(), 1.0);
        let model = ModelSpec::preset("llama-3.1-70b").unwrap();
        assert!(matches!(
            prefill(&model, &on_nvlink(&m, 1), 1, 0),
            Err(CostError::NoCompute { .. })
        ));
    }
}
