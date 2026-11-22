//! Analytical cost model.
//!
//! Prefill and decode phases are modelled per operator, in the style of GenZ and LLMCompass
//! (single-digit percent error on GPUs in their validations): the dense projections and attention
//! run as separate kernels one after the other, so each takes the larger of its compute time and
//! memory time and the two add, plus a small per-layer overhead. Tensor parallelism scales
//! compute and bandwidth linearly and adds two ring all-reduces of the activations per layer over
//! the scale-up link.

use agentforge_ir::{ModelSpec, Precision, Scalar};
use agentforge_machine::{Device, Link};

pub mod calibration;

/// Achieved fractions of peak and fixed overheads. The defaults are the grid point
/// [`calibration::fit`] picks for the bundled TensorRT-LLM rows (`agentforge cost calibrate --fit`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Efficiency {
    /// Fraction of peak compute reached by large prefill GEMMs.
    pub mfu_prefill: f64,
    /// Fraction of peak compute reached by decode GEMVs and small-batch GEMMs.
    pub mfu_decode: f64,
    /// Fraction of the 4-bit tensor-core peak that block-scaled FP4 GEMMs reach, relative to what
    /// 8-bit GEMMs reach of theirs. Applied to the compute rate when the weight element type is
    /// the 4-bit E2M1 (MXFP4, NVFP4). Fitted; the published rows cannot separate it from a
    /// Blackwell-wide derate because no B200 FP8 row exists on the same page, so read it as
    /// "Blackwell FP4 reaches this share of its paper peak".
    pub fp4_mfu_scale: f64,
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
            mfu_decode: 0.4,
            fp4_mfu_scale: 0.65,
            bw_util: 0.6,
            per_layer_overhead_s: 4e-5,
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
///
/// The dense projections (weight reads against GEMM FLOPs) and attention (KV traffic against
/// attention FLOPs) are two rooflines that run one after the other, so `seconds` is the sum of
/// their bounding times plus overhead and communication. `compute_s` and `memory_s` sum the two
/// parts; when both parts share a bound, `seconds` is the larger of them plus overhead and
/// communication.
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
    bound: &'static str,
}

#[derive(Clone, Copy)]
struct Work {
    flops: f64,
    bytes: f64,
}

#[derive(Clone, Copy)]
struct Rates {
    gemm_flops_per_s: f64,
    attention_flops_per_s: f64,
    bytes_per_s: f64,
}

impl Rates {
    fn new(model: &ModelSpec, target: &Target<'_>, mfu: f64) -> Result<Rates, CostError> {
        let (device, eff) = (target.device, &target.eff);
        let precision = model.weight_precision;
        let tp = f64::from(target.tp.max(1));
        let peak = device
            .tflops(precision)
            .ok_or_else(|| CostError::NoCompute {
                device: device.name.clone(),
                precision,
            })?
            * 1e12;
        let flops_per_s = peak * tp * mfu;
        let fp4_scale = if precision.scalar() == Scalar::F4E2M1 {
            eff.fp4_mfu_scale
        } else {
            1.0
        };
        Ok(Rates {
            gemm_flops_per_s: flops_per_s * fp4_scale,
            attention_flops_per_s: flops_per_s,
            bytes_per_s: device.memory_bw_bytes_per_s() * tp * eff.bw_util,
        })
    }
}

impl Phase {
    fn new(dense: Work, attention: Work, rates: Rates, overhead_s: f64, comm_s: f64) -> Phase {
        let parts = [
            (dense, rates.gemm_flops_per_s),
            (attention, rates.attention_flops_per_s),
        ]
        .map(|(w, flops_per_s)| (w.flops / flops_per_s, w.bytes / rates.bytes_per_s));
        let on_compute: f64 = parts.iter().filter(|(c, m)| c >= m).map(|(c, _)| c).sum();
        let on_memory: f64 = parts.iter().filter(|(c, m)| c < m).map(|(_, m)| m).sum();
        Phase {
            flops: dense.flops + attention.flops,
            bytes: dense.bytes + attention.bytes,
            compute_s: parts.iter().map(|(c, _)| c).sum(),
            memory_s: parts.iter().map(|(_, m)| m).sum(),
            overhead_s,
            comm_s,
            seconds: on_compute + on_memory + overhead_s + comm_s,
            bound: if on_compute >= on_memory {
                "compute"
            } else {
                "memory"
            },
        }
    }

    /// Which resource the larger share of the phase's critical path waits on.
    pub fn bound(&self) -> &'static str {
        self.bound
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
    #[error("one sequence needs {need_bytes:.3e} KV bytes but the budget is {budget_bytes:.3e}")]
    KvBudgetTooSmall { need_bytes: f64, budget_bytes: f64 },
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
    prefill_batch(model, target, 1, new_tokens, cached_prefix)
}

/// Cost of prefilling `sequences` independent sequences of `new_tokens` each behind
/// `cached_prefix` resident tokens.
pub fn prefill_batch(
    model: &ModelSpec,
    target: &Target<'_>,
    sequences: u64,
    new_tokens: u64,
    cached_prefix: u64,
) -> Result<Phase, CostError> {
    let eff = &target.eff;
    let rates = Rates::new(model, target, eff.mfu_prefill)?;
    let seqs = sequences as f64;
    let n = new_tokens as f64;
    let dense = Work {
        flops: seqs * 2.0 * model.active_params_per_token() as f64 * n,
        bytes: model.weight_bytes() as f64,
    };
    let attention = Work {
        flops: seqs * attention_flops(model, new_tokens, cached_prefix),
        bytes: seqs
            * (kv_bytes(model, cached_prefix) as f64 + model.kv_bytes_per_token() as f64 * n),
    };
    let overhead = eff.per_layer_overhead_s * f64::from(model.n_layers);
    let comm = all_reduce_seconds(model, target, seqs * n)?;
    Ok(Phase::new(dense, attention, rates, overhead, comm))
}

/// Cost of one decode step (one token per sequence) for `batch` sequences at `context_tokens`.
pub fn decode_step(
    model: &ModelSpec,
    target: &Target<'_>,
    context_tokens: u64,
    batch: u64,
) -> Result<Phase, CostError> {
    let eff = &target.eff;
    let batch = batch.max(1) as f64;
    let rates = Rates::new(model, target, eff.mfu_decode)?;
    let dense = Work {
        flops: batch * 2.0 * model.active_params_per_token() as f64,
        bytes: model.weight_bytes() as f64,
    };
    let attention = Work {
        flops: batch * attention_flops(model, 1, context_tokens),
        bytes: batch * kv_bytes(model, context_tokens) as f64,
    };
    let overhead = eff.per_layer_overhead_s * f64::from(model.n_layers);
    let comm = all_reduce_seconds(model, target, batch)?;
    Ok(Phase::new(dense, attention, rates, overhead, comm))
}

/// Offline throughput for one request shape at the throughput-optimal batch: every request has
/// `isl` input and `osl` output tokens, the batch is prefilled together and then decodes `osl`
/// steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteadyState {
    pub batch: u64,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub output_tokens_per_s: f64,
    /// `"compute"` or `"memory"`, from the decode phase.
    pub bound: &'static str,
}

/// The [`SteadyState`] with the highest output rate over batches of 1 to `max_batch` whose KV for
/// `isl + osl` tokens each fits in `kv_budget_bytes`. Decode runs at the midpoint context
/// `isl + osl / 2`: KV reads and attention FLOPs are linear in context, so the midpoint gives the
/// exact sum over the `osl` steps.
pub fn max_throughput(
    model: &ModelSpec,
    target: &Target<'_>,
    isl: u64,
    osl: u64,
    kv_budget_bytes: f64,
    max_batch: u64,
) -> Result<SteadyState, CostError> {
    let per_sequence = kv_bytes(model, isl + osl) as f64;
    let mut best: Option<SteadyState> = None;
    for batch in (1..=max_batch).take_while(|&b| b as f64 * per_sequence <= kv_budget_bytes) {
        let prefill_s = prefill_batch(model, target, batch, isl, 0)?.seconds;
        let step = decode_step(model, target, isl + osl / 2, batch)?;
        let decode_s = osl as f64 * step.seconds;
        let output_tokens_per_s = (batch * osl) as f64 / (prefill_s + decode_s);
        if best.is_none_or(|b| output_tokens_per_s > b.output_tokens_per_s) {
            best = Some(SteadyState {
                batch,
                prefill_s,
                decode_s,
                output_tokens_per_s,
                bound: step.bound(),
            });
        }
    }
    best.ok_or(CostError::KvBudgetTooSmall {
        need_bytes: per_sequence,
        budget_bytes: kv_budget_bytes,
    })
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
    fn batched_prefill_attends_within_each_sequence() {
        let model = ModelSpec::preset("llama-3.1-70b").unwrap();
        let m = h200();
        let t = on_nvlink(&m, 2);
        let one = prefill(&model, &t, 4096, 0).unwrap();
        assert_eq!(one, prefill_batch(&model, &t, 1, 4096, 0).unwrap());
        let batch = prefill_batch(&model, &t, 8, 4096, 0).unwrap();
        let joined = prefill(&model, &t, 8 * 4096, 0).unwrap();
        assert!((batch.flops - 8.0 * one.flops).abs() < 1e-9 * batch.flops);
        assert!(batch.flops < joined.flops);
        assert!(batch.bytes < 8.0 * one.bytes);
        assert_eq!(batch.comm_s, joined.comm_s);
    }

    #[test]
    fn max_throughput_fills_the_kv_budget_up_to_the_batch_cap() {
        let model = ModelSpec::preset("llama-3.1-70b").unwrap();
        let m = h200();
        let t = on_nvlink(&m, 2);
        let per_sequence = kv_bytes(&model, 128 + 2048) as f64;
        let ten = max_throughput(&model, &t, 128, 2048, 10.5 * per_sequence, 4096).unwrap();
        assert_eq!(ten.batch, 10);
        let capped = max_throughput(&model, &t, 128, 2048, 1e15, 64).unwrap();
        assert_eq!(capped.batch, 64);
        assert!(capped.output_tokens_per_s > ten.output_tokens_per_s);
        let expected = (64 * 2048) as f64 / (capped.prefill_s + capped.decode_s);
        assert!((capped.output_tokens_per_s - expected).abs() < 1e-9 * expected);
        assert!(matches!(
            max_throughput(&model, &t, 128, 2048, 0.5 * per_sequence, 4096),
            Err(CostError::KvBudgetTooSmall { .. })
        ));
    }

    #[test]
    fn transfer_math() {
        let s = transfer_seconds(38e9, 400.0, 0.0);
        assert!((s - 0.095).abs() < 1e-9);
        assert!(transfer_seconds(1.0, 0.0, 0.0).is_infinite());
    }

    #[test]
    fn fp4_mfu_scale_slows_only_fp4_gemms() {
        let m = Machine::builtin("b200").unwrap();
        let at = |fp4_mfu_scale| Target {
            device: &m.devices[0],
            tp: 1,
            link: None,
            eff: Efficiency {
                fp4_mfu_scale,
                ..Efficiency::default()
            },
        };
        let mut model = ModelSpec::preset("llama-3.1-70b").unwrap();
        model.weight_precision = Precision::Nvfp4;
        let full = prefill(&model, &at(1.0), 8192, 0).unwrap();
        let half = prefill(&model, &at(0.5), 8192, 0).unwrap();
        let gemm_flops = 2.0 * model.active_params_per_token() as f64 * 8192.0;
        let gemm_s = gemm_flops / (9000e12 * Efficiency::default().mfu_prefill);
        assert!((half.compute_s - full.compute_s - gemm_s).abs() < 1e-9 * gemm_s);
        model.weight_precision = Precision::Fp8;
        assert_eq!(
            prefill(&model, &at(0.5), 8192, 0).unwrap(),
            prefill(&model, &at(1.0), 8192, 0).unwrap()
        );
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
