//! The discrete-event engine.
//!
//! Model of the machine: one device class scaled linearly by `tp`; prefill is serialized on the
//! device, decode is batched (per-token time from the cost model with the batch size at the
//! start of the decode), and one transfer channel per device group moves KV between the device
//! and the tiers. Session KV lives in one of: device memory, a tier, in flight back to the
//! device, or nowhere (dropped, which forces a prefix recompute on the next step).
//!
//! Fairness between policies: pressure evictions are available to both policies and their
//! writes occupy the transfer channel before the arriving session can start, because memory is
//! only free once the copy has landed. The plan policy's advantage comes from knowing which tool
//! a session is waiting on and spilling during the waits the program predicts, not from a
//! cheaper eviction primitive.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use agentforge_compile::ResidencyAction;
use agentforge_cost::{self as cost, CostError, Efficiency};
use agentforge_ir::ModelSpec;
use agentforge_machine::{Device, MemoryTier};

use crate::policy::{Policy, PolicyContext, Victim};
use crate::report::{mean, percentile, SimReport};
use crate::trace::Trace;

/// What the simulator runs against.
#[derive(Debug, Clone)]
pub struct SimConfig {
    pub model: ModelSpec,
    pub device: Device,
    pub tiers: Vec<MemoryTier>,
    pub tp: u32,
    pub eff: Efficiency,
    /// Fraction of device memory reserved for activations and fragmentation.
    pub kv_reserve_fraction: f64,
}

impl SimConfig {
    pub fn new(model: ModelSpec, device: Device, tiers: Vec<MemoryTier>, tp: u32) -> SimConfig {
        SimConfig {
            model,
            device,
            tiers,
            tp: tp.max(1),
            eff: Efficiency::default(),
            kv_reserve_fraction: 0.10,
        }
    }

    pub fn total_device_bytes(&self) -> f64 {
        self.device.memory_bytes() * f64::from(self.tp)
    }

    /// Bytes available for KV after weights and the reserve.
    pub fn kv_capacity_bytes(&self) -> f64 {
        (self.total_device_bytes() * (1.0 - self.kv_reserve_fraction)
            - self.model.weight_bytes() as f64)
            .max(0.0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error(transparent)]
    Cost(#[from] CostError),
    #[error("model weights ({weight_gb:.1} GB) do not fit in {total_gb:.1} GB of device memory (tp = {tp})")]
    ModelDoesNotFit {
        weight_gb: f64,
        total_gb: f64,
        tp: u32,
    },
    #[error("session {session} needs {need_gb:.1} GB of KV, more than the {capacity_gb:.1} GB KV budget")]
    SessionTooLarge {
        session: u32,
        need_gb: f64,
        capacity_gb: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum KvState {
    Absent,
    OnDevice,
    OnTier(usize),
    InFlight { ready_at: f64 },
    Dropped,
}

#[derive(Debug, Clone, Copy)]
enum Event {
    StepArrival {
        sid: usize,
    },
    DecodeDone {
        sid: usize,
    },
    SpillDone {
        sid: usize,
        tier: usize,
        gen: u32,
        bytes: f64,
    },
    PrefetchDue {
        sid: usize,
        gen: u32,
    },
}

#[derive(Debug, Clone, Copy)]
struct Timed {
    t: f64,
    seq: u64,
    ev: Event,
}

impl PartialEq for Timed {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Timed {}

impl PartialOrd for Timed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timed {
    // Reversed so that the max-heap pops the earliest event first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .t
            .total_cmp(&self.t)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

#[derive(Debug, Clone)]
struct Sess {
    step_idx: usize,
    prefix_tokens: u64,
    /// Bytes of this session's KV accounted on the device or in a tier (0 when absent).
    kv_bytes: f64,
    state: KvState,
    last_use: f64,
    predicted_return: f64,
    gen: u32,
    active: bool,
    done: bool,
    spill_pending: bool,
    step_start: f64,
    prefetch_retries: u32,
}

struct DevState {
    kv_used: f64,
    kv_capacity: f64,
    prefill_free_at: f64,
    transfer_free_at: f64,
    decoders: Vec<(usize, f64)>,
}

#[derive(Default)]
struct Metrics {
    ttft: Vec<f64>,
    step_latency: Vec<f64>,
    recompute_events: u64,
    recompute_tokens: u64,
    foreground_kv_wait_s: f64,
    fetched_bytes: f64,
    prefetched_bytes: f64,
    spill_bytes: f64,
    hidden_spill_bytes: f64,
    evictions: u64,
    evicted_drops: u64,
    planned_drops: u64,
    capacity_waits: u64,
    prefetch_skipped: u64,
    device_busy_s: f64,
    prefill_tokens: u64,
    sessions_done: u32,
    makespan: f64,
}

/// Run one policy over a trace.
pub fn run(cfg: &SimConfig, trace: &Trace, policy: &dyn Policy) -> Result<SimReport, SimError> {
    let mut sim = Sim::new(cfg, trace, policy)?;
    sim.run_all()?;
    Ok(sim.report())
}

struct Sim<'a> {
    cfg: &'a SimConfig,
    trace: &'a Trace,
    policy: &'a dyn Policy,
    sess: Vec<Sess>,
    dev: DevState,
    tier_used: Vec<f64>,
    events: BinaryHeap<Timed>,
    seq: u64,
    m: Metrics,
    /// Exponential moving average of prefill-plus-decode time per step.
    step_hint_s: f64,
}

impl<'a> Sim<'a> {
    fn new(cfg: &'a SimConfig, trace: &'a Trace, policy: &'a dyn Policy) -> Result<Self, SimError> {
        let kv_capacity = cfg.kv_capacity_bytes();
        if kv_capacity <= 0.0 {
            return Err(SimError::ModelDoesNotFit {
                weight_gb: cfg.model.weight_bytes() as f64 / 1e9,
                total_gb: cfg.total_device_bytes() / 1e9,
                tp: cfg.tp,
            });
        }
        // Seed the step-duration hint from the cost model for a typical step.
        let typical_prefix = trace
            .sessions
            .iter()
            .map(|s| s.initial_prefix_tokens)
            .max()
            .unwrap_or(0)
            .max(1);
        let prefill = cost::prefill(
            &cfg.model,
            &cfg.device,
            875,
            typical_prefix,
            cfg.tp,
            &cfg.eff,
        )?;
        let decode =
            cost::decode_step(&cfg.model, &cfg.device, typical_prefix, 4, cfg.tp, &cfg.eff)?;
        let step_hint_s = prefill.seconds + 214.0 * decode.seconds;
        let sess = trace
            .sessions
            .iter()
            .map(|s| Sess {
                step_idx: 0,
                prefix_tokens: s.initial_prefix_tokens,
                kv_bytes: 0.0,
                state: KvState::Absent,
                last_use: s.arrival_s,
                predicted_return: s.arrival_s,
                gen: 0,
                active: false,
                done: s.steps.is_empty(),
                spill_pending: false,
                step_start: s.arrival_s,
                prefetch_retries: 0,
            })
            .collect();
        let mut sim = Sim {
            cfg,
            trace,
            policy,
            sess,
            dev: DevState {
                kv_used: 0.0,
                kv_capacity,
                prefill_free_at: 0.0,
                transfer_free_at: 0.0,
                decoders: Vec::new(),
            },
            tier_used: vec![0.0; cfg.tiers.len()],
            events: BinaryHeap::new(),
            seq: 0,
            m: Metrics::default(),
            step_hint_s,
        };
        for (sid, s) in trace.sessions.iter().enumerate() {
            if !s.steps.is_empty() {
                sim.push(s.arrival_s, Event::StepArrival { sid });
            }
        }
        Ok(sim)
    }

    fn push(&mut self, t: f64, ev: Event) {
        self.seq += 1;
        self.events.push(Timed {
            t,
            seq: self.seq,
            ev,
        });
    }

    fn kv_bytes_for(&self, tokens: u64) -> f64 {
        cost::kv_bytes(&self.cfg.model, tokens) as f64
    }

    fn tier_free(&self, i: usize) -> f64 {
        self.cfg.tiers[i].capacity_bytes() - self.tier_used[i]
    }

    fn tier_free_vec(&self) -> Vec<f64> {
        (0..self.cfg.tiers.len())
            .map(|i| self.tier_free(i))
            .collect()
    }

    fn transfer(&self, bytes: f64, bw_gbps: f64, latency_us: f64) -> f64 {
        cost::transfer_seconds(bytes, bw_gbps * f64::from(self.cfg.tp), latency_us)
    }

    fn recompute_seconds(&self, prefix_tokens: u64) -> Result<f64, CostError> {
        if prefix_tokens == 0 {
            return Ok(0.0);
        }
        Ok(cost::prefill(
            &self.cfg.model,
            &self.cfg.device,
            prefix_tokens,
            0,
            self.cfg.tp,
            &self.cfg.eff,
        )?
        .seconds)
    }

    fn decoders_active_at(&self, t: f64) -> usize {
        self.dev
            .decoders
            .iter()
            .filter(|(_, done)| *done > t)
            .count()
    }

    fn next_decode_completion(&self, after: f64) -> Option<f64> {
        self.dev
            .decoders
            .iter()
            .map(|(_, done)| *done)
            .filter(|done| *done > after)
            .min_by(f64::total_cmp)
    }

    fn context(&self, recompute_s: f64) -> (Vec<f64>, f64, f64) {
        (
            self.tier_free_vec(),
            self.dev.kv_capacity - self.dev.kv_used,
            recompute_s,
        )
    }

    fn run_all(&mut self) -> Result<(), SimError> {
        while let Some(Timed { t, ev, .. }) = self.events.pop() {
            self.m.makespan = self.m.makespan.max(t);
            match ev {
                Event::StepArrival { sid } => self.on_arrival(sid, t)?,
                Event::DecodeDone { sid } => self.on_decode_done(sid, t)?,
                Event::SpillDone {
                    sid,
                    tier,
                    gen,
                    bytes,
                } => self.on_spill_done(sid, tier, gen, bytes),
                Event::PrefetchDue { sid, gen } => self.on_prefetch_due(sid, gen, t),
            }
        }
        Ok(())
    }

    /// Evict idle sessions until `shortfall` bytes are free. Writes to tiers occupy the
    /// transfer channel; returns the time at which the freed memory is usable.
    fn evict(&mut self, shortfall: f64, now: f64, exclude: usize, recompute_s: f64) -> f64 {
        let cfg = self.cfg;
        let policy = self.policy;
        let mut victims: Vec<Victim> = self
            .sess
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                *i != exclude
                    && !s.active
                    && !s.done
                    && !s.spill_pending
                    && s.kv_bytes > 0.0
                    && match s.state {
                        KvState::OnDevice => true,
                        KvState::InFlight { ready_at } => ready_at <= now,
                        _ => false,
                    }
            })
            .map(|(i, s)| Victim {
                session: i,
                kv_bytes: s.kv_bytes,
                last_use: s.last_use,
                predicted_return: s.predicted_return,
            })
            .collect();
        if victims.is_empty() {
            return now;
        }
        let (tier_free, device_free, recompute_s) = self.context(recompute_s);
        {
            let cx = PolicyContext {
                now,
                tiers: &cfg.tiers,
                tier_free: &tier_free,
                tp: cfg.tp,
                device_free,
                device_capacity: self.dev.kv_capacity,
                recompute_s,
                step_hint_s: self.step_hint_s,
            };
            policy.rank_victims(&cx, &mut victims);
        }
        let mut freed = 0.0;
        let mut usable_at = now;
        for v in victims {
            if freed >= shortfall {
                break;
            }
            let dest = (0..cfg.tiers.len()).find(|&i| self.tier_free(i) >= v.kv_bytes);
            match dest {
                Some(i) => {
                    let tier = &cfg.tiers[i];
                    let start = now.max(self.dev.transfer_free_at);
                    let dur = self.transfer(v.kv_bytes, tier.write_bw_gbps, tier.latency_us);
                    self.dev.transfer_free_at = start + dur;
                    usable_at = usable_at.max(start + dur);
                    self.tier_used[i] += v.kv_bytes;
                    self.sess[v.session].state = KvState::OnTier(i);
                    self.m.evictions += 1;
                }
                None => {
                    self.sess[v.session].state = KvState::Dropped;
                    self.sess[v.session].kv_bytes = 0.0;
                    self.m.evicted_drops += 1;
                }
            }
            self.dev.kv_used -= v.kv_bytes;
            freed += v.kv_bytes;
        }
        usable_at
    }

    fn on_arrival(&mut self, sid: usize, now: f64) -> Result<(), SimError> {
        let cfg = self.cfg;
        if self.sess[sid].done {
            return Ok(());
        }
        let step = &self.trace.sessions[sid].steps[self.sess[sid].step_idx];
        let (new_tokens, out_tokens) = (step.new_tokens, step.output_tokens);
        let prefix = self.sess[sid].prefix_tokens;
        let kv_prefix = self.kv_bytes_for(prefix);
        let kv_after = self.kv_bytes_for(prefix + new_tokens + out_tokens);
        let kv_new = (kv_after - kv_prefix).max(0.0);
        let state = self.sess[sid].state;
        let resident = matches!(state, KvState::OnDevice | KvState::InFlight { .. });
        let need = if resident { kv_new } else { kv_prefix + kv_new };
        if need > self.dev.kv_capacity {
            return Err(SimError::SessionTooLarge {
                session: self.trace.sessions[sid].id,
                need_gb: need / 1e9,
                capacity_gb: self.dev.kv_capacity / 1e9,
            });
        }
        // Any pending spill or prefetch for this session is now stale.
        self.sess[sid].gen += 1;
        self.sess[sid].spill_pending = false;

        let mut ready = now;
        if self.dev.kv_used + need > self.dev.kv_capacity {
            let shortfall = self.dev.kv_used + need - self.dev.kv_capacity;
            let recompute_s = self.recompute_seconds(prefix)?;
            let usable_at = self.evict(shortfall, now, sid, recompute_s);
            if self.dev.kv_used + need > self.dev.kv_capacity {
                self.m.capacity_waits += 1;
                let retry_at = self
                    .next_decode_completion(now)
                    .unwrap_or(now + 0.25)
                    .max(now + 1e-6);
                self.push(retry_at, Event::StepArrival { sid });
                return Ok(());
            }
            if usable_at > now {
                self.m.foreground_kv_wait_s += usable_at - now;
                ready = usable_at;
            }
        }
        self.sess[sid].active = true;
        self.sess[sid].step_start = now;

        let mut prefill_tokens = new_tokens;
        let mut cached = prefix;
        match state {
            KvState::OnDevice => {}
            KvState::InFlight { ready_at } => {
                if ready_at > ready {
                    self.m.foreground_kv_wait_s += ready_at - ready;
                    ready = ready_at;
                }
            }
            KvState::OnTier(i) => {
                let bytes = self.sess[sid].kv_bytes;
                let tier = &cfg.tiers[i];
                let start = ready.max(self.dev.transfer_free_at);
                let dur = self.transfer(bytes, tier.read_bw_gbps, tier.latency_us);
                self.dev.transfer_free_at = start + dur;
                self.m.foreground_kv_wait_s += start + dur - ready;
                ready = start + dur;
                self.m.fetched_bytes += bytes;
                self.tier_used[i] -= bytes;
                self.dev.kv_used += bytes;
            }
            KvState::Absent | KvState::Dropped => {
                if prefix > 0 {
                    if state == KvState::Dropped {
                        self.m.recompute_events += 1;
                        self.m.recompute_tokens += prefix;
                    }
                    prefill_tokens += prefix;
                    cached = 0;
                }
                self.dev.kv_used += kv_prefix;
            }
        }
        self.dev.kv_used += kv_new;
        self.sess[sid].kv_bytes = kv_after;
        self.sess[sid].state = KvState::OnDevice;

        let phase = cost::prefill(
            &cfg.model,
            &cfg.device,
            prefill_tokens,
            cached,
            cfg.tp,
            &cfg.eff,
        )?;
        let start = ready.max(self.dev.prefill_free_at);
        let first_token = start + phase.seconds;
        self.dev.prefill_free_at = first_token;
        self.m.ttft.push(first_token - now);
        self.m.device_busy_s += phase.seconds;
        self.m.prefill_tokens += prefill_tokens;
        self.sess[sid].prefix_tokens = prefix + new_tokens;

        let batch = self.decoders_active_at(first_token) + 1;
        let ctx = prefix + new_tokens + out_tokens / 2;
        let per_token =
            cost::decode_step(&cfg.model, &cfg.device, ctx, batch as u64, cfg.tp, &cfg.eff)?
                .seconds;
        let duration = out_tokens as f64 * per_token;
        let done = first_token + duration;
        self.dev.decoders.push((sid, done));
        self.m.device_busy_s += duration / batch as f64;
        self.sess[sid].prefix_tokens += out_tokens;
        // Track the typical compute time of a step (prefill plus decode, excluding waits).
        let compute_s = phase.seconds + duration;
        self.step_hint_s = 0.9 * self.step_hint_s + 0.1 * compute_s;
        self.push(done, Event::DecodeDone { sid });
        Ok(())
    }

    fn on_decode_done(&mut self, sid: usize, now: f64) -> Result<(), SimError> {
        let cfg = self.cfg;
        let policy = self.policy;
        self.dev.decoders.retain(|(s, _)| *s != sid);
        self.m.step_latency.push(now - self.sess[sid].step_start);
        self.sess[sid].active = false;
        self.sess[sid].last_use = now;
        let step_idx = self.sess[sid].step_idx;
        let tool = self.trace.sessions[sid].steps[step_idx].tool.clone();
        self.sess[sid].step_idx += 1;
        let kv = self.sess[sid].kv_bytes;
        let Some(tw) = tool else {
            self.sess[sid].done = true;
            self.dev.kv_used -= kv;
            self.sess[sid].kv_bytes = 0.0;
            self.sess[sid].state = KvState::Absent;
            self.m.sessions_done += 1;
            return Ok(());
        };
        let return_at = now + tw.wait_s.max(0.0);
        self.sess[sid].predicted_return = now + tw.latency.median();
        let recompute_s = self.recompute_seconds(self.sess[sid].prefix_tokens)?;
        let (tier_free, device_free, recompute_s) = self.context(recompute_s);
        let action = {
            let cx = PolicyContext {
                now,
                tiers: &cfg.tiers,
                tier_free: &tier_free,
                tp: cfg.tp,
                device_free,
                device_capacity: self.dev.kv_capacity,
                recompute_s,
                step_hint_s: self.step_hint_s,
            };
            policy.on_tool_call(&cx, kv, &tw.latency)
        };
        match action {
            ResidencyAction::Keep => {}
            ResidencyAction::Drop => {
                self.dev.kv_used -= kv;
                self.sess[sid].kv_bytes = 0.0;
                self.sess[sid].state = KvState::Dropped;
                self.m.planned_drops += 1;
            }
            ResidencyAction::Spill {
                tier_index,
                prefetch_lead_s,
                ..
            } => {
                if tier_index < cfg.tiers.len() && self.tier_free(tier_index) >= kv && kv > 0.0 {
                    let tier = &cfg.tiers[tier_index];
                    let start = now.max(self.dev.transfer_free_at);
                    let dur = self.transfer(kv, tier.write_bw_gbps, tier.latency_us);
                    self.dev.transfer_free_at = start + dur;
                    let spill_done = start + dur;
                    self.tier_used[tier_index] += kv;
                    self.sess[sid].spill_pending = true;
                    self.sess[sid].prefetch_retries = 0;
                    let gen = self.sess[sid].gen;
                    self.push(
                        spill_done,
                        Event::SpillDone {
                            sid,
                            tier: tier_index,
                            gen,
                            bytes: kv,
                        },
                    );
                    // Prefetch so the KV is back before the 10th-percentile return.
                    let due =
                        (now + tw.latency.quantile(0.1) - prefetch_lead_s).max(spill_done + 1e-6);
                    self.push(due, Event::PrefetchDue { sid, gen });
                    self.m.spill_bytes += kv;
                    if spill_done <= return_at {
                        self.m.hidden_spill_bytes += kv;
                    }
                }
            }
        }
        self.push(return_at, Event::StepArrival { sid });
        Ok(())
    }

    fn on_spill_done(&mut self, sid: usize, tier: usize, gen: u32, bytes: f64) {
        let s = &self.sess[sid];
        if s.gen != gen || s.active || s.done || !s.spill_pending {
            // Stale: the session came back before the copy landed; release the reservation.
            self.tier_used[tier] -= bytes;
            return;
        }
        self.sess[sid].spill_pending = false;
        self.dev.kv_used -= bytes;
        self.sess[sid].state = KvState::OnTier(tier);
    }

    fn on_prefetch_due(&mut self, sid: usize, gen: u32, now: f64) {
        let cfg = self.cfg;
        let s = &self.sess[sid];
        if s.gen != gen || s.active || s.done {
            return;
        }
        let KvState::OnTier(i) = s.state else {
            return;
        };
        let bytes = s.kv_bytes;
        // Leave headroom for arrivals; a prefetch that fills the device only gets evicted again.
        if self.dev.kv_used + bytes <= 0.9 * self.dev.kv_capacity {
            let tier = &cfg.tiers[i];
            let start = now.max(self.dev.transfer_free_at);
            let dur = self.transfer(bytes, tier.read_bw_gbps, tier.latency_us);
            self.dev.transfer_free_at = start + dur;
            self.dev.kv_used += bytes;
            self.tier_used[i] -= bytes;
            self.sess[sid].state = KvState::InFlight {
                ready_at: start + dur,
            };
            self.m.prefetched_bytes += bytes;
        } else if self.sess[sid].prefetch_retries < 200 && now < self.sess[sid].predicted_return {
            self.sess[sid].prefetch_retries += 1;
            self.push(now + 1.0, Event::PrefetchDue { sid, gen });
        } else {
            self.m.prefetch_skipped += 1;
        }
    }

    fn report(&self) -> SimReport {
        SimReport {
            policy: self.policy.name().to_string(),
            sessions: self.trace.sessions.len() as u32,
            sessions_done: self.m.sessions_done,
            steps: self.m.ttft.len() as u64,
            makespan_s: self.m.makespan,
            ttft_mean_s: mean(&self.m.ttft),
            ttft_p50_s: percentile(&self.m.ttft, 0.5),
            ttft_p90_s: percentile(&self.m.ttft, 0.9),
            ttft_p99_s: percentile(&self.m.ttft, 0.99),
            step_p50_s: percentile(&self.m.step_latency, 0.5),
            step_p90_s: percentile(&self.m.step_latency, 0.9),
            step_p99_s: percentile(&self.m.step_latency, 0.99),
            recompute_events: self.m.recompute_events,
            recompute_tokens: self.m.recompute_tokens,
            foreground_kv_wait_s: self.m.foreground_kv_wait_s,
            fetched_gb: self.m.fetched_bytes / 1e9,
            prefetched_gb: self.m.prefetched_bytes / 1e9,
            spill_gb: self.m.spill_bytes / 1e9,
            hidden_spill_gb: self.m.hidden_spill_bytes / 1e9,
            evictions: self.m.evictions,
            evicted_drops: self.m.evicted_drops,
            planned_drops: self.m.planned_drops,
            capacity_waits: self.m.capacity_waits,
            prefetch_skipped: self.m.prefetch_skipped,
            device_busy_s: self.m.device_busy_s,
            prefill_tokens: self.m.prefill_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Lru, Plan};
    use crate::trace::{synthesize, SyntheticConfig};
    use agentforge_ir::{AgentProgram, Precision};
    use agentforge_machine::Machine;

    /// Llama 3.1 70B with FP8 KV on eight H200s: 80 sessions with 110k-token median prefixes
    /// need about 1.5 TB of KV against a 945 GB device budget, so idle sessions must leave the
    /// device, while the 2 TB host tier has room for all of them.
    fn pressure_config() -> SimConfig {
        let m = Machine::builtin("h200").unwrap();
        let mut model = ModelSpec::preset("llama-3.1-70b").unwrap();
        model.kv_precision = Precision::Fp8;
        let tiers = m.tiers.clone();
        SimConfig::new(model, m.devices[0].clone(), tiers, 8)
    }

    fn trace(sessions: u32, seed: u64) -> Trace {
        synthesize(
            &AgentProgram::coding_agent_default(),
            &SyntheticConfig {
                sessions,
                mean_steps: 20.0,
                arrival_rate_per_s: 0.5,
                seed,
                ..Default::default()
            },
        )
    }

    #[test]
    fn capacity_accounting() {
        let cfg = pressure_config();
        let cap = cfg.kv_capacity_bytes();
        assert!(cap > 800e9 && cap < 1000e9, "{cap}");
    }

    #[test]
    fn runs_are_deterministic_and_complete() {
        let cfg = pressure_config();
        let t = trace(48, 3);
        let a = run(&cfg, &t, &Lru).unwrap();
        let b = run(&cfg, &t, &Lru).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.sessions_done, a.sessions);
        assert_eq!(a.steps, t.total_steps());
        let p = run(&cfg, &t, &Plan).unwrap();
        assert_eq!(p.sessions_done, p.sessions);
        assert_eq!(p.steps, t.total_steps());
    }

    #[test]
    fn plan_beats_lru_under_pressure() {
        let cfg = pressure_config();
        let t = trace(80, 11);
        let lru = run(&cfg, &t, &Lru).unwrap();
        let plan = run(&cfg, &t, &Plan).unwrap();
        eprintln!(
            "foreground wait: plan {:.1}s vs lru {:.1}s; ttft p90: plan {:.3}s vs lru {:.3}s",
            plan.foreground_kv_wait_s, lru.foreground_kv_wait_s, plan.ttft_p90_s, lru.ttft_p90_s
        );
        assert!(
            plan.spill_gb > 0.0,
            "the plan should spill during long waits"
        );
        // The robust effect across seeds is fewer pressure evictions (30 to 40 percent fewer in
        // the M0 sweep); foreground waits and tail latency improve in most seeds but not all.
        assert!(
            plan.evictions < lru.evictions,
            "plan {} vs lru {}",
            plan.evictions,
            lru.evictions
        );
        assert!(
            plan.recompute_tokens <= lru.recompute_tokens,
            "plan {} vs lru {}",
            plan.recompute_tokens,
            lru.recompute_tokens
        );
    }

    #[test]
    fn model_that_does_not_fit_is_an_error() {
        let m = Machine::builtin("blackhole-p150").unwrap();
        let cfg = SimConfig::new(
            ModelSpec::preset("llama-3.1-70b").unwrap(),
            m.devices[0].clone(),
            m.tiers.clone(),
            1,
        );
        assert!(matches!(
            run(&cfg, &trace(2, 1), &Lru),
            Err(SimError::ModelDoesNotFit { .. })
        ));
    }
}
