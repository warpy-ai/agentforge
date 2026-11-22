//! The discrete-event engine.
//!
//! Model of the machine: one device class scaled linearly by `tp`; prefill is serialized on the
//! device, decode is batched (per-token time from the cost model with the batch size at the
//! start of the decode), and one transfer channel per device group moves KV between the device
//! and the tiers. By default a prefill stalls every decode running when it starts for its whole
//! duration (see [`SimConfig::prefill_stalls_decode`]). New sessions can be held at an
//! admission gate ([`SimConfig::admit_fraction`]). The transfer channel serves transfers in the
//! order they are issued, so a prefetch is issued only when, behind what the channel already
//! carries, it lands before the session's predicted return. Session KV lives in one of: device
//! memory, device memory mirrored (a write-through copy has landed the first part of it in a
//! tier, so evicting it writes only the rest), a tier, in flight back to the device, or nowhere
//! (dropped, which forces a prefix recompute on the next step).
//!
//! Fairness between policies: pressure evictions are available to every policy and their
//! writes occupy the transfer channel before the arriving session can start, because memory is
//! only free once the copy has landed. The plan policy's advantage comes from knowing which tool
//! a session is waiting on and spilling during the waits the program predicts, not from a
//! cheaper eviction primitive; the write-through policy makes evictions cheaper by paying for
//! the copies during every wait instead.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use agentforge_compile::ResidencyAction;
use agentforge_cost::{self as cost, CostError, Efficiency, Target};
use agentforge_ir::ModelSpec;
use agentforge_machine::{Device, Link, MemoryTier};

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
    /// The link the tensor-parallel group all-reduces over; required when `tp > 1`.
    pub link: Option<Link>,
    pub eff: Efficiency,
    /// Fraction of device memory reserved for activations and fragmentation.
    pub kv_reserve_fraction: f64,
    /// Whether a prefill stalls the decodes running when it starts. Chunked prefill in current
    /// engines lets decode progress one token per prefill chunk, which to first order is a stall
    /// of the prefill's duration; the decode batch's KV reads do not slow a compute-bound
    /// prefill, so the prefill keeps its own duration either way.
    pub prefill_stalls_decode: bool,
    /// Admission gate for new sessions: a session's first step starts only while the device KV
    /// in use, its own included, stays within this fraction of the KV budget, or once the
    /// device holds no KV at all. Until then the session waits in arrival order and the wait
    /// counts in its TTFT. `None` admits every session on arrival.
    pub admit_fraction: Option<f64>,
}

impl SimConfig {
    pub fn new(model: ModelSpec, device: Device, tiers: Vec<MemoryTier>, tp: u32) -> SimConfig {
        SimConfig {
            model,
            device,
            tiers,
            tp: tp.max(1),
            link: None,
            eff: Efficiency::default(),
            kv_reserve_fraction: 0.10,
            prefill_stalls_decode: true,
            admit_fraction: None,
        }
    }

    fn target(&self) -> Target<'_> {
        Target {
            device: &self.device,
            tp: self.tp,
            link: self.link.as_ref(),
            eff: self.eff,
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
    Dropped,
    OnDevice,
    Mirrored { tier: usize, landed_bytes: f64 },
    OnTier(usize),
    InFlight { ready_at: f64 },
}

#[derive(Debug, Clone, Copy)]
enum Event {
    SessionArrival {
        sid: usize,
    },
    StepArrival {
        sid: usize,
    },
    DecodeDone {
        sid: usize,
    },
    CopyDone {
        sid: usize,
        gen: u32,
        tier: usize,
        bytes: f64,
        release: bool,
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
    copying: bool,
    step_arrival: f64,
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
    free_evictions: u64,
    evicted_drops: u64,
    planned_drops: u64,
    capacity_waits: u64,
    sessions_queued: u64,
    admission_wait_s: f64,
    prefetch_skipped: u64,
    prefetch_late: u64,
    decode_stall_s: f64,
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
    admit_queue: VecDeque<usize>,
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
        let prefill = cost::prefill(&cfg.model, &cfg.target(), 875, typical_prefix)?;
        let decode = cost::decode_step(&cfg.model, &cfg.target(), typical_prefix, 4)?;
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
                copying: false,
                step_arrival: s.arrival_s,
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
            admit_queue: VecDeque::new(),
            events: BinaryHeap::new(),
            seq: 0,
            m: Metrics::default(),
            step_hint_s,
        };
        for (sid, s) in trace.sessions.iter().enumerate() {
            if !s.steps.is_empty() {
                sim.push(s.arrival_s, Event::SessionArrival { sid });
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

    fn unmirrored_bytes(&self, sid: usize, tier: usize) -> f64 {
        let s = &self.sess[sid];
        match s.state {
            KvState::Mirrored {
                tier: t,
                landed_bytes,
            } if t == tier => s.kv_bytes - landed_bytes,
            _ => s.kv_bytes,
        }
    }

    fn discard_mirror(&mut self, sid: usize) {
        if let KvState::Mirrored { tier, landed_bytes } = self.sess[sid].state {
            self.tier_used[tier] -= landed_bytes;
            self.sess[sid].state = KvState::OnDevice;
        }
    }

    fn claim_tier(&mut self, sid: usize, tier: usize) -> f64 {
        let write = self.unmirrored_bytes(sid, tier);
        if !matches!(self.sess[sid].state, KvState::Mirrored { tier: t, .. } if t == tier) {
            self.discard_mirror(sid);
        }
        self.tier_used[tier] += write;
        write
    }

    fn start_copy(
        &mut self,
        sid: usize,
        tier: usize,
        now: f64,
        release: bool,
        return_at: f64,
    ) -> Option<f64> {
        if tier >= self.cfg.tiers.len()
            || self.sess[sid].kv_bytes <= 0.0
            || self.tier_free(tier) < self.unmirrored_bytes(sid, tier)
        {
            return None;
        }
        let bytes = self.claim_tier(sid, tier);
        let t = &self.cfg.tiers[tier];
        let start = now.max(self.dev.transfer_free_at);
        let lands = start + self.transfer(bytes, t.write_bw_gbps, t.latency_us);
        self.dev.transfer_free_at = lands;
        self.sess[sid].copying = true;
        let gen = self.sess[sid].gen;
        self.push(
            lands,
            Event::CopyDone {
                sid,
                gen,
                tier,
                bytes,
                release,
            },
        );
        self.m.spill_bytes += bytes;
        if lands <= return_at {
            self.m.hidden_spill_bytes += bytes;
        }
        Some(lands)
    }

    fn recompute_seconds(&self, prefix_tokens: u64) -> Result<f64, CostError> {
        if prefix_tokens == 0 {
            return Ok(0.0);
        }
        Ok(cost::prefill(&self.cfg.model, &self.cfg.target(), prefix_tokens, 0)?.seconds)
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
                Event::SessionArrival { sid } => self.on_session_arrival(sid, t)?,
                Event::StepArrival { sid } => self.on_arrival(sid, t)?,
                Event::DecodeDone { sid } => self.on_decode_done(sid, t)?,
                Event::CopyDone {
                    sid,
                    gen,
                    tier,
                    bytes,
                    release,
                } => self.on_copy_done(sid, gen, tier, bytes, release),
                Event::PrefetchDue { sid, gen } => self.on_prefetch_due(sid, gen, t),
            }
            self.admit_queued(t)?;
        }
        Ok(())
    }

    fn admissible(&self, sid: usize) -> bool {
        let Some(fraction) = self.cfg.admit_fraction else {
            return true;
        };
        let step = &self.trace.sessions[sid].steps[0];
        let need =
            self.kv_bytes_for(self.sess[sid].prefix_tokens + step.new_tokens + step.output_tokens);
        self.dev.kv_used <= 0.0 || self.dev.kv_used + need <= fraction * self.dev.kv_capacity
    }

    fn on_session_arrival(&mut self, sid: usize, now: f64) -> Result<(), SimError> {
        if self.admit_queue.is_empty() && self.admissible(sid) {
            return self.on_arrival(sid, now);
        }
        self.admit_queue.push_back(sid);
        self.m.sessions_queued += 1;
        Ok(())
    }

    fn admit_queued(&mut self, now: f64) -> Result<(), SimError> {
        while let Some(&sid) = self.admit_queue.front() {
            if !self.admissible(sid) {
                break;
            }
            self.admit_queue.pop_front();
            self.m.admission_wait_s += now - self.sess[sid].step_arrival;
            self.on_arrival(sid, now)?;
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
                    && !s.copying
                    && s.kv_bytes > 0.0
                    && match s.state {
                        KvState::OnDevice | KvState::Mirrored { .. } => true,
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
                channel_backlog_s: (self.dev.transfer_free_at - now).max(0.0),
            };
            policy.rank_victims(&cx, &mut victims);
        }
        let mut freed = 0.0;
        let mut usable_at = now;
        for v in victims {
            if freed >= shortfall {
                break;
            }
            let dest = match self.sess[v.session].state {
                KvState::Mirrored { tier, landed_bytes }
                    if self.tier_free(tier) >= v.kv_bytes - landed_bytes =>
                {
                    Some(tier)
                }
                _ => (0..cfg.tiers.len()).find(|&i| self.tier_free(i) >= v.kv_bytes),
            };
            match dest {
                Some(i) => {
                    let write = self.claim_tier(v.session, i);
                    if write > 0.0 {
                        let tier = &cfg.tiers[i];
                        let start = now.max(self.dev.transfer_free_at);
                        let dur = self.transfer(write, tier.write_bw_gbps, tier.latency_us);
                        self.dev.transfer_free_at = start + dur;
                        usable_at = usable_at.max(start + dur);
                    } else {
                        self.m.free_evictions += 1;
                    }
                    self.sess[v.session].state = KvState::OnTier(i);
                    self.m.evictions += 1;
                }
                None => {
                    self.discard_mirror(v.session);
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
        let resident = matches!(
            state,
            KvState::OnDevice | KvState::Mirrored { .. } | KvState::InFlight { .. }
        );
        let need = if resident { kv_new } else { kv_prefix + kv_new };
        if kv_after > self.dev.kv_capacity {
            return Err(SimError::SessionTooLarge {
                session: self.trace.sessions[sid].id,
                need_gb: kv_after / 1e9,
                capacity_gb: self.dev.kv_capacity / 1e9,
            });
        }
        self.sess[sid].gen += 1;
        self.sess[sid].copying = false;

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

        let mut prefill_tokens = new_tokens;
        let mut cached = prefix;
        match state {
            KvState::OnDevice | KvState::Mirrored { .. } => {}
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
        if !matches!(state, KvState::Mirrored { .. }) {
            self.sess[sid].state = KvState::OnDevice;
        }

        let phase = cost::prefill(&cfg.model, &cfg.target(), prefill_tokens, cached)?;
        let start = ready.max(self.dev.prefill_free_at);
        let first_token = start + phase.seconds;
        self.dev.prefill_free_at = first_token;
        if cfg.prefill_stalls_decode {
            for (_, done) in self
                .dev
                .decoders
                .iter_mut()
                .filter(|(_, done)| *done > start)
            {
                *done += phase.seconds;
                self.m.decode_stall_s += phase.seconds;
            }
        }
        self.m.ttft.push(first_token - self.sess[sid].step_arrival);
        self.m.device_busy_s += phase.seconds;
        self.m.prefill_tokens += prefill_tokens;
        self.sess[sid].prefix_tokens = prefix + new_tokens;

        let batch = self.decoders_active_at(first_token) + 1;
        let ctx = prefix + new_tokens + out_tokens / 2;
        let per_token = cost::decode_step(&cfg.model, &cfg.target(), ctx, batch as u64)?.seconds;
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
        if let Some(&(_, done)) = self.dev.decoders.iter().find(|(s, _)| *s == sid) {
            if done > now {
                self.push(done, Event::DecodeDone { sid });
                return Ok(());
            }
        }
        self.dev.decoders.retain(|(s, _)| *s != sid);
        self.m.step_latency.push(now - self.sess[sid].step_arrival);
        self.sess[sid].active = false;
        self.sess[sid].last_use = now;
        let step_idx = self.sess[sid].step_idx;
        let tool = self.trace.sessions[sid].steps[step_idx].tool.clone();
        self.sess[sid].step_idx += 1;
        let kv = self.sess[sid].kv_bytes;
        let Some(tw) = tool else {
            self.sess[sid].done = true;
            self.discard_mirror(sid);
            self.dev.kv_used -= kv;
            self.sess[sid].kv_bytes = 0.0;
            self.sess[sid].state = KvState::Absent;
            self.m.sessions_done += 1;
            return Ok(());
        };
        let return_at = now + tw.wait_s.max(0.0);
        self.sess[sid].step_arrival = return_at;
        self.sess[sid].predicted_return = now + tw.latency.median();
        let recompute_s = self.recompute_seconds(self.sess[sid].prefix_tokens)?;
        let (tier_free, device_free, recompute_s) = self.context(recompute_s);
        let mirrored_bytes = match self.sess[sid].state {
            KvState::Mirrored { landed_bytes, .. } => landed_bytes,
            _ => 0.0,
        };
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
                channel_backlog_s: (self.dev.transfer_free_at - now).max(0.0),
            };
            policy.on_tool_call(&cx, kv, mirrored_bytes, &tw.latency)
        };
        match action {
            ResidencyAction::Keep => {}
            ResidencyAction::Drop => {
                self.discard_mirror(sid);
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
                if let Some(lands) = self.start_copy(sid, tier_index, now, true, return_at) {
                    let gen = self.sess[sid].gen;
                    // Prefetch so the KV is back before the 10th-percentile return.
                    let due = (now + tw.latency.quantile(0.1) - prefetch_lead_s).max(lands + 1e-6);
                    self.push(due, Event::PrefetchDue { sid, gen });
                }
            }
            ResidencyAction::WriteThrough { tier_index } => {
                self.start_copy(sid, tier_index, now, false, return_at);
            }
        }
        self.push(return_at, Event::StepArrival { sid });
        Ok(())
    }

    fn on_copy_done(&mut self, sid: usize, gen: u32, tier: usize, bytes: f64, release: bool) {
        if self.sess[sid].gen != gen {
            self.tier_used[tier] -= bytes;
            return;
        }
        self.sess[sid].copying = false;
        let kv = self.sess[sid].kv_bytes;
        self.sess[sid].state = if release {
            self.dev.kv_used -= kv;
            KvState::OnTier(tier)
        } else {
            KvState::Mirrored {
                tier,
                landed_bytes: kv,
            }
        };
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
        let (bytes, predicted_return) = (s.kv_bytes, s.predicted_return);
        let tier = &cfg.tiers[i];
        let dur = self.transfer(bytes, tier.read_bw_gbps, tier.latency_us);
        let lands = now.max(self.dev.transfer_free_at) + dur;
        if lands > predicted_return {
            self.m.prefetch_late += 1;
            return;
        }
        // Leave headroom for arrivals; a prefetch that fills the device only gets evicted again.
        if self.dev.kv_used + bytes <= 0.9 * self.dev.kv_capacity {
            self.dev.transfer_free_at = lands;
            self.dev.kv_used += bytes;
            self.tier_used[i] -= bytes;
            self.sess[sid].state = KvState::InFlight { ready_at: lands };
            self.m.prefetched_bytes += bytes;
            return;
        }
        let retry_at = self.dev.transfer_free_at.max(now + 1.0);
        if retry_at + dur <= predicted_return {
            self.push(retry_at, Event::PrefetchDue { sid, gen });
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
            free_evictions: self.m.free_evictions,
            evicted_drops: self.m.evicted_drops,
            planned_drops: self.m.planned_drops,
            capacity_waits: self.m.capacity_waits,
            sessions_queued: self.m.sessions_queued,
            admission_wait_s: self.m.admission_wait_s,
            prefetch_skipped: self.m.prefetch_skipped,
            prefetch_late: self.m.prefetch_late,
            decode_stall_s: self.m.decode_stall_s,
            device_busy_s: self.m.device_busy_s,
            prefill_tokens: self.m.prefill_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Lru, Plan, WriteThrough};
    use crate::trace::{synthesize, Session, SyntheticConfig, ToolWait, TraceStep};
    use agentforge_ir::{AgentProgram, LatencyModel, Precision};
    use agentforge_machine::Machine;

    /// Llama 3.1 70B with FP8 KV on eight H200s: 80 sessions with 110k-token median prefixes
    /// need about 1.5 TB of KV against a 945 GB device budget, so idle sessions must leave the
    /// device, while the 2 TB host tier has room for all of them.
    fn pressure_config() -> SimConfig {
        let m = Machine::builtin("h200").unwrap();
        let mut model = ModelSpec::preset("llama-3.1-70b").unwrap();
        model.kv_precision = Precision::Fp8;
        let tiers = m.tiers.clone();
        let mut cfg = SimConfig::new(model, m.devices[0].clone(), tiers, 8);
        cfg.link = m.fastest_link().cloned();
        cfg
    }

    fn trace(sessions: u32, seed: u64) -> Trace {
        synthesize(
            &AgentProgram::coding_agent_default(),
            &SyntheticConfig {
                sessions,
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
    fn ttft_includes_the_wait_for_device_memory() {
        let m = Machine::builtin("h200").unwrap();
        let model = ModelSpec::preset("llama-3.1-8b").unwrap();
        let mut cfg = SimConfig::new(model, m.devices[0].clone(), m.tiers.clone(), 1);
        cfg.kv_reserve_fraction = 0.8;
        let (prefix, new_tokens, output_tokens) = (60_000, 1_000, 200);
        let session = |id, arrival_s| Session {
            id,
            arrival_s,
            initial_prefix_tokens: prefix,
            steps: vec![TraceStep {
                new_tokens,
                output_tokens,
                tool: None,
            }],
        };
        let t = Trace {
            sessions: vec![session(0, 0.0), session(1, 0.001)],
        };
        let need = cost::kv_bytes(&cfg.model, prefix + new_tokens + output_tokens) as f64;
        let budget = cfg.kv_capacity_bytes();
        assert!(need > budget / 2.0 && need < budget, "{need} of {budget}");

        let r = run(&cfg, &t, &Lru).unwrap();

        let target = cfg.target();
        let first_prefill = cost::prefill(&cfg.model, &target, prefix + new_tokens, 0)
            .unwrap()
            .seconds;
        let first_decode = output_tokens as f64
            * cost::decode_step(&cfg.model, &target, prefix + new_tokens, 1)
                .unwrap()
                .seconds;
        assert_eq!(r.sessions_done, 2);
        assert!(r.capacity_waits >= 1, "{}", r.capacity_waits);
        assert!(
            r.ttft_p99_s >= first_prefill + first_decode,
            "second TTFT {:.3}s, first session held the memory for {:.3}s",
            r.ttft_p99_s,
            first_prefill + first_decode
        );
    }

    #[test]
    fn prefill_stalls_the_decodes_it_interrupts() {
        let mut cfg = pressure_config();
        let t = trace(80, 11);
        let stalled = run(&cfg, &t, &Lru).unwrap();
        cfg.prefill_stalls_decode = false;
        let overlapped = run(&cfg, &t, &Lru).unwrap();
        assert_eq!(stalled.steps, overlapped.steps);
        assert!(stalled.decode_stall_s > 0.0, "{}", stalled.decode_stall_s);
        assert_eq!(overlapped.decode_stall_s, 0.0);
        assert!(
            stalled.step_p50_s > overlapped.step_p50_s,
            "stalled {:.3}s vs overlapped {:.3}s",
            stalled.step_p50_s,
            overlapped.step_p50_s
        );
        assert_eq!(run(&cfg, &t, &Lru).unwrap(), overlapped);
    }

    #[test]
    fn write_through_makes_pressure_evictions_cheaper() {
        let cfg = pressure_config();
        let t = trace(80, 11);
        let lru = run(&cfg, &t, &Lru).unwrap();
        let wt = run(&cfg, &t, &WriteThrough).unwrap();
        assert_eq!(wt, run(&cfg, &t, &WriteThrough).unwrap());
        assert_eq!(wt.sessions_done, wt.sessions);
        assert_eq!(wt.steps, t.total_steps());
        assert!(
            wt.free_evictions > 0,
            "{} of {}",
            wt.free_evictions,
            wt.evictions
        );
        assert_eq!(wt.recompute_tokens, 0);
        assert!(
            wt.foreground_kv_wait_s < lru.foreground_kv_wait_s,
            "write-through {:.1}s vs lru {:.1}s",
            wt.foreground_kv_wait_s,
            lru.foreground_kv_wait_s
        );
    }

    #[test]
    fn admission_gate_queues_new_sessions_instead_of_crowding_the_device() {
        let mut cfg = pressure_config();
        let t = trace(80, 11);
        let open = run(&cfg, &t, &Lru).unwrap();
        cfg.admit_fraction = Some(0.7);
        let gated = run(&cfg, &t, &Lru).unwrap();
        assert_eq!(open.sessions_queued, 0);
        assert_eq!(open.admission_wait_s, 0.0);
        assert_eq!(gated.sessions_done, gated.sessions);
        assert_eq!(gated.steps, t.total_steps());
        assert!(gated.sessions_queued > 0);
        assert!(
            gated.capacity_waits < open.capacity_waits,
            "gated {} vs open {}",
            gated.capacity_waits,
            open.capacity_waits
        );
        let ttft_sum = gated.ttft_mean_s * gated.steps as f64;
        assert!(
            ttft_sum >= gated.admission_wait_s,
            "TTFT sum {ttft_sum:.1}s vs admission wait {:.1}s",
            gated.admission_wait_s
        );
    }

    #[test]
    fn session_that_outgrows_the_kv_budget_is_an_error() {
        let cfg = pressure_config();
        let per_token = cfg.model.kv_bytes_per_token() as f64;
        let fits = (0.6 * cfg.kv_capacity_bytes() / per_token) as u64;
        let step = |new_tokens| TraceStep {
            new_tokens,
            output_tokens: 8,
            tool: Some(ToolWait {
                name: "read".to_string(),
                wait_s: 1.0,
                latency: LatencyModel::Fixed { seconds: 1.0 },
            }),
        };
        let t = Trace {
            sessions: vec![Session {
                id: 7,
                arrival_s: 0.0,
                initial_prefix_tokens: 0,
                steps: vec![step(fits), step(fits), step(1)],
            }],
        };
        match run(&cfg, &t, &Lru) {
            Err(SimError::SessionTooLarge { session: 7, .. }) => {}
            other => panic!("expected SessionTooLarge, got {other:?}"),
        }
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
