# 0001: AgentForge compiles agent programs and models against explicit machine models

Status: accepted as the project thesis on 2026-10-30. Supersedes nothing. Research basis:
`../research/reports/AI inference silicon landscape 2026.md`.

## 1. Problem

Today's compilers stop at the kernel or at one model iteration. The serving layer owns the KV
cache, batching, prefill/decode placement and prefix reuse, in Python, over hand-written
per-vendor kernels. Agentic workloads make that split expensive: a median coding-agent step
carries about 119k prefix tokens, appends 875, generates 214, and is followed by a tool call
whose duration ranges from a second to minutes; prefix-cache hit rates are 95 to 96 percent, and
the misses, caused by long waits, cost seconds of recompute each. Heterogeneous prefill and
decode silicon is now shipping (Vera Rubin with Groq 3 LPX; Trainium with Cerebras) but is
configured by hand. NVIDIA and AMD chose different sub-byte block formats, so a quantized model
does not move between them natively. Async protocol bugs (TMEM, mbarrier) recur across the best
kernel libraries.

## 2. Thesis

AgentForge compiles the agent program and the model together against an explicit,
data-described machine model. The agent program tells the compiler what the serving layer today
has to guess: which prompt blocks are stable, when a session will come back, what it will append,
and which branches are likely. The machine model tells it where bytes can live and how fast they
move. From the two it plans KV residency, prefetch, placement and kernels before the first token
is served.

## 3. Architecture

Layers, top to bottom:

1. **Agent Program IR (`agentforge-ir::agent`).** Steps are LLM calls with structured prompt
   blocks (system, tools, history, tool result, user, reasoning) marked stable or not; transitions
   are tool calls with latency models, probabilistic branches, or end. KV block liveness is derived
   from it the way register liveness is derived from a control-flow graph.
2. **Model graph IR (`agentforge-ir::model`).** Architecture parameters with attention and MoE
   variants as structured descriptions (dense, grouped-query, latent, sliding-window, compressed)
   and numerics as first-class types, including block-scaled formats (element type, scale type,
   block size).
3. **Tile IR (`agentforge-ir::tile`, skeleton).** Tiles with explicit memory spaces (global,
   shared, register, tensor memory), async copies and MMA operations, parametrized by the machine
   model. Async protocols (barrier phases, tensor-memory allocation) are to be encoded in Rust
   types so misuse fails to compile.
4. **Machine model (`agentforge-machine`).** TOML descriptions: devices with class, compute per
   precision, memory capacity and bandwidth, on-chip SRAM and its bandwidth, transcendental
   throughput; memory tiers (host DRAM, NVMe, remote pools) with capacity, read and write bandwidth
   and latency; links. Bundled: `h100`, `h200`, `b200`, `mi355x`, `blackhole-p150`, `lpx-tier`.
5. **Cost model and simulator (`agentforge-cost`, `agentforge-sim`).** Per-operator roofline
   costs for prefill, decode and transfers (dense projections and attention as separate kernels,
   tensor-parallel all-reduces on the critical path), calibrated against published TensorRT-LLM
   throughput, and a discrete-event replay of agent sessions over the machine description with
   prefill/decode interference, a scheduled transfer channel and admission control. Recorded
   FreeInference and TraceLab traces replay through loaders. The same files drive the compiler,
   so every policy is testable before silicon.
6. **Compiler passes (`agentforge-compile`).** KV liveness, residency planning (Belady-style
   cost comparison over tiers), prefill/decode placement; e-graph rewrites of the model graph and
   tile programs come next.
7. **Backends (planned).** Emit public, stable low-level formats: PTX text or NVPTX IR, CUDA
   Tile IR bytecode on Blackwell and later, LLVM AMDGPU IR, SPIR-V, Metal Shading Language,
   tt-metal programs, Cranelift for CPUs. No MLIR dependency in the baseline.
8. **Runtime (`agentforge-runtime`).** Executor, KV block store with lifetimes from the plan,
   transports between tiers and devices; memory-safe by construction.

## 4. Novelty versus prior art

| Prior art | What it does | What AgentForge adds |
|---|---|---|
| llm-d (CNCF, v1.0 on 2026-10-20), NVIDIA Dynamo | Request-time routing, KV-aware placement, tiered offload and prefill/decode disaggregation above vLLM and SGLang; llm-d's agent-aware serving infers turn depth from a client header and a dispatch count, with an idle timer in place of tool durations | A plan computed before the request from the agent program and the machine model: residency, prefetch timed to the predicted return, placement across device classes; the plan can be consumed through their scheduler plugin and KV connector interfaces |
| Continuum (ICLR 2026) | KV time-to-live after tool calls, predicted at runtime | Liveness and tiering decided from the program, with prefetch and placement, not only pinning |
| CacheWise, ThunderAgent, Kairos, TokenCake | Runtime eviction and scheduling from tool metadata | Same information consumed at compile time and shared with kernel generation |
| SGLang frontend, Parrot | LM-program abstractions above a serving runtime | A compiler below them with an explicit machine model |
| Mirage Persistent Kernel, megakernels | Whole-model schedule inside one kernel, CUDA only | Whole-session schedule across tiers and device classes, multi-vendor |
| Luminal, TileLang, CubeCL | Search-based or portable kernel generation | The agent and machine layers above the kernel layer; numerics as verified types |

## 5. Feasibility

Reachable targets today: NVIDIA (PTX, Tile IR, cutile-rs), AMD (LLVM AMDGPU), Intel (SPIR-V, open
NPU compiler), Apple GPU (MSL), Tenstorrent (fully open), AMD XDNA 2 (Peano), Arm Ethos-U (Vela),
CPU. Closed: TPU, Trainium beyond NKI, Groq/LPX, Cerebras, SambaNova, Etched, d-Matrix, Apple
ANE, Qualcomm HMX; these are modelling targets through machine files.

First-order arithmetic for the median TraceLab step on a 70B grouped-query model (80 layers, 8 KV
heads, head dimension 128), FP16 KV cache:

| Quantity | Estimate |
|---|---|
| KV cache for the 119k prefix | about 38 GB (19 GB in FP8; far less with latent or compressed attention) |
| Recompute after a cache miss | about 1.7e16 FLOP, roughly 6 to 30 s depending on GPU and precision |
| Move over NVLink at 400 GB/s | about 0.1 s |
| Move from host DRAM over one PCIe 5 link | about 0.8 s |
| Move from one NVMe drive | about 5 s |

With a 4 to 5 percent miss rate the misses dominate tail latency. Continuum measured 1.1 to 3.7x
from a runtime heuristic; compile-time knowledge should match it and extends to placement.

### First simulation results

Superseded in part by "M1 simulator validation" below: the TTFT and step-latency rows in this
section exclude time spent waiting for device memory, and the cost defaults have since been
recalibrated, so the M1 tables are the ones to quote.

Setup (milestone M0): `agentforge sim run` replays synthetic coding-agent sessions shaped like
the TraceLab measurements (110k-token median initial prefix, 875 appended and 214 generated
tokens per step; per step the agent calls a fast tool 75 percent of the time with a 1.5 s median
wait, a build or test suite 15 percent at 45 s, or hands back to a human 10 percent at 180 s)
against Llama 3.1 70B with FP8 weights on eight H200 or B200 GPUs (tensor-parallel resources
scaled linearly). Two policies replay the same trace. `lru` keeps every session's KV on the
device until pressure forces least-recently-used eviction, whose write lands on the arriving
session's critical path, and fetches it back in the foreground. `plan` uses the per-tool latency
model from the Agent Program IR: it keeps the KV across short waits, spills it during long waits
and prefetches before the predicted return. The tunables below were found in this exercise and
are now defaults: spill only above 80 percent device occupancy; keep when the median wait is
shorter than the useful horizon (twice a typical step, or three times the cheapest fetch);
prefetch only while 10 percent of the device budget stays free.

**Regime D, the one the thesis targets** (FP8 KV, 80 sessions, 2 TB host tier, H200 x8,
seed 11):

model llama-3.1-70b (fp8 weights, fp8 KV) on h200-sxm x8 (945 GB KV budget); 80 sessions, 1719 steps

| metric | lru | plan |
|---|---:|---:|
| sessions done / total | 80 / 80 | 80 / 80 |
| steps | 1719 | 1719 |
| makespan (s) | 5725.7 | 5727.1 |
| TTFT mean (s) | 8.398 | 8.298 |
| TTFT p50 (s) | 0.063 | 0.063 |
| TTFT p90 (s) | 33.224 | 22.960 |
| TTFT p99 (s) | 112.230 | 113.058 |
| step latency p50 (s) | 1.71 | 1.73 |
| step latency p90 (s) | 36.35 | 27.82 |
| step latency p99 (s) | 113.82 | 115.43 |
| prefix recomputes | 0 | 0 |
| recomputed tokens | 0 | 0 |
| foreground KV wait (s) | 20.4 | 16.5 |
| fetched in foreground (GB) | 3726 | 3681 |
| prefetched (GB) | 0 | 3777 |
| spilled during waits (GB) | 0 | 5229 |
| spills hidden by the wait (GB) | 0 | 5229 |
| pressure evictions | 188 | 114 |
| evictions that dropped KV | 0 | 0 |
| planned drops | 0 | 0 |
| capacity waits | 1954 | 2107 |
| prefetches skipped | 0 | 8 |
| device busy (s) | 1716.2 | 1710.5 |
| prefill tokens | 11693402 | 11693402 |

Seed sweep for regime D:

| seed | TTFT mean (s) plan vs lru | TTFT p90 (s) plan vs lru | foreground KV wait (s) plan vs lru | pressure evictions plan vs lru |
|---|---|---|---|---|
| 1 | 9.570 vs 9.498  | 25.333 vs 22.044  | 16.1 vs 19.3  | 114 vs 182  |
| 11 | 8.298 vs 8.398  | 22.960 vs 33.224  | 16.5 vs 20.4  | 114 vs 188  |
| 23 | 9.071 vs 9.174  | 40.108 vs 37.589  | 20.8 vs 22.9  | 159 vs 220  |
| 42 | 11.290 vs 11.359  | 29.730 vs 36.044  | 12.8 vs 12.5  | 90 vs 129  |

**Regime A** (FP16 KV at 320 KB per token, 64 sessions, 2 TB host, H200 x8, seed 1):

model llama-3.1-70b (fp8 weights, fp16 KV) on h200-sxm x8 (945 GB KV budget); 64 sessions, 1328 steps

| metric | lru | plan |
|---|---:|---:|
| sessions done / total | 64 / 64 | 64 / 64 |
| steps | 1328 | 1328 |
| makespan (s) | 5355.1 | 5387.2 |
| TTFT mean (s) | 4.829 | 4.849 |
| TTFT p50 (s) | 0.066 | 0.069 |
| TTFT p90 (s) | 8.976 | 0.706 |
| TTFT p99 (s) | 77.736 | 79.545 |
| step latency p50 (s) | 1.98 | 2.04 |
| step latency p90 (s) | 17.70 | 14.23 |
| step latency p99 (s) | 78.58 | 81.17 |
| prefix recomputes | 0 | 0 |
| recomputed tokens | 0 | 0 |
| foreground KV wait (s) | 53.9 | 54.8 |
| fetched in foreground (GB) | 9252 | 11455 |
| prefetched (GB) | 0 | 6157 |
| spilled during waits (GB) | 0 | 11170 |
| spills hidden by the wait (GB) | 0 | 11170 |
| pressure evictions | 235 | 166 |
| evictions that dropped KV | 0 | 0 |
| planned drops | 0 | 0 |
| capacity waits | 2612 | 2884 |
| prefetches skipped | 0 | 38 |
| device busy (s) | 1878.4 | 1855.6 |
| prefill tokens | 9261042 | 9261042 |

**Regime B** (FP8 KV, 80 sessions, host tier cut to 128 GB so long waits must go to NVMe,
H200 x8, seed 11):

model llama-3.1-70b (fp8 weights, fp8 KV) on h200-sxm x8 (945 GB KV budget); 80 sessions, 1719 steps

| metric | lru | plan |
|---|---:|---:|
| sessions done / total | 80 / 80 | 80 / 80 |
| steps | 1719 | 1719 |
| makespan (s) | 5732.2 | 5848.9 |
| TTFT mean (s) | 8.815 | 11.252 |
| TTFT p50 (s) | 0.071 | 0.080 |
| TTFT p90 (s) | 33.224 | 34.688 |
| TTFT p99 (s) | 112.230 | 113.371 |
| step latency p50 (s) | 1.80 | 1.97 |
| step latency p90 (s) | 35.35 | 36.06 |
| step latency p99 (s) | 113.82 | 114.78 |
| prefix recomputes | 0 | 0 |
| recomputed tokens | 0 | 0 |
| foreground KV wait (s) | 1014.1 | 4723.2 |
| fetched in foreground (GB) | 3735 | 5942 |
| prefetched (GB) | 0 | 2993 |
| spilled during waits (GB) | 0 | 5160 |
| spills hidden by the wait (GB) | 0 | 5015 |
| pressure evictions | 188 | 192 |
| evictions that dropped KV | 0 | 0 |
| planned drops | 0 | 0 |
| capacity waits | 1954 | 3016 |
| prefetches skipped | 0 | 26 |
| device busy (s) | 1719.2 | 1743.7 |
| prefill tokens | 11693402 | 11693402 |

**Regime C** (B200 x8, FP8 KV, 80 sessions, 128 GB host; little contention, seed 11):

model llama-3.1-70b (fp8 weights, fp8 KV) on b200-sxm x8 (1225 GB KV budget); 80 sessions, 1719 steps

| metric | lru | plan |
|---|---:|---:|
| sessions done / total | 80 / 80 | 80 / 80 |
| steps | 1719 | 1719 |
| makespan (s) | 5395.1 | 5393.8 |
| TTFT mean (s) | 1.395 | 1.415 |
| TTFT p50 (s) | 0.028 | 0.030 |
| TTFT p90 (s) | 5.672 | 5.672 |
| TTFT p99 (s) | 20.181 | 20.181 |
| step latency p50 (s) | 1.10 | 1.12 |
| step latency p90 (s) | 7.19 | 7.19 |
| step latency p99 (s) | 21.23 | 21.23 |
| prefix recomputes | 0 | 0 |
| recomputed tokens | 0 | 0 |
| foreground KV wait (s) | 5.8 | 9.1 |
| fetched in foreground (GB) | 212 | 412 |
| prefetched (GB) | 0 | 1891 |
| spilled during waits (GB) | 0 | 2303 |
| spills hidden by the wait (GB) | 0 | 2303 |
| pressure evictions | 12 | 0 |
| evictions that dropped KV | 0 | 0 |
| planned drops | 0 | 0 |
| capacity waits | 0 | 0 |
| prefetches skipped | 0 | 0 |
| device busy (s) | 1222.5 | 1211.9 |
| prefill tokens | 11693402 | 11693402 |

Reading of the results:

1. **Where the thesis holds.** With a fast, large host tier (regime D) the plan cuts pressure
   evictions by 30 to 40 percent in every seed, lowers foreground KV waits in three of four
   seeds, and improves p90 time to first token in two of four. The information that does the
   work is which tool the agent called: a policy that only saw the blended wait distribution
   (median 2 s, p90 90 s) spilled across one-second file reads and lost to LRU.
2. **What the thesis does not fix.** Mean TTFT barely moves in any pressured regime because
   the dominant cost is queueing for device memory (thousands of capacity waits); no residency
   policy changes how much KV the active sessions need. Admission control, smaller KV (FP8,
   latent or compressed attention) and placement onto more or bigger devices do.
3. **KV size is the first-order lever.** With FP16 KV at 110k-token contexts (regime A),
   decode traffic alone saturates the device: 24 resident sessions each stream about 36 GB of
   KV per generated token. The plan still halves p90 step latency pressure (TTFT p90 0.7 s
   against 9.0 s) but p99 is unchanged. This is the same conclusion the industry reached in
   2026 with FP8 KV, multi-head latent attention and DeepSeek V4's compressed attention.
4. **Where the plan loses.** When tiers are scarce and slow (regime B) the plan is worse:
   every extra transfer competes for one channel whose NVMe side writes at 24 GB/s, and KV
   prefetched too early is evicted again by new arrivals, doubling the traffic. The planner
   has to treat the transfer channel as a scheduled resource and the prefetch as an
   admission decision, not a timer. Without contention (regime C) the plan only adds a few
   seconds of exposed fetches over 1,700 steps; the 80 percent occupancy threshold keeps that
   small but not zero.

Implications carried into the roadmap: M1 added channel-aware planning, a write-through offload
baseline (the strongest reactive policy in production systems), admission control in the
simulator, and loaders for the public FreeInference and TraceLab traces so the regime mix can be
measured rather than assumed. The KV-size lever moves placement and numerics (M5, M6) up in
priority relative to residency alone.

### M1 simulator validation (2026-11-22)

M1 changed the simulator in five ways and re-ran the four regimes. The complete tables are in
[m1-regimes-2026-11-22.md](m1-regimes-2026-11-22.md), produced by `scripts/m1-regimes.sh`.

1. **Calibrated cost model.** Dense projections and attention are priced as two rooflines that
   run one after the other, tensor-parallel all-reduces (two per layer, ring, over the per-direction
   link bandwidth) are on the critical path, and offline throughput for a request shape is the
   throughput-optimal batch under the KV budget. The efficiency knobs were fitted by grid search
   against 33 published TensorRT-LLM rows (Llama 3.3 70B, which has the Llama 3.1 70B
   architecture: FP8 on H200 TP 2, NVFP4 on B200 TP 1, FP8 on H100 TP 2; eleven input/output
   shapes each; `agentforge cost calibrate`). Result: every row within 12.2 percent (the worst
   is H100 128/4096 at -12.15), geometric mean 5.3 percent. The fit on the 22 H200 and B200 rows
   alone selects the same grid point, so the H100 errors are out of sample; the ignored test
   `h200_b200_fit_reproduces_the_stored_knobs` (`cargo test --release -p agentforge-cost --
   --ignored`) reruns that fit. One mechanism was needed beyond the roofline: block-scaled FP4
   GEMMs reach 0.65 of the share of their paper peak that FP8 GEMMs reach; without it the
   roofline alone fits each device to about 10 and 12 percent but the joint fit stops at 24.5,
   with every B200 row high and every H200 row low. No published B200 FP8 row exists on the same
   page, so this knob cannot be told apart from a Blackwell-wide derate. Two knobs sit on the
   edge of their grid, `bw_util` at the 0.60 floor and the per-layer overhead at the 40 us
   ceiling; probing one step past each edge raises the worst error (0.55 gives 18.5 percent,
   60 us gives 16.4), so the optimum is bracketed, but 0.60 is low for HBM streaming and is
   probably absorbing attention-kernel inefficiency. The M1 target of 15 percent is met for
   throughput only: no published time-to-first-token series for this model exists. The
   long-input rows (2048/128, 5000/500, 20000/2000), where prefill is most of the request time,
   sit within 10 percent, and only 2048/128 is prefill-bound in the model's own labels.
2. **Honest latency.** M0 measured time to first token from the last retry of a step that had
   waited for device memory, so capacity waits never reached the TTFT or step-latency rows. M1
   measures from the moment the step became runnable. Measured at the commit that made this
   change alone (a983b56, regime D seed 11, M0 cost defaults), mean TTFT rose 28 percent for LRU
   and 38 percent for the plan and the plan's p90 lead over LRU fell from 13.9 s to 3.9 s; that
   run is not in the committed tables. The M0 headline that the plan improves p90 TTFT therefore
   rested partly on a measurement gap.
3. **Prefill stalls decode.** The device is one timeline: a prefill of duration P delays every
   decode running when it starts by P (chunked prefill lets decode advance one token per chunk,
   which is the same to first order). This moves time from the prefill queue into step latency.
   The 170 to 260 s TTFT p99 in regimes A, B and D is the cold-start prefill queue: histories of
   roughly 50k to 230k tokens (the 5th to 95th percentile of the synthetic initial prefix)
   arriving every two seconds, not memory. Regime C on B200 sits at 44 to 56 s.
4. **Write-through baseline.** A third policy copies each session's new KV to the host tier during
   every tool wait and keeps the device copy, so a later pressure eviction frees memory without a
   write. It is what a KV connector does in vLLM or LMCache, and it knows nothing about the agent.
5. **Channel-aware planning and admission control.** The plan's write must land inside the median
   wait behind the channel's current backlog or the KV stays; a prefetch that the channel would
   land after the predicted return is refused instead of queued; new sessions can be gated on
   device occupancy (`--admit-fraction`).

Regime D (FP8 KV, 80 sessions, 2 TB host, H200 x8) over seeds 1, 11, 23 and 42, plan against
LRU and write-through:

| metric | lru | write-through | plan |
|---|---|---|---|
| pressure evictions, four seeds | 231, 230, 246, 172 | 240, 232, 216, 176 | 173, 140, 155, 122 |
| evictions with no write | 0 | all | 0 |
| foreground KV wait (s) | 24.0, 23.3, 24.9, 17.5 | 13.9, 14.7, 12.4, 10.5 | 22.3, 20.6, 21.0, 15.3 |
| step latency p90 (s) | 42.8, 37.9, 49.4, 69.7 | 47.2, 41.9, 54.4, 78.9 | 41.2, 34.4, 38.1, 66.1 |
| TTFT p90 (s) | 6.1, 1.2, 5.6, 9.1 | 4.7, 3.1, 5.2, 9.1 | 6.0, 4.0, 3.8, 7.9 |
| TTFT p99 (s) | 246, 223, 179, 260 | 250, 199, 209, 218 | 236, 187, 199, 243 |
| TTFT mean (s) | 10.8, 8.5, 8.7, 12.4 | 10.4, 8.5, 8.9, 12.2 | 10.7, 8.6, 9.3, 12.7 |

Reading of the M1 results:

1. **What survives.** The plan still evicts 25 to 39 percent less under pressure in every seed,
   has the lowest p90 step latency in every seed, and lowers foreground KV waits against LRU in
   every seed. Its p90 and p99 TTFT beat LRU in three of four seeds each.
2. **What does not.** Write-through has the lowest foreground KV wait in every seed of every
   pressured regime (evictions cost it nothing). Mean TTFT is within 0.6 s across the three
   policies in every seed; the thesis still does not move it, because queueing for device memory
   and the prefill queue dominate.
3. **Where the plan loses.** With a scarce, slow host tier (regime B, 128 GB) the plan's
   foreground wait is 6,082 s against 2,180 s for LRU and 1,300 s for write-through; its p90 TTFT
   is 47 s against 29 and 27. Channel-aware rules did not change a single decision in regimes B
   or D: in an instrumented run of regime B (not in the committed tables) the backlog peaked at
   35 s while the shortest long wait has a 45 s median, so the rule never binds. What hurts is that queued spill writes delay other sessions' foreground fetches,
   and no rule prices that. The same holds on B200 with a 128 GB host (regime C, now contended
   under honest TTFT and interference): write-through 929 s, LRU 1,647 s, plan 4,519 s.
4. **The TraceLab-shaped workload.** With the published per-step, per-session and wait
   distributions (`--program tracelab`), 40 sessions on eight H200s with FP8 KV barely contend:
   the policies tie, and TTFT p99 of 140 to 190 s is the first prefill of histories up to the
   1M-token cap (TraceLab's Table 8 puts the Claude Code prefix p99 at 918k tokens). Pressure in
   the published mix comes from session count, not from the step shape.
5. **A simulator bug.** A session whose context outgrew the KV budget made the engine retry
   forever once nothing else could be evicted; it is now the `SessionTooLarge` error, and the
   synthesizer caps context growth at 1M tokens, above TraceLab's p99 prefix. The cap clips the
   sampled distribution rather than modelling compaction.

What the simulator still does not model: scheduler and kernel-launch overhead beyond a per-layer
constant, the chunk size of chunked prefill (a decode with fewer tokens left than the prefill has
chunks is stalled too long), prediction error beyond the log-normal draw, cross-session prefix
sharing, and multi-device placement (one device class per run). The next planning rule to try
is pricing a spill write by the foreground fetches it delays, which is what regime B exposes.

## 6. Risks

NVIDIA's EULA and closed assemblers; Tile IR closed to contributions; divergent vendor numerics;
closed decode tiers; attention targets that move every few months; Luminal and Qualcomm's
Modular as competitors; llm-d and Dynamo absorbing agent-aware policies into the routing layer,
which caps the gain left for compile-time planning unless M1 measures against their policies
rather than LRU; alpha-quality Rust MLIR bindings (avoided in the baseline).

## 7. Decisions

- Rust 2021 edition, toolchain pinned to 1.90.0, crate prefix `agentforge-`.
- No MLIR or LLVM dependency in the baseline; backends will emit text or bytecode formats.
- Machine descriptions are TOML files embedded in `agentforge-machine` and loadable from disk.
- The simulator uses a zero-dependency SplitMix64 generator with Box-Muller sampling so runs are
  reproducible from a seed.
- Simulator-first development: no backend is started before its machine file and cost model
  exist and the simulator can replay a public trace against it.
- Residency tunables set by the M0 simulation: spill above 80 percent device occupancy, keep
  when the median wait is under twice a step or three times a fetch, prefetch only with 10
  percent headroom, prefetch timed to the 10th-percentile return. M1 added: the write must
  land inside the median wait behind the channel backlog, and a prefetch the channel would land
  after the predicted return is refused.
- Cost-model efficiencies are the grid point that minimises the worst error over the bundled
  TensorRT-LLM rows (`agentforge cost calibrate --fit`), and the rows are data under
  `crates/agentforge-cost/calibration/`; a new device or precision gets rows before it gets
  trusted numbers. Link bandwidth in machine files is per direction.
- Research is dated 2026-10-30 and should be refreshed, not redone.
