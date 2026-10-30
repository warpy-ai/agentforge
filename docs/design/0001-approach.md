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
   and latency; links. Bundled: `h200`, `b200`, `mi355x`, `blackhole-p150`, `lpx-tier`.
5. **Cost model and simulator (`agentforge-cost`, `agentforge-sim`).** Roofline-style phase
   costs for prefill, decode and transfers, and a discrete-event replay of agent sessions over the
   machine description. The same files drive the compiler, so every policy is testable before
   silicon.
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

Implications carried into the roadmap: M1 adds channel-aware planning, a write-through offload
baseline (the strongest reactive policy in production systems), admission control in the
simulator, and replay of the public FreeInference and TraceLab traces so the regime mix is
measured rather than assumed. The KV-size lever moves placement and numerics (M5, M6) up in
priority relative to residency alone.

What the simulator does not model yet: tensor-parallel communication (device resources scale
linearly with `tp`), scheduler and kernel-launch overhead beyond a per-layer constant, batching
interference between prefill and decode, prediction error beyond the log-normal draw, and
multi-device placement (one device class per run).

## 6. Risks

NVIDIA's EULA and closed assemblers; Tile IR closed to contributions; divergent vendor numerics;
closed decode tiers; attention targets that move every few months; Luminal and Qualcomm's
Modular as competitors; alpha-quality Rust MLIR bindings (avoided in the baseline).

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
  percent headroom, prefetch timed to the 10th-percentile return.
- Research is dated 2026-10-30 and should be refreshed, not redone.
