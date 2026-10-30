# AI inference silicon landscape 2026

Research report for the AgentForge project. Date: 2026-10-30. Sources: the eight notes files in
`../notes/AI inference silicon landscape 2026/` (cited there with URLs); this report cites the
notes by number.

## Executive summary

- **Inference silicon split along the prefill/decode axis in 2026, and the split is shipping.**
  NVIDIA dropped the Rubin CPX prefill GPU from its roadmap and pairs Vera Rubin GPUs with Groq 3
  LPX racks of SRAM-only LPUs for decode; AWS pairs Trainium3 with Cerebras CS-3 on Bedrock;
  d-Matrix's SRAM-centric Corsair reached volume shipment and its successor plugs into NVLink
  Fusion (notes 01, 02, 03).
- **Custom ASICs are the fastest-growing inference hardware, and all of them are closed.**
  TPU Ironwood, Trainium3, Maia 200, MTIA 300 and Huawei Ascend run in production; projected
  2026 shipment growth is about 45 percent for ASICs against 16 percent for GPUs; none exposes
  an ISA, only vendor DSLs or graph formats (note 02).
- **Software converged on tile-level IRs, and the neutral layers changed owners.** NVIDIA
  open-sourced CUDA Tile IR but accepts no contributions and ships Rust kernel bindings; Triton
  added Gluon; Helion 1.0 reached GA under the Linux Foundation; Qualcomm bought Modular (Mojo
  and MAX) for about 3.1 to 3.9 billion dollars; Groq's technology went to NVIDIA (notes 03, 05).
- **The serving layer, not the compiler, owns the dominant costs.** KV cache placement,
  batching, disaggregation and prefix reuse live in Python schedulers (vLLM, SGLang, Dynamo) over
  hand-written per-vendor kernels; every 2025 to 2026 agentic-serving system is a runtime policy
  around a black-box model (note 06).
- **Workloads are agentic and prefix-heavy.** A median coding-agent step carries about 119k
  prefix tokens, appends 875 and generates 214, with 95 to 96 percent prefix-cache hit rates and
  misses caused by long idle gaps; decode runs at 1 to 2 FLOP per byte on hardware whose ridge
  point is near 590 (notes 06, 08).
- **What is missing** is a vendor-neutral tile IR with a stable binary form, a compiler that
  sees the agent program, planned placement across heterogeneous silicon, numerics as verified
  types, and a simulator-first development path. Rust has the parts (CubeCL, cutile-rs, cudarc,
  wgpu, Cranelift, egglog) but no such compiler (notes 05, 07, 08).
- **Recommendation:** AgentForge should compile the agent program and the model together
  against an explicit, data-described machine model, with a tile-level kernel layer that emits
  public low-level formats. Feasibility is supported by reachable targets (NVIDIA, AMD, Intel,
  Apple GPU, Tenstorrent, open NPUs, CPU) and by first-order arithmetic on the public traces.

## 1. Hardware landscape

### 1.1 Datacenter GPUs (note 01)

Shipping inference GPUs in October 2026 are NVIDIA B200/GB200 and B300/GB300 (180 to 288 GB
HBM3e, 8 TB/s, 9 to 15 PF dense FP4) and AMD MI355X (288 GB, 8 TB/s, 10 PF FP4, 1,400 W). Vera
Rubin NVL72 entered production shipments in August to October 2026 with specs cut on
2026-10-29 (288 GB HBM4 at 19.2 TB/s, NVLink 6 at 3 TB/s per GPU). AMD's MI455X/Helios (432 GB
HBM4, 23.3 TB/s, 40 PF MXFP4) is in a slow ramp with large commitments from OpenAI, Oracle, Meta
and Anthropic. Intel's Gaudi 3 is end of line; the Intel GPU D270 (LPDDR5X, 1.3 PF FP8) is
sampling.

Compiler-relevant facts: Blackwell adds 256 KB of Tensor Memory per SM, 2-SM MMA shapes, block
scaling with E4M3 scales per 16 elements (NVFP4) or E8M0 per 32 (MX), and a datapath asymmetry
of roughly 8,192 MMA operations to 16 transcendental operations to 128 bytes of shared-memory
bandwidth per SM cycle. CDNA4 raised LDS to 160 KB per CU and supports MX scaling only.
Practitioners report a recurring class of TMEM and mbarrier protocol bugs across CUTLASS,
DeepGEMM and TileLang and call for formal semantics of the async protocols.

Openness: PTX is documented and SASS is not; `ptxas` and `tileiras` are closed; NVIDIA's EULA
bars translation layers, so the lowest legal entry is PTX text or NVPTX IR into `ptxas` or the
driver JIT, with CUDA Tile IR bytecode a Blackwell-only second path. AMD publishes ISA documents
and an upstream LLVM backend, the most open of the three. Intel offers SPIR-V into its compiler
with no Xe3 ISA documentation.

### 1.2 SRAM decode tiers (notes 01, 03)

NVIDIA's Groq 3 LPX rack holds 256 LPUs with 500 MB of SRAM each (150 TB/s per LPU, about 315
PF FP8 per rack) and is described as in full production; Dynamo routes work between Rubin GPUs
and LPUs. Cerebras (44 GB on-wafer SRAM; IPO May 2026, 510 million dollars of 2025 revenue
concentrated in two customers) is the decode tier of the AWS service announced in March 2026.
d-Matrix Corsair (digital in-memory compute) ships in volume since June 2026; Raptor adds 3D
stacked DRAM and NVLink Fusion. These products settle a question the research community debated
for years: SRAM-centric decode is now a tier next to GPUs rather than a replacement for them.
All of them are closed to outside compilers.

### 1.3 Hyperscaler ASICs (note 02)

Google rents and sells Ironwood TPUs to Anthropic (up to one million chips) and Meta; AWS made
Trn3 UltraServers generally available in December 2025 (144 chips, 20.7 TB HBM3e, 362 MXFP8
PFLOPS) and promoted the NKI kernel interface to stable; Microsoft runs Maia 200 in production
and announced Maia 300; Meta published a four-generation MTIA roadmap with the 450 and 500 aimed
at generative-AI inference; Huawei ships Atlas 950 and 960 SuperPoDs of thousands of Ascend NPUs
with an open-sourced CANN stack and Triton and vLLM support. OpenAI's Broadcom accelerators are
due from the second half of 2026. Entry points for outsiders are StableHLO through PJRT, NKI
kernels, Triton for Maia and AscendC or Triton-Ascend.

### 1.4 Startups (note 03)

Capital concentrated on 2027 delivery: Positron (875 million dollars at a 5 billion valuation),
MatX (500 million), Etched (500 million, first racks 2026), Tenstorrent (1.5 billion at 6.2
billion, with Galaxy Blackhole shipping in volume), Furiosa, Rebellions, Extropic (Z1 early
access 2027). Model-specific silicon was absorbed: AMD agreed to acquire Taalas in August 2026.
Tenstorrent is the only vendor whose kernel SDK, compiler, firmware and device library are open.

### 1.5 Edge (note 04)

AMD XDNA 2 (Peano, MLIR-AIE), Intel NPU (open driver and compiler plugin) and Arm Ethos-U (Vela)
have open compilers; Apple's Neural Engine and Qualcomm's HMX matrix unit are closed. Apple's M5
adds matrix units inside GPU cores reachable through Metal 4. On-device decode of 8B models runs
at single-digit to low-tens tokens per second on NPUs because it is bandwidth-bound.

### 1.6 Openness summary

| Target | Lowest entry for an external open-source compiler |
|---|---|
| NVIDIA GPUs | PTX text or NVPTX IR; CUDA Tile IR bytecode (Blackwell+); cutile-rs for Rust kernels |
| AMD Instinct | LLVM AMDGPU IR, HIP source, HSA code objects; public ISA documents |
| Intel GPU and NPU | SPIR-V via Level Zero or Vulkan; open NPU compiler plugin |
| Apple | Metal Shading Language, MPSGraph, Core ML; ANE closed |
| Tenstorrent | tt-metal host API and kernels; tt-mlir; Rust `luwen`, community crates |
| AMD XDNA 2, Intel NPU, Arm Ethos-U | Peano and MLIR-AIE; Level Zero plugin; Vela |
| TPU, Trainium, Maia, Ascend | Vendor DSLs only (Pallas, NKI, Triton, AscendC) |
| Groq/LPX, Cerebras, SambaNova, Etched, d-Matrix, Apple ANE, Qualcomm HMX | Closed |

## 2. Software stack

### 2.1 Tile-level convergence (note 05)

The stack is now framework, graph IR, tile-level DSL or IR, vendor IR or ISA, and the tile level
is where vendors compete. CUDA Tile IR was open-sourced in December 2025 with a public spec,
bytecode and Python API, but contributions are not accepted and only NVIDIA GPUs are targets;
NVIDIA's cutile-rs brings the tile model to stable Rust. Triton 3.9 added Gluon for Blackwell and
Rubin. Helion 1.0 compiles a Python DSL to Triton with planned Pallas, CuTe DSL and Metal
lowerings. TileLang became the language of DeepSeek V4's day-0 kernels. IREE 3.12 and TVM remain
the open graph-to-kernel compilers; adding a backend to either is still a C++ MLIR project
measured in vendor-team-years, as Tenstorrent's tt-mlir shows.

### 2.2 Ownership (notes 03, 05)

Two best-funded "write once, run on any accelerator" efforts changed hands within a year:
Modular to Qualcomm (closed July 2026; Mojo 1.0 is Apache-2.0 but external compiler
contributions are not yet accepted) and Groq to NVIDIA (December 2025). Neutral ownership of the
portability layer is now scarce.

### 2.3 The serving layer (note 06)

vLLM, SGLang and Dynamo own scheduling, paged KV, prefix caching, prefill/decode disaggregation
and KV tiering through LMCache, Mooncake, FlexKV and NIXL; TensorRT-LLM dropped its own engine
for PyTorch. The compiler-to-server surface is the attention kernel interface over paged KV plus
KV connector APIs. The model's kernel target moves every few months: DeepSeek V4 pairs
sliding-window and compressed attention in every layer with native FP4 MoE; Qwen3-Next and Kimi
Linear mix recurrent layers with attention; speculative decoding accepts 60 to 80 percent of
draft tokens.

### 2.4 Search and LLM-generated kernels (note 05)

Equality saturation with egglog is used in 2025 to 2026 papers for instruction selection,
verification and LLM-guided strategy synthesis, but no production tensor compiler runs it as the
core optimizer. Under strict evaluation (KernelBench-Verified, July 2026) no frontier model
averages above 1x over PyTorch TF32, and benchmark correctness checks were found to be weak.

### 2.5 Rust (note 07)

Burn and CubeCL provide a Rust kernel language with CUDA, ROCm, Metal, Vulkan, WebGPU and CPU
backends; NVIDIA's cutile-rs is in production use by Hugging Face's Grout and mistral.rs;
Cranelift offers declarative instruction selection and a production register allocator; egglog
offers equality saturation; Tenstorrent has official and community Rust device crates. Luminal
is a funded Rust inference compiler with search-based code generation on CUDA and Metal. No Rust
compiler targets a non-GPU accelerator in production.

## 3. Workloads and economics (notes 06, 08)

| Metric | Value |
|---|---|
| Median coding-agent step (TraceLab) | 119k prefix tokens, 875 appended, 214 generated; 8.9 tool calls per request |
| Prefix-cache hit rate (TraceLab) | 95.2 to 96.2 percent; misses after long human idle time |
| Decode arithmetic intensity | 1 to 2 FLOP per byte at batch 1, ridge near 590 on H100 |
| Compute versus DRAM bandwidth growth | 3.0x versus 1.6x per two years |
| Google token volume | 3.2 quadrillion per month at I/O 2026, about 7x year over year |
| Continuum (ICLR 2026) | 1.12 to 3.66x lower delay from KV time-to-live after tool calls |
| Throughput accounting | reused-prefix tokens inflate per-GPU throughput about 5x |

Open traces with block-level prefixes exist (FreeInference, TraceLab, LMCache), and analytical
cost models are accurate to single-digit percent on GPUs (GenZ 1.9 to 5.8, LLMCompass 4.1, Vidur
under 9).

## 4. What is established

1. Decode is bandwidth-bound and prefill compute-bound; hardware now specializes accordingly.
2. Tile-level IRs with explicit memory spaces and async copies are the right abstraction for
   tensor cores; every vendor has one.
3. Paged KV, prefix caching, continuous batching and prefill/decode disaggregation are standard
   serving techniques; KV tiering to host DRAM and NVMe is a product category.
4. Block-scaled sub-byte formats are standard, but NVIDIA and AMD chose different block sizes
   and scale encodings.
5. Analytical cost models predict GPU inference time well enough to rank design choices.

## 5. What is missing

1. A vendor-neutral tile IR with a stable binary form that reaches non-NVIDIA silicon.
2. A compiler that sees the agent program. Sessions, tool-wait gaps and branch structure that
   the agent knows are re-inferred by servers from independent HTTP calls.
3. Planned placement and KV movement across heterogeneous prefill and decode silicon; today it
   is configured per deployment by hand.
4. Numerics as first-class, verified types (block formats, scale layouts, accumulation order);
   verified rewrite systems exist but are unused by shipping compilers.
5. A simulator-first development path: accurate open cost models exist, but no compiler is
   developed against a virtual target and validated on silicon later.
6. Formal treatment of async protocols (TMEM, mbarrier) whose misuse is a recurring bug class.
7. A Rust compiler for inference silicon; the ecosystem has the parts, not the whole.

## 6. Candidate approaches and recommendation

| Candidate | Summary | Assessment |
|---|---|---|
| A. Agent-program plus model co-compilation against a machine model | Treat the agent step program as a compiled artifact; derive KV liveness, residency, prefetch and placement from it; compile the model's kernels for the same machine description | Addresses gaps 2, 3, 5; novel as a combination; measurable on public traces before any backend exists |
| B. Vendor-neutral tile IR with declarative machine descriptions and e-graph search | The kernel layer: tile IR with explicit memory tiers, numerics types, egglog rewrites, backends emitting PTX, Tile IR, LLVM AMDGPU, SPIR-V, MSL, tt-metal | Addresses gaps 1, 4, 6; overlaps with Luminal, TileLang and CubeCL; needed by A for the single-device cost model and codegen |
| C. Open compiler for SRAM dataflow silicon | Static whole-program scheduling for LPU-class machines | Unreachable: every such vendor is closed; only expressible through the machine model and simulator |
| D. LLM-agent-driven autotuning | Agents generate and optimize kernels | Not a compiler replacement under verified evaluation; useful only as a search heuristic behind strict oracles |

Recommendation: A is the thesis and B is its kernel layer; C stays a modelling target; D is a
future search heuristic. The project's name fits: it forges the agent program and the model into
one schedule for the silicon.

## 7. Feasibility

Targets: an open compiler can reach NVIDIA, AMD, Intel, Apple GPUs, Tenstorrent, AMD and Intel
NPUs, Arm Ethos-U and CPUs today (section 1.6); SRAM decode tiers, TPU, Trainium beyond NKI,
Apple ANE and Qualcomm HMX are modelling targets only.

First-order arithmetic on TraceLab's median step with a 70B grouped-query model (80 layers, 8 KV
heads, head dimension 128): KV is 320 KB per token in FP16, so about 38 GB for the 119k prefix
(19 GB in FP8, far less with latent or compressed attention). Recomputing it after a miss costs
about 1.7e16 FLOP, roughly 6 to 30 s depending on GPU and precision. Moving it instead takes
about 0.1 s over NVLink at 400 GB/s, 0.8 s from host DRAM over one PCIe 5 link, 5 s from one
NVMe drive. At a 4 to 5 percent miss rate, misses dominate tail latency and consume a large share
of compute; a plan that spills to the right tier before a predictable tool wait and prefetches
before the next step turns seconds of recompute into sub-second transfers. Continuum's measured
1.1 to 3.7x from a runtime heuristic is consistent with this estimate, and compile-time knowledge
extends it to placement across device classes.

The baseline workspace in this repository includes a discrete-event trace-replay simulator driven
by the same machine descriptions the compiler will use, so the claim is measured, not asserted.
The first runs (`docs/design/0001-approach.md`, "First simulation results") show 30 to 40 percent
fewer pressure evictions and better p90 latency in most seeds when the host tier is fast and
large, no gain without contention, and losses when tiers are scarce and slow; mean latency under
pressure is set by queueing for device memory, so KV size, placement and admission control rank
above residency alone in the roadmap.

## 8. Risks

NVIDIA's EULA and closed assemblers bound what a third-party backend may do; Tile IR is closed to
contributions; NVIDIA and AMD numerics diverge; the fastest-growing decode tiers are closed; the
attention kernel target changes every few months; Luminal and Qualcomm's Modular are funded
competitors in the same space; Rust's MLIR bindings are alpha, so the baseline avoids MLIR.

## Sources

The notes in `../notes/AI inference silicon landscape 2026/`: `01-datacenter-gpus.md` (143 URLs,
written by a research subagent), `02-hyperscaler-asics.md`, `03-startup-silicon.md`,
`04-edge-npus.md`, `05-compiler-stacks.md`, `06-serving-runtimes-and-workloads.md`,
`07-rust-ecosystem.md`, `08-workloads-costmodels-correctness.md`.
