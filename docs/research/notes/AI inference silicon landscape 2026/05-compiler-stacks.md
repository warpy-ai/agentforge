# Compiler stacks and kernel DSLs for AI accelerators (state as of 2026-10-30)

Method: direct web research by the coordinating session. Facts are dated; vendor claims are
marked. URLs are the sources used.

## 1. The layering that has become standard

Framework (PyTorch, JAX) -> graph IR (StableHLO, TOSA, Torch FX / Inductor IR, ONNX, TVM Relax)
-> tile-level DSL or IR (Triton, CUDA Tile IR, CuTe DSL, TileLang, Helion, Pallas/Mosaic, NKI,
Mojo kernels, CubeCL) -> vendor IR / ISA (PTX -> SASS; LLVM AMDGPU -> CDNA ISA; SPIR-V; Metal
Shading Language -> AIR; TT-Metal C++ -> RISC-V Tensix firmware).

The 2025-2026 shift: the **tile level** is where the industry converged and where vendors now
compete to own the IR. Three independent tile IRs appeared or matured within twelve months
(NVIDIA CUDA Tile IR, Triton's Gluon, PyTorch Helion), and TileLang became the kernel language
used for DeepSeek V4 day-0 kernels and NVIDIA NeMo AutoModel sparse attention.

## 2. Project-by-project status (October 2026)

| Project | Owner / governance | License | Hardware reach | 2026 status | Openness to a third-party compiler |
|---|---|---|---|---|---|
| CUDA Tile IR (`NVIDIA/cuda-tile`) | NVIDIA | Apache-2.0 with LLVM exceptions | NVIDIA only; Blackwell (sm_100) and newer for the Triton backend | Open-sourced 2025-12-04; releases track the CUDA toolkit (latest tag v13.3.4, 2026-08-27). MLIR dialect, bytecode format, Python API, conformance suite. Bytecode is JIT-compiled by the CUDA driver at launch or AOT via `tileiras` to cubin. | Spec and dialect are public, so an external compiler can **emit Tile IR bytecode** as a target; `tileiras` itself is closed (file 01). README: "We are currently not accepting external contributions." https://github.com/NVIDIA/cuda-tile , https://www.phoronix.com/news/NVIDIA-CUDA-Tile-IR-Open-Source , https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/ |
| CUDA Rust: `cutile-rs` (tile model) and `cuda-oxide` (SIMT) | NVIDIA | open source | NVIDIA, compute capability 8.0+ for cutile-rs | Announced 2026-09. cutile-rs on crates.io, builds on stable Rust, used by Hugging Face's Grout Qwen3 engine (batch-1 decode 171 tok/s Qwen3-4B on RTX 5090, 82 tok/s Qwen3-32B on B200, authors' numbers) and mistral.rs; cuda-oxide is early alpha on a pinned nightly. NVIDIA says APIs will change through 2027. | A vendor-backed Rust path from Rust source to tensor cores exists for the first time. https://www.marktechpost.com/2026/09/08/nvidia-announces-cuda-rust-with-cuda-oxide-simt-and-cutile-rs-tile-for-compile-time-safe-gpu-kernels/ , https://github.com/nvlabs/cutile-rs/releases |
| Triton (`triton-lang/triton`) | OpenAI-led, open community | MIT | NVIDIA, AMD; third-party backends: Intel XPU, triton-shared (CPU/MLIR), Triton-Ascend (Huawei), Microsoft Maia | 3.8.0 and 3.9.0 in 2026. Gluon (lower-level, hardware-explicit layer) gained Blackwell Cluster Launch Control and 256-bit global loads; a `gluon.language.nvidia.rubin` module exists. Out-of-tree backends register devices, runtimes and profilers. | Backends are plug-ins but must track Triton's MLIR internals release by release. https://newreleases.io/project/github/triton-lang/triton/release/v3.8.0 , https://newreleases.io/project/pypi/triton/release/3.9.0 |
| Helion | Meta -> Linux Foundation (contributed 2026-03) | BSD-style (PyTorch family) | Lowers to Triton: NVIDIA, AMD MI300X+, CPU via Triton-CPU; planned lowering to Pallas, CuTe DSL, Metal | 1.0 GA announced 2026-04-07 at PyTorch Conference Europe. Autotuner explores PID swizzling, loop order, persistent-kernel strategies. | Python DSL above Triton. https://pypi.org/project/helion , https://pytorch.org/projects/helion/ |
| TileLang (`tile-ai/tilelang`) | Microsoft Research + Peking University, open community | MIT | NVIDIA (V100 to H100 and consumer), AMD MI250/MI300X, Apple Metal (2025-10), CuTe DSL backend (2025-12) | ICLR 2026 paper; v0.1.13 covering 2026-07/08. Used for DeepSeek V4 sparse attention kernels and NVIDIA NeMo AutoModel. Built on TVM infrastructure. | Open and multi-vendor, Python-embedded. https://mlanthology.org/iclr/2026/wang2026iclr-tilelang/ , https://github.com/tile-ai/tilelang/discussions/2848 |
| Mojo + MAX (Modular) | **Qualcomm** (acquisition signed 2026-06-21, closed 2026-07-28/29, all-stock, about 3.1 to 3.9 B USD) | Mojo 1.0 under Apache-2.0 (2026-08), compiler included; MAX source-available | NVIDIA, AMD (MI300X/MI355X), Apple Silicon; Qualcomm targets expected | External contributions to the compiler not yet accepted; target end of 2026. Vendor benchmarks: FLUX.2-dev on MI355X 1.25x torch.compile, about 4% behind B200. | The "hardware-agnostic layer" is now owned by a silicon vendor. https://www.nasdaq.com/press-release/qualcomm-completes-acquisition-modular-2026-07-29 , https://www.sec.gov/Archives/edgar/data/0000804328/000080432826000086/qcom-20260628.htm , https://modular.com/blog/mojo-open-source |
| IREE (`iree-org/iree`) | OpenXLA / Linux Foundation; AMD is the main corporate backer | Apache-2.0 with LLVM exceptions | CPU (LLVM), CUDA, ROCm/HIP, Vulkan/SPIR-V, Metal, WebGPU, AMD AIE NPUs via `nod-ai/iree-amd-aie` | 3.12 (2026-10-13): rewritten Vulkan HAL, experimental AMD HSA HAL, CDNA5 / MI455X and RDNA3 support, rewritten CPU execution path. | Fully open MLIR stack; adding a backend means C++ MLIR work. https://phoronix.com/news/IREE-3.12-Released , https://github.com/nod-ai/iree-amd-aie |
| Apache TVM / Relax / MLC-LLM | Apache Software Foundation | Apache-2.0 | CPU, CUDA, ROCm, Vulkan, Metal, WebGPU, OpenCL | TVM v0.24 line in 2026; MLC-LLM about 23k stars (secondary). TileLang reuses TVM codegen. | Momentum moved to TileLang and MLC rather than TVM core. https://www.everydev.ai/tools/apache-tvm |
| Tenstorrent tt-mlir / tt-forge | Tenstorrent | Apache-2.0 | Wormhole, Blackhole | TTIR (named ops, StableHLO/TOSA-like) -> TTNN dialect (mirrors tt-nn API, explicit deallocation) or D2M generic path -> TTKernel / TTMetal. Front ends: PyTorch, JAX, ONNX via tt-forge. | The only accelerator vendor whose whole compiler stack, kernel SDK and firmware are open. https://docs.tenstorrent.com/tt-mlir/overview.html |
| Mirage Persistent Kernel (MPK) | CMU, Tsinghua, NVIDIA and others | open source (`mirage-project/mirage`) | NVIDIA A100, H100, B200 | arXiv 2512.22219 (2025-12-22). Compiles a model into SM-level task graphs run inside one persistent megakernel; 1.0 to 1.7x over SGLang/vLLM in the paper; README claims 1.2 to 6.7x latency reduction. | Research grade, CUDA only; shows whole-model scheduling inside one kernel is viable. https://arxiv.org/abs/2512.22219 |
| tinygrad | tiny corp | MIT | NVIDIA incl. Blackwell, AMD incl. MI350 with its own userspace driver "AM" and PM4 submission, Mesa NIR/NAK path (0.12), Apple eGPU driver TinyGPU (approved 2026-03-31) | Focus in 2026: large-model training and GPU infrastructure; Exabox pre-orders for 2027; no confirmed custom silicon tape-out. | Precedent for a small team owning the full stack down to the driver. https://www.phoronix.com/news/Tinygrad-0.11-Released , https://www.phoronix.com/news/Tiny-Sovereign-Stack-AMD-Close |

## 3. Search, superoptimization and LLM-generated kernels

- Equality saturation is moving from papers to tooling: egglog used for tensor instruction
  selection on accelerators (arXiv 2512.02371), for verifying sharded computational graphs in a
  production distributed framework (arXiv 2509.10694), and for LLM-guided EqSat strategy
  synthesis with up to 11.89x speed-up over unguided saturation on an XLA testbed (arXiv
  2604.17364, 2026-04). A parallel, customizable EqSat engine (CC 2026) reports beating egglog on
  four workloads. No production tensor compiler ships EqSat as its core optimizer yet.
  https://arxiv.org/pdf/2512.02371 , https://arxiv.org/pdf/2509.10694 , https://arxiv.org/pdf/2604.17364
- LLM-generated kernels do not yet beat compilers under strict evaluation. KernelBench-Verified
  (arXiv 2607.16241, 2026-07, H200): no frontier model's average speed-up over PyTorch TF32
  exceeds 1x at any difficulty level; 28% of the best model's correct kernels raise peak memory.
  KernelBench-X (arXiv 2605.04956): 46.6% of correct kernels are slower than eager;
  cross-hardware speed-up variance up to 21.4x. KernelBench's `allclose` check is passed by
  all-zeros on 4 of 60 level-1 problems (arXiv 2609.21058). Agentic-RL systems self-report much
  higher numbers (unverified). CANN-Bench (arXiv 2607.20518) extends this to Huawei NPUs.
  https://arxiv.org/pdf/2607.16241 , https://arxiv.org/html/2605.04956v1 , https://arxiv.org/abs/2609.21058

## 4. Entry points an independent Rust compiler can use (lowest practical level)

| Target | Entry point | Notes |
|---|---|---|
| NVIDIA | PTX text or NVPTX LLVM IR into `ptxas` or the driver JIT; CUDA Tile IR bytecode (public spec; Blackwell+); Rust kernels via cutile-rs / cuda-oxide | SASS undocumented; EULA bars translation layers (file 01). |
| AMD | LLVM IR for the AMDGPU backend (open ISA PDFs, open LLVM), HIP source via hipcc, or HSA code objects | IREE and tinygrad both do this from outside ROCm's Python stack. |
| Intel GPU / NPU | SPIR-V via Level Zero or Vulkan; OpenVINO IR | SPIR-V emitters exist in Rust (`rspirv`, `naga`). |
| Apple | Metal Shading Language source compiled at runtime; MPSGraph; Core ML packages | ANE closed; GPU reachable. |
| Tenstorrent | tt-metal host API (C++; community `libtt-metal-cxx` cxx bridge) with C++ kernel sources compiled by tt-metal; `luwen` (official Rust low-level device comms); `metalium-rs` (community native-Rust stack for Blackhole); tt-mlir flatbuffers | Fully documented, fully open. |
| Any Vulkan device, browsers | SPIR-V / WGSL via `wgpu`/`naga` | Portability layer; cooperative-matrix extensions narrow the tensor-core gap. |
| CPU | Cranelift (Rust, ISLE instruction selection, regalloc2), LLVM via `inkwell`, or C source | Cranelift is pure Rust and fast to compile. |
| Huawei Ascend, AWS Trainium, Google TPU | Triton-Ascend / AscendC; NKI Python kernels; Pallas or StableHLO via PJRT | No bytecode-level entry for an external compiler; only vendor DSL emission. |

## 5. What practitioners say is missing

1. **No vendor-neutral tile-level IR with a stable binary form.** Tile IR is NVIDIA-only and
   closed to contributions; Triton's internals move every release; TOSA and StableHLO are
   graph-level and leave scheduling to each vendor.
2. **Backend cost is still measured in vendor-team-years.** Tenstorrent needed a dedicated
   multi-dialect MLIR compiler plus a kernel SDK plus front ends; Triton third-party backends
   are maintained by the hardware vendors themselves.
3. **Kernel-level compilation ignores the serving schedule.** Megakernel work shows large gains
   from scheduling the whole forward pass, but it stops at one model iteration and one vendor.
4. **Numerics are second-class.** FP4 variants (NVFP4 vs OCP MXFP4), block scaling and
   accumulation order are handled per kernel, not as IR types with verified semantics.
5. **LLM kernel generation is not yet a compiler replacement**, but it is a usable search
   heuristic when paired with strict verification.
6. **Neutral ownership is scarce.** The two best-funded "write once, run on any accelerator"
   efforts changed hands in 2025-2026: Modular to Qualcomm; Groq's technology licensed to NVIDIA.

## 6. Implications recorded for the AgentForge design

- Emit to public, stable, low-level formats (PTX, Tile IR bytecode, LLVM AMDGPU IR, SPIR-V, MSL,
  tt-metal) rather than to moving Python DSLs; reuse vendor compilers for register allocation and
  tensor-core instruction selection where they are open or specified.
- Own the layer nobody owns: a tile IR with explicit memory-tier placement and an explicit
  machine model, plus the cross-request schedule.
- Use search (e-graphs) plus verification rather than LLM-only kernel generation.
