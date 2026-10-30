# Rust ecosystem for ML inference, GPU programming and compiler construction (as of 2026-10-30)

Method: direct web research by the coordinating session. Activity figures are from third-party
trackers and marked as such; verify on GitHub before relying on them.

## 1. Inference frameworks and compilers written in Rust

| Project | What it is | 2026 status | License | Reusable for AgentForge? |
|---|---|---|---|---|
| Burn + CubeCL (`tracel-ai`) | Deep-learning framework; CubeCL is a Rust-embedded GPU compute language compiling to CUDA, ROCm/HIP, Metal, Vulkan/SPIR-V, WebGPU/WGSL and CPU SIMD; autotuning built in | Burn 0.21.0 (2026-05-07), 0.22 pre-release; CubeCL 0.10.0 (2026-05); Burn 0.20 (2026-01) introduced "CubeK" multi-platform kernels; LibTorch backend deprecated in 0.22 | MIT / Apache-2.0 | Yes: the only mature Rust kernel language with five GPU backends. https://phoronix.com/news/burn-0.20-released , https://docs.rs/burn/=0.22.0/ |
| Candle (Hugging Face) | Minimalist tensor framework with CPU, CUDA, Metal, WASM backends | Actively maintained; no formal 2026 release notes found in search | MIT / Apache-2.0 | Model loaders, safetensors, quantized kernels. |
| Luminal | Rust ML compiler: 15 primitive ops (Log2, Exp2, Sin, Sqrt, Recip, Add, Mul, Mod, LessThan, SumReduce, MaxReduce, Iota, Gather, Scatter, Cast), ahead-of-time compilation, symbolic shapes, egglog-based search ("the best heuristic is no heuristic"), CUDA and Metal backends, ROCm and "more inference accelerator backends" on the roadmap, PyTorch backend integration | Seed 5.3 M USD led by Felicis (2025-11); SEC filing to raise up to 29.2 M USD (2026-10). Vendor claims: 36k tok/s GPT-OSS-120B on 8xH100 vs 28k TensorRT-LLM and 26k vLLM; Q8 Llama 3 8B at about 80% of H100 peak | MIT or Apache-2.0 | Closest existing project in spirit; also a competitor. https://github.com/luminal-ai/luminal , https://dealroom.co/news/165150-luminal-raises-29-2m-to-speed-up-ai-inference-with-a-compiler-first-appr/ |
| mistral.rs; Hugging Face Grout | Rust LLM inference engines | Both reported on NVIDIA's cutile-rs (2026-09); Grout is a Qwen3 engine built with NVIDIA (batch-1 decode 171 tok/s Qwen3-4B on RTX 5090, 82 tok/s Qwen3-32B on B200; authors' numbers; arXiv 2606.15991) | MIT / Apache-2.0 | Reference for Rust serving ergonomics and kernel call sites. https://github.com/nvlabs/cutile-rs/releases |
| tract (Sonos), ort (ONNX Runtime bindings), Ratchet, wonnx | CPU/ONNX inference, WebGPU inference | Mature niche tools | Apache-2.0 / MIT | ONNX ingestion patterns. |

## 2. GPU and accelerator access from Rust

| Crate / project | Capability | Status |
|---|---|---|
| `cutile-rs`, `cuda-oxide` (NVIDIA, 2026-09) | Write NVIDIA kernels in Rust; tile model on stable Rust (CC 8.0+), SIMT model on nightly | Alpha; vendor-backed; APIs will change through 2027. |
| Rust-CUDA (`rust-gpu/rust-cuda`, `rustc_codegen_nvvm`, `cust`) | Rust -> PTX via NVVM; host API | Rebooted 2025; goals: C++ interop, PTX back-end collaboration. https://www.phoronix.com/news/Rust-CUDA-Project-Reboot |
| rustc `nvptx64-nvidia-cuda` target | Rust -> PTX through LLVM NVPTX | Tier 2, nightly, `no_std`; minimum SM 7.0 / PTX ISA 7.0 reported from Rust 1.97 (secondary source). |
| `cudarc` | Safe CUDA driver/runtime bindings, PTX/cubin module loading, cuBLAS/cuDNN FFI | Widely used (Candle, Burn). |
| `wgpu` + `naga` | WebGPU API; WGSL/SPIR-V/MSL/HLSL/GLSL code generation and validation | Mature; the most portable path. |
| `rspirv`, `rust-gpu` | SPIR-V builder; Rust -> SPIR-V compiler | rust-gpu community-maintained. |
| `ash`, `metal-rs` / `objc2-metal`, Level Zero bindings | Vulkan, Metal, oneAPI host APIs | Mature. |
| Tenstorrent: `luwen` (official), `libtt-metal-cxx` (community, cxx.rs bridge to the tt-metal host API: devices, kernels, circular buffers, MeshDevice), `metalium-rs` (community native-Rust stack for Blackhole avoiding C++ entirely), `infinitum` issue #23 (planned tt-metal crate) | Device communication and host-side programming of Tenstorrent parts | No official Rust binding for tt-metal; tt-metal itself at v0.80.0-dev (2026-09-22). https://docs.tenstorrent.com/tt-awesome/entry/libtt-metal-cxx/ , https://docs.tenstorrent.com/tt-awesome/entry/luwen/ , https://github.com/silvanshade-org/infinitum/issues/23 |

## 3. Compiler infrastructure in Rust

| Library | Role | Status |
|---|---|---|
| Cranelift (`cranelift-codegen`, ISLE, `regalloc2`) | CPU code generator used by Wasmtime; ISLE declarative instruction-selection DSL for all backends since 2022; regalloc2 0.15.1 (2026-04) widened VReg limits for very large functions | Production quality for x86-64, aarch64, riscv64, s390x. https://cranelift.dev/ , https://cfallin.org/blog/2022/06/09/cranelift-regalloc2/ |
| `egg`, `egglog` | Equality saturation; egglog unifies Datalog and EqSat, Rust implementation | Active research-grade tools used by several 2025-2026 tensor-compiler papers. |
| `melior` (`mlir-rs/melior`) | Safe wrapper over the MLIR C API; `melior-macro` generates dialect bindings from TableGen | Alpha, unstable API, tracks LLVM/MLIR 22; about 508 stars, last push 2026-04 (tracker). https://gittrend.io/repo/mlir-rs/melior |
| `inkwell`, `llvm-sys` | LLVM bindings | Stable but pinned to specific LLVM majors; build friction. |
| `salsa`, `chumsky`, `logos`, `indexmap`, `petgraph` | Incremental computation, parsing, graph utilities | Mature. |
| `half`, FP8 crates, `std::simd` (nightly) | f16/bf16, FP8 emulation, portable SIMD | FP4 / MX block formats have no standard crate; must be defined in-project. |

## 4. Observations

- Nobody has shipped a Rust compiler that targets a non-GPU AI accelerator in production. The
  closest artifacts are Tenstorrent community crates and Luminal's GPU work.
- NVIDIA's 2026 move to first-party Rust (cutile-rs) removes the "no sanctioned Rust path to
  tensor cores" objection on NVIDIA hardware.
- The pieces to reuse: CubeCL (kernel language and backends), cudarc / wgpu / ash (host APIs),
  Cranelift ISLE and regalloc2 (declarative lowering and allocation, CPU backend), egglog
  (search), naga/rspirv (SPIR-V), safetensors/GGUF loaders from Candle.
- The pieces to build: the IR and machine model, FP4/MX numerics types, backends that emit PTX,
  Tile IR bytecode, LLVM AMDGPU IR, MSL and tt-metal programs, and the runtime.
