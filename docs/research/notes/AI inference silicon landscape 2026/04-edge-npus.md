# Edge and on-device inference silicon and software (state as of 2026-10-30)

Method: direct web research by the coordinating session. "vendor" marks vendor claims; "(gap)"
marks items not verified against a primary source.

## 1. Takeaway

Edge openness splits cleanly by vendor: AMD (XDNA 2 via Peano and MLIR-AIE), Intel (NPU driver
and compiler plugin) and Arm (Ethos-U with Vela) publish open compilers an outside project can
build on; Apple's Neural Engine and Qualcomm's HMX matrix unit remain closed, reachable only
through Core ML, Metal, or the QNN SDK. On-device LLM decode is bandwidth-bound and slow (single
digits to low tens of tokens per second for 8B models); prefill is where NPUs shine.

## 2. Per-vendor status

| Vendor / NPU | 2026 status | Open entry for an outside compiler | Sources |
|---|---|---|---|
| Apple M5 (Neural Accelerators in each GPU core; 16-core Neural Engine) | MLX reaches the GPU Neural Accelerators through Metal 4 TensorOps and Metal Performance Primitives (macOS 26.2+); Apple reports up to 4x faster time-to-first-token vs M4 (vendor; third-party 3.3x to 4.06x across Qwen 1.7B to 14B and GPT-OSS 20B) | Metal Shading Language and Metal tensors (WWDC26 session 330), MPSGraph, Core ML; ANE closed | https://machinelearning.apple.com/research/exploring-llms-mlx-m5 , https://blakecrosley.com/blog/metal-machine-learning-2026 |
| Qualcomm Hexagon (Snapdragon 8 Elite Gen 5; X2 Elite) | NPU 37% faster generation over generation, INT2 support; shipments ramp Q1 2026; Gen 6 teased with an "Element Accelerator" and larger shared memory | HVX documented and upstream in LLVM; HMX matrix unit has no public ISA or API (Qualcomm engineer on the TVM forum: no support for non-public APIs); QNN / AI Engine Direct is the closed path; an MLIR-based Hexagon NPU compiler was presented at the 2025 LLVM Developers' Meeting; community `ggml-hexagon` work exists | https://discuss.tvm.apache.org/t/hmx-support-on-htp-using-tvm-runtime/17457 , https://llvm.org/devmtg/2025-10/slides/quick_talks/baskaran_slama.pdf , https://counterpointresearch.com/de/insights/qualcomm-snapdragon-8-elite-gen-5-redefines-mobile-performance-and-ai-efficiency |
| AMD XDNA 2 (Ryzen AI) | MLIR-AIE v1.3 (2026-03-17) added a C++ `aiecc`; IRON Python API; Peano open-source single-core LLVM compiler (no Vitis needed for XDNA/XDNA 2); MLIR-AIR above; ASPLOS 2026 tutorial (2026-03-23); `amdxdna` Linux driver upstream | Open: Peano LLVM backend, MLIR-AIE, IRON | https://phoronix.com/news/MLIR-AIE-1.3 , https://www.amd.com/content/dam/amd/en/documents/solutions/ai/iron-aie-api-for-ryzen-ai-npu-tutorial-asplos-2026.pdf |
| Intel NPU 5 (Panther Lake) | `ivpu` kernel driver since Linux 6.13; firmware in linux-firmware Dec 2025; MIT-licensed user-space driver 1.38 (Sept 2026, Ubuntu 26.04, Linux 7.0); compiler plugin pinned by tag; LiteRT names it NPU5010 with JIT or AOT compilation | Open stack from kernel driver through Level Zero to OpenVINO | https://www.phoronix.com/news/Intel-Linux-NPU-Driver-1.38 , https://github.com/intel/linux-npu-driver/releases , https://developers.google.cn/edge/litert/next/intel |
| Arm Ethos-U85 | 128 to 2,048 MACs, transformer support; Arm says 20% more energy efficient than U55/U65 (vendor); ExecuTorch 1.0 made the Ethos-U backend production quality | Open: Vela ahead-of-time compiler (TFLite or TOSA in, command streams out; PyPI `ethos-u-vela` 4.x), open drivers, RTL available under Arm's program; KleidiAI is a separate CPU micro-kernel library | https://pypi.org/project/ethos-u-vela/ , https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85 |
| Google Tensor G5 / Edge TPU, MediaTek, Samsung Exynos, Hailo, Axelera, NXP, Rockchip, Raspberry Pi AI HAT, NVIDIA Jetson Thor | Not verified in this pass (gap) | | |

## 3. Portable layers and standards

| Layer | 2026 status | Source |
|---|---|---|
| ExecuTorch 1.0 | GA 2025-10-22; backends: Core ML, MPS, XNNPACK, Vulkan, Qualcomm (92 of 116 HTP ops), MediaTek, Samsung Exynos, NXP Neutron (i.MX RT700 only), OpenVINO, Cadence, Arm Ethos-U (production) and VGF | https://newreleases.io/project/pypi/executorch/release/1.0.0 , https://docs.pytorch.org/executorch/1.0/backends-overview.html |
| WebNN | Origin trial in Chrome 146 (Feb 2026) through Chrome 149; not production-ready; WebGPU plus ONNX Runtime Web or transformers.js is the production path | https://phoronix.com/news/chrome-146-beta , https://www.utsubo.com/blog/frontier-web-apis-2026-production-ready |
| Vulkan ML | Cooperative-matrix extensions in 2026: `VK_QCOM_cooperative_matrix_conversion` (Jan), `VK_NV_cooperative_matrix_decode_vector` (May), `VK_EXT_cooperative_matrix_maintenance1` (Aug, NVIDIA, Qualcomm, Intel, Arm); Khronos ML working group: data-graph extension exposing the TOSA operator set and tensor resources | https://www.phoronix.com/news/Vulkan-1.4.359 , https://www.khronos.org/developers/linkto/machine-learning-neural-rendering-and-the-future-of-open-graphics-slides |
| llama.cpp / ggml | GPU backends Vulkan, CUDA/ROCm/MUSA, Metal, SYCL, OpenCL, WebGPU; NPU backends CANN (Huawei), zDNN (IBM), Hexagon (Qualcomm); per-op capability reporting with CPU fallback; Kompute removed (Vulkanised 2026 talk by Jeff Bolz, NVIDIA) | https://vulkan.org/user/pages/09.events/vulkanised-2026/0930-Jeff-Bolz-NVIDIA.pptx.pdf |
| TOSA, StableHLO, ONNX | TOSA is the input of Vela and the Vulkan data-graph work; StableHLO is the PJRT input; ONNX remains the interchange format for ONNX Runtime execution providers | see above |
| Android NNAPI, DirectML, Windows ML | Not verified in this pass (gap) | |

## 4. On-device LLM performance (directional, not a controlled benchmark)

| Setting | Figure | Source |
|---|---|---|
| 8B model decode on Snapdragon X Elite NPU (QNN/Genie) | about 5 to 10 tokens/s | https://runaihome.com/blog/npu-vs-gpu-local-llm-2026/ |
| 8B Llama 3.1 Q4 on RTX 4060 (desktop reference) | about 52 tokens/s | https://www.techplained.com/best-gpu-for-llms |
| Gemma 3n E2B on Dimensity 9500 NPU | about 1,600 tokens/s prefill vs 28 tokens/s decode | same as first row |
| Research prefill kernels on NPUs (mllm-NPU, 2024) | above 1,000 tokens/s prefill for about 2B models | https://arxiv.org/abs/2407.05858v1 |
| Snapdragon 8 Elite Gen 5 | vendor claim up to 220 tokens/s (model unspecified) | vendor |

Decode is bounded by memory bandwidth divided by model bytes, so TOPS ratings predict little.

## 5. Practitioner pain points

Fragmentation per vendor delegate; quantization constraints per NPU; poor transformer coverage
forcing GPU or CPU fallback; undocumented matrix units (HMX) that only the vendor SDK can use;
duplicated kernels per backend in llama.cpp and ExecuTorch.
