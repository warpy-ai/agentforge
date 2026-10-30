# LLM serving runtimes, inference techniques and agentic workloads (state as of 2026-10-30)

Method: direct web research by the coordinating session. Vendor claims marked.

## 1. Engines and what they own

| Engine | Ownership of the stack | 2026 notes | Hardware portability mechanism |
|---|---|---|---|
| vLLM | Python scheduler, paged KV, continuous batching, hand-written / FlashInfer / Triton kernels | V1 engine; hardware plugin system (RFC originated with the Huawei Ascend team) registers platforms via Python entry points; Intel Gaudi plugin v1.22.2 is V1-based and replaces the old fork; AMD ZenDNN "zentorch" CPU plugin (2026-04) | `vllm.platform_plugins`: Executor / Worker / Runner / Attention backends accept plugins. https://vllm.ai/blog/2025-05-12-hardware-plugin , https://docs.vllm.ai/projects/gaudi/en/0.13.0/dev_guide/plugin_system.html |
| SGLang | RadixAttention prefix cache, LM-program frontend, router, PD disaggregation with Mooncake and NIXL transfer backends | 2026 Q2 roadmap (issue #21703): PD disaggregation "widely used in production"; items include heterogeneous instances per pool, long-context vs short-context prefill roles, and a **session-based, load-aware routing strategy for agentic serving** | Hand-written kernels per vendor; AMD ships ATOM for MI355X. https://github.com/sgl-project/sglang/issues/21703 |
| NVIDIA Dynamo | Disaggregated serving framework (Rust and Python, Apache-2.0); KV Block Manager (KVBM): allocate -> register -> match lifecycle; NIXL transfer library for GPU P2P, RDMA, NVLink, SSD; KV-aware routing | v1.0 to v1.2 docs in 2026; KVBM integrates LMCache, FlexKV, SGLang HiCache; ICMSP (CES 2026) extends KV into NVMe with a 4-tier hierarchy | NVIDIA-centric; backends vLLM, SGLang, TensorRT-LLM. https://docs.nvidia.com/dynamo/v1.2.0/components/kvbm.md , https://www.blocksandfiles.com/ai-ml/2026/03/30/nvidia-and-its-partners-kv-cache-extenders/5209284 |
| TensorRT-LLM | NVIDIA kernels and PyTorch runtime | 1.2 removed the TensorRT engine backend; PyTorch only (file 01) | NVIDIA only |
| LMCache + Mooncake | KV cache storage and transfer: L1 CPU shared-memory pool, L2 persistent store; backends CPU RAM, SSD, Redis/Valkey, Mooncake Store, InfiniStore, S3, NIXL, GDS | LMCache multiprocess architecture (2026-04); multi-node P2P CPU memory sharing production (2026-01); FlexKV (Tencent + NVIDIA) added Mooncake transfer (2026-01-28) | Engine-agnostic connectors for vLLM, SGLang, Dynamo. https://github.com/lmcache/lmcache , https://lmcache.ai/en/lmcache-x-mooncake-unite-to-pioneer-kvcache-centric-llm-serving-system/ |
| llama.cpp / ggml | Single C/C++ codebase; backends report op support and fall back to CPU | GPU backends: Vulkan, CUDA/ROCm/MUSA, Metal, SYCL, OpenCL, WebGPU; NPU backends: CANN, zDNN, Hexagon; Kompute removed | The de facto portable backend layer for edge. https://vulkan.org/user/pages/09.events/vulkanised-2026/0930-Jeff-Bolz-NVIDIA.pptx.pdf |

Integration surface between compiler and serving layer today: attention kernel interfaces with
paged KV layouts (FlashInfer, FlashMLA, vendor kernels), custom-op registration, and KV connector
APIs (LMCache/Mooncake/NIXL). Schedulers remain Python; kernels remain hand-written per vendor.

## 2. Model architecture trends that move the kernel target

- DeepSeek V4 (launched 2026-04-24): every layer pairs sliding-window attention over the last 128
  tokens with compressed attention (C4: top-512 sparse over 4:1-compressed KV; C128: dense over
  128:1-compressed KV); native FP4 MoE experts; day-0 kernels from FlashMLA, FlashInfer, DeepGEMM
  and TileLang; validated on NVIDIA and Huawei Ascend; third-party figures: about 27% of V3.2's
  per-token FLOPs and about 10% of its KV cache at equal context. Hopper lacks FP4 tensor cores.
  https://www.lmsys.org/blog/2026-04-25-deepseek-v4 , https://www.theregister.com/2026/04/24/deepseek_v4/
- Hybrid linear attention: Qwen3-Next (3:1 Gated DeltaNet to gated attention), Kimi Linear (Kimi
  Delta Attention, channel-wise gating, MLA for the full-attention layers, KDA kernels released,
  about 75% KV reduction claimed). Fixed-size recurrent state changes KV memory from linear in
  context to constant for most layers. https://alphaxiv.org/abs/2510.26692
- Speculative decoding: EAGLE-3 acceptance commonly reported 0.6 to 0.8; AMD ATOM measured MTP on
  Kimi K2.6 at 59.6% acceptance, 2.79 tokens per forward pass with 3 draft tokens (2026-05,
  gsm8k). https://rocm.docs.amd.com/projects/atom/en/latest/eagle3_speculative_decoding.html
- Consequence: the "transformer kernel" is not fixed. Sparse/compressed attention, recurrent
  layers, MoE routing and FP4 block formats all arrived within 18 months; a compiler that
  hard-codes one attention form ages in months.

## 3. Agentic workloads: measured properties

| Source | Finding |
|---|---|
| TraceLab (UW SyFI, arXiv 2606.30560, 2026-06; Claude Code and Codex sessions; about 4,300 sessions, 350k LLM steps, 430k tool calls) | Median generation: about 119k prefix tokens, 875 appended tokens, 214 output tokens; prefix-cache hit rate 95.2% and 96.2% for the two harness groups; 8.9 tool calls per request; main misses are user-initiated steps after long human idle time. https://tracelab.cs.washington.edu/ , https://arxiv.org/html/2606.30560v1 |
| FreeInference agentic trace (Harvard; 16 weeks, 12,002 sessions, 267 accounts, 14 harnesses, 1.19 M LLM requests, 1.21 M tool calls) | Block-level prefixes included for cache replay. https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace |
| LMCache agentic traces (787 sessions, 24,881 iterations; IBM OTel conversion from SWE-bench, GAIA, WildClaw) | Multi-turn prefix growth to 10k+ tokens. https://huggingface.co/datasets/sammshen/lmcache-agentic-traces |
| Provider hit rates (third-party measurements) | Kimi K2.6 86 to 98% vs GLM 5.1 47 to 91% on coding agents (arXiv 2609.26779); DeepSeek V4 Flash about 98% for coding agents (blog); one coding session log 99.82% over 435 M input tokens (blog). No official provider disclosure found. |
| SemiAnalysis InferenceX v3 "AgentX" | Agentic replay benchmark; results vary by model: GLM 5.3 744B at 108 tok/s/user: GB200 NVL72 15,236 tok/s per chip at 0.03 USD per M tokens vs MI355X 5,478 at 0.08; Qwen 3.5 397B-A17B reversed (206 vs 873 tok/s per GPU at 137 tok/s/user); AgentX reported "every single AMD backend is unsupported" for context-parallel attention (file 01). https://inferencex.semianalysis.com/compare/glm-5-3-gb200-vs-mi355x , https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat |
| NVIDIA TensorRT-LLM blog 27 | Counting reused-prefix tokens inflates per-GPU throughput about 5x versus counting freshly computed tokens; proposes job-level metrics for agentic serving. https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog27_Evaluating_Agentic_Serving_with_Trace_Replay_and_Job_Level_Metrics.html |

Agentic serving research (2025-2026): Continuum (ICLR 2026, arXiv 2511.02230; KV time-to-live
from predicted tool durations and turn count, built on vLLM; 1.12 to 3.66x lower delay and 1.10
to 3.22x throughput on SWE-Bench and BFCL with Llama 3.1 8B and 70B; a follow-up block-level
eviction, arXiv 2606.02964, adds 4.4 to 18.1% latency reduction), CacheWise (arXiv 2606.16824;
prefix-aware scheduling and reuse-aware eviction driven by tool-call metadata for coding agents),
"Learning Agent Execution for KV-Cache Management in Agentic Serving" (arXiv 2608.14624), "A
Policy-Driven Runtime Layer for Agentic LLM Serving" (arXiv 2605.27744), "Ask the Tool, Don't
Guess" (arXiv 2609.18849; the serving system reads tool progress), ThunderAgent (arXiv
2602.13692), Kairos, TokenCake, CONCUR (arXiv 2601.22705; congestion-based concurrency control
for agentic batch inference), "Stateful Inference for Low-Latency Multi-Agent Tool Calling"
(arXiv 2605.26289), "Agentic AI Workload Characteristics" (arXiv 2605.26297).
Pattern: all of them live in the Python scheduler and treat the model as a black box; none
changes what is compiled.

## 4. Benchmarks and economics

- MLPerf Inference v6.0 (2026-04) added an Interactive scenario on DeepSeek-R1 (5x minimum token
  rate, 1.3x shorter TTFT than Server, speculative decoding allowed) and GPT-OSS-120B; only NVIDIA
  submitted across all five new benchmarks; AMD submitted MI355X on GPT-OSS-120B and WAN-2.2.
  MLPerf Inference v6.1 (2026-10): AMD's "broadest submission" (vendor blog).
  https://nand-research.com/mlperf-inference-6-0-software-gains-broadening-competition-shake-things-up/ ,
  https://www.amd.com/en/blogs/2026/amd-delivers-its-broadest-mlperf-inference-6-1-submission.html
- SemiAnalysis InferenceMAX renamed InferenceX; v2 (2026-02-16) Blackwell vs AMD vs Hopper; v3
  "AgentX" agentic. NVIDIA GTC 2026 cited it for GB300 NVL72 "50x tokens per watt, 35x lower cost
  per token" vs Hopper (vendor framing). https://inferencex.semianalysis.com/blog
- Demand: Google processes over 3.2 quadrillion tokens per month (I/O 2026), up about 7x year
  over year; Goldman Sachs (2026-05) projects 47 quadrillion per month by 2028.
  https://www.shacknews.com/article/149205/google-3-2-quadrillion-monthly-ai-tokens

## 5. Open problems named by maintainers and researchers

1. KV cache is the first-class resource but is managed outside the compiler, by Python
   schedulers and storage tiers; nothing plans residency from program structure.
2. Agentic requests arrive as independent HTTP calls; the serving layer has to infer session
   structure, tool-wait gaps and branch structure that the agent program already knows.
3. Every accelerator needs its own attention kernel for every attention variant; plugin systems
   lower the integration cost but not the kernel cost.
4. Determinism is wanted (RL, evals, agents) but costs throughput; batch-invariant kernels exist
   as a research artifact, not a default.
5. Throughput accounting is contested: reused-prefix tokens vs computed tokens differ about 5x.
