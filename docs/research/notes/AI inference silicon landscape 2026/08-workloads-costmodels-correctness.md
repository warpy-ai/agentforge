# Workload trends, performance models and correctness (as of 2026-10-30)

Method: direct web research by the coordinating session.

## A. Workload facts a compiler and runtime must respond to

| Fact | Value | Source |
|---|---|---|
| Agentic coding median request shape | about 119k prefix tokens, 875 new input tokens, 214 output tokens; 8.9 tool calls per request | TraceLab (arXiv 2606.30560) https://tracelab.cs.washington.edu/ |
| Prefix-cache hit rate, coding agents | 95 to 96% (Claude Code, Codex); misses dominated by long human idle time | TraceLab |
| Decode arithmetic intensity | about 1 to 2 FLOP/byte at batch 1 vs an H100 ridge near 590 FLOP/byte; decode is bandwidth-bound, prefill compute-bound | https://pub.towardsai.net/why-an-ai-model-only-uses-0-34-of-the-gpu-compute-how-gpus-actually-work-part-2-639e9a7bd97b , arXiv 2608.28048 |
| Compute vs bandwidth growth | peak server FLOPS about 3.0x per 2 years; DRAM bandwidth about 1.6x per 2 years (Gholami et al., IEEE Micro 2024); 2025 projection 2.7x FLOPS per 2 years | https://howaiworks.ai/glossary/memory-wall , https://files.futurememorystorage.com/proceedings/2025/20250807_DRAM-302-1_Greenberg-fnl.pdf |
| HBM4 generation | NVIDIA Rubin: 288 GB HBM4 at 19.2 TB/s final (22 TB/s preliminary; file 01); SK hynix 16-high HBM4 mass production 2H 2026 | file 01; https://pdpspectra.com/blog/hbm4-memory-ai-accelerators-2026 |
| Token demand | Google 3.2 quadrillion tokens per month (I/O 2026), about 7x year over year; Gemini 22 B tokens per minute (Q2 2026 call) | https://www.shacknews.com/article/149205/google-3-2-quadrillion-monthly-ai-tokens |
| Open trace datasets | FreeInference (Harvard, 1.19 M requests, block-level prefixes), TraceLab (UW), LMCache agentic traces (plus IBM OTel conversion), Exgentic (1,781 OTel traces), SemiAnalysis AgentX replay (prompt-free) | see file 06 |

## B. Performance models and simulators

| Tool | Scope | Reported accuracy | Notes |
|---|---|---|---|
| Vidur (Microsoft, MLSys 2024) | Request-level LLM serving simulation, profiling-based operator models | below 9% latency error in the paper; below 5% claimed on the poster | https://arxiv.org/abs/2405.05465 |
| GenZ LLM Analyzer | Analytical single-iteration LLM model across parallelism and hardware parameters | 1.85% to 5.82% geometric-mean error vs NVIDIA HGX, Intel Gaudi2, AMD MI300X and SambaNova SN40L | open source `abhibambhaniya/GenZ-LLM-Analyzer`; https://www.alphaxiv.org/overview/2406.01698v3 |
| LLMCompass (Princeton, ISCA 2024) | Hardware evaluation framework parameterized by systolic array and buffer design | 10.9% average operator error; 4.1% average error for LLM inference on A100, MI210, TPUv3 | https://arxiv.org/html/2312.03134v1 |
| TokenSim (arXiv 2503.08415) | Hardware and software exploration for LLM inference systems | n/a | |
| Timeloop / Accelergy, SCALE-Sim, ASTRA-sim, MAESTRO, NeuSight | Accelerator and distributed-system simulators with YAML architecture descriptions | varies | Timeloop's YAML architecture spec is the most reused open hardware-description format. |

No evidence found of a compiler project developed against a simulator first and real silicon
later; Tenstorrent's tt-mlir carries op cost models but develops on hardware. The analytical
models above are accurate enough (single-digit percent) to rank design choices for prefill and
decode on GPUs; none is calibrated for SRAM-centric or dataflow chips.

## C. Correctness, determinism, verification

- Nondeterminism: Thinking Machines (Horace He, 2025-09-10) attributes run-to-run variation to
  batch-size-dependent reduction order; batch-invariant RMSNorm, matmul and attention give 1
  unique completion in 1,000 runs vs 80 for stock vLLM, at a throughput cost (one secondary
  source says about 34%). Adoption: open-source `batch_invariant_ops` only; no engine ships it by
  default. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference
- Verified rewrites: TensorRight (POPL 2025) proves 115 of 175 XLA algebraic-simplifier rules
  for arbitrary rank and size with SMT. egglog-based verification of sharded computational graphs
  in a production distributed framework (arXiv 2509.10694). No production tensor compiler runs
  translation validation on every compile. https://par.nsf.gov/biblio/10568086
- Benchmark correctness checks are weak: KernelBench's `allclose` passes an all-zeros output on
  4 of 60 level-1 problems (arXiv 2609.21058); any kernel-search or LLM-generation loop needs
  stronger oracles (property tests, reference numerics per format, bit-exact modes).
- Numerics as types: FP8 (E4M3/E5M2), OCP MX block formats (E8M0 scale per 32 elements) and
  NVFP4 (E4M3 scale per 16 elements, 512-byte scale atoms on Blackwell; file 01) differ in block
  size and scale encoding; AMD CDNA4 supports MX only; no open IR treats them as verified types
  with defined rounding and accumulation order.
- Async protocols: a recurring class of TMEM and mbarrier protocol bugs across CUTLASS,
  DeepGEMM and TileLang, and a 2026 call to formalize TMEM, descriptor and async semantics
  (arXiv 2609.36800; file 01).
- Security: multi-tenant isolation and prompt-cache side channels argue for a memory-safe
  runtime (gap: 2024 to 2025 prompt-cache side-channel papers not re-verified in this pass).

## D. Open problems in researchers' words (paraphrased, with sources above)

1. Simulators are validated on GPUs; no public calibrated model exists for SRAM-centric or
   dataflow chips.
2. Determinism, memory safety and multi-tenant isolation are wanted together and traded off
   separately.
3. Verification of tensor rewrites is tractable (TensorRight) but unused by compilers that ship.
4. Asymmetric datapaths (MMA vs transcendental vs shared-memory bandwidth) force algorithm and
   pipeline co-design that no cost model captures by default.
