# Hyperscaler, big-tech and national custom inference ASICs (state as of 2026-10-30)

Method: direct web research by the coordinating session. "vendor" marks vendor claims; "(gap)"
marks items not verified against a primary source. URLs are the sources used.

## 1. Takeaway

Every hyperscaler now runs its own inference silicon in production, and in 2026 the parts
became externally visible products: Google rents and sells Ironwood TPUs to Anthropic and Meta,
AWS pairs Trainium3 with Cerebras for a disaggregated Bedrock service, Microsoft runs Maia 200
and announced Maia 300, Meta published a four-generation MTIA roadmap, and Huawei ships
SuperPoDs of thousands of Ascend NPUs with an open-sourced CANN stack. None of these parts
exposes an ISA to outside compilers; the only entry points are vendor DSLs and graph formats.

## 2. Per-vendor status

| Vendor / part | Status (date) | Architecture and scale facts | Programming entry for outsiders | Sources |
|---|---|---|---|---|
| Google TPU v7 Ironwood | Inference generation in production; Anthropic deal (Oct 2025): Google to invest up to 40 B USD, up to 1 million Ironwood chips and 5 GW over five years (press); Meta agreed in Feb 2026 to rent TPUs via Google Cloud; Google researchers reportedly queue for internal capacity | Systolic-array TPU with ICI/OCS interconnect (details per Google) | XLA/StableHLO via PJRT; Pallas/Mosaic TPU kernels; ISA closed; no bytecode-level entry | https://www.maginative.com/article/anthropic-secures-1m-google-tpus-while-keeping-amazon-as-primary-training-partner/ , https://siliconangle.com/2026/02/26/google-meta-reportedly-strike-new-multibillion-dollar-ai-chip-deal/ , https://thenextweb.com/news/google-tpu-compute-internal-researchers-anthropic |
| AWS Trainium3 / Trn3 UltraServer | Generally available 2025-12-02 | Up to 144 Trainium3 chips per UltraServer, 20.7 TB HBM3e, 706 TB/s aggregate, 362 MXFP8 PFLOPS, NeuronSwitch-v1 (vendor); per chip about 2.52 PF FP8, 144 GB HBM3e, 4.9 TB/s (secondary); AWS claims 4.4x Trn2 (vendor) | Neuron SDK; NKI (Neuron Kernel Interface) stable 0.3.0 with Neuron SDK 2.29 (2026-04-17) including a NumPy CPU simulator; AWS product pages call the NKI compiler open source and MLIR-based, developer docs are inconsistent (gap: confirm repository and license) | https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/ , https://www.businesswire.com/news/home/20251201046353/en/ , https://softprom.com/aws-neuron-sdk-2290-nki-stable-cpu-simulator-expanded-library |
| AWS Trainium4 | Announced | 6x FP4, 3x FP8, 4x memory bandwidth vs Trn3; NVLink Fusion support (vendor) | as above | https://introl.com/blog/aws-trainium-inferentia-silicon-ecosystem-guide-2025 |
| AWS Project Rainier | Activated Oct 2025 | about 500k Trainium2 chips for Anthropic training | n/a | same |
| AWS + Cerebras | Announced 2026-03-13; service planned 2H 2026 on Bedrock | Disaggregated inference: Trainium prefill, Cerebras CS-3 decode, connected over EFA on Nitro | n/a | https://secure.businesswire.com/news/home/20260313406341/en/ , https://siliconangle.com/2026/03/13/aws-will-bring-cerebras-wafer-size-wse-3-chip-cloud-platform/ |
| Microsoft Maia 200 | In production since Jan 2026 in US Central (Iowa), then US West 3 (Arizona); serves Microsoft 365 Copilot and OpenAI models | Microsoft claims about 30% better performance per dollar than prior fleet hardware (vendor) | Triton backend for Maia; closed hardware | https://news.microsoft.com/source/emea/features/maia-200-microsoft-ai-accelerator-azure-4/ , https://hothardware.com/news/microsoft-unveils-maia-200-ai-chips |
| Microsoft Maia 300 | Unveiled 2026-10-12 | Third generation; Microsoft says Maia 200 experience shaped it | as above | https://www.datacenterknowledge.com/data-center-chips/microsoft-pushes-custom-ai-chips-with-azure-maia-300 |
| Meta MTIA 300 / 400 / 450 / 500 | Unveiled 2026-03-11; MTIA 300 in production for ranking and recommendation; 400 tested, deployment pending; 450 and 500 target generative-AI inference, 450 doubles HBM bandwidth vs 400; cadence about one chip per six months; Broadcom deal extended Apr 2026 | n/a | Closed; internal PyTorch stack | https://www.datacenterdynamics.com/en/news/meta-unveils-next-four-generations-of-its-mtia-chip/ , https://techwireasia.com/2026/04/meta-extends-broadcom-deal-to-develop-ai-chips/ |
| OpenAI + Broadcom | 10 GW of custom accelerators, deployments from 2H 2026 to end of 2029, internal only, Broadcom Ethernet; no confirmed first deployment found (gap) | OpenAI designs, Broadcom manufactures and integrates | Closed | https://www.maginative.com/article/openai-and-broadcom-to-co-design-custom-ai-chips-deploy-10-gw-by-2029/ |
| Apple server silicon ("Baltra") | Reported Dec 2024 with Broadcom for 2026 mass production; no 2026 confirmation found (gap); Private Cloud Compute runs on M-series today | n/a | Closed | https://www.theregister.com/2024/12/12/apple_ai_chip_broadcom/ |
| Huawei Ascend | Atlas 950 SuperPoD (up to 8,192 NPUs over UnifiedBus) shown MWC 2026 and reported in large-scale commercial use; Atlas 960E SuperPoD (Sept 2026): up to 4,096 NPUs, 8 EFLOPS FP8, 1 PB HBM (vendor); roadmap: Ascend 960DT Q1 2027 (pulled in), 960PR Q3 2027, 970 in 2028, 980 in 2029 | NPU-based SuperPoDs; export controls shape supply | CANN open-sourced; Huawei says 61% of CANN developers are external and more than 90 third-party projects are supported including PyTorch, Triton (Triton-Ascend), vLLM and veRL (company claims); DeepSeek V4 validated on Ascend | https://www.huawei.com/en/news/2026/9/hc-ascend960-supernode , https://www.crnasia.com/news/2026/artificial-intelligence/huawei-outlines-ascend-960-roadmap-and-million-processor-ai , https://www.techradar.com/pro/huawei-debuts-its-atlas-950-ai-superpod-at-mwc-2026-taking-the-ai-data-center-fight-to-nvidia-and-amd |
| Other Chinese parts (Cambricon, Baidu Kunlun, Alibaba T-Head, Moore Threads, Biren, Hygon) | Not researched in depth (gap) | | | |
| Tesla | Dojo cancelled Aug 2025; AI5/AI6 inference chips for vehicles (not researched further) | | | |

## 3. Market share and volume estimates (secondary sources, definitions differ)

| Claim | Figure | Source |
|---|---|---|
| Custom ASIC shipment growth 2026 vs GPU | +44.6% vs +16.1% (TrendForce via secondary write-up) | https://introl.com/blog/custom-silicon-inflection-2026-hyperscaler-asics-nvidia-gpu |
| ASIC vs merchant GPU volume | Omdia projects ASIC shipments exceed GPU shipments by 2028 | same |
| Digital AI ASIC market 2026 | 60 to 70 B USD (JPMorgan) | https://www.kucoin.com/news/flash/jpmorgan-ai-asic-shipments-to-surpass-gpus-by-2027 |
| NVIDIA share of AI accelerators | about 70% (Tom's Hardware) to above 90% (Introl), depending on definition | https://www.tomshardware.com/tech-industry/semiconductors/custom-ai-asics-examined-from-broadcom-to-mtia |
| Custom ASIC share of internal hyperscaler workloads | about 15 to 20% today (MarketsandMarkets) | https://www.marketsandmarkets.com/Market-Reports/custom-ai-asic-market-130951669.html |

## 4. What this means for an outside compiler

- No hyperscaler part offers a documented ISA, a public bytecode or an open driver. The
  reachable layers are StableHLO through PJRT (TPU), NKI Python kernels (Trainium), Triton
  (Maia), and AscendC, Triton-Ascend or CANN graph APIs (Ascend).
- The AWS plus Cerebras service is the first public cloud product whose architecture is
  defined by heterogeneous prefill and decode silicon. Placement across device classes is now a
  product configuration, not a research idea.
- Hyperscalers describe their software challenge as the same one the open ecosystem has: the
  need for a kernel language below the graph compiler (NKI, Pallas, Triton-for-Maia) because
  graph compilers alone could not reach peak performance on new attention and MoE variants.
