# Startup and novel-architecture inference silicon (state as of 2026-10-30)

Method: direct web research by the coordinating session. "vendor" marks vendor claims; "(gap)"
marks items not verified against a primary source.

## 1. Takeaway

The 2025 to 2026 story is consolidation around SRAM-centric decode tiers: Groq's technology
went to NVIDIA and ships as the Groq 3 LPX rack, Cerebras became the decode tier of an AWS
service and went public, d-Matrix reached volume shipments and joined NVLink Fusion, and the
one model-specific chip company with silicon (Taalas) was bought by AMD. A second wave
(Positron, MatX, Etched, Extropic, Furiosa, Rebellions) raised large rounds in 2026 on
pre-production or early silicon. Only Tenstorrent publishes its whole software stack and
firmware; every other startup is closed to outside compilers.

## 2. Company status table

| Company / part | Architecture class and target | Status (date) | Software openness | Sources |
|---|---|---|---|---|
| Groq LPU; NVIDIA Groq 3 LPX | Deterministic, SRAM-only, statically scheduled tensor streaming; decode latency | NVIDIA took a non-exclusive license and hired most engineering in Dec 2025 (13.0 B USD at closing plus 4 B within a year per NVIDIA's annual report; press: about 20 B); GroqCloud continues under CEO Simon Edwards; Senate scrutiny of the structure; NVIDIA's LPX rack: 256 LPUs, 500 MB SRAM each, 150 TB/s per LPU, about 315 PF FP8 per rack, "in full production" Aug 2026 (see file 01) | Closed; compiler is the product | https://www.sec.gov/Archives/edgar/data/0001045810/000104581026000038/a2026-annualxreportxwebxfi.pdf , https://x.com/GroqInc/status/2003945400181608843 , https://www.nvidia.com/en-us/data-center/lpx/ |
| Cerebras WSE-3 / CS-3 | Wafer-scale, 44 GB on-wafer SRAM; batch-1 decode speed | Nasdaq IPO May 2026 (CBRS) at 185 USD, 5.5 to 6.4 B USD raised; 2025 revenue 510 M USD (+76%); about 86% of revenue from two UAE-linked customers; OpenAI deal above 20 B USD; AWS decode-tier partnership (Mar 2026); vendor claims up to 15x GPU speed | Closed | https://techcrunch.com/2026/05/14/cerebras-raises-5-5b-kicking-off-2026s-ipo-season-with-a-bang , https://www.sec.gov/Archives/edgar/data/0002021728/000162828026044941/cbrsannouncesfinancialresu.htm |
| SambaNova SN40L / SN50 | Reconfigurable dataflow unit with three-tier memory | Intel acquisition talks (about 1.6 B USD) stalled Jan 2026; Series E 350 M USD (Feb 2026); Series F first close 1 B USD at 11 B USD (Jul 2026); SN50 shipments targeted 2H 2026 with SoftBank first; IPO considered | Closed | https://www.datacenterdynamics.com/en/news/sambanova-seeking-500m-in-funding-after-acquisition-talks-with-intel-stall-report , https://dealroom.co/news/137657-sambanova-banks-1b-series-f-first-close-at-11b-valuation/ |
| Tenstorrent Blackhole / Galaxy | RISC-V Tensix cores with per-core SRAM, GDDR6; general-purpose AI | Series E 1.5 B USD at 6.2 B USD post-money (2026-10-19, Trace Capital and Innovation X); Galaxy Blackhole GA 2026-04-28 (32 Blackhole processors, 23 PF FP8 per system), "shipping in volume" by May 2026 with a 36-Galaxy supercluster; Blackhole p150 card: 140 Tensix cores with 1.5 MB SRAM each, 32 GB GDDR6 at 512 GB/s, 774 TFLOPS FP8 (vendor), 300 W, four 800G ports | Fully open: TT-Metalium kernel SDK (v0.80.0-dev, 2026-09-22), TT-NN, TT-MLIR (TTIR, TTNN, D2M, TTKernel, TTMetal dialects), TT-Forge (PyTorch, JAX, ONNX); official Rust `luwen`; community `libtt-metal-cxx` and `metalium-rs` | https://datacenterdynamics.com/en/news/ai-chip-startup-tenstorrent-raises-15bn-at-62bn-valuation , https://docs.tenstorrent.com/tt-mlir/overview.html , https://github.com/tenstorrent/tt-metal/releases , https://docs.tenstorrent.com/tt-awesome/entry/libtt-metal-cxx/ |
| Etched Sohu | Transformer-only ASIC sold as liquid-cooled racks | Series B 500 M USD (Jan 2026, Stripes), total about 800 M; more than 1 B USD in signed contracts, Jane Street the confirmed customer; shipping status conflicting between sources (first racks summer 2026 vs pre-production); vendor claims 500k tokens/s on Llama 70B per rack, no independent benchmark | Closed | https://seedtable.com/companies/etched-ai/funding-rounds/series-b-2026-01 , https://www.fast.io/resources/etched-ai-review-2026.md |
| d-Matrix Corsair / Raptor | Digital in-memory compute, SRAM-centric; low-latency decode | Corsair in full production and volume shipments from June 2026 (about 90% US customers); Series C 275 M USD at 2 B USD (Nov 2025), total about 450 to 500 M; Raptor (4 nm, 3D-stacked DRAM) tape-out by end 2026, MGX rack systems Q4 2027; joined NVLink Fusion 2026-09-10; acquired Wallaroo.ai 2026-08-03 | Closed (Aviator software stack) | https://www.cnbc.com/2026/06/09/nvidia-d-matrix-chip-production-microsoft.html , https://shattered.io/d-matrix-nvlink-fusion-raptor-nvidia-2026/ |
| Taalas HC1 / HC2 | Model hard-wired into silicon; LoRA and context size adjustable | HC1 (Feb 2026): Llama 3.1 8B on an 815 mm2 TSMC 6 nm die, about 17,000 tokens/s per user (vendor), demonstrator; HC2 for about 20B models; AMD agreed to acquire Taalas 2026-08-06 (price undisclosed; 219 M USD raised) | Closed | https://www.nextplatform.com/2026/02/19/taalas-etches-ai-models-onto-transistors-to-rocket-boost-inference/ , https://siliconangle.com/2026/08/06/amd-acquires-taalas-hardwire-ai-models-silicon/ |
| Positron Asimov / Titan | Memory-centric inference accelerator | Series C 875 M USD at 5 B USD post-money (2026-09-10, NEA and Jim Clark); Asimov tape-out end 2026 on TSMC N3P, production 2H 2027 | Closed | https://www.datacenterdynamics.com/en/news/positron-ai-raises-875m-at-a-5bn-valuation-to-support-develop-of-next-generation-hardware/ |
| MatX One | LLM-first programmable accelerator | Series B 500 M USD (2026-02-24); first product shipping 2027; one report says Anthropic walked away from a 7 B USD acquisition and explores a partnership (single source) | Closed | https://news.bloomberglaw.com/private-equity/ai-chip-startup-matx-raises-500-million-to-compete-with-nvidia |
| Furiosa RNGD | Tensor contraction processor | Sought 300 to 500 M USD Series D (Jan 2026) for second-generation RNGD; LG contract; rejected Meta's 800 M USD bid; 2027 IPO eyed | Closed | https://www.bloomberg.com/news/articles/2026-01-19/ai-chip-startup-furiosaai-plans-500-million-round-before-ipo |
| Rebellions | NPU | Series C 250 M USD at 1.4 B (Nov 2025); 166 M USD from South Korea's National Growth Fund (Jul 2026); pre-IPO | Closed | https://autotechinsight.spglobal.com/news/5283567/rebellions-advances-pre-ipo-fundraising-fueled-by-strong-investor-interest |
| Qualcomm AI200 / AI250 | Rack-scale inference built on Hexagon NPU IP, 768 GB LPDDR per card, near-memory compute | AI200 slated for 2026, AI250 for 2027; Humain 200 MW commitment from 2026; no confirmed 2026 shipments found (gap) | Closed | https://www.sdxcentral.com/news/qualcomm-launches-ai200-and-ai250-chip-offering-targeting-inferencing-workloads-at-rack-scale/ |
| EnCharge EN100 | Analog in-memory compute, edge | 150 TOPS at 1 W claimed (vendor); 100 M USD Series B | Closed | https://www.datacenterdynamics.com/en/news/encharge-ai-launches-its-analog-in-memory-en100-ai-accelerator/ |
| Mythic | Analog flash compute-in-memory | Acquired Videantis (Jan 2026) | Closed | https://www.mythic.ai/technology |
| Extropic | Thermodynamic sampling units (probabilistic bits) | XTR-0 desktop dev kits shipped to early adopters; Z1 tape-out with early access 2027 (hardware page still says 2026); 75 M USD CHIPS R&D letter of intent (2026-07-29); the 10,000x efficiency claim is modeled, not measured | Closed | https://extropic.ai/writing/from-one-to-one-billion |
| Lightmatter, Ayar Labs, Celestial AI | Photonic interconnect and memory fabrics | Lightmatter Passage M1000; partner GUC tape-out on TSMC N3P (Jul 2026); Ayar Labs TeraPHY rack demo at Computex 2026; Celestial AI acquired by Marvell | n/a (interconnect) | https://www.digitimes.com.tw/about_us/article.asp?news_key=1150022 |
| SK hynix AiM / AiMX, Samsung PIM | Processing-in-memory DRAM | AiMX accelerator card and LLM demo server at AI Infra Summit 2026 (prototype since 2023); no Samsung 2026 status found (gap) | n/a | https://www.remio.ai/post/sk-hynix-processing-in-memory-takes-aim-at-ai-bottlenecks-beyond-gpu-and-hbm |
| Untether AI, Graphcore (SoftBank), Esperanto, Axelera | Not verified in this pass (gap) | | | |

## 3. Patterns

1. SRAM-centric decode won a place next to GPUs rather than instead of them: Groq inside
   NVIDIA's rack, Cerebras behind Trainium on Bedrock, d-Matrix on NVLink Fusion.
2. Model-specific silicon was absorbed (Taalas to AMD); programmable LLM-first chips (MatX,
   Positron, Etched racks) are funded on 2027 delivery.
3. The compiler is the product for every SRAM or dataflow startup, and none of them opens it.
4. Tenstorrent is the only vendor whose kernel SDK, compiler, firmware and device library are
   open, which makes it the natural first non-GPU target for an open compiler.
