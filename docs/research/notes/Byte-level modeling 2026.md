# Byte-level modeling and AgentForge (as of 2026-11-06)

Method: direct web research by the coordinating session, prompted by Percy Liang's CS336 lecture 1
(Spring 2026 edition). Numbers in section D are first-order arithmetic over the repository's own
model and machine files, not measurements.

## A. What Percy Liang has said and published

| Item | Finding | Source |
|---|---|---|
| CS336 lecture 1 (2025 edition source; summaries of the 2026 edition report the same framing) | Tokenization is "a necessary evil, maybe one day we'll just do it from bytes". Byte tokenization has a "terrible" compression ratio, so sequences are too long for quadratic attention. Tokenizer-free approaches (ByT5, MegaByte, BLT, T-FREE) "use bytes directly, promising, but have not yet been scaled up to the frontier." The course then builds a BPE tokenizer. | https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py |
| Liang-authored byte-level or tokenizer research | None found on arXiv, Scholar or the CRFM page. A "Liang et al., 2023" cited for the vocabulary bottleneck is Davis Liang (Meta, XLM-V), not Percy Liang. | arXiv author search, 2026-11-06 |
| HELM (Liang et al., 2022, TMLR 2023) | Uses bits-per-byte as the language-modeling metric because it is invariant to the tokenizer. This is the one Liang contribution directly reusable here: a byte-vs-token evaluation must report bits-per-byte, not perplexity. | https://arxiv.org/pdf/2211.09110 |
| Stanford byte-level work (Potts and Manning group, not Liang) | MrT5 (Kallini et al., ICLR 2025): a learned delete gate in ByT5's encoder cuts sequence length by up to 75% at similar accuracy. Fast Byte Latent Transformer (Kallini, Pagnoni, Zettlemoyer, Potts, Iyer and others, May 2026): block diffusion and self-speculation cut BLT's generation memory-bandwidth cost by over 50%, with a quality trade-off. | https://arxiv.org/abs/2410.20771 ; https://arxiv.org/abs/2605.08044 |

Conclusion: the lecture frames the open problem; the research program is elsewhere. Nothing to cite
from Liang beyond the lecture quote and HELM's metric choice.

## B. State of the field, with what changed since mid-2025

| Work | Date | Claim | Status of the claim |
|---|---|---|---|
| MegaByte (Meta) | 2023 | Fixed 8-byte patches, local decoder | Superseded by learned and entropy patching |
| SpaceByte (Rice) | 2024 | Patch at whitespace; byte Transformer and MegaByte need about 10x the training FLOPs of a subword Transformer | Single-author result, consistent with Cartesia part II below |
| BLT, Byte Latent Transformer (Meta) | Dec 2024, ACL 2025 | Entropy-based patching from a small byte LM; latent Transformer over patches; 8B scale matches Llama 3 at matched FLOPs | Decoding is byte-by-byte and slow; addressed by Fast BLT (May 2026) |
| H-Net (Hwang, Wang, Gu; CMU and Cartesia) | Jul 2025 | Learned, end-to-end dynamic chunking; one-stage byte H-Net matches a BPE Transformer above 1B; two-stage scales better with data | Independent replication not found; HF Transformers integration in progress |
| Cartesia "From Tokens to Bytes I" | 2026-10-28 | One-stage H-Net: 1.28x compute efficiency on validation loss, about 1.8x (Paloma) and 2.1x (OLMES) on downstream suites, median 2.0x on in-distribution code and 3.1x on unseen languages. Two-stage 3.67B H-Net reaches SmolLM3-3B's six-task average after about 1T training tokens instead of about 11T, compute per byte matched within 2%. Main network sees one position per about 7.6 bytes; outer stages are four Mamba-2 layers each side. | Authors' own numbers; model sizes and data differ from the baseline, which they acknowledge. Improved model not yet released. |
| Cartesia "From Tokens to Bytes II" | 2026-10-28 | Compute to reach the same OLMES score, relative to H-Net: BLT-like 1.84x, BPE 2.06x, BPE without pretokenization 3.82x, pure byte Transformer 9.60x. Byte models transfer best from loss to downstream. Open: whether the transfer advantage survives post-training. | Authors' own numbers |
| Bolmo (Minixhofer, Hofmann and others; "byteifying" OLMo) | Dec 2025, v2 Feb 2026 | Fully open byte-level models: Bolmo 7B (7.63B) converted from Olmo 3 7B and Bolmo 1B (1.47B) from OLMo 2 1B, by a two-stage procedure (distill local encoder, decoder and boundary predictor with the global model frozen, then train end to end) on under 1% of a pretraining budget. Comes close to the parent on standard tasks and far ahead on character tasks (CUTE 78.6 vs 56.9 at 7B). Reports single-sequence inference: about 125 bytes/s decode vs 150 for the parent at the same compression, prefill about 1 s for 72k bytes vs 0.8 s, and a crossover where the byte model becomes cheaper above about 6.6 bytes per patch. | The right open model for solo experiments; ByT5 is obsolete for this purpose. The only published inference numbers in this table, and they are single-sequence, not serving-level. |
| Fast BLT | May 2026 | See section A | |
| "Byte Language Models: Scaling, Emergent Abstractions, and Information Allocation" (Wang, Luo, Zhang, Wu; East China Normal and Fudan) | 2026-10-05 | Plain byte Transformers (no hierarchy; hybrid Gated DeltaNet and softmax attention following Qwen3.5, hashed 5 to 8-gram embeddings, and "token superposition" training on bags of four tokens for the first 30% of updates) at 200M to 3B dense plus MoE to 16.6B total. At matched parameter count the byte models have lower bits-per-byte across the sweep and beat the subword models on every downstream task, with CUTE near 99 vs 70 to 81. At matched training FLOPs the subword models are better, and a looped subword control closes most of the parameter-matched gap. Up to 25% of the byte model's layers can run on 4x compressed representations with no retraining. Speculative decoding with same-family drafts accepts 3.4x more tokens per target pass, but accepted bytes per pass are equal to the subword model's. | Authors' own. The headline "bytes beat subwords" is parameter-matched; the FLOP-matched result agrees with Cartesia part II. The byte advantage in speculative decoding is a token-count artifact, not more text per pass. |

What this changes in the project list that prompted this note:

1. "Learnable dynamic patching" is no longer a gap. H-Net did it and Cartesia scaled it. The pure
   byte Transformer is about 9x less compute-efficient than a hierarchical model at matched
   compute, so "end-to-end byte-level" in practice means hierarchical with a learned chunker.
   The one dissent, the October 2026 ECNU and Fudan paper, does not contradict this: its byte
   models win at matched parameters and lose at matched FLOPs, and its own looped-subword control
   shows the parameter-matched win is mostly extra compute per byte of text. What it adds is that
   a flat byte model with hybrid linear attention learns chunk-like structure on its own, which is
   evidence for the hierarchical designs rather than against them.
2. Inference slowness is partially addressed on the decoder side (Fast BLT), and Bolmo reports
   single-sequence decode and prefill numbers, but neither Cartesia post says anything about
   inference, and no work reports KV cache, residency or batched serving behaviour for any
   byte-level model. That is the open gap.
3. The evaluation and post-training questions remain open and are cheap: Bolmo against its OLMo
   parent isolates tokenization with everything else held fixed.

Sources: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1 ,
https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2 , https://arxiv.org/abs/2507.07955 ,
https://aclanthology.org/2025.acl-long.453.pdf , https://arxiv.org/pdf/2404.14408 ,
https://arxiv.org/html/2512.15586v2 , https://arxiv.org/html/2610.05978v1

## C. What AgentForge does and does not offer this work

AgentForge does not shorten sequences. It plans where KV lives and when it moves, from the agent
program and a machine description (`design/0001-approach.md`). For a byte-level model the inputs
that change are: positions per byte at each level of the hierarchy, which levels hold a growing
KV cache versus fixed SSM state, and the per-step cost of each level. All three are inputs to
`agentforge-cost` (KV bytes per token, attention kind, weight bytes) and to the trace generator
(`agentforge-sim::trace`, which samples prefix, appended and generated token counts). None of
this exists yet as a model variant; today `agentforge-ir::model` describes one flat Transformer
with a single attention kind (dense, grouped-query, latent, sliding-window or compressed) applied
to every layer. Every byte-level design in section B mixes layer kinds: H-Net puts Mamba-2 in its
outer stages, the ECNU and Fudan models interleave Gated DeltaNet with softmax attention, and
Bolmo pairs small local networks with a global one. Fixed-state layers (SSM or linear attention)
hold a constant number of bytes per sequence instead of a growing cache, so the model IR needs
per-layer or per-stage kinds, including a fixed-state kind, before any of section D can be
simulated rather than hand-computed.

## D. First-order arithmetic for the median agent step

Setup from the design doc: 70B grouped-query model, 80 layers, 8 KV heads, head dimension 128,
320 KB of FP16 KV per attended position; H200 at 4.8 TB/s and 141 GB. Median TraceLab step:
119k BPE-token prefix, 875 appended, 214 generated. Bytes per BPE token taken as 4.57 (the rate
Cartesia reports for their tokenizer on their mix), so the prefix is about 544 kB.

Positions held in a growing attention cache, and their FP16 KV, for the 544 kB prefix:

| Architecture | Positions per byte | Cache positions | FP16 KV | Fits one H200? |
|---|---|---|---|---|
| BPE Transformer (baseline) | 1 / 4.57 | 119k | 39 GB | yes |
| Pure byte Transformer, softmax attention in every layer | 1 | 544k | 178 GB | no (89 GB in FP8, 63% of the device) |
| Pure byte Transformer, hybrid 3:1 linear to softmax attention (ECNU and Fudan design) | 1 | 544k, cached in one layer of four | about 45 GB plus fixed linear-attention state | yes |
| BLT (patch size 4.5 to 8) | 1 / 4.5 to 1 / 8 | 68k to 121k latent, plus windowed local caches | 22 to 40 GB | yes |
| H-Net, one stage (compression matched to BPE) | 1 / 4.57 | 119k main, SSM state in encoder and decoder | 39 GB plus megabytes of state | yes |
| H-Net, two stage (Cartesia part II) | 1 / 7.6 | about 72k main, sliding-window attention plus SSM state in the inner stage | about 23 GB plus a window | yes |

Decode traffic per emitted byte, batch one, FP8 weights (70 GB), memory-bound:

| Architecture | Weight bytes read per byte emitted | Sequential steps per BPE-token equivalent | Note |
|---|---|---|---|
| BPE Transformer | 70 GB / 4.57 = 15 GB | 1 | about 15 ms per token on H200 |
| Pure byte Transformer | 70 GB | 4.57 | about 15 ms per byte, 4.6x the baseline. Hybrid attention does not change this row: weight traffic, not KV traffic, dominates at batch one. Same-family speculative drafting does not either: the ECNU and Fudan paper finds accepted bytes per target pass equal to the subword model's, so the 4.6x more target passes per unit of text remain. |
| H-Net | decoder and encoder stage weights per byte (under 1 GB for four Mamba-2 layers), main network per chunk | 4.57 small steps plus 1 large | about the baseline's bandwidth cost plus a few percent; latency is bounded by launch overhead of the small steps, not bandwidth. The H-Net paper describes exactly this generation procedure and likens it to speculative decoding: the outer networks step on every byte, the main network only at chunk boundaries. |
| BLT with Fast BLT decoding | over 50% less than BLT per the paper | fewer than 4.57 | quality trade-off |

Measured anchor for the BLT-style rows: Bolmo 7B, single sequence, decodes at about 125 bytes/s
against 150 bytes/s for its Olmo 3 parent at the same compression, and becomes cheaper than the
parent above about 6.6 bytes per patch. Agent tool output is mostly code and logs, which chunk
longer than prose, so the median agent step may sit on the favourable side of that crossover.

Three consequences for the compiler, none of which the public byte-level work addresses:

1. **KV grows by bytes times a content-dependent ratio, not by a fixed amount per token.**
   Cartesia reports H-Net compresses code more than BPE and Chinese or Arabic less. The residency
   planner today budgets by token counts from the trace. A byte-level plan must budget by bytes
   with a ratio predicted per prompt block; the Agent Program IR already types blocks (tool
   result, user, reasoning), so the prediction has somewhere to live.
2. **Prefix caching survives learned chunking.** H-Net's routing module is causal by
   construction: the boundary probability at position t is a cosine similarity between the
   projected encoder outputs at t and t-1, and the paper states that dynamic chunking "maintains
   causality by computing boundary probabilities based only on current and previous
   representations". BLT's entropy model is a causal byte LM. So chunks and KV before an edit
   point stay valid. Only the chunk containing the
   boundary can change. This needs stating and simulating because it is the first objection a
   systems reviewer raises.
3. **The prompt boundary problem disappears.** With BPE, appending text after a cached prefix
   can change the last token at the join, invalidating the last block. A byte-level model has no
   such join. Cartesia part II names this as open for BPE in practice; agent traces with 8.9 tool
   calls per request are where it would show up.

## E. Projects a single researcher can do, revised

| Project | Needs | Deliverable | Why it is open |
|---|---|---|---|
| 1. Byte-hierarchical model variants in the simulator | This repository, no GPU | Per-layer or per-stage attention kinds in `agentforge-ir::model`, including a fixed-state kind for SSM and linear-attention layers, then a `hierarchy` variant (per stage: positions per byte, layer kinds, layer cost), per-stage prefill and decode in `agentforge-cost`, byte-denominated traces, and regimes A and D replayed for BPE, pure byte (softmax and hybrid attention), BLT and H-Net one and two stage. Report KV GB, decode ms per byte, TTFT p90, foreground KV wait. | Only single-sequence inference numbers exist (Bolmo); no KV, residency or batched serving analysis of any byte-level model (section B). |
| 2. Robustness benchmark, bytes versus tokens | One GPU, Bolmo 7B versus Olmo 3 7B and Bolmo 1B versus OLMo 2 1B | Corruption suite (bit flips, byte deletions, UTF-8 misalignment, OCR-style substitutions) at graded rates; bits-per-byte (HELM) and task accuracy; the corruption rate at which each model's accuracy halves. | Each Bolmo is converted from its OLMo parent with the global network initialised from it, so the pair isolates the byte interface better than any two independently trained models. The original project 4, still open. |
| 3. Post-training on bytes | One GPU, Bolmo | DPO on byte-level versus token-level loss; whether Cartesia's loss-to-downstream transfer advantage survives post-training. | Named as open in Cartesia part II. The original project 2, narrowed to the question the field is asking. |
| 4. Program-aware chunking | Later, after 1 | A chunker conditioned on prompt block type from the Agent Program IR, compared with content-only chunking. | Speculative; content-only chunking is already solved by H-Net. Lowest priority. |

Order: 1 first, because it needs no hardware and produces the numbers the other three are
evaluated against. 2 and 3 share a model setup and can follow together.

## F. Caveats

- Section D uses Cartesia's 4.57 bytes per token. On code-heavy agent traces the Llama 3
  tokenizer's ratio differs; the simulator should sample it per block rather than fix it.
- The H-Net decode column assumes the small stages are the only per-byte work, which matches the
  paper's generation procedure, but the released code has no serving-optimized decode path, so
  the launch overhead per byte step is an assumption to measure, not a fact. Bolmo's 125 versus
  150 bytes/s is the only measured byte-level decode rate found, and it is batch one.
- All efficiency multipliers in section B are the authors' own; no independent replication of
  the October 2026 H-Net results was found.
