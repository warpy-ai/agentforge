# 0002: Byte-level language models at GPT-2-small scale, with an inference contract for AgentForge

Status: draft, 2026-11-06. This spec is for a separate repository (working name `bytelm`). Copy it
into that repository as `SPEC.md`. AgentForge consumes only what section 9 defines. Background and
sources: `../research/notes/Byte-level modeling 2026.md`.

Every number marked *estimate* is first-order arithmetic to be replaced by a measurement. Every
item marked *verify* was not confirmed against a primary source when this spec was written.

## 1. Goal

Train three language models that differ only in how they consume text, on the same bytes of data,
and measure three things for each:

1. **Quality**, in bits per byte (BPB) and on two downstream tasks.
2. **Inference cost** on one GPU: prefill latency, decode rate, peak memory, and cache bytes per
   sequence, broken down by stage.
3. **Compression by content type**: how many bytes each main-network position covers on prose,
   code, tool logs and non-Latin text.

Then export each model's architecture, measurements and reference tensors in the format of
section 9, so AgentForge can (a) calibrate its cost model against real latencies, (b) simulate the
same architecture at 7B and 70B under agent workloads, and (c) test bit-exact execution of a tiny
variant in its CPU backend (AgentForge milestone M2).

## 2. Non-goals

- Beating GPT-2 or any published model. The comparison is internal and controlled.
- Instruction tuning, RLHF, DPO, tool use, or agent behaviour. A 124M model cannot run an agent.
- Two-stage H-Net, BLT, MegaByte, or hybrid linear attention in the first version (section 12
  lists them as extensions).
- Serving optimisation (custom kernels, speculative decoding). Measure the reference
  implementation honestly and record what was and was not optimised.
- Any change to AgentForge. This repository writes files; AgentForge reads them later.

## 3. The three models

All three share one Transformer block so that only the input interface differs. Use the block from
the H-Net reference code (`goombalab/hnet`, MIT license): pre-norm RMSNorm, rotary position
embeddings, SwiGLU feed-forward, no biases. If its `HNet` class accepts an isotropic layout such
as `["T12"]`, build all three from it; otherwise write a thin isotropic wrapper around the same
`Block` class (*verify* which applies).

| ID | Input unit | Vocabulary | Context | Main network | Outer networks | Params *estimate* |
|---|---|---|---|---|---|---|
| `bpe` | GPT-2 BPE token | 50,257 | 896 tokens | 12 x T, d 768, 12 heads, SwiGLU width 2048 | none | 124M with tied embeddings (85M blocks + 38.6M embedding) |
| `byte` | UTF-8 byte | 256 | 4,096 bytes | same as `bpe` | none | 85M |
| `hnet` | UTF-8 byte, chunked | 256 | 4,096 bytes | same as `bpe`, run on chunks | encoder 4 x Mamba-2 and decoder 4 x Mamba-2, d 512 | about 100M |

Context lengths follow the H-Net paper's matching rule (8,192 bytes against 1,792 GPT-2 tokens,
a ratio of 4.57), so all three see about the same span of text per sequence.

Tiny variants for AgentForge bit-exact tests (section 9.3), same structure, same data pipeline:

| ID | Main network | Outer networks | Context |
|---|---|---|---|
| `bpe-tiny` | 2 x T, d 128, 4 heads, SwiGLU 384 | none | 128 tokens |
| `byte-tiny` | same | none | 512 bytes |
| `hnet-tiny` | same | 1 x Mamba-2 each side, d 64 | 512 bytes |

### 3.1 `hnet` configuration

In the reference code's config format:

```json
{
  "arch_layout": ["m4", ["T12"], "m4"],
  "d_model": [512, 768],
  "d_intermediate": [0, 2048],
  "vocab_size": 256,
  "ssm_cfg": { "chunk_size": 256, "d_conv": 4, "d_state": 128, "expand": 2 },
  "attn_cfg": { "num_heads": [8, 12], "rotary_emb_dim": [32, 32], "window_size": [1023, -1] },
  "tie_embeddings": false
}
```

The outer `attn_cfg` entries are unused because the outer layers are Mamba-2 only; they are kept
because the code expects one entry per stage. Rotary dimension 32 is half the head dimension of
64, as in the published configs.

### 3.2 Dynamic chunking, as published (implement from the reference code, test against this)

For encoder outputs `x_t`:

- Router: `q_t = W_q x_t`, `k_t = W_k x_t`,
  `p_t = 0.5 * (1 - cos(q_t, k_{t-1}))`, `p_1 = 1`, boundary `b_t = 1[p_t >= 0.5]`.
- Downsampler: keep the vectors where `b_t = 1`; the kept `p_t` go with them.
- Smoothing on the way back: `z_bar_t = P_t * z_hat_t + (1 - P_t) * z_bar_{t-1}`.
- Upsampler: confidence `c_t = p_t` if `b_t = 1` else `1 - p_t`; straight-through
  `STE(c_t) = c_t + stopgrad(1 - c_t)`; each chunk's vector repeats until the next boundary;
  output `STE(c_t) * z_tilde_t`.
- Residual: decoder input `= Dechunk(main output) + Linear(x_t)`, the linear initialised near 0.
- Ratio loss: with `F = mean(b_t)`, `G = mean(p_t)` and target ratio `N`,
  `L_ratio = N/(N-1) * ((N-1) * F * G + (1-F) * (1-G))`.
  Total loss `L = L_next_byte + 0.03 * L_ratio`. Use `N = 6`; the paper's one-stage model with
  `N = 6` reached 4.8 bytes per chunk, close to GPT-2 BPE's 4.6.
- An RMSNorm at the end of the encoder, the main network and the decoder.

The router is causal by construction: `p_t` depends only on positions `t` and `t-1`. Section 7.2
turns this into a test, because AgentForge's prefix-caching argument depends on it.

### 3.3 Per-stage learning rates for `hnet`

The paper multiplies the base learning rate of stage `s` by
`lambda_s = sqrt(N_gpt * (prod_{i>=s} N_i / prod_{i>=0} N_i) * D_S / D_s)`, with `N_gpt = 4.6`
bytes per GPT-2 token, `N_S = 1` and `D_s` the stage's width. For this model (`N_0 = 6`,
`D_0 = 512`, `D_1 = 768`): `lambda_0 = sqrt(4.6 * 1.5) = 2.63` for the encoder and decoder,
`lambda_1 = sqrt(4.6 / 6) = 0.88` for the main network (*verify* the formula against the paper's
Appendix C before relying on it; if a sweep is affordable, also try 2.0 and 1.0).

## 4. Data

- **Training corpus:** FineWeb-Edu, `HuggingFaceFW/fineweb-edu`, subset `sample-10BT` (about
  10B GPT-2 tokens, about 45B UTF-8 bytes *estimate*).
- **Held-out set:** 10M bytes from a FineWeb-Edu shard that is not in `sample-10BT`, fixed once and
  hashed (record the SHA-256 in the repo).
- **Shards:** byte shards store `uint8` with vocabulary 256 and documents separated by the byte
  sequence `\n\n`, so byte models never need an id above 255. BPE shards store `uint16` GPT-2 ids
  with `<|endoftext|>` between documents. The separator bytes are excluded from BPB in both.
- **Same bytes for every model.** Both shard types are produced from one ordered document stream.
  Each byte shard records the document range it covers; BPE shards cover the same ranges.
- **Tiers:**

| Tier | Bytes | Purpose | Hardware |
|---|---|---|---|
| S | 1B | Pipeline and smoke tests; the tiny variants | one GPU, or Apple MPS for `bpe` and `byte` only |
| M | 45B (all of `sample-10BT`) | The real comparison | rented H100 |

- **Domain probe sets** for compression measurement (section 6.3), 2M bytes each:
  prose (from the held-out set), code (Python and Rust from a permissively licensed code set, for
  example `bigcode/the-stack-smol`, *verify*), tool and shell logs (agent trajectories, for
  example SWE-agent trajectories on Hugging Face, *verify* name and license), Chinese and Arabic
  (FineWeb-2, *verify* subset names).

## 5. Training

Shared settings:

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas 0.9 and 0.95, eps 1e-8, weight decay 0.1 on matrices only |
| Peak learning rate | 6e-4 (scaled per stage for `hnet`, section 3.3) |
| Schedule | warmup-stable-decay: 10% linear warmup, then constant, then 20% decay (inverse square root, as in the H-Net paper) |
| Gradient clipping | 1.0 global norm |
| Batch | 512 sequences: 2.10M bytes for `byte` and `hnet`, 459k tokens for `bpe` |
| Steps, Tier M | 45B bytes / 2.10M bytes = about 21,450 for all three |
| Precision | bf16 autocast, fp32 master weights and optimizer state |
| Seed | fixed per run, recorded |
| Checkpoints | every 1B bytes seen, plus final |

Gradient accumulation reaches the batch size on whatever hardware is available; record the
micro-batch.

**Comparison axis.** All three models see the same bytes once. They spend different compute.
Record training FLOPs at every checkpoint (section 6.4) so BPB can be plotted against both bytes
seen and FLOPs. Do not claim a winner on one axis only.

Cost *estimate* for Tier M on one H100 at 400 TFLOP/s sustained:

| Model | Training FLOP | Hours |
|---|---|---|
| `bpe` | about 8.5e18 | about 6 |
| `byte` | about 4e19 (attention at 4,096 positions is half of it) | about 28 |
| `hnet` | about 1e19 | about 7 |

Total about 40 GPU-hours plus evaluation; at 2.5 to 3 USD per H100-hour, roughly 100 to 150 USD.
If that is too much, halve Tier M to 22.5B bytes for all three; the comparison stays valid.

H-Net's Mamba-2 layers need the `mamba_ssm` CUDA kernels, so `hnet` cannot train on Apple
silicon. `mamba_ssm` also ships pure-PyTorch reference implementations of the scan; use them only
for CPU unit tests (*verify* the function names in the installed version).

## 6. Evaluation and measurement

### 6.1 Bits per byte

`BPB = (sum of next-unit negative log-likelihoods in nats) / (ln 2 * number of UTF-8 bytes)`,
over the whole held-out set. For `bpe`, sum token NLLs over the same text and divide by its byte
count. Evaluate with a sliding window, stride half the context, scoring only the second half of
each window after the first. Never report perplexity across models.

### 6.2 Downstream

- **HellaSwag**, zero-shot: score each ending by total NLL divided by the ending's UTF-8 byte
  length, for all three models. GPT-2 124M scores about 29% to 31% depending on the
  implementation; the `bpe` model should land in that range (sanity check, not a target).
- **CUTE** (character manipulation): the byte models should win clearly; if not, look for a bug.

### 6.3 Compression by content type (`hnet` only)

On each domain probe set: bytes per chunk (mean and the distribution), and boundary positions
for 20 sampled windows saved for inspection. For `bpe`, record bytes per token on the same sets
for comparison. This table is the input AgentForge needs to budget KV by bytes.

### 6.4 FLOP accounting

One function per model, unit-tested against `torch.utils.flop_counter.FlopCounterMode` on a
single forward pass to within 5%:

- Linear layers: 2 FLOP per multiply-accumulate in the forward pass, times 3 for training.
- Attention scores and values: `4 * n_layers * context * d_model` forward per position, halved
  for causal masking. State whether the halving is applied.
- Mamba-2 layers: count from the layer's projections plus the scan; take the scan from the flop
  counter if it is not attributed.
- `hnet`: outer stages run per byte, the main network per chunk; multiply by the measured
  chunks per byte, not the target.

### 6.5 Inference measurement

One H100 (record the exact SKU, driver, CUDA and PyTorch versions), bf16 weights and caches,
eager mode first; `torch.compile` and CUDA graphs only as separately labelled runs.

| Measurement | Grid |
|---|---|
| Prefill latency | prompt bytes 512, 1,024, 2,048, 4,096 (for `bpe`, the same text tokenised); batch 1, 8, 32 |
| Decode rate | greedy, 512 bytes generated after a 2,048-byte prompt; batch 1, 8, 32; report bytes per second per sequence and in aggregate |
| Peak memory | from `torch.cuda.max_memory_allocated`, per grid point |
| Cache bytes per sequence | sum of the sizes of all inference-cache tensors after the prompt, split into growing KV and fixed state |
| `hnet` stage split | per decode step: encoder, router, main network (only on boundary steps), decoder, in milliseconds; and main-network steps per byte generated |
| Kernel launches per byte | from the PyTorch profiler, for batch 1 decode |

Timing: CUDA events, 10 warm-up runs, then 50 timed runs; report median and 90th percentile.

Correctness gate before any timing: incremental decode with the cache must reproduce full-forward
logits (section 7.1). A fast but wrong cache is the most likely silent bug in this project.

## 7. Tests that must pass

1. **Cache equivalence.** For each model, logits from incremental decoding over 256 steps match
   a single full forward pass, max absolute difference below 1e-3 in fp32 and 2e-2 in bf16.
2. **Router causality.** For `hnet`, perturbing bytes after position `t` leaves `b_1..b_t` and the
   encoder outputs up to `t` unchanged, bit for bit in fp32. This is the property AgentForge's
   prefix caching relies on.
3. **BPB consistency.** On a 1MB text, the `bpe` BPB computed through tokens equals the value from
   summing per-token NLL by hand; for byte models, BPB equals mean next-byte NLL divided by ln 2.
4. **Data alignment.** Byte and BPE shards decode to identical text for the same document range.
5. **FLOP counter** within 5% of `FlopCounterMode` (section 6.4).
6. **Export schema.** Every exported `model-spec.json` validates against the JSON Schema checked
   into the repository (section 9.1).
7. **Reference reproducibility.** Regenerating section 9.3's reference tensors twice gives
   identical files.

## 8. Repository layout

```
bytelm/
  SPEC.md                    this document
  CLAUDE.md                  section 11, for AI assistants
  pyproject.toml             Python 3.11+, managed with uv
  configs/                   one YAML per model and tier (bpe-M.yaml, hnet-tiny-S.yaml, ...)
  src/bytelm/
    data/                    download, shard, align, domain probe sets
    models/                  block wrapper, isotropic model, hnet adapter
    train/                   loop, optimizer groups with per-stage learning rates, checkpointing
    eval/                    bpb, hellaswag, cute, compression probes
    infer/                   cache, decode loop, timing harness
    flops/                   per-model FLOP accounting
    export/                  model-spec writer, JSON Schema, reference tensors
  scripts/                   thin entry points, one per milestone
  tests/                     section 7
  results/                   committed: tables and plots; never checkpoints
  exports/                   committed: section 9 files for each model
```

Checkpoints and shards live outside the repository (object storage or a local data directory),
referenced by path and hash.

## 9. Export contract for AgentForge

This is the only interface between the two repositories. Version it; AgentForge will reject an
unknown `schema_version`.

### 9.1 `exports/<model-id>/model-spec.json`

```json
{
  "schema_version": "0.1",
  "model_id": "hnet",
  "family": "hnet",
  "input_unit": "byte",
  "vocab_size": 256,
  "context_bytes": 4096,
  "params": { "total": 0, "embedding": 0, "per_stage": [0, 0, 0] },
  "stages": [
    {
      "index": 0,
      "role": "encoder",
      "d_model": 512,
      "positions_per_byte": 1.0,
      "layers": [
        { "kind": "mamba2", "count": 4, "d_state": 128, "d_conv": 4, "expand": 2,
          "n_heads": 0, "params_per_layer": 0,
          "state_bytes_per_seq_per_layer": 0 }
      ]
    },
    {
      "index": 1,
      "role": "main",
      "d_model": 768,
      "positions_per_byte": { "prose": 0.0, "code": 0.0, "logs": 0.0, "zh": 0.0, "ar": 0.0 },
      "layers": [
        { "kind": "attention", "variant": "mha", "count": 12, "n_heads": 12, "n_kv_heads": 12,
          "head_dim": 64, "window": null, "params_per_layer": 0,
          "kv_bytes_per_position_per_layer": 0 },
        { "kind": "mlp", "variant": "swiglu", "count": 12, "d_intermediate": 2048,
          "params_per_layer": 0 }
      ]
    },
    { "index": 2, "role": "decoder", "d_model": 512, "positions_per_byte": 1.0,
      "layers": [ { "kind": "mamba2", "count": 4, "...": "as the encoder" } ] }
  ],
  "routing": { "target_ratio": 6, "ratio_loss_weight": 0.03 },
  "precision": { "weights": "bf16", "cache": "bf16" },
  "flops_per_byte_forward": { "prose": 0.0, "code": 0.0 },
  "training": { "bytes_seen": 0, "flops": 0, "bpb_heldout": 0.0, "git_commit": "" },
  "measured": {
    "device": "h100-sxm-80gb",
    "software": { "torch": "", "cuda": "", "driver": "" },
    "prefill": [ { "batch": 1, "prompt_bytes": 4096, "ms_p50": 0.0, "ms_p90": 0.0, "peak_mem_gb": 0.0 } ],
    "decode": [ { "batch": 1, "context_bytes": 2048, "bytes_per_s_p50": 0.0,
                  "stage_ms": { "encoder": 0.0, "router": 0.0, "main": 0.0, "decoder": 0.0 },
                  "main_steps_per_byte": 0.0, "kernel_launches_per_byte": 0.0 } ],
    "cache_bytes_per_seq": { "kv": 0, "fixed_state": 0, "at_context_bytes": 2048 }
  }
}
```

Rules:
- `bpe` and `byte` have one stage with `role: "main"`; for `bpe`, `positions_per_byte` is the
  measured tokens per byte per domain.
- Layer `kind` values in version 0.1: `attention`, `mlp`, `mamba2`. A new kind needs a schema
  version bump.
- Every quantity is measured or computed by code in this repository, never typed by hand.
- Zeros above are placeholders for the schema example only.

### 9.2 `exports/<model-id>/results.md`

The BPB, downstream and compression tables for that model, and a plot of held-out BPB against
training FLOPs with all three models on it.

### 9.3 `exports/<tiny-model-id>/reference/`

For the tiny variants only, generated in fp32 on CPU with deterministic algorithms enabled:

- `weights.safetensors`: all parameters, fp32, names as in the PyTorch module.
- `input.bin`: a fixed 256-byte UTF-8 prompt (128 BPE tokens for `bpe-tiny`), stored raw.
- `outputs.safetensors`: final logits for every position; hidden states after every layer;
  for `hnet-tiny`, `p_t`, `b_t` and the chunk indices.
- `decode.safetensors`: logits for 32 greedy decode steps from the prompt, with the cache.
- `manifest.json`: SHA-256 of every file, PyTorch version, and the exact module code commit.

AgentForge's M2 backend will reproduce these within a tolerance it defines.

## 10. Milestones

| ID | Scope | Done when |
|---|---|---|
| P0 | Repository skeleton, data download, aligned byte and BPE shards for Tier S, held-out set and probe sets | Test 4 passes; held-out hash committed |
| P1 | Models and training loop; tiny variants trained on Tier S | Tests 1, 2, 5 pass; tiny losses fall smoothly |
| P2 | Full-size models on Tier S, evaluation pipeline | All three reach a held-out BPB below 1.5 *estimate*; test 3 passes; HellaSwag runs end to end |
| P3 | Tier M training | Three final checkpoints, BPB against FLOPs plot, no unexplained loss spikes |
| P4 | Evaluation | BPB, HellaSwag, CUTE and compression tables for all three |
| P5 | Inference measurement | Section 6.5 grid complete, test 1 re-run on final checkpoints |
| P6 | Export | Tests 6 and 7 pass; `exports/` complete for all six models; hand back to AgentForge |

P0 to P2 can run on modest hardware; only P3 and P5 need the rented H100.

## 11. Instructions for AI assistants (copy into `CLAUDE.md`)

- Read `SPEC.md` before changing code. If a change contradicts it, update the spec in the same
  commit and say why.
- Compare models in bits per byte only. Never compare perplexity across tokenizers.
- Every number in `results/` or `exports/` must be produced by a script in this repository; do
  not type numbers by hand.
- Do not change the shared Transformer block for one model only; the three models must differ
  only in their input interface.
- Do not add custom CUDA kernels or speculative decoding to the inference path; measure the
  reference implementation.
- Before timing anything, run the cache-equivalence test on that checkpoint.
- Record hardware and software versions with every measurement.
- When a published hyperparameter is uncertain, mark it in the config with a comment and a
  `verify` note rather than guessing silently.
- Keep checkpoints and shards out of git.

## 12. Extensions after version 0.1

In rough priority order, each a schema-compatible addition:

1. **Hybrid flat byte model:** `byte` with Gated DeltaNet in three of every four layers, following
   arXiv 2610.05978; adds a `gated_deltanet` layer kind. Tests whether a flat byte model with fixed
   state is competitive on inference cost.
2. **Two-stage H-Net** with outer ratios near 4 and 2; adds a second chunking stage.
3. **Robustness suite:** graded bit flips, byte deletions, UTF-8 misalignment and OCR-style
   substitutions on the held-out set; BPB and HellaSwag against corruption rate.
4. **Bolmo 1B against OLMo 2 1B:** the same inference measurement on a public pair at larger
   scale, as an external check on the 124M results.

## 13. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| `hnet` reference code has no training loop | P1 takes longer | Write the loop here; reuse only the modules |
| Router collapses to all or no boundaries | `hnet` degenerates to `byte` or to a tiny model | Log `F` and `G` every step; ratio loss weight 0.03 as published; stop the run if `F` leaves 0.1 to 0.35 |
| Per-stage learning rates wrong | Unstable or slow `hnet` training | Section 3.3 values first, then 2.0 and 1.0 on Tier S |
| `byte` attention at 4,096 positions dominates cost | Tier M overruns budget | Use PyTorch SDPA flash attention; halve Tier M if needed |
| Cache bug | Inference numbers meaningless | Test 1 is a gate before every timing run |
| Data misalignment between byte and BPE shards | Unfair comparison | Test 4, and document ranges stored in shard headers |

## 14. References gathered for this spec

Status column: **read** means the primary source was opened and the facts used here were taken
from it; **search** means the item appeared in search results only and its claims are the
search summary's, not checked; **cited** means named by another source and not opened.

### 14.1 Starting point: Percy Liang and CS336

| Source | Status | Used for |
|---|---|---|
| CS336 lecture 1 source, Spring 2025: https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py | read | "Tokenization is a necessary evil"; tokenizer-free models "not yet scaled up to the frontier" |
| HELM, Liang et al. 2022, TMLR 2023: https://arxiv.org/abs/2211.09110 | search | Bits per byte as a tokenizer-invariant metric (section 6.1) |
| CS336 Spring 2026 lecture 1 summaries: https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-JuoVZk/ , https://yuqing-zhou.github.io/learning-blog/cs336/2026/04/24/cs336-lec01-2/ , https://medium.com/@fahey_james/building-language-models-from-scratch-why-stanfords-cs336-lecture-1-on-tokenization-still-matters-63b0a79fbc58 , https://realwujing.github.io/linux/drivers/gpu/stanford-cs336/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E4%BB%8E0%E5%88%B01%EF%BD%9C%E7%AC%AC%E4%B8%80%E8%AE%B2%EF%BC%9A%E6%A6%82%E8%BF%B0%E5%92%8Ctokenization | search | Secondary only; the 2026 edition keeps the same framing |

Finding: no byte-level or tokenizer paper by Percy Liang was found. The "Liang et al., 2023"
cited for the vocabulary bottleneck is Davis Liang (Meta), not Percy Liang.

### 14.2 Architecture used in this spec

| Source | Status | Used for |
|---|---|---|
| H-Net, Hwang, Wang, Gu, "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", 2025: https://arxiv.org/abs/2507.07955 (HTML v2: https://arxiv.org/html/2507.07955v2) | read | Router, downsampler, smoothing, STE upsampler, ratio loss, alpha 0.03, N = 6, per-stage learning rates, network RMSNorm, 8,192 bytes against 1,792 tokens, causality statement, generation procedure (sections 3.2, 3.3, 5) |
| H-Net reference code, MIT: https://github.com/goombalab/hnet | read | Requirements (PyTorch 2.5.1 or later, `mamba_ssm` from source), inference only, `generate.py` |
| H-Net config `hnet_1stage_L`: https://raw.githubusercontent.com/goombalab/hnet/main/configs/hnet_1stage_L.json | read | Config format in section 3.1 |
| H-Net in Hugging Face Transformers, PR: https://github.com/huggingface/transformers/pull/50167 | search | Checkpoints `cartesia-ai/hnet_1stage_L` and `hnet_2stage_XL` loadable through Transformers |
| Cartesia, "From Tokens to Bytes I: Scaling Hierarchical Networks", 2026-10-28: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1 | read | One-stage H-Net 1.28x compute efficiency; two-stage 3.67B matches SmolLM3-3B with about 1T vs 11T tokens; 4.57 bytes per BPE token; no inference discussion |
| Cartesia, "From Tokens to Bytes II: Learning from Raw Data", 2026-10-28: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2 | read | Compute multipliers vs H-Net: BLT-like 1.84x, BPE 2.06x, BPE without pretokenization 3.82x, byte Transformer 9.60x |
| dnaHNet, 2026: https://arxiv.org/pdf/2602.10603 | search | H-Net applied to genomic sequences |

### 14.3 Other byte-level models and their inference

| Source | Status | Used for |
|---|---|---|
| Bolmo, Minixhofer et al., "Byteifying the Next Generation of Language Models", 2025, v2 2026: https://arxiv.org/abs/2512.15586 (HTML v2: https://arxiv.org/html/2512.15586v2) | read | Bolmo 7B from Olmo 3 7B, 1B from OLMo 2 1B; two-stage conversion; about 125 vs 150 bytes/s decode; crossover above 6.6 bytes per patch (extension 4) |
| Wang, Luo, Zhang, Wu, "Byte Language Models: Scaling, Emergent Abstractions, and Information Allocation", 2026-10-05: https://arxiv.org/abs/2610.05978 (HTML: https://arxiv.org/html/2610.05978v1) | read | Flat byte Transformers with hybrid Gated DeltaNet; win at matched parameters, lose at matched FLOPs; speculative decoding gain is in tokens, not bytes (extension 1) |
| Byte Latent Transformer (BLT), Pagnoni et al., ACL 2025: https://aclanthology.org/2025.acl-long.453.pdf | search | Entropy patching; source of hash n-gram embeddings |
| Fast Byte Latent Transformer, Kallini et al., 2026: https://arxiv.org/abs/2605.08044 | read | BLT-D, BLT-S, BLT-DV; over 50% lower generation memory-bandwidth cost than BLT |
| MrT5, Kallini, Murty, Manning, Potts, Csordás, ICLR 2025: https://arxiv.org/abs/2410.20771 | search | Learned delete gate in ByT5, up to 75% shorter sequences |
| SpaceByte, Slagle, 2024: https://arxiv.org/abs/2404.14408 | search | Byte Transformers need about 10x subword training FLOPs; whitespace patching |
| ByT5, MegaByte, T-FREE | cited | Named in CS336 lecture 1 as tokenizer-free approaches; not opened |
| "Bridging the Gap for Tokenizer-Free Language Models", 2019: https://arxiv.org/abs/1908.10322 | search | Early 40-layer byte Transformer on lm1b |
| Hierarchical autoregressive transformers, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/file/7f69eee2fafab3743a3d2ae5981c949f-Paper-Conference.pdf | search | Byte and word hierarchical model |
| ByteFlow, 2026: https://www.alphaxiv.org/abs/2603.03583 | search | Adaptive byte compression without a tokenizer |
| Byte-level vs BPE, autoregressive and masked diffusion, 2026: https://arxiv.org/html/2605.12928v1 | search | Autoregressive byte models approach BPE parity from 48M to 1.2B |
| Byte-level distillation from token teachers, 2026: https://www.alphaxiv.org/abs/2609.12303 | search | Distillation of byte students |
| "The Efficiency Gap in Byte Modeling": https://arxivlens.com/paperview/details/the-efficiency-gap-in-byte-modeling-3996-5d9ca417 | search | Not reviewed |
| Kronecker Embeddings, 2026: https://arxiv.org/pdf/2605.29459 | search | Byte-level structured token embeddings |

### 14.4 Tokenization research relevant to the comparison

| Source | Status | Used for |
|---|---|---|
| Exact Byte-Level Probabilities, ICLR 2025: https://openreview.net/pdf?id=zGej22CBnS | search | Converting tokenised model probabilities to byte probabilities (BPB for `bpe`) |
| Sampling from Your Language Model One Byte at a Time, 2025: https://arxiv.org/abs/2506.14123 | search | Prompt boundary problem for BPE models |
| Broken Tokens? Non-canonical tokenizations, 2025: https://arxiv.org/pdf/2506.19004 | search | Robustness of token models to unusual segmentations |
| Language Model Tokenizers Introduce Unfairness Between Languages, 2023: https://arxiv.org/pdf/2305.15425 | search | Per-language compression differences (Chinese and Arabic probe sets) |
| Back to Bytes: Revisiting Tokenization Through UTF-8, 2025: https://arxiv.org/abs/2510.16987 | search | UTF8Tokenizer, a minimal byte tokenizer |
| Bit-level BPE: Below the byte boundary, 2025: https://arxiv.org/pdf/2506.07541 | search | Sub-byte merging of UTF-8 high bits |
| Decoupling the Benefits of Subword Tokenization, 2026: https://arxiv.org/abs/2604.27263 | search | Reproducing subword effects inside byte training |
| Compute Optimal Tokenization, 2026: https://arxivlens.com/PaperView/Details/compute-optimal-tokenization-6282-8edb77ae (arXiv 2605.01188) | search | Tokenizer choice under compute budgets |
| TokEval, 2026: https://arxiv.org/pdf/2608.18062 | search | Tokenizer evaluation suite |
| How Long Is a Piece of String?, 2026: https://arxiv.org/abs/2601.11518 | search | Tokenizer variation across models and domains |
| How Much is Enough? Tokenizer training data, 2025: https://arxiv.org/pdf/2502.20273 | search | Diminishing returns of tokenizer training data |
| The Foundations of Tokenization, 2024: https://arxiv.org/html/2407.11606v3 | search | Formal treatment of tokenization |
| PathPiece, EMNLP 2024: https://aclanthology.org/2024.emnlp-main.40.pdf | search | Fewer tokens do not imply better downstream results; 64 trained models |
| snimu, "tokens vs bytes", 2025: https://snimu.github.io/2025/03/07/tokens-vs-bytes.html | search | Small-scale mixed byte and token experiments |

### 14.5 Data and baselines

| Source | Status | Used for |
|---|---|---|
| FineWeb-Edu, `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | cited | Training corpus (section 4); also the H-Net paper's corpus |
| The Stack smol, `bigcode/the-stack-smol` | cited, verify | Code probe set |
| FineWeb-2 | cited, verify | Chinese and Arabic probe sets |
| SWE-agent trajectories on Hugging Face | cited, verify | Tool and shell log probe set |
| Karpathy, llm.c and nanoGPT GPT-2 124M reproduction | cited, from memory | 10B FineWeb tokens in about 90 minutes on 8xA100; AdamW 0.9 and 0.95, learning rate 6e-4, weight decay 0.1; HellaSwag about 29% to 31% |

### 14.6 AgentForge context

| Source | Used for |
|---|---|
| `../research/notes/Byte-level modeling 2026.md` | Field survey, KV and decode arithmetic for the median agent step, project list |
| `0001-approach.md` | AgentForge thesis, simulator regimes, 70B model and H200 figures |
| `../research/notes/AI inference silicon landscape 2026/08-workloads-costmodels-correctness.md` | TraceLab agent workload shape (119k prefix, 875 appended, 214 generated tokens) |
