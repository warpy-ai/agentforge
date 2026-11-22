# AgentForge

AgentForge is an open-source compiler and runtime, written in Rust, for AI inference silicon.
Its thesis: compile the **agent program** and the **model** together against an explicit,
data-described **machine model**, so that KV-cache residency, prefetch, prefill/decode placement
and kernels are planned from what the agent already knows instead of being guessed by a serving
scheduler one HTTP request at a time.

Status: research, baseline and simulator validation (M0 and M1, October to November 2026). The
research report, the design document, a trace-replay simulator and a cost model calibrated
against published TensorRT-LLM throughput are in this repository; no hardware backend exists yet.

## Why

- Agentic workloads dominate inference growth. A median coding-agent step carries about 119k
  prefix tokens, appends 875 and generates 214; prefix-cache hit rates are 95 to 96 percent, and
  each miss costs seconds of recompute (TraceLab, June 2026).
- Silicon split along the prefill/decode axis in 2026 (Vera Rubin with Groq 3 LPX racks,
  Trainium with Cerebras), but placement across device classes is configured by hand.
- Tile-level IRs converged but are vendor-owned; the neutral layers changed owners (Modular to
  Qualcomm, Groq to NVIDIA); numerics diverge (NVFP4 versus OCP MX); no compiler sees the agent.

The full argument, with sources, is in
[docs/research/reports/AI inference silicon landscape 2026.md](docs/research/reports/AI%20inference%20silicon%20landscape%202026.md)
and the design in [docs/design/0001-approach.md](docs/design/0001-approach.md).

## Layout

| Crate | Role |
|---|---|
| `agentforge-ir` | Numerics as types (including block-scaled FP4/FP6/FP8 formats), model specs, the Agent Program IR, tile IR skeleton |
| `agentforge-machine` | TOML machine descriptions: devices, memory tiers, links; bundled `h100`, `h200`, `b200`, `mi355x`, `blackhole-p150`, `lpx-tier` |
| `agentforge-cost` | Per-operator roofline cost model for prefill, decode, tensor-parallel communication and transfers, calibrated against 33 published TensorRT-LLM rows |
| `agentforge-compile` | KV liveness, Belady-style residency planning that prices the transfer channel, prefill/decode placement |
| `agentforge-sim` | Discrete-event trace replay with prefill/decode interference and admission control; LRU, write-through and program-aware policies; loaders for the FreeInference and TraceLab traces |
| `agentforge-runtime` | KV block store, transport and executor traits |
| `agentforge-cli` | The `agentforge` binary |

## Quickstart

```bash
cargo build --workspace
```

```bash
cargo test --workspace
```

```bash
cargo run -p agentforge-cli -- machines list
```

```bash
cargo run -p agentforge-cli -- cost prefill --model llama-3.1-70b --machine h200 --tp 8 --tokens 119000 --prefix 0
```

```bash
cargo run -p agentforge-cli -- plan residency --machine h200 --tp 8 --prefix-tokens 119000 --wait-median 45
```

```bash
cargo run -p agentforge-cli -- sim run --model llama-3.1-70b --machine h200 --tp 8 --sessions 64 --steps 20 --policy all --seed 1
```

```bash
cargo run --release -p agentforge-cli -- cost calibrate
```

```bash
cargo run --release -p agentforge-cli -- sim run --machine h200 --tp 8 --kv-precision fp8 --sessions 40 --program tracelab --policy all --seed 1
```

To replay a public trace, decompress it (`zstd -d` or `gunzip`) and pass `--trace <file>
--trace-format freeinference` or `--trace-format tracelab`.

## Simulation results

The simulator compares three KV residency policies on synthetic coding-agent sessions (110k-token
median prefixes, per-tool latency for file reads, builds and human turns) over Llama 3.1 70B on
eight H200s: `lru` (reactive), `write-through` (copies new KV to the host during every tool wait,
so evictions write nothing) and `plan` (program-aware: spills during the waits the agent program
predicts and prefetches before the predicted return). After the M1 fixes, which count time
spent waiting for device memory in TTFT and let prefills stall running decodes:

- with FP8 KV and a 2 TB host tier the plan evicts 25 to 39 percent less than LRU under pressure
  in every seed, has the lowest p90 step latency in every seed, and beats LRU's p99 TTFT in
  three of four seeds;
- write-through has the lowest foreground KV wait in every pressured regime, and the plan's p90
  TTFT beats LRU in only two of four seeds; mean TTFT is within 0.6 s across all three policies,
  because queueing for device memory and the prefill queue dominate it;
- with a scarce, slow host tier the plan loses to both baselines: its spill writes delay other
  sessions' foreground fetches, and the planner does not yet price that.

The cost model behind the simulator predicts 33 published TensorRT-LLM throughput rows (Llama
3.3 70B on H200, B200 and H100) within 12.1 percent, and the knobs fitted on H200 and B200 alone
predict the H100 rows within 12.2 percent. Tables, seeds and the reading of the results are in
the design document's "M1 simulator validation" section; `scripts/m1-regimes.sh` reproduces them.

## Documentation

- [docs/README.md](docs/README.md): index.
- [docs/ROADMAP.md](docs/ROADMAP.md): milestones M0 to M6.
- [docs/research/notes/](docs/research/notes/): the eight per-area research notes with sources.

## License

Apache-2.0. See [LICENSE](LICENSE).
