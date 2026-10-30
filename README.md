# AgentForge

AgentForge is an open-source compiler and runtime, written in Rust, for AI inference silicon.
Its thesis: compile the **agent program** and the **model** together against an explicit,
data-described **machine model**, so that KV-cache residency, prefetch, prefill/decode placement
and kernels are planned from what the agent already knows instead of being guessed by a serving
scheduler one HTTP request at a time.

Status: research and baseline (October 2026). The research report, the design document and a
working trace-replay simulator are in this repository; no hardware backend exists yet.

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
| `agentforge-machine` | TOML machine descriptions: devices, memory tiers, links; bundled `h200`, `b200`, `mi355x`, `blackhole-p150`, `lpx-tier` |
| `agentforge-cost` | Roofline-style cost model for prefill, decode and transfers |
| `agentforge-compile` | KV liveness, Belady-style residency planning, prefill/decode placement |
| `agentforge-sim` | Discrete-event trace replay comparing an LRU baseline with the program-aware plan |
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
cargo run -p agentforge-cli -- sim run --model llama-3.1-70b --machine h200 --tp 8 --sessions 64 --steps 20 --policy both --seed 1
```

## First simulation results

The M0 simulator compares a reactive LRU baseline with the program-aware residency plan on
synthetic coding-agent sessions (110k-token median prefixes, per-tool latency for file reads,
builds and human turns) over Llama 3.1 70B on eight H200s:

- with FP8 KV and a 2 TB host tier the plan cuts pressure evictions by 30 to 40 percent in
  every seed and improves foreground KV waits and p90 time to first token in most seeds;
- mean latency barely moves under pressure, because queueing for device memory dominates and
  no residency policy changes how much KV the active sessions need;
- with FP16 KV at 110k tokens decode traffic alone saturates the device, and with scarce, slow
  tiers the plan loses by saturating the transfer channel.

Full tables, seeds and the reading of the results are in the design document's "First simulation
results" section. Reproduce with:

```bash
cargo run --release -p agentforge-cli -- sim run --machine h200 --tp 8 --kv-precision fp8 --sessions 80 --steps 20 --policy both --seed 11
```

## Documentation

- [docs/README.md](docs/README.md): index.
- [docs/ROADMAP.md](docs/ROADMAP.md): milestones M0 to M6.
- [docs/research/notes/](docs/research/notes/): the eight per-area research notes with sources.

## License

Apache-2.0. See [LICENSE](LICENSE).
