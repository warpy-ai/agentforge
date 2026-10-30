# Roadmap

| Milestone | Scope | Done when |
|---|---|---|
| M0 Baseline (2026-10) | Research notes and report, design doc, Cargo workspace with IR, machine model, cost model, compiler passes, trace simulator, runtime traits, CLI | `cargo test`, `cargo clippy -D warnings` and `cargo fmt --check` pass; `agentforge sim run` compares LRU and program-aware residency |
| M1 Simulator validation | Loader for the FreeInference block-level trace and TraceLab distributions; calibration of the cost model against published GenZ and LLMCompass numbers; TP communication and prefill/decode interference; channel-aware residency planning (the transfer channel as a scheduled resource, prefetch as an admission decision); a write-through offload baseline; admission control | Simulated TTFT and throughput within 15 percent of published measurements for Llama 3.1 70B on H200 and B200 |
| M2 CPU backend | Tile IR to Cranelift for a tiny transformer; numerics reference implementations for FP8, MXFP4, NVFP4 | A 2-layer model runs end to end with bit-exact comparison against the reference |
| M3 NVIDIA backend | PTX emission and cutile-rs kernels; KV block store with planned lifetimes; agent-aware runtime on one GPU | Replays an agentic trace on real hardware with the simulator's predicted ordering of policies confirmed |
| M4 Tenstorrent backend | tt-metal programs through the host API; Blackhole machine file validated on hardware | Same tiny model and trace replay on a Blackhole card |
| M5 Heterogeneous placement | Prefill-class and decode-class devices in one plan; KV transfer scheduling over links | Simulator and (where hardware allows) measurement of a two-class deployment |
| M6 Numerics verification | Block formats as verified IR types; e-graph rewrites with checked equivalence; batch-invariant kernel mode | Rewrite rules proven or property-tested against reference semantics; deterministic mode passes 1,000-run identity test |
