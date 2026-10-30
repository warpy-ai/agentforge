# Research notes: AI inference silicon landscape 2026

Per-area research notes gathered on 2026-10-30 for the AgentForge design.
Each file is dated and cites its sources; vendor claims are marked as such.
The synthesized report lives in `../../reports/AI inference silicon landscape 2026.md`.

| File | Area | Written by |
|---|---|---|
| `01-datacenter-gpus.md` | NVIDIA, AMD, Intel datacenter GPUs | research subagent (about 70 web calls, 143 cited URLs) |
| `02-hyperscaler-asics.md` | Google, AWS, Microsoft, Meta, OpenAI, Apple, Chinese custom silicon | coordinating session |
| `03-startup-silicon.md` | Groq, Cerebras, SambaNova, Tenstorrent, Etched, d-Matrix, Taalas, Positron, MatX and others | coordinating session |
| `04-edge-npus.md` | Apple, Qualcomm, AMD, Intel, Arm NPUs and portable layers | coordinating session |
| `05-compiler-stacks.md` | Tile IRs, graph compilers, search-based and LLM-generated kernels | coordinating session |
| `06-serving-runtimes-and-workloads.md` | Serving engines, inference techniques, agentic workloads, benchmarks | coordinating session |
| `07-rust-ecosystem.md` | Rust frameworks, GPU and accelerator access, compiler infrastructure | coordinating session |
| `08-workloads-costmodels-correctness.md` | Workload facts, simulators and cost models, determinism and verification | coordinating session |

Method note: seven of the eight planned research subagents could not be launched in the
original session, so files 02 to 08 were researched directly by the coordinating session with
about 60 web searches. They are shorter than file 01 and lean on search summaries rather than
full page reads; items marked "(gap)" were not verified against a primary source.
