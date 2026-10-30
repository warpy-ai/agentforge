// Copy and numbers for the landing page. Sources: README.md, docs/design/0001-approach.md,
// docs/ROADMAP.md and crates/agentforge-machine/machines/*.toml in the repository root.

export const repo = 'https://github.com/warpy-ai/agentforge';

export const links = {
  repo,
  designDoc: `${repo}/blob/main/docs/design/0001-approach.md`,
  report: `${repo}/blob/main/docs/research/reports/AI%20inference%20silicon%20landscape%202026.md`,
  notes: `${repo}/tree/main/docs/research/notes/AI%20inference%20silicon%20landscape%202026`,
  roadmap: `${repo}/blob/main/docs/ROADMAP.md`,
  contributing: `${repo}/blob/main/CONTRIBUTING.md`,
  license: `${repo}/blob/main/LICENSE`,
  machines: `${repo}/tree/main/crates/agentforge-machine/machines`,
  docs: `${repo}/tree/main/docs`,
};

export const nav = [
  { label: 'Thesis', href: '#thesis' },
  { label: 'Architecture', href: '#architecture' },
  { label: 'Results', href: '#results' },
  { label: 'Machines', href: '#machines' },
  { label: 'Roadmap', href: '#roadmap' },
];

export const heroStats = [
  { value: '119K', label: 'Prefix tokens carried' },
  { value: '875', label: 'Tokens appended' },
  { value: '214', label: 'Tokens generated' },
  { value: '95–96%', label: 'Prefix-cache hit rate' },
];

export const tickerItems = [
  'KV residency',
  'Prefetch',
  'Prefill / decode placement',
  'Block-scaled FP4 · FP6 · FP8',
  'TOML machine models',
  'Trace-replay simulator',
  'No MLIR dependency',
  'Written in Rust',
  'Apache-2.0',
];

export const problems = [
  {
    index: '01.1',
    tag: 'Workloads',
    title: 'Agentic workloads dominate inference growth',
    body:
      'A median coding-agent step carries about 119k prefix tokens, appends 875 and generates 214. Prefix-cache hit rates are 95 to 96 percent, and each miss, caused by a long tool wait, costs seconds of recompute.',
    source: 'Source: TraceLab, June 2026',
  },
  {
    index: '01.2',
    tag: 'Silicon',
    title: 'Hardware split along the prefill/decode axis',
    body:
      'Vera Rubin with Groq 3 LPX racks, Trainium with Cerebras: heterogeneous prefill and decode devices shipped in 2026. Placement across device classes is still configured by hand.',
    source: 'Source: research notes 01 to 03',
  },
  {
    index: '01.3',
    tag: 'Software',
    title: 'Tile IRs converged, and nobody sees the agent',
    body:
      'Tile-level IRs are vendor-owned, the neutral layers changed owners (Modular to Qualcomm, Groq to NVIDIA), and NVFP4 versus OCP MX numerics diverge. No compiler looks past one model iteration.',
    source: 'Source: research note 05',
  },
];

export type LayerStatus = 'baseline' | 'skeleton' | 'traits' | 'planned';

export const layers: {
  id: string;
  name: string;
  crate: string;
  role: string;
  status: LayerStatus;
}[] = [
  {
    id: 'L1',
    name: 'Agent Program IR',
    crate: 'agentforge-ir::agent',
    role: 'Steps are LLM calls with structured prompt blocks marked stable or not; transitions are tool calls with latency models, probabilistic branches, or end. KV block liveness is derived from it.',
    status: 'baseline',
  },
  {
    id: 'L2',
    name: 'Model graph IR',
    crate: 'agentforge-ir::model',
    role: 'Attention and MoE variants as structured descriptions (dense, grouped-query, latent, sliding-window, compressed); numerics as first-class types, including block-scaled formats.',
    status: 'baseline',
  },
  {
    id: 'L3',
    name: 'Tile IR',
    crate: 'agentforge-ir::tile',
    role: 'Tiles with explicit memory spaces (global, shared, register, tensor memory), async copies and MMA operations. Async protocols are to be encoded in Rust types so misuse fails to compile.',
    status: 'skeleton',
  },
  {
    id: 'L4',
    name: 'Machine model',
    crate: 'agentforge-machine',
    role: 'TOML descriptions: devices with class, compute per precision, memory capacity and bandwidth, on-chip SRAM; memory tiers with capacity, bandwidth and latency; links.',
    status: 'baseline',
  },
  {
    id: 'L5',
    name: 'Cost model and simulator',
    crate: 'agentforge-cost · agentforge-sim',
    role: 'Roofline-style phase costs for prefill, decode and transfers, and a discrete-event replay of agent sessions over the machine description.',
    status: 'baseline',
  },
  {
    id: 'L6',
    name: 'Compiler passes',
    crate: 'agentforge-compile',
    role: 'KV liveness, residency planning (Belady-style cost comparison over tiers), prefill/decode placement. E-graph rewrites of the model graph and tile programs come next.',
    status: 'baseline',
  },
  {
    id: 'L7',
    name: 'Backends',
    crate: 'PTX · CUDA Tile IR · LLVM AMDGPU · SPIR-V · MSL · tt-metal · Cranelift',
    role: 'Emit public, stable low-level formats. No MLIR dependency in the baseline.',
    status: 'planned',
  },
  {
    id: 'L8',
    name: 'Runtime',
    crate: 'agentforge-runtime',
    role: 'Executor, KV block store with lifetimes from the plan, transports between tiers and devices; memory-safe by construction.',
    status: 'traits',
  },
];

export const crates = [
  'agentforge-ir',
  'agentforge-machine',
  'agentforge-cost',
  'agentforge-compile',
  'agentforge-sim',
  'agentforge-runtime',
  'agentforge-cli',
];

export const results = {
  setup: [
    ['Model', 'llama-3.1-70b · FP8 weights · FP8 KV'],
    ['Machine', 'h200-sxm ×8 · 945 GB KV budget · 2 TB host tier'],
    ['Trace', '80 synthetic coding-agent sessions · 1719 steps · seed 11'],
    ['Policies', 'lru (reactive, device until pressure) vs plan (program-aware)'],
  ],
  // Regime D, seed 11. Bars are indexed to the LRU value.
  rows: [
    { metric: 'TTFT p90', unit: 's', lru: 33.224, plan: 22.96, fmt: 1 },
    { metric: 'Step latency p90', unit: 's', lru: 36.35, plan: 27.82, fmt: 1 },
    { metric: 'Foreground KV wait', unit: 's', lru: 20.4, plan: 16.5, fmt: 1 },
    { metric: 'Pressure evictions', unit: '', lru: 188, plan: 114, fmt: 0 },
    { metric: 'TTFT mean', unit: 's', lru: 8.398, plan: 8.298, fmt: 2 },
  ],
  table: [
    ['TTFT mean (s)', '8.398', '8.298'],
    ['TTFT p50 (s)', '0.063', '0.063'],
    ['TTFT p90 (s)', '33.224', '22.960'],
    ['TTFT p99 (s)', '112.230', '113.058'],
    ['Step latency p50 (s)', '1.71', '1.73'],
    ['Step latency p90 (s)', '36.35', '27.82'],
    ['Step latency p99 (s)', '113.82', '115.43'],
    ['Foreground KV wait (s)', '20.4', '16.5'],
    ['Fetched in foreground (GB)', '3726', '3681'],
    ['Prefetched (GB)', '0', '3777'],
    ['Spilled during waits (GB)', '0', '5229'],
    ['Pressure evictions', '188', '114'],
    ['Capacity waits', '1954', '2107'],
    ['Device busy (s)', '1716.2', '1710.5'],
    ['Makespan (s)', '5725.7', '5727.1'],
  ],
  seeds: 'Seed sweep (1, 11, 23, 42): pressure evictions fall 30 to 40 percent in every seed; foreground KV waits fall in three of four; p90 TTFT improves in two of four.',
  readings: [
    {
      tag: 'Where it holds',
      body: 'With a fast, large host tier the plan cuts pressure evictions by 30 to 40 percent in every seed and lowers foreground KV waits in three of four. The information doing the work is which tool the agent called: a policy that only saw the blended wait distribution lost to LRU.',
    },
    {
      tag: 'What it does not fix',
      body: 'Mean TTFT barely moves in any pressured regime, because the dominant cost is queueing for device memory. No residency policy changes how much KV the active sessions need; admission control, smaller KV and placement onto bigger devices do.',
    },
    {
      tag: 'Where it loses',
      body: 'With scarce, slow tiers (a 128 GB host tier, NVMe writing at 24 GB/s) every extra transfer competes for one channel. The plan saturates it and loses to LRU, which is why channel-aware planning is the first M1 item.',
    },
  ],
  // First-order arithmetic for the median step on a 70B grouped-query model, FP16 KV.
  arithmetic: {
    maxSeconds: 30,
    rows: [
      { label: 'Recompute after a cache miss', detail: '≈1.7e16 FLOP', from: 6, to: 30, text: '6 to 30 s', emphasis: true },
      { label: 'Move from one NVMe drive', detail: '38 GB', from: 0, to: 5, text: '≈5 s', emphasis: false },
      { label: 'Move from host DRAM over PCIe 5', detail: 'one x16 link', from: 0, to: 0.8, text: '≈0.8 s', emphasis: false },
      { label: 'Move over NVLink', detail: '400 GB/s', from: 0, to: 0.1, text: '≈0.1 s', emphasis: false },
    ],
  },
};

export const machines = [
  {
    file: 'h200',
    vendor: 'NVIDIA H200 SXM',
    count: '×8',
    class: 'gpu',
    memory: '141 GB HBM3e',
    bandwidth: '4.8 TB/s',
    sram: '50 MB',
    fp8: '1979 TFLOPS',
    link: 'NVLink 4 · 900 GB/s',
    tiers: 'host 2 TB · NVMe 30 TB',
    note: 'No FP4 tensor cores; FP4 weights run through the FP8 path.',
    tone: 'paper',
  },
  {
    file: 'b200',
    vendor: 'NVIDIA HGX B200',
    count: '×8',
    class: 'gpu',
    memory: '180 GB HBM3e',
    bandwidth: '8.0 TB/s',
    sram: '164 MB',
    fp8: '4500 TFLOPS',
    link: 'NVLink 5 · 1800 GB/s',
    tiers: 'host 2 TB · NVMe 30 TB',
    note: 'NVFP4 at 9000 TFLOPS; E4M3 scales per 16 elements.',
    tone: 'dark',
  },
  {
    file: 'mi355x',
    vendor: 'AMD Instinct MI355X',
    count: '×8',
    class: 'gpu',
    memory: '288 GB HBM3e',
    bandwidth: '8.0 TB/s',
    sram: '296 MB',
    fp8: '5000 TFLOPS',
    link: 'Infinity Fabric · 1000 GB/s',
    tiers: 'host 2 TB · NVMe 30 TB',
    note: 'OCP MX block scaling only; NVFP4 falls back to the MXFP4 rate.',
    tone: 'paper',
  },
  {
    file: 'blackhole-p150',
    vendor: 'Tenstorrent Blackhole p150',
    count: '×1',
    class: 'gpu',
    memory: '32 GB GDDR6',
    bandwidth: '0.512 TB/s',
    sram: '210 MB',
    fp8: '774 TFLOPS',
    link: '800G ×4 · 400 GB/s',
    tiers: 'host 512 GB',
    note: '140 Tensix cores with 1.5 MB SRAM each. The only fully open accelerator stack.',
    tone: 'dark',
  },
  {
    file: 'lpx-tier',
    vendor: 'SRAM decode tier · Groq 3 LPX shape',
    count: '×256',
    class: 'sram_lpu',
    memory: '0.5 GB SRAM per LPU',
    bandwidth: '150 TB/s',
    sram: '500 MB',
    fp8: '1230 TFLOPS',
    link: 'Rack fabric · 2500 GB/s',
    tiers: 'DDR5 pool 12 TB',
    note: 'Modelling target only; the hardware is closed.',
    tone: 'paper',
  },
];

export const machineExcerpt = `name = "h200"
description = "NVIDIA H200 SXM node: eight GPUs on NVLink 4, PCIe Gen5 host links, host DRAM and NVMe tiers"

[[devices]]
name = "h200-sxm"
class = "gpu"
count = 8
memory_gb = 141
memory_bw_tbps = 4.8
on_chip_sram_mb = 50

[devices.compute_tflops]
fp16 = 989
fp8 = 1979

[[tiers]]
name = "host_dram"
capacity_gb = 2000
read_bw_gbps = 50
write_bw_gbps = 50
latency_us = 20

[[links]]
name = "nvlink4"
bw_gbps = 900
latency_us = 2`;

export const quickstart = [
  { comment: 'build every crate', cmd: 'cargo build --workspace' },
  { comment: 'run the test suite', cmd: 'cargo test --workspace' },
  { comment: 'list the bundled machine descriptions', cmd: 'cargo run -p agentforge-cli -- machines list' },
  {
    comment: 'cost a 119k-token prefill on eight H200s',
    cmd: 'cargo run -p agentforge-cli -- cost prefill --model llama-3.1-70b --machine h200 --tp 8 --tokens 119000 --prefix 0',
  },
  {
    comment: 'decide residency for a prefix across a 45 s median tool wait',
    cmd: 'cargo run -p agentforge-cli -- plan residency --machine h200 --tp 8 --prefix-tokens 119000 --wait-median 45',
  },
  {
    comment: 'replay 64 synthetic sessions and compare both policies',
    cmd: 'cargo run -p agentforge-cli -- sim run --model llama-3.1-70b --machine h200 --tp 8 --sessions 64 --steps 20 --policy both --seed 1',
  },
];

export const roadmap = [
  {
    id: 'M0',
    title: 'Baseline',
    when: '2026-10',
    scope: 'Research notes and report, design doc, Cargo workspace with IR, machine model, cost model, compiler passes, trace simulator, runtime traits, CLI.',
    done: 'cargo test, clippy and fmt pass; sim run compares LRU and program-aware residency.',
    status: 'current',
  },
  {
    id: 'M1',
    title: 'Simulator validation',
    when: 'next',
    scope: 'FreeInference and TraceLab trace loaders, calibration against GenZ and LLMCompass, TP interference, channel-aware residency, write-through baseline, admission control.',
    done: 'Simulated TTFT and throughput within 15 percent of published measurements for Llama 3.1 70B on H200 and B200.',
    status: 'planned',
  },
  {
    id: 'M2',
    title: 'CPU backend',
    when: 'planned',
    scope: 'Tile IR to Cranelift for a tiny transformer; reference numerics for FP8, MXFP4, NVFP4.',
    done: 'A 2-layer model runs end to end, bit-exact against the reference.',
    status: 'planned',
  },
  {
    id: 'M3',
    title: 'NVIDIA backend',
    when: 'planned',
    scope: 'PTX emission and cutile-rs kernels; KV block store with planned lifetimes; agent-aware runtime on one GPU.',
    done: 'An agentic trace replays on real hardware with the simulator’s predicted ordering of policies confirmed.',
    status: 'planned',
  },
  {
    id: 'M4',
    title: 'Tenstorrent backend',
    when: 'planned',
    scope: 'tt-metal programs through the host API; Blackhole machine file validated on hardware.',
    done: 'The same tiny model and trace replay on a Blackhole card.',
    status: 'planned',
  },
  {
    id: 'M5',
    title: 'Heterogeneous placement',
    when: 'planned',
    scope: 'Prefill-class and decode-class devices in one plan; KV transfer scheduling over links.',
    done: 'Simulation, and measurement where hardware allows, of a two-class deployment.',
    status: 'planned',
  },
  {
    id: 'M6',
    title: 'Numerics verification',
    when: 'planned',
    scope: 'Block formats as verified IR types; e-graph rewrites with checked equivalence; batch-invariant kernel mode.',
    done: 'Rewrite rules proven or property-tested; deterministic mode passes a 1,000-run identity test.',
    status: 'planned',
  },
];

export const targets = {
  open: ['NVIDIA', 'AMD', 'Intel', 'Apple GPU', 'Tenstorrent', 'AMD XDNA 2', 'Arm Ethos-U', 'CPU'],
  modelled: ['TPU', 'Trainium', 'Groq / LPX', 'Cerebras', 'SambaNova', 'Etched', 'd-Matrix'],
};
