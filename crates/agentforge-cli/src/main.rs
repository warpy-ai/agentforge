//! `agentforge`: command-line access to the machine model, cost model, planning passes and the
//! trace-replay simulator.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use agentforge_compile::{
    kv_liveness, place, plan_residency, DeviceState, ResidencyAction, TierBudget,
};
use agentforge_cost::calibration::{self, Calibration};
use agentforge_cost::{self as cost, Efficiency, Phase};
use agentforge_ir::{AgentProgram, LatencyModel, ModelSpec, Precision};
use agentforge_machine::Machine;
use agentforge_sim::{
    render_table, run, synthesize, Lru, Plan, Policy, SimConfig, SimReport, SyntheticConfig, Trace,
    WriteThrough,
};

#[derive(Parser)]
#[command(
    name = "agentforge",
    version,
    about = "AgentForge: research tools for compiling agent programs and models onto AI inference silicon"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List or show the bundled machine descriptions.
    Machines {
        #[command(subcommand)]
        command: MachinesCommand,
    },
    /// List or show the bundled model presets.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Estimate prefill and decode costs on a machine.
    Cost {
        #[command(subcommand)]
        command: CostCommand,
    },
    /// Compiler planning passes: KV liveness, residency, placement.
    Plan {
        #[command(subcommand)]
        command: PlanCommand,
    },
    /// Trace-replay simulation of agentic sessions.
    Sim {
        #[command(subcommand)]
        command: SimCommand,
    },
}

#[derive(Subcommand)]
enum MachinesCommand {
    /// List bundled machines.
    List,
    /// Print a machine description as TOML.
    Show { name: String },
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// List bundled model presets.
    List,
    /// Print a model preset as JSON with derived sizes.
    Show { name: String },
}

/// Model, machine and parallelism selection shared by several commands.
#[derive(Args, Clone)]
struct TargetArgs {
    /// Model preset name.
    #[arg(long, default_value = "llama-3.1-70b")]
    model: String,
    /// Bundled machine name or path to a machine TOML file.
    #[arg(long, default_value = "h200")]
    machine: String,
    /// Device name within the machine (default: the first device).
    #[arg(long)]
    device: Option<String>,
    /// Tensor-parallel degree; device resources scale linearly and all-reduces run over the
    /// machine's fastest link.
    #[arg(long, default_value_t = 1)]
    tp: u32,
    /// Override the model's weight precision (fp16, bf16, fp8, mxfp4, nvfp4, ...).
    #[arg(long)]
    weight_precision: Option<String>,
    /// Override the model's KV cache precision.
    #[arg(long)]
    kv_precision: Option<String>,
}

#[derive(Subcommand)]
enum CostCommand {
    /// Prefill `tokens` new tokens behind `prefix` cached tokens.
    Prefill {
        #[command(flatten)]
        target: TargetArgs,
        #[arg(long, default_value_t = 875)]
        tokens: u64,
        #[arg(long, default_value_t = 119_000)]
        prefix: u64,
    },
    /// One decode step for `batch` sequences at `context` tokens.
    Decode {
        #[command(flatten)]
        target: TargetArgs,
        #[arg(long, default_value_t = 119_000)]
        context: u64,
        #[arg(long, default_value_t = 1)]
        batch: u64,
    },
    /// Compare predicted offline throughput with the bundled TensorRT-LLM measurements.
    Calibrate {
        /// Largest batch the sweep tries; 2048 is trtllm-bench's default maximum batch size.
        #[arg(long, default_value_t = 2048)]
        max_batch: u64,
        /// Fraction of device memory kept out of the KV budget; 0.10 matches TensorRT-LLM's
        /// default free-memory fraction of 0.9.
        #[arg(long, default_value_t = 0.10)]
        reserve: f64,
        /// Grid-search the efficiency knobs for the smallest worst-row error instead of
        /// printing the table for the defaults.
        #[arg(long)]
        fit: bool,
    },
}

#[derive(Subcommand)]
enum PlanCommand {
    /// KV liveness of the default coding-agent program.
    Liveness,
    /// Residency decision for a prefix across a log-normal tool wait.
    Residency {
        #[command(flatten)]
        target: TargetArgs,
        #[arg(long, default_value_t = 119_000)]
        prefix_tokens: u64,
        #[arg(long, default_value_t = 45.0)]
        wait_median: f64,
        #[arg(long, default_value_t = 0.7)]
        wait_sigma: f64,
        /// Device KV pressure in 0..=1.
        #[arg(long, default_value_t = 0.0)]
        pressure: f64,
        /// Typical step duration in seconds; derived from the cost model when omitted.
        #[arg(long)]
        step_hint: Option<f64>,
    },
    /// Prefill and decode placement across a machine's device classes.
    Placement {
        #[command(flatten)]
        target: TargetArgs,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PolicyChoice {
    Lru,
    WriteThrough,
    Plan,
    /// lru and plan.
    Both,
    /// lru, write-through and plan.
    All,
}

#[derive(Subcommand)]
enum SimCommand {
    /// Replay a synthetic or recorded trace and compare KV residency policies.
    Run {
        #[command(flatten)]
        target: TargetArgs,
        /// Number of synthetic sessions.
        #[arg(long, default_value_t = 64)]
        sessions: u32,
        /// Mean steps per synthetic session.
        #[arg(long, default_value_t = 20.0)]
        steps: f64,
        /// Session arrival rate per second.
        #[arg(long, default_value_t = 0.5)]
        arrival_rate: f64,
        /// Median initial prefix in tokens (0 = the program's prefix only).
        #[arg(long, default_value_t = 110_000)]
        prefix_median: u64,
        /// Log-normal sigma of the initial prefix.
        #[arg(long, default_value_t = 0.45)]
        prefix_sigma: f64,
        #[arg(long, value_enum, default_value_t = PolicyChoice::Both)]
        policy: PolicyChoice,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Replay a JSONL trace instead of synthesizing one.
        #[arg(long)]
        trace: Option<PathBuf>,
        /// Save the trace that was run as JSONL.
        #[arg(long)]
        save_trace: Option<PathBuf>,
        /// Override the capacity of the first (host DRAM) tier in GB.
        #[arg(long)]
        host_gb: Option<f64>,
        /// Let decode run at full speed during a prefill instead of stalling for its duration.
        #[arg(long)]
        decode_overlaps_prefill: bool,
        /// Admit a new session only while the device KV in use, its own included, stays within
        /// this fraction (0 to 1] of the KV budget; later sessions queue in arrival order.
        /// Every session is admitted on arrival when omitted.
        #[arg(long)]
        admit_fraction: Option<f64>,
        /// Print reports as JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
}

fn load_machine(name_or_path: &str) -> Result<Machine> {
    let path = Path::new(name_or_path);
    if path.is_file() {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading machine file {}", path.display()))?;
        return Machine::from_toml(&text)
            .with_context(|| format!("parsing machine file {}", path.display()));
    }
    Machine::builtin(name_or_path).map_err(|e| anyhow!(e))
}

fn parse_precision(value: &str) -> Result<Precision> {
    value.parse::<Precision>().map_err(|e| anyhow!(e))
}

fn load_model(target: &TargetArgs) -> Result<ModelSpec> {
    let mut model = ModelSpec::preset(&target.model).ok_or_else(|| {
        anyhow!(
            "unknown model preset '{}'; known presets: {}",
            target.model,
            ModelSpec::preset_names().join(", ")
        )
    })?;
    if let Some(p) = &target.weight_precision {
        model.weight_precision = parse_precision(p)?;
    }
    if let Some(p) = &target.kv_precision {
        model.kv_precision = parse_precision(p)?;
    }
    Ok(model)
}

struct Resolved {
    model: ModelSpec,
    machine: Machine,
    device_index: usize,
}

fn resolve(target: &TargetArgs) -> Result<Resolved> {
    let model = load_model(target)?;
    let machine = load_machine(&target.machine)?;
    let device_index = match &target.device {
        Some(name) => machine
            .devices
            .iter()
            .position(|d| &d.name == name)
            .ok_or_else(|| anyhow!("machine '{}' has no device '{}'", machine.name, name))?,
        None => 0,
    };
    Ok(Resolved {
        model,
        machine,
        device_index,
    })
}

impl Resolved {
    fn cost_target(&self, tp: u32) -> cost::Target<'_> {
        cost::Target {
            device: &self.machine.devices[self.device_index],
            tp,
            link: self.machine.fastest_link(),
            eff: Efficiency::default(),
        }
    }
}

fn print_phase(label: &str, phase: &Phase) {
    println!("{label}");
    println!("  flops            {:.3e}", phase.flops);
    println!("  bytes            {:.3e}", phase.bytes);
    println!("  compute time     {:.4} s", phase.compute_s);
    println!("  memory time      {:.4} s", phase.memory_s);
    println!("  overhead         {:.4} s", phase.overhead_s);
    println!("  communication    {:.4} s", phase.comm_s);
    println!(
        "  total            {:.4} s ({}-bound)",
        phase.seconds,
        phase.bound()
    );
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Machines { command } => machines(command),
        Command::Models { command } => models(command),
        Command::Cost { command } => cost_cmd(command),
        Command::Plan { command } => plan_cmd(command),
        Command::Sim { command } => sim_cmd(command),
    }
}

fn machines(command: MachinesCommand) -> Result<()> {
    match command {
        MachinesCommand::List => {
            for name in Machine::builtin_names() {
                let m = Machine::builtin(name)?;
                let devices: Vec<String> = m
                    .devices
                    .iter()
                    .map(|d| {
                        format!(
                            "{} x{} ({} GB, {} TB/s)",
                            d.name, d.count, d.memory_gb, d.memory_bw_tbps
                        )
                    })
                    .collect();
                println!("{name:16} {}", devices.join("; "));
                println!("{:16} {}", "", m.description);
            }
            Ok(())
        }
        MachinesCommand::Show { name } => {
            let m = load_machine(&name)?;
            print!("{}", m.to_toml()?);
            Ok(())
        }
    }
}

fn models(command: ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::List => {
            println!(
                "{:16} {:>8} {:>8} {:>10} {:>12} {:>7} {:>6}",
                "name", "total B", "active B", "weights GB", "KV B/token", "weight", "kv"
            );
            for m in ModelSpec::presets() {
                println!(
                    "{:16} {:>8.1} {:>8.1} {:>10.1} {:>12} {:>7} {:>6}",
                    m.name,
                    m.total_params() as f64 / 1e9,
                    m.active_params_per_token() as f64 / 1e9,
                    m.weight_bytes() as f64 / 1e9,
                    m.kv_bytes_per_token(),
                    m.weight_precision,
                    m.kv_precision
                );
            }
            Ok(())
        }
        ModelsCommand::Show { name } => {
            let m =
                ModelSpec::preset(&name).ok_or_else(|| anyhow!("unknown model preset '{name}'"))?;
            println!("{}", serde_json::to_string_pretty(&m)?);
            println!("total params:        {}", m.total_params());
            println!("active params/token: {}", m.active_params_per_token());
            println!("weight bytes:        {}", m.weight_bytes());
            println!("kv bytes/token:      {}", m.kv_bytes_per_token());
            Ok(())
        }
    }
}

fn cost_cmd(command: CostCommand) -> Result<()> {
    match command {
        CostCommand::Prefill {
            target,
            tokens,
            prefix,
        } => {
            let r = resolve(&target)?;
            let device = &r.machine.devices[r.device_index];
            let phase = cost::prefill(&r.model, &r.cost_target(target.tp), tokens, prefix)?;
            print_phase(
                &format!(
                    "prefill of {tokens} tokens behind {prefix} cached tokens: {} on {} x{}",
                    r.model.name, device.name, target.tp
                ),
                &phase,
            );
            let kv = cost::kv_bytes(&r.model, prefix + tokens) as f64 / 1e9;
            println!("  KV after step    {kv:.1} GB");
            Ok(())
        }
        CostCommand::Decode {
            target,
            context,
            batch,
        } => {
            let r = resolve(&target)?;
            let device = &r.machine.devices[r.device_index];
            let phase = cost::decode_step(&r.model, &r.cost_target(target.tp), context, batch)?;
            print_phase(
                &format!(
                    "decode step for {batch} sequences at {context} tokens: {} on {} x{}",
                    r.model.name, device.name, target.tp
                ),
                &phase,
            );
            println!("  per-user rate    {:.1} tokens/s", 1.0 / phase.seconds);
            println!(
                "  aggregate rate   {:.1} tokens/s",
                batch as f64 / phase.seconds
            );
            Ok(())
        }
        CostCommand::Calibrate {
            max_batch,
            reserve,
            fit,
        } => calibrate(max_batch, reserve, fit),
    }
}

fn calibrate(max_batch: u64, reserve: f64, fit: bool) -> Result<()> {
    let cal = Calibration::builtin();
    if fit {
        let (eff, max_error) = calibration::fit(&cal, reserve, max_batch)?;
        let results = cal.evaluate(&eff, reserve, max_batch)?;
        println!("{eff:#?}");
        println!("max |error|      {:.1}%", 100.0 * max_error);
        println!(
            "geomean |error|  {:.1}%",
            100.0 * calibration::geomean_abs_error(&results)
        );
        return Ok(());
    }
    let results = cal.evaluate(&Efficiency::default(), reserve, max_batch)?;
    println!("{} ({})", cal.name, cal.metric);
    println!();
    println!("| machine | weights | tp | isl | osl | published tok/s | predicted tok/s | batch | bound | error |");
    println!("|---|---|---:|---:|---:|---:|---:|---:|---|---:|");
    for r in &results {
        println!(
            "| {} | {} | {} | {} | {} | {:.0} | {:.0} | {} | {} | {:+.1}% |",
            r.row.machine,
            r.row.weight_precision,
            r.row.tp,
            r.row.isl,
            r.row.osl,
            r.row.tokens_per_s,
            r.predicted.output_tokens_per_s,
            r.predicted.batch,
            r.predicted.bound,
            100.0 * r.error
        );
    }
    println!();
    println!(
        "max |error| {:.1}%, geomean |error| {:.1}%",
        100.0 * calibration::max_abs_error(&results),
        100.0 * calibration::geomean_abs_error(&results)
    );
    Ok(())
}

fn plan_cmd(command: PlanCommand) -> Result<()> {
    match command {
        PlanCommand::Liveness => {
            let program = AgentProgram::coding_agent_default();
            println!(
                "program: {} (prefix {} tokens)",
                program.name,
                program.prefix_tokens()
            );
            for l in kv_liveness(&program) {
                println!(
                    "step {}: tool '{}' ({:.0}%) expected gap {:.1} s (p10 {:.1}, p50 {:.1}, p90 {:.1})",
                    l.step,
                    l.tool,
                    100.0 * l.probability,
                    l.expected_gap_s,
                    l.p10_s,
                    l.p50_s,
                    l.p90_s
                );
            }
            Ok(())
        }
        PlanCommand::Residency {
            target,
            prefix_tokens,
            wait_median,
            wait_sigma,
            pressure,
            step_hint,
        } => {
            let r = resolve(&target)?;
            let cost_target = r.cost_target(target.tp);
            let kv = cost::kv_bytes(&r.model, prefix_tokens) as f64;
            let recompute = cost::prefill(&r.model, &cost_target, prefix_tokens, 0)?;
            let step_hint_s = match step_hint {
                Some(s) => s,
                None => {
                    let step_prefill = cost::prefill(&r.model, &cost_target, 875, prefix_tokens)?;
                    let step_decode = cost::decode_step(&r.model, &cost_target, prefix_tokens, 4)?;
                    step_prefill.seconds + 214.0 * step_decode.seconds
                }
            };
            let budgets: Vec<TierBudget<'_>> = r
                .machine
                .tiers
                .iter()
                .map(|t| TierBudget {
                    tier: t,
                    free_bytes: t.capacity_bytes(),
                })
                .collect();
            let wait = LatencyModel::LogNormal {
                median_s: wait_median,
                sigma: wait_sigma,
            };
            let d = plan_residency(
                kv,
                &wait,
                &budgets,
                target.tp,
                recompute.seconds,
                DeviceState {
                    pressure,
                    step_hint_s,
                    channel_backlog_s: 0.0,
                },
            );
            println!("KV for {prefix_tokens} tokens: {:.1} GB", kv / 1e9);
            println!("recompute: {:.2} s", d.recompute_s);
            println!("typical step: {step_hint_s:.2} s");
            match &d.action {
                ResidencyAction::Keep => println!("decision: keep on device"),
                ResidencyAction::Spill { tier, prefetch_lead_s, .. } => println!(
                    "decision: spill to '{tier}' during the wait, prefetch {prefetch_lead_s:.2} s before the predicted return"
                ),
                ResidencyAction::WriteThrough { tier_index } => println!(
                    "decision: keep on device and write through to '{}' during the wait",
                    r.machine.tiers[*tier_index].name
                ),
                ResidencyAction::Drop => println!("decision: drop and recompute on return"),
            }
            if let Some(f) = d.fetch_s {
                println!("fetch: {f:.3} s");
            }
            if let Some(w) = d.write_s {
                println!("write: {w:.3} s");
            }
            println!("rationale: {}", d.rationale);
            Ok(())
        }
        PlanCommand::Placement { target } => {
            let r = resolve(&target)?;
            let plan = place(&r.model, &r.machine);
            println!("prefill device: {}", plan.prefill_device);
            println!("decode device:  {}", plan.decode_device);
            println!(
                "kv link:        {}",
                plan.kv_link.as_deref().unwrap_or("none (same device)")
            );
            println!("rationale:      {}", plan.rationale);
            Ok(())
        }
    }
}

fn sim_cmd(command: SimCommand) -> Result<()> {
    match command {
        SimCommand::Run {
            target,
            sessions,
            steps,
            arrival_rate,
            prefix_median,
            prefix_sigma,
            policy,
            seed,
            trace,
            save_trace,
            host_gb,
            decode_overlaps_prefill,
            admit_fraction,
            json,
        } => {
            if let Some(f) = admit_fraction {
                let valid = f > 0.0 && f <= 1.0;
                if !valid {
                    bail!("--admit-fraction must be in (0, 1], got {f}");
                }
            }
            let r = resolve(&target)?;
            let device = r.machine.devices[r.device_index].clone();
            let mut tiers = r.machine.tiers.clone();
            if let Some(gb) = host_gb {
                match tiers.first_mut() {
                    Some(t) => t.capacity_gb = gb,
                    None => bail!("machine '{}' has no memory tiers", r.machine.name),
                }
            }
            let mut cfg = SimConfig::new(r.model.clone(), device, tiers, target.tp);
            cfg.link = r.machine.fastest_link().cloned();
            cfg.prefill_stalls_decode = !decode_overlaps_prefill;
            cfg.admit_fraction = admit_fraction;
            let trace_data = match &trace {
                Some(path) => {
                    let text = fs::read_to_string(path)
                        .with_context(|| format!("reading trace {}", path.display()))?;
                    Trace::from_jsonl(&text)
                        .with_context(|| format!("parsing trace {}", path.display()))?
                }
                None => synthesize(
                    &AgentProgram::coding_agent_default(),
                    &SyntheticConfig {
                        sessions,
                        steps: agentforge_sim::StepCount::Geometric { mean: steps },
                        arrival_rate_per_s: arrival_rate,
                        initial_prefix_median: prefix_median,
                        initial_prefix_sigma: prefix_sigma,
                        seed,
                        ..Default::default()
                    },
                ),
            };
            if let Some(path) = &save_trace {
                fs::write(path, trace_data.to_jsonl()?)
                    .with_context(|| format!("writing trace {}", path.display()))?;
            }
            let policies: Vec<Box<dyn Policy>> = match policy {
                PolicyChoice::Lru => vec![Box::new(Lru)],
                PolicyChoice::WriteThrough => vec![Box::new(WriteThrough)],
                PolicyChoice::Plan => vec![Box::new(Plan)],
                PolicyChoice::Both => vec![Box::new(Lru), Box::new(Plan)],
                PolicyChoice::All => vec![Box::new(Lru), Box::new(WriteThrough), Box::new(Plan)],
            };
            let mut reports: Vec<SimReport> = Vec::new();
            for p in &policies {
                reports.push(run(&cfg, &trace_data, p.as_ref())?);
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&reports)?);
            } else {
                println!(
                    "model {} ({} weights, {} KV) on {} x{} ({:.0} GB KV budget); {} sessions, {} steps",
                    r.model.name,
                    r.model.weight_precision,
                    r.model.kv_precision,
                    cfg.device.name,
                    cfg.tp,
                    cfg.kv_capacity_bytes() / 1e9,
                    trace_data.sessions.len(),
                    trace_data.total_steps()
                );
                println!();
                print!("{}", render_table(&reports));
            }
            Ok(())
        }
    }
}
