//! The Agent Program IR.
//!
//! An agent program is the loop a serving system otherwise has to infer from independent HTTP
//! requests: which prompt blocks are stable across steps, what each step appends and generates,
//! which tool runs next and how long it tends to take, and which branches are likely. The
//! compiler derives KV block liveness from it the way register liveness is derived from a
//! control-flow graph.
//!
//! The information that matters most for KV planning is *which tool* a step calls: a file read
//! returns in a second, a test suite in a minute, a human in minutes. [`Transition::ToolChoice`]
//! carries that per-tool knowledge; a serving layer that only sees requests cannot have it.

use serde::{Deserialize, Serialize};

pub type StepId = u32;

/// The role of a prompt block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    System,
    Tools,
    History,
    ToolResult,
    User,
    Reasoning,
}

/// A contiguous block of prompt tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptBlock {
    pub kind: BlockKind,
    pub tokens: u64,
    /// Identical across steps, so its KV is reusable as a prefix.
    pub stable: bool,
}

/// A latency distribution for a tool call or a human turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LatencyModel {
    Fixed { seconds: f64 },
    LogNormal { median_s: f64, sigma: f64 },
    Uniform { min_s: f64, max_s: f64 },
    Mixture { components: Vec<MixtureComponent> },
}

/// One weighted component of a [`LatencyModel::Mixture`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixtureComponent {
    pub weight: f64,
    pub model: LatencyModel,
}

impl LatencyModel {
    /// Expected value in seconds.
    pub fn mean(&self) -> f64 {
        match self {
            LatencyModel::Fixed { seconds } => *seconds,
            LatencyModel::LogNormal { median_s, sigma } => median_s * (0.5 * sigma * sigma).exp(),
            LatencyModel::Uniform { min_s, max_s } => 0.5 * (min_s + max_s),
            LatencyModel::Mixture { components } => {
                let total: f64 = components.iter().map(|c| c.weight).sum();
                if total <= 0.0 {
                    return 0.0;
                }
                components
                    .iter()
                    .map(|c| c.weight * c.model.mean())
                    .sum::<f64>()
                    / total
            }
        }
    }

    /// Cumulative distribution function at `x` seconds.
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            LatencyModel::Fixed { seconds } => {
                if x >= *seconds {
                    1.0
                } else {
                    0.0
                }
            }
            LatencyModel::LogNormal { median_s, sigma } => {
                if x <= 0.0 {
                    0.0
                } else {
                    normal_cdf((x.ln() - median_s.ln()) / sigma)
                }
            }
            LatencyModel::Uniform { min_s, max_s } => {
                if max_s <= min_s {
                    return if x >= *min_s { 1.0 } else { 0.0 };
                }
                ((x - min_s) / (max_s - min_s)).clamp(0.0, 1.0)
            }
            LatencyModel::Mixture { components } => {
                let total: f64 = components.iter().map(|c| c.weight).sum();
                if total <= 0.0 {
                    return 1.0;
                }
                components
                    .iter()
                    .map(|c| c.weight * c.model.cdf(x))
                    .sum::<f64>()
                    / total
            }
        }
    }

    /// Quantile function: the smallest `x` with `cdf(x) >= q`.
    pub fn quantile(&self, q: f64) -> f64 {
        let q = q.clamp(1e-6, 1.0 - 1e-6);
        match self {
            LatencyModel::Fixed { seconds } => *seconds,
            LatencyModel::LogNormal { median_s, sigma } => median_s * (sigma * probit(q)).exp(),
            LatencyModel::Uniform { min_s, max_s } => min_s + (max_s - min_s) * q,
            LatencyModel::Mixture { components } => {
                if components.is_empty() {
                    return 0.0;
                }
                // The mixture quantile lies between the smallest and largest component quantile.
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for c in components {
                    let v = c.model.quantile(q);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                if hi - lo < 1e-9 {
                    return lo;
                }
                for _ in 0..80 {
                    let mid = 0.5 * (lo + hi);
                    if self.cdf(mid) >= q {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                hi
            }
        }
    }

    /// Median in seconds.
    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }
}

/// Standard normal CDF via the Abramowitz-Stegun 7.1.26 erf approximation (error below 1.5e-7).
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Inverse standard normal CDF (Acklam's rational approximation, relative error about 1e-9).
fn probit(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// A tool invocation that follows an LLM step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub latency: LatencyModel,
    /// Tokens the tool result appends to the next step's input, when not already counted in
    /// the next step's `new_input_tokens`.
    #[serde(default)]
    pub result_tokens: u64,
}

/// One possible tool call after a step and the probability the agent chooses it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolArm {
    pub probability: f64,
    pub call: ToolCall,
    pub then: StepId,
}

/// One arm of a probabilistic branch between steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchArm {
    pub probability: f64,
    pub then: StepId,
}

/// What happens after a step's generation completes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Transition {
    /// Always the same tool.
    Tool {
        call: ToolCall,
        then: StepId,
    },
    /// One of several tools, chosen by the agent; the runtime learns which one at call time.
    ToolChoice {
        arms: Vec<ToolArm>,
    },
    /// Another LLM step without a tool in between.
    Branch {
        arms: Vec<BranchArm>,
    },
    End,
}

impl Transition {
    /// The tool calls this transition may make, with their probabilities.
    pub fn tool_calls(&self) -> Vec<(f64, &ToolCall)> {
        match self {
            Transition::Tool { call, .. } => vec![(1.0, call)],
            Transition::ToolChoice { arms } => {
                let total: f64 = arms.iter().map(|a| a.probability).sum();
                arms.iter()
                    .map(|a| {
                        (
                            if total > 0.0 {
                                a.probability / total
                            } else {
                                0.0
                            },
                            &a.call,
                        )
                    })
                    .collect()
            }
            Transition::Branch { .. } | Transition::End => Vec::new(),
        }
    }

    /// The wait distribution a policy would see if it did not know which tool was chosen.
    pub fn blended_latency(&self) -> Option<LatencyModel> {
        let calls = self.tool_calls();
        match calls.as_slice() {
            [] => None,
            [(_, call)] => Some(call.latency.clone()),
            many => Some(LatencyModel::Mixture {
                components: many
                    .iter()
                    .map(|(p, call)| MixtureComponent {
                        weight: *p,
                        model: call.latency.clone(),
                    })
                    .collect(),
            }),
        }
    }
}

/// One LLM call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub id: StepId,
    /// Tokens appended to the context before this call (user text, tool results, reasoning).
    pub new_input_tokens: u64,
    pub output_tokens: u64,
    pub transition: Transition,
}

/// A whole agent program: a stable prefix and a graph of steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentProgram {
    pub name: String,
    pub prefix: Vec<PromptBlock>,
    pub steps: Vec<Step>,
    pub entry: StepId,
}

impl AgentProgram {
    /// Tokens in the initial prefix.
    pub fn prefix_tokens(&self) -> u64 {
        self.prefix.iter().map(|b| b.tokens).sum()
    }

    /// Tokens in the stable part of the prefix.
    pub fn stable_prefix_tokens(&self) -> u64 {
        self.prefix
            .iter()
            .filter(|b| b.stable)
            .map(|b| b.tokens)
            .sum()
    }

    pub fn step(&self, id: StepId) -> Option<&Step> {
        self.steps.iter().find(|s| s.id == id)
    }

    /// A coding-agent loop shaped like the TraceLab measurements (University of Washington,
    /// June 2026): about 20k tokens of stable prefix, 875 appended and 214 generated tokens per
    /// step, and a choice between fast tools (reads, edits, short commands), long-running tools
    /// (builds, test suites) and human turns, each with its own latency model.
    pub fn coding_agent_default() -> AgentProgram {
        let arm = |probability: f64, name: &str, median_s: f64, sigma: f64| ToolArm {
            probability,
            call: ToolCall {
                name: name.to_string(),
                latency: LatencyModel::LogNormal { median_s, sigma },
                result_tokens: 0,
            },
            then: 0,
        };
        AgentProgram {
            name: "coding-agent-default".to_string(),
            prefix: vec![
                PromptBlock {
                    kind: BlockKind::System,
                    tokens: 3_000,
                    stable: true,
                },
                PromptBlock {
                    kind: BlockKind::Tools,
                    tokens: 6_000,
                    stable: true,
                },
                PromptBlock {
                    kind: BlockKind::History,
                    tokens: 11_000,
                    stable: true,
                },
            ],
            steps: vec![Step {
                id: 0,
                new_input_tokens: 875,
                output_tokens: 214,
                transition: Transition::ToolChoice {
                    arms: vec![
                        arm(0.75, "read_or_edit", 1.5, 0.8),
                        arm(0.15, "build_or_test", 45.0, 0.7),
                        arm(0.10, "human_turn", 180.0, 1.0),
                    ],
                },
            }],
            entry: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lognormal_quantiles_and_mean() {
        let m = LatencyModel::LogNormal {
            median_s: 10.0,
            sigma: 1.0,
        };
        assert!((m.median() - 10.0).abs() < 1e-3);
        assert!((m.mean() - 10.0 * 0.5f64.exp()).abs() < 1e-6);
        assert!(m.quantile(0.9) > m.quantile(0.1));
        assert!((m.cdf(m.quantile(0.3)) - 0.3).abs() < 1e-4);
    }

    #[test]
    fn blended_latency_is_a_mixture_between_its_components() {
        let p = AgentProgram::coding_agent_default();
        let blended = p.steps[0].transition.blended_latency().unwrap();
        let q20 = blended.quantile(0.2);
        let q50 = blended.quantile(0.5);
        let q90 = blended.quantile(0.9);
        assert!(q20 < q50 && q50 < q90, "{q20} {q50} {q90}");
        assert!(q50 > 1.0 && q50 < 10.0, "median {q50}");
        assert!(q90 > 30.0, "p90 {q90}");
        assert!((blended.cdf(q50) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn default_program_tools_are_distinguishable() {
        let p = AgentProgram::coding_agent_default();
        assert_eq!(p.prefix_tokens(), 20_000);
        assert_eq!(p.stable_prefix_tokens(), 20_000);
        let calls = p.steps[0].transition.tool_calls();
        assert_eq!(calls.len(), 3);
        let total: f64 = calls.iter().map(|(p, _)| p).sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(calls[0].1.latency.median() < 2.0);
        assert!(calls[2].1.latency.median() > 100.0);
    }
}
