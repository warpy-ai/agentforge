//! Trace schema, JSONL loading and a synthetic generator driven by an agent program.

use agentforge_ir::{AgentProgram, LatencyModel, Transition};
use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// A tool wait between two steps: the tool the agent called, the realized duration and the
/// latency model the program declares for that tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolWait {
    pub name: String,
    pub wait_s: f64,
    pub latency: LatencyModel,
}

/// One LLM step of a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceStep {
    pub new_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub tool: Option<ToolWait>,
}

/// One agent session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub id: u32,
    pub arrival_s: f64,
    pub initial_prefix_tokens: u64,
    pub steps: Vec<TraceStep>,
}

/// A set of sessions.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Trace {
    pub sessions: Vec<Session>,
}

impl Trace {
    pub fn total_steps(&self) -> u64 {
        self.sessions.iter().map(|s| s.steps.len() as u64).sum()
    }

    /// One JSON object per line, one line per session.
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        let mut out = String::new();
        for s in &self.sessions {
            out.push_str(&serde_json::to_string(s)?);
            out.push('\n');
        }
        Ok(out)
    }

    pub fn from_jsonl(text: &str) -> Result<Trace, serde_json::Error> {
        let mut sessions = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            sessions.push(serde_json::from_str(line)?);
        }
        Ok(Trace { sessions })
    }
}

/// Parameters of the synthetic generator.
///
/// Defaults follow the TraceLab measurements of coding agents (University of Washington, June
/// 2026): the median step carries about 119k prefix tokens while appending about 875, so most
/// steps belong to sessions that resume long histories. Sessions therefore start from a
/// log-normal initial prefix with a 110k-token median (never below the program's own stable
/// prefix) and grow by the program's per-step tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticConfig {
    pub sessions: u32,
    pub mean_steps: f64,
    pub max_steps: u32,
    pub arrival_rate_per_s: f64,
    /// Median of the initial prefix in tokens; `0` uses the program's prefix only.
    pub initial_prefix_median: u64,
    pub initial_prefix_sigma: f64,
    pub max_initial_prefix: u64,
    pub new_tokens_sigma: f64,
    pub output_tokens_sigma: f64,
    pub seed: u64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        SyntheticConfig {
            sessions: 64,
            mean_steps: 20.0,
            max_steps: 200,
            arrival_rate_per_s: 0.5,
            initial_prefix_median: 110_000,
            initial_prefix_sigma: 0.45,
            max_initial_prefix: 400_000,
            new_tokens_sigma: 0.6,
            output_tokens_sigma: 0.7,
            seed: 1,
        }
    }
}

/// Pick an index with the given weights.
fn pick_weighted(rng: &mut Rng, weights: &[f64]) -> Option<usize> {
    let total: f64 = weights.iter().sum();
    if weights.is_empty() || total <= 0.0 {
        return None;
    }
    let mut r = rng.next_f64() * total;
    for (i, w) in weights.iter().enumerate() {
        if r < *w {
            return Some(i);
        }
        r -= w;
    }
    Some(weights.len() - 1)
}

/// Sample sessions from an agent program.
pub fn synthesize(program: &AgentProgram, cfg: &SyntheticConfig) -> Trace {
    let mut rng = Rng::new(cfg.seed);
    let mut t = 0.0;
    let mut sessions = Vec::with_capacity(cfg.sessions as usize);
    for id in 0..cfg.sessions {
        t += rng.exponential(1.0 / cfg.arrival_rate_per_s.max(1e-9));
        let initial_prefix_tokens = if cfg.initial_prefix_median == 0 {
            program.prefix_tokens()
        } else {
            let sampled = rng
                .lognormal(cfg.initial_prefix_median as f64, cfg.initial_prefix_sigma)
                .round()
                .clamp(0.0, cfg.max_initial_prefix as f64) as u64;
            sampled.max(program.prefix_tokens())
        };
        let n_steps = rng.geometric(cfg.mean_steps, cfg.max_steps);
        let mut step_id = program.entry;
        let mut steps = Vec::with_capacity(n_steps as usize);
        for i in 0..n_steps {
            let Some(step) = program.step(step_id) else {
                break;
            };
            let new_tokens = rng
                .lognormal(step.new_input_tokens.max(1) as f64, cfg.new_tokens_sigma)
                .round()
                .max(16.0) as u64;
            let output_tokens = rng
                .lognormal(step.output_tokens.max(1) as f64, cfg.output_tokens_sigma)
                .round()
                .max(8.0) as u64;
            let last = i + 1 == n_steps;
            let tool = if last {
                None
            } else {
                match &step.transition {
                    Transition::Tool { call, then } => {
                        step_id = *then;
                        Some(ToolWait {
                            name: call.name.clone(),
                            wait_s: rng.sample_latency(&call.latency),
                            latency: call.latency.clone(),
                        })
                    }
                    Transition::ToolChoice { arms } => {
                        let weights: Vec<f64> = arms.iter().map(|a| a.probability).collect();
                        pick_weighted(&mut rng, &weights).map(|i| {
                            let arm = &arms[i];
                            step_id = arm.then;
                            ToolWait {
                                name: arm.call.name.clone(),
                                wait_s: rng.sample_latency(&arm.call.latency),
                                latency: arm.call.latency.clone(),
                            }
                        })
                    }
                    Transition::Branch { arms } => {
                        let weights: Vec<f64> = arms.iter().map(|a| a.probability).collect();
                        pick_weighted(&mut rng, &weights).map(|i| {
                            step_id = arms[i].then;
                            ToolWait {
                                name: "branch".to_string(),
                                wait_s: 0.0,
                                latency: LatencyModel::Fixed { seconds: 0.0 },
                            }
                        })
                    }
                    Transition::End => None,
                }
            };
            let ends = tool.is_none();
            steps.push(TraceStep {
                new_tokens,
                output_tokens,
                tool,
            });
            if ends {
                break;
            }
        }
        sessions.push(Session {
            id,
            arrival_s: t,
            initial_prefix_tokens,
            steps,
        });
    }
    Trace { sessions }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_trace_is_deterministic_and_well_formed() {
        let p = AgentProgram::coding_agent_default();
        let cfg = SyntheticConfig {
            sessions: 10,
            seed: 5,
            ..Default::default()
        };
        let a = synthesize(&p, &cfg);
        let b = synthesize(&p, &cfg);
        assert_eq!(a, b);
        assert_eq!(a.sessions.len(), 10);
        let mut names = std::collections::HashSet::new();
        for s in &a.sessions {
            assert!(s.initial_prefix_tokens >= p.prefix_tokens());
            assert!(!s.steps.is_empty());
            assert!(s.steps.last().unwrap().tool.is_none());
            for st in &s.steps[..s.steps.len() - 1] {
                let tool = st.tool.as_ref().expect("intermediate steps have tools");
                names.insert(tool.name.clone());
            }
        }
        assert!(names.contains("read_or_edit"), "{names:?}");
    }

    #[test]
    fn jsonl_round_trip() {
        let p = AgentProgram::coding_agent_default();
        let t = synthesize(
            &p,
            &SyntheticConfig {
                sessions: 3,
                ..Default::default()
            },
        );
        let text = t.to_jsonl().unwrap();
        let back = Trace::from_jsonl(&text).unwrap();
        assert_eq!(t, back);
    }
}
