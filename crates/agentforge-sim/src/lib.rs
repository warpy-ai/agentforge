//! Trace-replay simulation of agentic sessions over a machine description.
//!
//! The simulator replays sessions of LLM steps separated by tool waits against one device class
//! (scaled by tensor parallelism), with host and NVMe tiers, and compares KV residency policies:
//! a reactive LRU policy that only evicts under pressure, and the program-aware plan produced by
//! `agentforge-compile`, which spills during predictable waits and prefetches before the
//! predicted return. Everything is deterministic given a seed.

pub mod engine;
pub mod policy;
pub mod report;
pub mod rng;
pub mod trace;

pub use engine::{run, SimConfig, SimError};
pub use policy::{by_name, Lru, Plan, Policy, PolicyContext, Victim};
pub use report::{mean, percentile, render_table, SimReport};
pub use rng::Rng;
pub use trace::{synthesize, Session, SyntheticConfig, ToolWait, Trace, TraceStep};
