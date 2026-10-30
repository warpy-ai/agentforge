//! Intermediate representations for AgentForge.
//!
//! Three layers live here:
//! - [`dtype`]: numeric formats as first-class types, including block-scaled sub-byte formats;
//! - [`model`]: model architecture descriptions sufficient for cost modelling and KV planning;
//! - [`agent`]: the Agent Program IR, which describes the loop of LLM calls, tool calls and
//!   branches that a serving system otherwise has to infer from independent requests.
//!
//! [`tile`] holds the documented skeleton of the tile-level IR used by future backends.

pub mod agent;
pub mod dtype;
pub mod model;
pub mod tile;

pub use agent::{
    AgentProgram, BlockKind, LatencyModel, MixtureComponent, PromptBlock, Step, StepId, ToolCall,
    Transition,
};
pub use dtype::{BlockFormat, ParsePrecisionError, Precision, Scalar, ScaleKind};
pub use model::{AttentionKind, ModelSpec, MoeSpec};
