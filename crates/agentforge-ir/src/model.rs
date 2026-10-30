//! Model architecture descriptions sufficient for cost modelling and KV planning.

use crate::dtype::Precision;
use serde::{Deserialize, Serialize};

/// Attention family.
///
/// The baseline describes one family per model; models that mix families per layer (DeepSeek
/// V4 pairs sliding-window and compressed attention in every layer) are approximated by the
/// dominant one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionKind {
    /// Multi-head attention with one KV head per query head.
    Dense,
    /// Grouped-query attention sharing `n_kv_heads` KV heads.
    Gqa,
    /// Multi-head latent attention: KV is a per-layer latent vector plus a RoPE part.
    Mla { latent_dim: u32, rope_dim: u32 },
    /// Sliding-window attention over the last `window` tokens.
    SlidingWindow { window: u32 },
    /// Compressed attention: KV compressed `ratio` to one, queries attend to `top_k` entries.
    Compressed { ratio: u32, top_k: u32 },
}

impl AttentionKind {
    /// Number of KV entries that must be stored for a context of `ctx` tokens.
    pub fn stored_kv_tokens(&self, ctx: u64) -> u64 {
        match *self {
            AttentionKind::SlidingWindow { window } => ctx.min(u64::from(window)),
            AttentionKind::Compressed { ratio, .. } => ctx.div_ceil(u64::from(ratio.max(1))),
            _ => ctx,
        }
    }

    /// Number of KV entries a query attends to at context length `ctx`.
    pub fn attended_tokens(&self, ctx: u64) -> u64 {
        match *self {
            AttentionKind::SlidingWindow { window } => ctx.min(u64::from(window)),
            AttentionKind::Compressed { ratio, top_k } => {
                ctx.div_ceil(u64::from(ratio.max(1))).min(u64::from(top_k))
            }
            _ => ctx,
        }
    }
}

/// Mixture-of-experts configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoeSpec {
    pub n_experts: u32,
    pub active_experts: u32,
    #[serde(default)]
    pub shared_experts: u32,
    pub expert_d_ff: u32,
}

/// A decoder-only transformer description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Dense feed-forward width (ignored for the expert layers of MoE models).
    pub d_ff: u32,
    pub vocab: u32,
    #[serde(default)]
    pub moe: Option<MoeSpec>,
    pub attention: AttentionKind,
    pub weight_precision: Precision,
    pub kv_precision: Precision,
    /// Published total parameter count, when the generic formula is too coarse.
    #[serde(default)]
    pub total_params_override: Option<u64>,
    /// Published active parameters per token, when the generic formula is too coarse.
    #[serde(default)]
    pub active_params_override: Option<u64>,
}

impl ModelSpec {
    fn attention_params_per_layer(&self) -> u64 {
        let d = u64::from(self.d_model);
        let hd = u64::from(self.head_dim);
        let q = d * u64::from(self.n_heads) * hd;
        let kv = 2 * d * u64::from(self.n_kv_heads) * hd;
        let o = u64::from(self.n_heads) * hd * d;
        q + kv + o
    }

    fn ffn_params_per_layer(&self, active_only: bool) -> u64 {
        let d = u64::from(self.d_model);
        match self.moe {
            None => 3 * d * u64::from(self.d_ff),
            Some(moe) => {
                let experts = if active_only {
                    u64::from(moe.active_experts + moe.shared_experts)
                } else {
                    u64::from(moe.n_experts + moe.shared_experts)
                };
                3 * d * u64::from(moe.expert_d_ff) * experts + d * u64::from(moe.n_experts)
            }
        }
    }

    /// Total stored parameters (embeddings and LM head included).
    pub fn total_params(&self) -> u64 {
        if let Some(total) = self.total_params_override {
            return total;
        }
        let per_layer = self.attention_params_per_layer() + self.ffn_params_per_layer(false);
        u64::from(self.n_layers) * per_layer + 2 * u64::from(self.vocab) * u64::from(self.d_model)
    }

    /// Parameters touched by one token's forward pass (LM head included, embedding lookup not).
    pub fn active_params_per_token(&self) -> u64 {
        if let Some(active) = self.active_params_override {
            return active;
        }
        let per_layer = self.attention_params_per_layer() + self.ffn_params_per_layer(true);
        u64::from(self.n_layers) * per_layer + u64::from(self.vocab) * u64::from(self.d_model)
    }

    /// Bytes of weights at the configured weight precision.
    pub fn weight_bytes(&self) -> u64 {
        self.weight_precision.bytes_for(self.total_params())
    }

    /// KV elements stored per token across all layers.
    pub fn kv_elements_per_token(&self) -> u64 {
        let layers = u64::from(self.n_layers);
        match self.attention {
            AttentionKind::Mla {
                latent_dim,
                rope_dim,
            } => layers * u64::from(latent_dim + rope_dim),
            _ => layers * 2 * u64::from(self.n_kv_heads) * u64::from(self.head_dim),
        }
    }

    /// KV bytes stored per token at the configured KV precision.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_precision.bytes_for(self.kv_elements_per_token())
    }

    /// Names of the bundled presets.
    pub fn preset_names() -> &'static [&'static str] {
        &[
            "llama-3.1-8b",
            "llama-3.1-70b",
            "qwen3-32b",
            "deepseek-v3",
            "gpt-oss-120b",
        ]
    }

    /// A bundled preset by name.
    pub fn preset(name: &str) -> Option<ModelSpec> {
        let spec = match name {
            "llama-3.1-8b" => ModelSpec {
                name: name.to_string(),
                n_layers: 32,
                d_model: 4096,
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                d_ff: 14336,
                vocab: 128_256,
                moe: None,
                attention: AttentionKind::Gqa,
                weight_precision: Precision::Bf16,
                kv_precision: Precision::Fp16,
                total_params_override: None,
                active_params_override: None,
            },
            "llama-3.1-70b" => ModelSpec {
                name: name.to_string(),
                n_layers: 80,
                d_model: 8192,
                n_heads: 64,
                n_kv_heads: 8,
                head_dim: 128,
                d_ff: 28672,
                vocab: 128_256,
                moe: None,
                attention: AttentionKind::Gqa,
                weight_precision: Precision::Fp8,
                kv_precision: Precision::Fp16,
                total_params_override: None,
                active_params_override: None,
            },
            "qwen3-32b" => ModelSpec {
                name: name.to_string(),
                n_layers: 64,
                d_model: 5120,
                n_heads: 64,
                n_kv_heads: 8,
                head_dim: 128,
                d_ff: 25600,
                vocab: 151_936,
                moe: None,
                attention: AttentionKind::Gqa,
                weight_precision: Precision::Bf16,
                kv_precision: Precision::Fp16,
                total_params_override: None,
                active_params_override: None,
            },
            "deepseek-v3" => ModelSpec {
                name: name.to_string(),
                n_layers: 61,
                d_model: 7168,
                n_heads: 128,
                n_kv_heads: 128,
                head_dim: 128,
                d_ff: 18432,
                vocab: 129_280,
                moe: Some(MoeSpec {
                    n_experts: 256,
                    active_experts: 8,
                    shared_experts: 1,
                    expert_d_ff: 2048,
                }),
                attention: AttentionKind::Mla {
                    latent_dim: 512,
                    rope_dim: 64,
                },
                weight_precision: Precision::Fp8,
                kv_precision: Precision::Fp16,
                total_params_override: Some(671_000_000_000),
                active_params_override: Some(37_000_000_000),
            },
            "gpt-oss-120b" => ModelSpec {
                name: name.to_string(),
                n_layers: 36,
                d_model: 2880,
                n_heads: 64,
                n_kv_heads: 8,
                head_dim: 64,
                d_ff: 2880,
                vocab: 201_088,
                moe: Some(MoeSpec {
                    n_experts: 128,
                    active_experts: 4,
                    shared_experts: 0,
                    expert_d_ff: 2880,
                }),
                attention: AttentionKind::Gqa,
                weight_precision: Precision::Mxfp4,
                kv_precision: Precision::Bf16,
                total_params_override: Some(117_000_000_000),
                active_params_override: Some(5_100_000_000),
            },
            _ => return None,
        };
        Some(spec)
    }

    /// All bundled presets.
    pub fn presets() -> Vec<ModelSpec> {
        Self::preset_names()
            .iter()
            .filter_map(|n| Self::preset(n))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn within(actual: u64, expected: u64, tolerance: f64) -> bool {
        let a = actual as f64;
        let e = expected as f64;
        ((a - e) / e).abs() <= tolerance
    }

    #[test]
    fn llama_70b_kv_bytes_per_token() {
        let m = ModelSpec::preset("llama-3.1-70b").unwrap();
        assert_eq!(m.kv_bytes_per_token(), 327_680);
    }

    #[test]
    fn llama_param_counts_match_published() {
        let m70 = ModelSpec::preset("llama-3.1-70b").unwrap();
        assert!(
            within(m70.total_params(), 70_550_000_000, 0.03),
            "{}",
            m70.total_params()
        );
        let m8 = ModelSpec::preset("llama-3.1-8b").unwrap();
        assert!(
            within(m8.total_params(), 8_030_000_000, 0.03),
            "{}",
            m8.total_params()
        );
    }

    #[test]
    fn mla_kv_is_small() {
        let m = ModelSpec::preset("deepseek-v3").unwrap();
        let kv = m.kv_bytes_per_token();
        assert!((69_000..=71_000).contains(&kv), "{kv}");
    }

    #[test]
    fn attention_kinds_bound_context() {
        let swa = AttentionKind::SlidingWindow { window: 128 };
        assert_eq!(swa.stored_kv_tokens(10_000), 128);
        let comp = AttentionKind::Compressed {
            ratio: 4,
            top_k: 512,
        };
        assert_eq!(comp.stored_kv_tokens(10_000), 2_500);
        assert_eq!(comp.attended_tokens(10_000), 512);
        assert_eq!(AttentionKind::Gqa.attended_tokens(10_000), 10_000);
    }

    #[test]
    fn all_presets_load() {
        assert_eq!(ModelSpec::presets().len(), ModelSpec::preset_names().len());
    }
}
