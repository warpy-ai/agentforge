//! Numeric formats as first-class types.
//!
//! Inference silicon disagrees on sub-byte formats: NVIDIA's NVFP4 pairs E2M1 elements with an
//! E4M3 scale per 16 elements, while the OCP Microscaling (MX) formats used by AMD CDNA4 pair
//! them with an E8M0 scale per 32. A compiler that wants to move a quantized model between
//! vendors has to carry the block format, not just "fp4", in its types.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Scalar element formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scalar {
    F64,
    F32,
    /// TensorFloat-32: 19 significant bits stored in a 32-bit container.
    Tf32,
    Bf16,
    F16,
    F8E4M3,
    F8E5M2,
    F6E2M3,
    F6E3M2,
    F4E2M1,
    I32,
    I8,
    I4,
    U8,
    Bool,
}

impl Scalar {
    /// Storage bits per element.
    pub const fn bits(self) -> u32 {
        match self {
            Scalar::F64 => 64,
            Scalar::F32 | Scalar::I32 | Scalar::Tf32 => 32,
            Scalar::Bf16 | Scalar::F16 => 16,
            Scalar::F8E4M3 | Scalar::F8E5M2 | Scalar::I8 | Scalar::U8 => 8,
            Scalar::F6E2M3 | Scalar::F6E3M2 => 6,
            Scalar::F4E2M1 | Scalar::I4 => 4,
            Scalar::Bool => 1,
        }
    }

    pub const fn is_float(self) -> bool {
        !matches!(
            self,
            Scalar::I32 | Scalar::I8 | Scalar::I4 | Scalar::U8 | Scalar::Bool
        )
    }
}

/// Encodings used for per-block scale factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleKind {
    /// Unsigned power-of-two scale used by the OCP MX formats.
    E8M0,
    /// Unsigned E4M3 scale used by NVIDIA's NVFP4.
    E4M3,
    /// Unsigned E5M3 scale introduced with PTX ISA 9.4.
    E5M3,
}

impl ScaleKind {
    /// Storage bits per scale factor.
    pub const fn bits(self) -> u32 {
        8
    }
}

/// A block-scaled format: `block_size` elements share one scale factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockFormat {
    pub element: Scalar,
    pub scale: ScaleKind,
    pub block_size: u32,
}

impl BlockFormat {
    pub const MXFP4: BlockFormat = BlockFormat {
        element: Scalar::F4E2M1,
        scale: ScaleKind::E8M0,
        block_size: 32,
    };
    pub const MXFP6: BlockFormat = BlockFormat {
        element: Scalar::F6E2M3,
        scale: ScaleKind::E8M0,
        block_size: 32,
    };
    pub const MXFP8: BlockFormat = BlockFormat {
        element: Scalar::F8E4M3,
        scale: ScaleKind::E8M0,
        block_size: 32,
    };
    pub const NVFP4: BlockFormat = BlockFormat {
        element: Scalar::F4E2M1,
        scale: ScaleKind::E4M3,
        block_size: 16,
    };

    /// Average storage bits per element including the amortized scale.
    pub fn bits_per_element(&self) -> f64 {
        f64::from(self.element.bits()) + f64::from(self.scale.bits()) / f64::from(self.block_size)
    }
}

/// Deployment precision names used in model specs and machine compute tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Fp32,
    Bf16,
    Fp16,
    Fp8,
    Mxfp8,
    Mxfp6,
    Mxfp4,
    Nvfp4,
    Int8,
    Int4,
}

impl Precision {
    pub const ALL: [Precision; 10] = [
        Precision::Fp32,
        Precision::Bf16,
        Precision::Fp16,
        Precision::Fp8,
        Precision::Mxfp8,
        Precision::Mxfp6,
        Precision::Mxfp4,
        Precision::Nvfp4,
        Precision::Int8,
        Precision::Int4,
    ];

    /// The key used in machine files and on the command line.
    pub const fn key(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Bf16 => "bf16",
            Precision::Fp16 => "fp16",
            Precision::Fp8 => "fp8",
            Precision::Mxfp8 => "mxfp8",
            Precision::Mxfp6 => "mxfp6",
            Precision::Mxfp4 => "mxfp4",
            Precision::Nvfp4 => "nvfp4",
            Precision::Int8 => "int8",
            Precision::Int4 => "int4",
        }
    }

    pub const fn block_format(self) -> Option<BlockFormat> {
        match self {
            Precision::Mxfp8 => Some(BlockFormat::MXFP8),
            Precision::Mxfp6 => Some(BlockFormat::MXFP6),
            Precision::Mxfp4 => Some(BlockFormat::MXFP4),
            Precision::Nvfp4 => Some(BlockFormat::NVFP4),
            _ => None,
        }
    }

    pub const fn scalar(self) -> Scalar {
        match self {
            Precision::Fp32 => Scalar::F32,
            Precision::Bf16 => Scalar::Bf16,
            Precision::Fp16 => Scalar::F16,
            Precision::Fp8 | Precision::Mxfp8 => Scalar::F8E4M3,
            Precision::Mxfp6 => Scalar::F6E2M3,
            Precision::Mxfp4 | Precision::Nvfp4 => Scalar::F4E2M1,
            Precision::Int8 => Scalar::I8,
            Precision::Int4 => Scalar::I4,
        }
    }

    pub const fn is_block_scaled(self) -> bool {
        self.block_format().is_some()
    }

    /// Average storage bits per element, scale factors included.
    pub fn bits_per_element(self) -> f64 {
        match self.block_format() {
            Some(block) => block.bits_per_element(),
            None => f64::from(self.scalar().bits()),
        }
    }

    /// Bytes needed to store `elements` values, rounded up.
    pub fn bytes_for(self, elements: u64) -> u64 {
        (elements as f64 * self.bits_per_element() / 8.0).ceil() as u64
    }

    /// Compute-table keys to try, from the exact format to the closest fallback a device may
    /// execute the format on (for example NVFP4 weights dequantized to FP8 on Hopper).
    pub fn compute_fallbacks(self) -> &'static [&'static str] {
        match self {
            Precision::Fp32 => &["fp32", "tf32", "fp16"],
            Precision::Bf16 => &["bf16", "fp16"],
            Precision::Fp16 => &["fp16", "bf16"],
            Precision::Fp8 => &["fp8", "fp16", "bf16"],
            Precision::Mxfp8 => &["mxfp8", "fp8", "fp16"],
            Precision::Mxfp6 => &["mxfp6", "mxfp4", "fp8", "fp16"],
            Precision::Mxfp4 => &["mxfp4", "nvfp4", "fp4", "fp8", "fp16"],
            Precision::Nvfp4 => &["nvfp4", "mxfp4", "fp4", "fp8", "fp16"],
            Precision::Int8 => &["int8", "fp8", "fp16"],
            Precision::Int4 => &["int4", "mxfp4", "nvfp4", "fp8", "fp16"],
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}

/// Error returned when a precision name is not recognized.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown precision '{0}' (expected one of fp32, bf16, fp16, fp8, mxfp8, mxfp6, mxfp4, nvfp4, int8, int4)")]
pub struct ParsePrecisionError(pub String);

impl FromStr for Precision {
    type Err = ParsePrecisionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        Precision::ALL
            .iter()
            .copied()
            .find(|p| p.key() == lower)
            .ok_or_else(|| ParsePrecisionError(s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_formats_amortize_scales() {
        assert!((BlockFormat::NVFP4.bits_per_element() - 4.5).abs() < 1e-12);
        assert!((BlockFormat::MXFP4.bits_per_element() - 4.25).abs() < 1e-12);
        assert!((BlockFormat::MXFP8.bits_per_element() - 8.25).abs() < 1e-12);
    }

    #[test]
    fn precision_round_trips_through_keys() {
        for p in Precision::ALL {
            assert_eq!(p.key().parse::<Precision>().unwrap(), p);
            assert_eq!(p.to_string(), p.key());
        }
        assert!("fp3".parse::<Precision>().is_err());
    }

    #[test]
    fn bytes_round_up() {
        assert_eq!(Precision::Mxfp4.bytes_for(32), 17);
        assert_eq!(Precision::Fp16.bytes_for(3), 6);
        assert_eq!(Precision::Int4.bytes_for(3), 2);
    }
}
