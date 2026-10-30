//! Tile IR skeleton.
//!
//! This module documents the shape of the tile-level IR that future backends lower from. It is
//! deliberately small: the baseline does not generate kernels yet, and the types exist so the
//! compiler, cost model and machine model agree on vocabulary from the start.
//!
//! Design notes carried over from the research phase:
//! - memory spaces are explicit because every 2025 to 2026 tile language (Triton/Gluon, CUDA
//!   Tile IR, TileLang, CuTe DSL, CubeCL) made them explicit;
//! - tensor memory is a first-class space because Blackwell's TMEM has its own allocation and
//!   lane rules and a recurring class of protocol bugs;
//! - async copies and barriers are operations with phases so their protocol can later be
//!   encoded in Rust types.

use crate::dtype::Precision;
use serde::{Deserialize, Serialize};

/// Where a tile lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySpace {
    /// Device memory (HBM, GDDR, LPDDR).
    Global,
    /// Shared memory, LDS or per-core SRAM.
    Shared,
    /// Registers.
    Register,
    /// Blackwell-style tensor memory.
    TensorMemory,
    /// Host memory reachable over a link.
    Host,
}

/// A two-dimensional tile type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileType {
    pub rows: u32,
    pub cols: u32,
    pub dtype: Precision,
    pub space: MemorySpace,
}

impl TileType {
    pub fn bytes(&self) -> u64 {
        self.dtype
            .bytes_for(u64::from(self.rows) * u64::from(self.cols))
    }
}

/// Identifier of a tile value inside a kernel.
pub type TileId = u32;

/// Tile-level operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum TileOp {
    /// Asynchronous copy between memory spaces, completed by a barrier phase.
    CopyAsync {
        src: TileId,
        dst: TileId,
        barrier: u32,
    },
    /// Wait for a barrier phase.
    BarrierWait { barrier: u32, phase: u32 },
    /// Matrix multiply-accumulate `acc += a * b` with explicit accumulation precision.
    Mma {
        a: TileId,
        b: TileId,
        acc: TileId,
        accumulate_in: Precision,
    },
    /// Row or column reduction.
    Reduce {
        src: TileId,
        dst: TileId,
        axis: u8,
        kind: ReduceKind,
    },
    /// Element-wise operation.
    Elementwise {
        srcs: Vec<TileId>,
        dst: TileId,
        kind: ElementwiseKind,
    },
    /// Re-quantize a tile into another precision.
    Convert { src: TileId, dst: TileId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduceKind {
    Sum,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementwiseKind {
    Add,
    Mul,
    Exp2,
    Recip,
    Silu,
}

/// A kernel: typed tiles and a list of operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileKernel {
    pub name: String,
    pub tiles: Vec<TileType>,
    pub ops: Vec<TileOp>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_bytes_account_for_block_scales() {
        let t = TileType {
            rows: 128,
            cols: 128,
            dtype: Precision::Nvfp4,
            space: MemorySpace::Shared,
        };
        assert_eq!(t.bytes(), 9_216);
    }
}
