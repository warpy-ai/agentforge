//! Runtime traits and a reference in-memory KV block store.
//!
//! The runtime is deliberately thin in the baseline: it fixes the interfaces the compiler's
//! plans will drive (block stores with explicit tiers, transports with completions, executors
//! for compiled steps) and provides an in-memory store that is useful for tests and for the
//! planned CPU backend.

use std::collections::HashMap;

use agentforge_ir::Step;

/// Content hash of a KV block (hash of the token prefix it covers).
pub type BlockHash = u64;

/// A memory tier identifier; `0` is device memory by convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TierId(pub u8);

impl TierId {
    pub const DEVICE: TierId = TierId(0);
    pub const HOST: TierId = TierId(1);
    pub const DISK: TierId = TierId(2);
}

/// One KV block.
#[derive(Debug, Clone, PartialEq)]
pub struct KvBlock {
    pub hash: BlockHash,
    pub tokens: u32,
    pub bytes: u64,
    pub tier: TierId,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StoreError {
    #[error("block of {bytes} bytes exceeds store capacity of {capacity} bytes")]
    TooLarge { bytes: u64, capacity: u64 },
}

/// A store of KV blocks with bounded capacity.
pub trait KvBlockStore {
    fn capacity_bytes(&self) -> u64;
    fn used_bytes(&self) -> u64;
    /// Insert a block, evicting least recently used blocks as needed; returns the evicted blocks.
    fn insert(&mut self, block: KvBlock) -> Result<Vec<KvBlock>, StoreError>;
    /// Look a block up and mark it used.
    fn lookup(&mut self, hash: BlockHash) -> Option<&KvBlock>;
    /// Remove a block.
    fn release(&mut self, hash: BlockHash) -> Option<KvBlock>;
    /// Remove the least recently used block.
    fn evict_lru(&mut self) -> Option<KvBlock>;
}

/// In-memory block store with LRU eviction.
#[derive(Debug)]
pub struct MemKvStore {
    capacity: u64,
    used: u64,
    clock: u64,
    blocks: HashMap<BlockHash, (KvBlock, u64)>,
}

impl MemKvStore {
    pub fn new(capacity_bytes: u64) -> MemKvStore {
        MemKvStore {
            capacity: capacity_bytes,
            used: 0,
            clock: 0,
            blocks: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl KvBlockStore for MemKvStore {
    fn capacity_bytes(&self) -> u64 {
        self.capacity
    }

    fn used_bytes(&self) -> u64 {
        self.used
    }

    fn insert(&mut self, block: KvBlock) -> Result<Vec<KvBlock>, StoreError> {
        if block.bytes > self.capacity {
            return Err(StoreError::TooLarge {
                bytes: block.bytes,
                capacity: self.capacity,
            });
        }
        self.release(block.hash);
        let mut evicted = Vec::new();
        while self.used + block.bytes > self.capacity {
            match self.evict_lru() {
                Some(b) => evicted.push(b),
                None => break,
            }
        }
        self.used += block.bytes;
        self.clock += 1;
        self.blocks.insert(block.hash, (block, self.clock));
        Ok(evicted)
    }

    fn lookup(&mut self, hash: BlockHash) -> Option<&KvBlock> {
        self.clock += 1;
        let clock = self.clock;
        let entry = self.blocks.get_mut(&hash)?;
        entry.1 = clock;
        Some(&entry.0)
    }

    fn release(&mut self, hash: BlockHash) -> Option<KvBlock> {
        let (block, _) = self.blocks.remove(&hash)?;
        self.used -= block.bytes;
        Some(block)
    }

    fn evict_lru(&mut self) -> Option<KvBlock> {
        let victim = self
            .blocks
            .iter()
            .min_by_key(|(_, (_, last))| *last)
            .map(|(h, _)| *h)?;
        self.release(victim)
    }
}

/// A completion time for an asynchronous copy, in seconds of simulated or wall time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Completion {
    pub ready_at_s: f64,
}

/// Moves bytes between tiers.
pub trait Transport {
    fn copy(&mut self, bytes: u64, from: TierId, to: TierId) -> Completion;
}

/// A transport that models a single channel of fixed bandwidth.
#[derive(Debug, Clone, Copy)]
pub struct ModelTransport {
    pub bw_bytes_per_s: f64,
    pub now_s: f64,
    pub busy_until_s: f64,
}

impl ModelTransport {
    pub fn new(bw_bytes_per_s: f64) -> ModelTransport {
        ModelTransport {
            bw_bytes_per_s,
            now_s: 0.0,
            busy_until_s: 0.0,
        }
    }
}

impl Transport for ModelTransport {
    fn copy(&mut self, bytes: u64, _from: TierId, _to: TierId) -> Completion {
        let start = self.now_s.max(self.busy_until_s);
        let ready = start + bytes as f64 / self.bw_bytes_per_s.max(1.0);
        self.busy_until_s = ready;
        Completion { ready_at_s: ready }
    }
}

/// Result of executing one step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutcome {
    pub output_tokens: u64,
    pub seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExecError {
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Executes compiled steps.
pub trait Executor {
    fn run_step(&mut self, step: &Step) -> Result<StepOutcome, ExecError>;
}

/// Placeholder executor until a backend exists.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullExecutor;

impl Executor for NullExecutor {
    fn run_step(&mut self, _step: &Step) -> Result<StepOutcome, ExecError> {
        Err(ExecError::Unsupported(
            "no backend is compiled into this build".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(hash: u64, bytes: u64) -> KvBlock {
        KvBlock {
            hash,
            tokens: 16,
            bytes,
            tier: TierId::DEVICE,
        }
    }

    #[test]
    fn lru_eviction_order() {
        let mut store = MemKvStore::new(100);
        assert!(store.insert(block(1, 40)).unwrap().is_empty());
        assert!(store.insert(block(2, 40)).unwrap().is_empty());
        assert!(store.lookup(1).is_some());
        let evicted = store.insert(block(3, 40)).unwrap();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].hash, 2);
        assert_eq!(store.used_bytes(), 80);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn too_large_is_rejected() {
        let mut store = MemKvStore::new(10);
        assert!(matches!(
            store.insert(block(1, 11)),
            Err(StoreError::TooLarge { .. })
        ));
        assert!(store.is_empty());
    }

    #[test]
    fn transport_serializes_copies() {
        let mut t = ModelTransport::new(1e9);
        let a = t.copy(1_000_000_000, TierId::DEVICE, TierId::HOST);
        let b = t.copy(1_000_000_000, TierId::HOST, TierId::DEVICE);
        assert!((a.ready_at_s - 1.0).abs() < 1e-9);
        assert!((b.ready_at_s - 2.0).abs() < 1e-9);
    }

    #[test]
    fn null_executor_refuses() {
        let program = agentforge_ir::AgentProgram::coding_agent_default();
        let mut exec = NullExecutor;
        assert!(exec.run_step(&program.steps[0]).is_err());
    }
}
