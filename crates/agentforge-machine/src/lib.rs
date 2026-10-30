//! Machine descriptions: the data the compiler, cost model and simulator share.
//!
//! A machine is a set of devices (GPUs, SRAM LPUs, NPUs, CPUs) with compute per precision,
//! device-attached memory and on-chip SRAM, plus memory tiers reachable from the devices (host
//! DRAM, NVMe, remote pools) and links. Descriptions are TOML files; five are bundled and more
//! can be loaded from disk. Numbers carry their sources in the file; vendor peaks are marked.

use std::collections::BTreeMap;

use agentforge_ir::Precision;
use serde::{Deserialize, Serialize};

/// Broad device class, used by placement heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceClass {
    Gpu,
    SramLpu,
    Npu,
    Cpu,
}

fn default_count() -> u32 {
    1
}

/// One device type in a machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub name: String,
    pub class: DeviceClass,
    /// Number of identical devices in the machine.
    #[serde(default = "default_count")]
    pub count: u32,
    /// Dense peak TFLOPS keyed by precision name (`fp16`, `fp8`, `nvfp4`, ...).
    pub compute_tflops: BTreeMap<String, f64>,
    /// Device-attached memory capacity in GB (10^9 bytes).
    pub memory_gb: f64,
    /// Device-attached memory bandwidth in TB/s.
    pub memory_bw_tbps: f64,
    /// On-chip SRAM (L2, LDS, Tensix SRAM, LPU SRAM) in MB.
    #[serde(default)]
    pub on_chip_sram_mb: f64,
    /// On-chip SRAM bandwidth in TB/s, when published.
    #[serde(default)]
    pub on_chip_sram_bw_tbps: Option<f64>,
    /// Transcendental (special function) throughput in TOPS, when published.
    #[serde(default)]
    pub sfu_tops: Option<f64>,
    #[serde(default)]
    pub tdp_w: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
}

impl Device {
    /// Peak TFLOPS usable for `precision`, following the precision's fallback chain.
    pub fn tflops(&self, precision: Precision) -> Option<f64> {
        precision
            .compute_fallbacks()
            .iter()
            .find_map(|key| self.compute_tflops.get(*key).copied())
    }

    pub fn memory_bytes(&self) -> f64 {
        self.memory_gb * 1e9
    }

    pub fn memory_bw_bytes_per_s(&self) -> f64 {
        self.memory_bw_tbps * 1e12
    }
}

/// A memory tier reachable from the devices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryTier {
    pub name: String,
    /// Capacity in GB (10^9 bytes).
    pub capacity_gb: f64,
    /// Read bandwidth into one device in GB/s.
    pub read_bw_gbps: f64,
    /// Write bandwidth from one device in GB/s.
    pub write_bw_gbps: f64,
    #[serde(default)]
    pub latency_us: f64,
    #[serde(default)]
    pub notes: Option<String>,
}

impl MemoryTier {
    pub fn capacity_bytes(&self) -> f64 {
        self.capacity_gb * 1e9
    }
}

/// A link between devices or between a device and the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub name: String,
    pub bw_gbps: f64,
    #[serde(default)]
    pub latency_us: f64,
    #[serde(default)]
    pub notes: Option<String>,
}

/// A complete machine description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub sources: Vec<String>,
    pub devices: Vec<Device>,
    #[serde(default)]
    pub tiers: Vec<MemoryTier>,
    #[serde(default)]
    pub links: Vec<Link>,
}

#[derive(Debug, thiserror::Error)]
pub enum MachineError {
    #[error("failed to parse machine description: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("failed to serialize machine description: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("invalid machine '{machine}': {reason}")]
    Invalid { machine: String, reason: String },
    #[error("unknown builtin machine '{0}'; known machines: {1}")]
    UnknownBuiltin(String, String),
}

const BUILTINS: &[(&str, &str)] = &[
    ("h200", include_str!("../machines/h200.toml")),
    ("b200", include_str!("../machines/b200.toml")),
    ("mi355x", include_str!("../machines/mi355x.toml")),
    (
        "blackhole-p150",
        include_str!("../machines/blackhole-p150.toml"),
    ),
    ("lpx-tier", include_str!("../machines/lpx-tier.toml")),
];

impl Machine {
    /// Parse and validate a TOML description.
    pub fn from_toml(text: &str) -> Result<Machine, MachineError> {
        let machine: Machine = toml::from_str(text)?;
        machine.validate()?;
        Ok(machine)
    }

    /// Serialize to TOML.
    pub fn to_toml(&self) -> Result<String, MachineError> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Names of the bundled machines.
    pub fn builtin_names() -> Vec<&'static str> {
        BUILTINS.iter().map(|(name, _)| *name).collect()
    }

    /// A bundled machine by name.
    pub fn builtin(name: &str) -> Result<Machine, MachineError> {
        match BUILTINS.iter().find(|(n, _)| *n == name) {
            Some((_, text)) => Machine::from_toml(text),
            None => Err(MachineError::UnknownBuiltin(
                name.to_string(),
                Self::builtin_names().join(", "),
            )),
        }
    }

    /// Structural validation: positive quantities, unique names, at least one device.
    pub fn validate(&self) -> Result<(), MachineError> {
        let invalid = |reason: String| MachineError::Invalid {
            machine: self.name.clone(),
            reason,
        };
        if self.name.trim().is_empty() {
            return Err(invalid("name is empty".to_string()));
        }
        if self.devices.is_empty() {
            return Err(invalid("no devices".to_string()));
        }
        let mut names = std::collections::HashSet::new();
        for d in &self.devices {
            if !names.insert(d.name.as_str()) {
                return Err(invalid(format!("duplicate device name '{}'", d.name)));
            }
            if d.count == 0 {
                return Err(invalid(format!("device '{}' has count 0", d.name)));
            }
            if d.memory_gb <= 0.0 || d.memory_bw_tbps <= 0.0 {
                return Err(invalid(format!(
                    "device '{}' needs positive memory_gb and memory_bw_tbps",
                    d.name
                )));
            }
            if d.compute_tflops.is_empty() {
                return Err(invalid(format!(
                    "device '{}' has no compute entries",
                    d.name
                )));
            }
            if let Some((key, value)) = d.compute_tflops.iter().find(|(_, v)| **v <= 0.0) {
                return Err(invalid(format!(
                    "device '{}' compute entry '{key}' must be positive, got {value}",
                    d.name
                )));
            }
        }
        let mut tier_names = std::collections::HashSet::new();
        for t in &self.tiers {
            if !tier_names.insert(t.name.as_str()) {
                return Err(invalid(format!("duplicate tier name '{}'", t.name)));
            }
            if t.capacity_gb <= 0.0 || t.read_bw_gbps <= 0.0 || t.write_bw_gbps <= 0.0 {
                return Err(invalid(format!(
                    "tier '{}' needs positive capacity and bandwidths",
                    t.name
                )));
            }
        }
        for l in &self.links {
            if l.bw_gbps <= 0.0 {
                return Err(invalid(format!(
                    "link '{}' needs positive bandwidth",
                    l.name
                )));
            }
        }
        Ok(())
    }

    /// A device by name, or the first device when `name` is `None`.
    pub fn device(&self, name: Option<&str>) -> Option<&Device> {
        match name {
            Some(n) => self.devices.iter().find(|d| d.name == n),
            None => self.devices.first(),
        }
    }

    pub fn tier(&self, name: &str) -> Option<&MemoryTier> {
        self.tiers.iter().find(|t| t.name == name)
    }

    /// Tiers ordered shallow to deep by read bandwidth (fastest first).
    pub fn tiers_by_speed(&self) -> Vec<&MemoryTier> {
        let mut tiers: Vec<&MemoryTier> = self.tiers.iter().collect();
        tiers.sort_by(|a, b| b.read_bw_gbps.total_cmp(&a.read_bw_gbps));
        tiers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_builtins_parse_and_validate() {
        for name in Machine::builtin_names() {
            let m = Machine::builtin(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(m.name, name);
            assert!(!m.devices.is_empty());
            assert!(!m.sources.is_empty(), "{name} has no sources");
        }
    }

    #[test]
    fn h200_numbers() {
        let m = Machine::builtin("h200").unwrap();
        let d = m.device(None).unwrap();
        assert!((d.memory_bw_tbps - 4.8).abs() < 1e-9);
        assert_eq!(d.tflops(Precision::Fp8), Some(1979.0));
        // No FP4 units on Hopper: NVFP4 falls back to the FP8 rate.
        assert_eq!(d.tflops(Precision::Nvfp4), Some(1979.0));
        assert!(m.tier("host_dram").is_some());
    }

    #[test]
    fn unknown_builtin_is_an_error() {
        assert!(matches!(
            Machine::builtin("nope"),
            Err(MachineError::UnknownBuiltin(..))
        ));
    }

    #[test]
    fn validation_rejects_bad_devices() {
        let text = r#"
name = "bad"
[[devices]]
name = "d"
class = "gpu"
memory_gb = 0
memory_bw_tbps = 1
[devices.compute_tflops]
fp16 = 1
"#;
        assert!(matches!(
            Machine::from_toml(text),
            Err(MachineError::Invalid { .. })
        ));
    }

    #[test]
    fn round_trips_through_toml() {
        let m = Machine::builtin("b200").unwrap();
        let text = m.to_toml().unwrap();
        let back = Machine::from_toml(&text).unwrap();
        assert_eq!(m, back);
    }
}
