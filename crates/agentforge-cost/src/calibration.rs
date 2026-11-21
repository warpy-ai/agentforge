//! Calibration of the cost model against published serving measurements.
//!
//! A calibration file lists measured offline throughputs, one model on one machine per row.
//! [`Calibration::evaluate`] predicts each row with [`max_throughput`] and reports the relative
//! error; [`fit`] grid-searches the [`Efficiency`] knobs that minimise the worst row's error.

use agentforge_ir::{ModelSpec, Precision};
use agentforge_machine::{Machine, MachineError};
use serde::Deserialize;

use crate::{max_throughput, CostError, Efficiency, SteadyState, Target};

/// A set of published measurements and where they came from.
#[derive(Debug, Clone, Deserialize)]
pub struct Calibration {
    pub name: String,
    pub source: String,
    pub retrieved: String,
    pub metric: String,
    pub rows: Vec<CalibrationRow>,
}

/// One published measurement: output tokens per second for a fixed request shape.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CalibrationRow {
    /// Bundled machine name.
    pub machine: String,
    /// Bundled model preset name.
    pub model: String,
    pub weight_precision: Precision,
    pub kv_precision: Precision,
    pub tp: u32,
    pub isl: u64,
    pub osl: u64,
    pub tokens_per_s: f64,
}

/// A row's prediction and its relative error.
#[derive(Debug, Clone, PartialEq)]
pub struct RowResult {
    pub row: CalibrationRow,
    pub predicted: SteadyState,
    /// `predicted / published - 1`.
    pub error: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error(transparent)]
    Cost(#[from] CostError),
    #[error(transparent)]
    Machine(#[from] MachineError),
    #[error("unknown model preset '{0}'")]
    UnknownModel(String),
}

impl Calibration {
    /// The bundled TensorRT-LLM rows for Llama 3.3 70B on H200 and B200.
    pub fn builtin() -> Calibration {
        Calibration::from_toml(include_str!(
            "../calibration/trtllm-perf-overview-2025-09-15.toml"
        ))
        .expect("the bundled calibration file parses")
    }

    pub fn from_toml(text: &str) -> Result<Calibration, toml::de::Error> {
        toml::from_str(text)
    }

    /// Predict every row with `eff`, reserving `reserve_fraction` of device memory for
    /// activations and the runtime, and sweeping batches up to `max_batch`.
    pub fn evaluate(
        &self,
        eff: &Efficiency,
        reserve_fraction: f64,
        max_batch: u64,
    ) -> Result<Vec<RowResult>, CalibrationError> {
        self.rows
            .iter()
            .map(|row| evaluate_row(row, eff, reserve_fraction, max_batch))
            .collect()
    }
}

fn evaluate_row(
    row: &CalibrationRow,
    eff: &Efficiency,
    reserve_fraction: f64,
    max_batch: u64,
) -> Result<RowResult, CalibrationError> {
    let machine = Machine::builtin(&row.machine)?;
    let device = &machine.devices[0];
    let mut model = ModelSpec::preset(&row.model)
        .ok_or_else(|| CalibrationError::UnknownModel(row.model.clone()))?;
    model.weight_precision = row.weight_precision;
    model.kv_precision = row.kv_precision;
    let target = Target {
        device,
        tp: row.tp,
        link: machine.fastest_link(),
        eff: *eff,
    };
    let budget = f64::from(row.tp) * device.memory_bytes() * (1.0 - reserve_fraction)
        - model.weight_bytes() as f64;
    let predicted = max_throughput(&model, &target, row.isl, row.osl, budget, max_batch)?;
    Ok(RowResult {
        row: row.clone(),
        predicted,
        error: predicted.output_tokens_per_s / row.tokens_per_s - 1.0,
    })
}

/// The largest absolute relative error.
pub fn max_abs_error(results: &[RowResult]) -> f64 {
    results.iter().map(|r| r.error.abs()).fold(0.0, f64::max)
}

/// The geometric mean of `1 + |error|`, minus one.
pub fn geomean_abs_error(results: &[RowResult]) -> f64 {
    let log_sum: f64 = results.iter().map(|r| r.error.abs().ln_1p()).sum();
    (log_sum / results.len() as f64).exp() - 1.0
}

/// Grid that [`fit`] searches: inclusive `(low, high)` ranges stepped by [`FIT_STEP`], and a list
/// of per-layer overheads. `link_util` stays at its default because the TP 1 and TP 2 rows cannot
/// pin it.
pub const FIT_MFU_PREFILL: (f64, f64) = (0.30, 0.80);
pub const FIT_MFU_DECODE: (f64, f64) = (0.20, 0.80);
pub const FIT_BW_UTIL: (f64, f64) = (0.60, 0.95);
pub const FIT_STEP: f64 = 0.05;
pub const FIT_PER_LAYER_OVERHEAD_S: [f64; 4] = [5e-6, 1e-5, 2e-5, 4e-5];

fn grid((low, high): (f64, f64)) -> impl Iterator<Item = f64> + Clone {
    let steps = ((high - low) / FIT_STEP).round() as u32;
    (0..=steps).map(move |i| ((low + f64::from(i) * FIT_STEP) * 100.0).round() / 100.0)
}

/// The grid point with the smallest maximum absolute error over `cal`'s rows, and that error.
pub fn fit(
    cal: &Calibration,
    reserve_fraction: f64,
    max_batch: u64,
) -> Result<(Efficiency, f64), CalibrationError> {
    let link_util = Efficiency::default().link_util;
    let mut best: Option<(Efficiency, f64)> = None;
    for mfu_prefill in grid(FIT_MFU_PREFILL) {
        for mfu_decode in grid(FIT_MFU_DECODE) {
            for bw_util in grid(FIT_BW_UTIL) {
                for per_layer_overhead_s in FIT_PER_LAYER_OVERHEAD_S {
                    let eff = Efficiency {
                        mfu_prefill,
                        mfu_decode,
                        bw_util,
                        per_layer_overhead_s,
                        link_util,
                    };
                    let error = max_abs_error(&cal.evaluate(&eff, reserve_fraction, max_batch)?);
                    if best.is_none_or(|(_, e)| error < e) {
                        best = Some((eff, error));
                    }
                }
            }
        }
    }
    Ok(best.expect("the fit grid is not empty"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_rows_parse() {
        let cal = Calibration::builtin();
        assert_eq!(cal.rows.len(), 22);
        assert!(cal
            .rows
            .iter()
            .any(|r| r.weight_precision == Precision::Nvfp4));
        assert!(cal.rows.iter().all(|r| r.kv_precision == Precision::Fp8));
    }

    #[test]
    fn grid_points_land_on_hundredths() {
        let points: Vec<f64> = grid(FIT_BW_UTIL).collect();
        assert_eq!(points, [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]);
    }
}
