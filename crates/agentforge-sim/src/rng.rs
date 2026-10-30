//! A small deterministic generator (SplitMix64) with the samplers the trace generator needs.

use agentforge_ir::LatencyModel;

#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn lognormal(&mut self, median: f64, sigma: f64) -> f64 {
        median * (sigma * self.normal()).exp()
    }

    pub fn exponential(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.next_f64()).ln()
    }

    /// Number of steps in `1..=max` with the given mean (geometric distribution).
    pub fn geometric(&mut self, mean: f64, max: u32) -> u32 {
        if mean <= 1.0 {
            return 1;
        }
        let p = 1.0 / mean;
        let u = 1.0 - self.next_f64();
        let k = (u.ln() / (1.0 - p).ln()).floor() as u32 + 1;
        k.clamp(1, max.max(1))
    }

    /// Draw a realized duration from a latency model.
    pub fn sample_latency(&mut self, model: &LatencyModel) -> f64 {
        match model {
            LatencyModel::Fixed { seconds } => *seconds,
            LatencyModel::LogNormal { median_s, sigma } => self.lognormal(*median_s, *sigma),
            LatencyModel::Uniform { min_s, max_s } => min_s + (max_s - min_s) * self.next_f64(),
            LatencyModel::Mixture { components } => {
                let total: f64 = components.iter().map(|c| c.weight).sum();
                if components.is_empty() || total <= 0.0 {
                    return 0.0;
                }
                let mut r = self.next_f64() * total;
                for c in components {
                    if r < c.weight {
                        return self.sample_latency(&c.model);
                    }
                    r -= c.weight;
                }
                let last = &components[components.len() - 1];
                self.sample_latency(&last.model)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            let x = a.next_f64();
            assert_eq!(x, b.next_f64());
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn lognormal_median_is_close() {
        let mut rng = Rng::new(7);
        let mut v: Vec<f64> = (0..20_000).map(|_| rng.lognormal(10.0, 1.0)).collect();
        v.sort_by(f64::total_cmp);
        let median = v[v.len() / 2];
        assert!((median - 10.0).abs() < 0.5, "{median}");
    }

    #[test]
    fn geometric_mean_is_close() {
        let mut rng = Rng::new(3);
        let n = 20_000;
        let sum: u64 = (0..n).map(|_| u64::from(rng.geometric(20.0, 10_000))).sum();
        let mean = sum as f64 / n as f64;
        assert!((mean - 20.0).abs() < 1.0, "{mean}");
    }
}
