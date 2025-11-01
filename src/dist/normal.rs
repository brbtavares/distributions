use crate::dist::{Continuous, DistError, Distribution, Moments};
use crate::{num, rng::RngCore};

#[derive(Debug, Clone, Copy)]
pub struct Normal {
    mu: f64,
    sigma: f64,
    inv_sigma: f64,
    norm: f64, // 1/(sigma*sqrt(2*pi))
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistError> {
        if !(sigma > 0.0 && sigma.is_finite() && mu.is_finite()) {
            return Err(DistError::InvalidParameter);
        }
        let inv_sigma = 1.0 / sigma;
        let norm = num::INV_SQRT_2PI * inv_sigma;
        Ok(Self {
            mu,
            sigma,
            inv_sigma,
            norm,
        })
    }
    #[inline]
    pub fn mean_param(&self) -> f64 {
        self.mu
    }
    #[inline]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl Distribution for Normal {
    type Value = f64;
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) * self.inv_sigma;
        num::standard_normal_cdf(z)
    }
    fn in_support(&self, x: f64) -> bool {
        x.is_finite()
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        // Box-Muller polar (Marsaglia) without external dependencies.
        loop {
            let u1 = 2.0 * rng.next_f64() - 1.0; // (-1,1)
            let u2 = 2.0 * rng.next_f64() - 1.0;
            let s = u1 * u1 + u2 * u2;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            let factor = (-2.0 * s.ln() / s).sqrt();
            return self.mu + self.sigma * u1 * factor;
        }
    }
}

impl Continuous for Normal {
    fn pdf(&self, x: f64) -> f64 {
        if !self.in_support(x) {
            return 0.0;
        }
        let z = (x - self.mu) * self.inv_sigma;
        self.norm * (-0.5 * z * z).exp()
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        self.mu + self.sigma * num::standard_normal_inv_cdf(p)
    }
}

impl Moments for Normal {
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn kurtosis(&self) -> f64 {
        0.0
    }
    fn entropy(&self) -> f64 {
        0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * self.sigma * self.sigma).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn normal_basic() {
        let n = Normal::new(0.0, 1.0).unwrap();
        assert!((n.pdf(0.0) - 0.3989422804014327).abs() < 1e-12);
        // CDF approximation via erf has typical error ~1e-7; use generous tolerance.
        assert!((n.cdf(0.0) - 0.5).abs() < 2e-6);
        let q = n.inv_cdf(0.975);
        assert!((q - 1.959963).abs() < 5e-4);
    }

    #[test]
    fn cdf_symmetry() {
        let n = Normal::new(0.0, 1.0).unwrap();
        let z = 0.7;
        let f = n.cdf(z);
        let f_neg = n.cdf(-z);
        assert!((f + f_neg - 1.0).abs() < 3e-15);
    }

    #[test]
    fn moments_higher() {
        let n = Normal::new(0.0, 2.0).unwrap();
        assert_eq!(n.skewness(), 0.0);
        assert_eq!(n.kurtosis(), 0.0);
        assert_eq!(n.kurtosis_full(), 3.0);
    }

    #[test]
    fn sampling_determinism() {
        let n = Normal::new(0.0, 1.0).unwrap();
        let mut rng1 = SplitMix64::seed_from_u64(42);
        let mut rng2 = SplitMix64::seed_from_u64(42);
        let x1 = n.sample(&mut rng1);
        let x2 = n.sample(&mut rng2);
        assert_eq!(x1.to_bits(), x2.to_bits());
    }

    #[test]
    fn entropy_normal() {
        let n = Normal::new(0.0, 2.0).unwrap();
        let expected = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * 4.0).ln();
        assert!((n.entropy() - expected).abs() < 1e-12);
    }
}
