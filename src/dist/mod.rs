//! Collection of probability distributions.
//! This module groups all distribution implementations under `dist`.
use crate::rng;

pub mod bernoulli;
pub mod beta;
pub mod binomial;
pub mod chisquared;
pub mod exponential;
pub mod gamma;
pub mod geometric;
pub mod lognormal;
pub mod normal;
pub mod poisson;
pub mod uniform;
/// Basic moments available for a distribution.
pub trait Moments {
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
}

/// Basic trait for distributions.
pub trait Distribution {
    type Value;
    fn cdf(&self, x: Self::Value) -> f64;
    fn sample<R: rng::RngCore>(&self, rng: &mut R) -> Self::Value;
    fn in_support(&self, x: Self::Value) -> bool;
}

/// Trait for continuous real-valued distributions.
pub trait Continuous: Distribution<Value = f64> {
    /// Returns f(x) (density / pdf).
    fn pdf(&self, x: f64) -> f64;
    /// Quantile: F^{-1}(p) for p in (0,1).
    fn inv_cdf(&self, p: f64) -> f64;
}

/// Trait for discrete distributions over {0,1} or small integers.
pub trait Discrete: Distribution<Value = i64> {
    /// pmf(x)
    fn pmf(&self, x: Self::Value) -> f64;
    fn inv_cdf(&self, p: f64) -> Self::Value;
}

/// Error returned when constructing distributions with invalid parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistError {
    InvalidParameter,
}

#[cfg(test)]
mod tests {
    use crate::dist::{Continuous, Distribution};
    use crate::dist::{normal::Normal, uniform::Uniform};
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
    fn uniform_basic() {
        let u = Uniform::new(2.0, 5.0).unwrap();
        assert!((u.pdf(3.0) - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(u.cdf(1.0), 0.0);
        assert_eq!(u.cdf(6.0), 1.0);
        assert!((u.cdf(2.0) - 0.0).abs() < 1e-15);
        assert!((u.cdf(3.5) - 0.5).abs() < 1e-15);
        let q = u.inv_cdf(0.3);
        assert!((q - 2.9).abs() < 1e-15);
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
}
