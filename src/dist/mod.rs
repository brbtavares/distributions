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
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// Skewness (standardized 3rd central moment).
    fn skewness(&self) -> f64;
    /// Excess kurtosis (kurtosis − 3).
    fn kurtosis(&self) -> f64;
    /// Full kurtosis (i.e., excess kurtosis + 3).
    fn kurtosis_full(&self) -> f64 {
        self.kurtosis() + 3.0
    }
    /// Entropy: Shannon (discrete) or differential entropy (continuous), in nats.
    fn entropy(&self) -> f64;
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
