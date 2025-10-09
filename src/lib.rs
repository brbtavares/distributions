//! Probability distributions library with no external dependencies.
//! Focus: numerical accuracy, clear API, and extensibility.
//!
//! Initial features:
//! - Internal pseudo-random number generator (SplitMix64)
//! - Generic trait `ContinuousDistribution`
//! - Distributions: Uniform, Normal, Exponential
//! - Simple discrete distribution: Bernoulli
//! - PDF, CDF, inverse CDF (quantile), mean and variance
//! - Allocation-free sampling
//!
//! Quick examples:
//! ```
//! use distributions::{rng::SplitMix64, normal::Normal, ContinuousDistribution};
//! let normal = Normal::new(0.0, 1.0).unwrap();
//! let mut rng = SplitMix64::seed_from_u64(123);
//! let x = normal.sample(&mut rng);
//! let p = normal.pdf(0.0);
//! let c = normal.cdf(0.0);
//! let q = normal.inv_cdf(0.975); // ~ 1.96
//! assert!((p - 0.39894228).abs() < 1e-7);
//! assert!((c - 0.5).abs() < 2e-6); // tolerance due to erf approximation
//! assert!((q - 1.95996).abs() < 5e-3);
//! ```

pub mod rng;
pub mod dist;

// Back-compat re-exports at crate root
pub use dist::{uniform, normal, exponential, bernoulli};
pub use dist::{Uniform, Normal, Exponential, Bernoulli};

/// Trait for continuous real-valued distributions.
pub trait ContinuousDistribution {
    /// Returns f(x) (density / pdf).
    fn pdf(&self, x: f64) -> f64;
    /// Returns F(x) (CDF).
    fn cdf(&self, x: f64) -> f64;
    /// Quantile: F^{-1}(p) for p in (0,1).
    fn inv_cdf(&self, p: f64) -> f64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    /// Draw a single sample using the provided RNG.
    fn sample<R: rng::RngCore>(&self, rng: &mut R) -> f64;
}

/// Trait for discrete distributions over {0,1} or small integers.
pub trait DiscreteDistribution {
    type Value: Copy + core::fmt::Debug + PartialEq;
    /// pmf(x)
    fn pmf(&self, x: Self::Value) -> f64;
    fn cdf(&self, x: Self::Value) -> f64;
    fn inv_cdf(&self, p: f64) -> Self::Value;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn sample<R: rng::RngCore>(&self, rng: &mut R) -> Self::Value;
}

/// Error returned when constructing distributions with invalid parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistError {
    InvalidParameter,
}

/// Frequently used numerical constants.
pub(crate) mod consts {
    pub const SQRT_2: f64 = 1.41421356237309504880168872420969808_f64;
    pub const INV_SQRT_2: f64 = 1.0 / SQRT_2;
    pub const SQRT_2PI: f64 = 2.50662827463100050241576528481104525_f64; // sqrt(2*pi)
    pub const INV_SQRT_2PI: f64 = 1.0 / SQRT_2PI; // 1 / sqrt(2*pi)
    pub const LN_2: f64 = 0.69314718055994530941723212145817657_f64;
}

/// Internal math helper functions.
pub(crate) mod math {
    use super::consts::{INV_SQRT_2PI};

    /// Standard normal PDF.
    #[inline]
    pub fn standard_normal_pdf(z: f64) -> f64 {
        (-0.5 * z * z).exp() * INV_SQRT_2PI
    }

    /// Fast approximation of erf(x) (Abramowitz & Stegun 7.1.26).
    pub fn erf(x: f64) -> f64 {
    // Preserve sign.
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }

    /// Standard normal CDF via erf.
    pub fn standard_normal_cdf(z: f64) -> f64 {
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    }

    /// Standard normal inverse CDF (probit) using Peter J. Acklam's rational approximation.
    /// Typical absolute error < 4.5e-4 in double precision.
    pub fn standard_normal_inv_cdf(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "p must be in (0,1)");

    // Coefficients (Acklam 2003). See public documentation.
        const A: [f64; 6] = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        const B: [f64; 5] = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        const C: [f64; 6] = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        const D: [f64; 4] = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];
        const P_LOW: f64 = 0.02425;
        const P_HIGH: f64 = 1.0 - P_LOW;

        let q: f64;
        let r: f64;
        let x: f64;
    if p < P_LOW { // Lower tail region
            q = (-2.0 * p.ln()).sqrt();
            x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
                ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
            return -x;
        }
    if p > P_HIGH { // Upper tail region
            q = (-2.0 * (1.0 - p).ln()).sqrt();
            x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
                ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
            return x;
        }
    // Central region
        q = p - 0.5;
        r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q /
            (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{normal::Normal, uniform::Uniform};
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
        assert!((u.pdf(3.0) - 1.0/3.0).abs() < 1e-15);
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
