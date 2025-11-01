use super::gamma::Gamma;
use crate::dist::{Continuous, DistError, Distribution, Moments};
use crate::rng::RngCore;

/// Chi-squared with v degrees of freedom: equivalent to Gamma(k=v/2, theta=2).
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared {
    v: f64,
    gamma: Gamma,
}

impl ChiSquared {
    pub fn new(v: f64) -> Result<Self, DistError> {
        if !(v > 0.0) || !v.is_finite() {
            return Err(DistError::InvalidParameter);
        }
        let gamma = Gamma::new(v / 2.0, 2.0)?;
        Ok(Self { v, gamma })
    }
    #[inline]
    pub fn dof(&self) -> f64 {
        self.v
    }
}

impl Distribution for ChiSquared {
    type Value = f64;
    fn cdf(&self, x: f64) -> f64 {
        self.gamma.cdf(x)
    }
    fn in_support(&self, x: f64) -> bool {
        self.gamma.in_support(x)
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        self.gamma.sample(rng)
    }
}

impl Continuous for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        self.gamma.pdf(x)
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        self.gamma.inv_cdf(p)
    }
}

impl Moments for ChiSquared {
    fn mean(&self) -> f64 {
        self.v
    }
    fn variance(&self) -> f64 {
        2.0 * self.v
    }
    fn skewness(&self) -> f64 {
        (8.0 / self.v).sqrt()
    }
    fn kurtosis(&self) -> f64 {
        12.0 / self.v
    }
    fn entropy(&self) -> f64 {
        self.gamma.entropy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moments() {
        let x2 = ChiSquared::new(4.0).unwrap();
        assert!((x2.mean() - 4.0).abs() < 1e-12);
        assert!((x2.variance() - 8.0).abs() < 1e-12);
    }
    #[test]
    fn moments_higher() {
        let x2 = ChiSquared::new(4.0).unwrap();
        assert!((x2.skewness() - 2f64.sqrt()).abs() < 1e-12);
        assert!((x2.kurtosis() - 3.0).abs() < 1e-12);
    }
}
