use crate::dist::{Continuous, DistError, Distribution, Moments};
use crate::rng::RngCore;

#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    lambda: f64,
}

impl Exponential {
    pub fn new(lambda: f64) -> Result<Self, DistError> {
        if !(lambda > 0.0 && lambda.is_finite()) {
            return Err(DistError::InvalidParameter);
        }
        Ok(Self { lambda })
    }
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Distribution for Exponential {
    type Value = f64;
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (-self.lambda * x).exp()
        }
    }
    fn in_support(&self, x: f64) -> bool {
        x >= 0.0 && x.is_finite()
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        let u = rng.next_f64();
        -u.ln() / self.lambda
    }
}

impl Continuous for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if self.in_support(x) {
            self.lambda * (-self.lambda * x).exp()
        } else {
            0.0
        }
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        debug_assert!(p > 0.0 && p < 1.0);
        -(1.0 - p).ln() / self.lambda
    }
}

impl Moments for Exponential {
    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }
    fn variance(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }
    fn skewness(&self) -> f64 {
        2.0
    }
    fn kurtosis(&self) -> f64 {
        6.0
    }
    fn entropy(&self) -> f64 {
        1.0 - self.lambda.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_values() {
        let e = Exponential::new(2.0).unwrap();
        assert!((e.mean() - 0.5).abs() < 1e-15);
        assert!((e.variance() - 0.25).abs() < 1e-15);
        assert!((e.cdf(0.0) - 0.0).abs() < 1e-15);
        assert!((e.pdf(0.0) - 2.0).abs() < 1e-12);
        assert!((e.skewness() - 2.0).abs() < 1e-15);
        assert!((e.kurtosis() - 6.0).abs() < 1e-15);
    }
}
