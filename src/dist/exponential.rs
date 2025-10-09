use crate::dist::{Continuous, DistError};
use crate::rng::RngCore;

#[derive(Debug, Clone, Copy)]
pub struct Exponential { lambda: f64 }

impl Exponential {
    pub fn new(lambda: f64) -> Result<Self, DistError> {
        if !(lambda > 0.0 && lambda.is_finite()) { return Err(DistError::InvalidParameter); }
        Ok(Self { lambda })
    }
    #[inline] pub fn lambda(&self) -> f64 { self.lambda }
}

impl Continuous for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.lambda * (-self.lambda * x).exp() }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { 1.0 - (-self.lambda * x).exp() }
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        debug_assert!(p > 0.0 && p < 1.0);
        - (1.0 - p).ln() / self.lambda
    }
    fn mean(&self) -> f64 { 1.0 / self.lambda }
    fn variance(&self) -> f64 { 1.0 / (self.lambda * self.lambda) }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        let u = rng.next_f64();
        -u.ln() / self.lambda
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
    }
}
