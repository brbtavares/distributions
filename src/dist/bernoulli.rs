use crate::dist::{Discrete, DistError};
use crate::rng::RngCore;

#[derive(Debug, Clone, Copy)]
pub struct Bernoulli { p: f64 }

impl Bernoulli {
    pub fn new(p: f64) -> Result<Self, DistError> {
        if !(p >= 0.0 && p <= 1.0 && p.is_finite()) { return Err(DistError::InvalidParameter); }
        Ok(Self { p })
    }
    pub fn p(&self) -> f64 { self.p }
}

impl Discrete for Bernoulli {
    type Value = u8; // 0 or 1
    fn pmf(&self, x: Self::Value) -> f64 {
        match x { 0 => 1.0 - self.p, 1 => self.p, _ => 0.0 }
    }
    fn cdf(&self, x: Self::Value) -> f64 {
        match x { 0 => 1.0 - self.p, 1 => 1.0, _ => if x > 1 { 1.0 } else { 0.0 } }
    }
    fn inv_cdf(&self, p: f64) -> Self::Value {
        debug_assert!(p >= 0.0 && p <= 1.0);
        if p < 1.0 - self.p { 0 } else { 1 }
    }
    fn mean(&self) -> f64 { self.p }
    fn variance(&self) -> f64 { self.p * (1.0 - self.p) }
    fn sample<R: RngCore>(&self, rng: &mut R) -> Self::Value { if rng.next_f64() < self.p { 1 } else { 0 } }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pmf_cdf() {
        let b = Bernoulli::new(0.3).unwrap();
        assert!((b.pmf(1) - 0.3).abs() < 1e-15);
        assert!((b.pmf(0) - 0.7).abs() < 1e-15);
        assert!((b.cdf(0) - 0.7).abs() < 1e-15);
        assert!((b.cdf(1) - 1.0).abs() < 1e-15);
    }
}
