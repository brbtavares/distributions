use crate::dist::{Discrete, DistError, Distribution, Moments};
use crate::rng::RngCore;

#[derive(Debug, Clone, Copy)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Result<Self, DistError> {
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(DistError::InvalidParameter);
        }
        Ok(Self { p })
    }
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution for Bernoulli {
    type Value = i64;
    fn cdf(&self, x: Self::Value) -> f64 {
        match x {
            x if x < 0 => 0.0,
            0 => 1.0 - self.p,
            1 => 1.0,
            _ => 1.0,
        }
    }
    fn in_support(&self, x: Self::Value) -> bool {
        x == 0 || x == 1
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> Self::Value {
        if rng.next_f64() < self.p { 1 } else { 0 }
    }
}

impl Discrete for Bernoulli {
    fn pmf(&self, x: Self::Value) -> f64 {
        if self.in_support(x) {
            if x == 1 { self.p } else { 1.0 - self.p }
        } else {
            0.0
        }
    }
    fn inv_cdf(&self, p: f64) -> Self::Value {
        debug_assert!((0.0..=1.0).contains(&p));
        if p < 1.0 - self.p { 0 } else { 1 }
    }
}

impl Moments for Bernoulli {
    fn mean(&self) -> f64 {
        self.p
    }
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
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
