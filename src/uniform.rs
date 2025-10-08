use crate::{ContinuousDistribution, DistError, rng::RngCore};

#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    a: f64,
    b: f64,
    inv_width: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Result<Self, DistError> {
        if !(a < b && a.is_finite() && b.is_finite()) { return Err(DistError::InvalidParameter); }
        let inv_width = 1.0 / (b - a);
        Ok(Self { a, b, inv_width })
    }
    #[inline] pub fn a(&self) -> f64 { self.a }
    #[inline] pub fn b(&self) -> f64 { self.b }
}

impl ContinuousDistribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b { 0.0 } else { self.inv_width }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a { 0.0 } else if x >= self.b { 1.0 } else { (x - self.a) * self.inv_width }
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        debug_assert!((0.0..=1.0).contains(&p));
        self.a + (self.b - self.a) * p
    }
    fn mean(&self) -> f64 { 0.5 * (self.a + self.b) }
    fn variance(&self) -> f64 { (self.b - self.a).powi(2) / 12.0 }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 { self.a + (self.b - self.a) * rng.next_f64() }
}
