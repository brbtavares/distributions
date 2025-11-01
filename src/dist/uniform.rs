use crate::dist::{Continuous, DistError, Distribution, Moments};
use crate::rng::RngCore;

#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    a: f64,
    b: f64,
    inv_width: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Result<Self, DistError> {
        if !(a < b && a.is_finite() && b.is_finite()) {
            return Err(DistError::InvalidParameter);
        }
        let inv_width = 1.0 / (b - a);
        Ok(Self { a, b, inv_width })
    }
    #[inline]
    pub fn a(&self) -> f64 {
        self.a
    }
    #[inline]
    pub fn b(&self) -> f64 {
        self.b
    }
}

impl Distribution for Uniform {
    type Value = f64;
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a {
            0.0
        } else if x >= self.b {
            1.0
        } else {
            (x - self.a) * self.inv_width
        }
    }
    fn in_support(&self, x: f64) -> bool {
        x >= self.a && x <= self.b && x.is_finite()
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        self.a + (self.b - self.a) * rng.next_f64()
    }
}

impl Continuous for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if self.in_support(x) {
            self.inv_width
        } else {
            0.0
        }
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        debug_assert!((0.0..=1.0).contains(&p));
        self.a + (self.b - self.a) * p
    }
}

impl Moments for Uniform {
    fn mean(&self) -> f64 {
        0.5 * (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn kurtosis(&self) -> f64 {
        -6.0 / 5.0
    }
    fn entropy(&self) -> f64 {
        (self.b - self.a).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::uniform::Uniform;

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
    fn moments_higher() {
        let u = Uniform::new(-1.0, 3.0).unwrap();
        assert_eq!(u.skewness(), 0.0);
        assert!((u.kurtosis() + 6.0 / 5.0).abs() < 1e-15);
    }

    #[test]
    fn entropy_uniform() {
        let u = Uniform::new(2.0, 5.0).unwrap();
        assert!((u.entropy() - (3.0f64).ln()).abs() < 1e-15);
    }
}
