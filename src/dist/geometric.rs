use crate::dist::{Discrete, DistError, Distribution, Moments};
use crate::rng::RngCore;

/// Geometric(p) over k=1,2,... counts trials until first success.
#[derive(Debug, Clone, Copy)]
pub struct Geometric {
    p: f64,
}

impl Geometric {
    pub fn new(p: f64) -> Result<Self, DistError> {
        if !(0.0..=1.0).contains(&p) || p == 0.0 || !p.is_finite() {
            return Err(DistError::InvalidParameter);
        }
        Ok(Self { p })
    }
    #[inline]
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution for Geometric {
    type Value = i64;
    fn cdf(&self, k: i64) -> f64 {
        if k < 1 {
            return 0.0;
        }
        1.0 - (1.0 - self.p).powi(k as i32)
    }
    fn in_support(&self, k: i64) -> bool {
        k >= 1
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> i64 {
        // Inverse CDF: k = ceil(log(1-u)/log(1-p))
        let u = rng.next_f64();
        let q = 1.0 - self.p;
        let k = ((1.0 - u).ln() / q.ln()).ceil() as i64;
        if k < 1 { 1 } else { k }
    }
}

impl Discrete for Geometric {
    fn pmf(&self, k: i64) -> f64 {
        if !self.in_support(k) {
            return 0.0;
        }
        self.p * (1.0 - self.p).powi((k - 1) as i32)
    }
    fn inv_cdf(&self, p: f64) -> i64 {
        debug_assert!((0.0..=1.0).contains(&p));
        if p <= 0.0 {
            return 1;
        }
        if p >= 1.0 {
            return i64::MAX;
        }
        let q = 1.0 - self.p;
        ((1.0 - p).ln() / q.ln()).ceil() as i64
    }
}

impl Moments for Geometric {
    fn mean(&self) -> f64 {
        1.0 / self.p
    }
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pmf_values() {
        let g = Geometric::new(0.25).unwrap();
        assert!((g.pmf(1) - 0.25).abs() < 1e-15);
        assert!((g.pmf(2) - 0.25 * 0.75).abs() < 1e-15);
    }
    #[test]
    fn cdf_values() {
        let g = Geometric::new(0.5).unwrap();
        assert!((g.cdf(1) - 0.5).abs() < 1e-15);
        assert!((g.cdf(2) - 0.75).abs() < 1e-15);
    }
}
