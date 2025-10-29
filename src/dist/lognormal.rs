use crate::dist::{Continuous, DistError, Distribution, Moments};
use crate::rng::RngCore;
use crate::dist::normal::Normal;

/// Lognormal with parameters (mu, sigma) where ln(X) ~ Normal(mu, sigma).
#[derive(Debug, Clone, Copy)]
pub struct LogNormal {
    mu: f64,
    sigma: f64,
    normal: Normal,
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistError> {
        if !(sigma > 0.0 && sigma.is_finite() && mu.is_finite()) {
            return Err(DistError::InvalidParameter);
        }
        let normal = Normal::new(mu, sigma)?;
        Ok(Self { mu, sigma, normal })
    }
    #[inline] pub fn mu(&self) -> f64 { self.mu }
    #[inline] pub fn sigma(&self) -> f64 { self.sigma }
}

impl Distribution for LogNormal {
    type Value = f64;
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 || !x.is_finite() { return 0.0; }
        self.normal.cdf(x.ln())
    }
    fn in_support(&self, x: f64) -> bool { x > 0.0 && x.is_finite() }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        (self.normal.sample(rng)).exp()
    }
}

impl Continuous for LogNormal {
    fn pdf(&self, x: f64) -> f64 {
        if !self.in_support(x) { return 0.0; }
        let z = x.ln();
        self.normal.pdf(z) / x
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        debug_assert!((0.0..=1.0).contains(&p));
        self.normal.inv_cdf(p).exp()
    }
}

impl Moments for LogNormal {
    fn mean(&self) -> f64 { (self.mu + 0.5 * self.sigma * self.sigma).exp() }
    fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        ((2.0 * s2).exp() - (s2).exp()) * (self.mu + s2).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pdf_cdf_basic() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        assert!(ln.pdf(1.0) > 0.0);
        assert!((ln.cdf(1.0) - 0.5).abs() < 2e-6);
    }
}
