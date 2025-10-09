use crate::{ContinuousDistribution, DistError, rng::RngCore, consts::{INV_SQRT_2PI}, math};

#[derive(Debug, Clone, Copy)]
pub struct Normal {
    mu: f64,
    sigma: f64,
    inv_sigma: f64,
    norm: f64, // 1/(sigma*sqrt(2*pi))
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistError> {
        if !(sigma > 0.0 && sigma.is_finite() && mu.is_finite()) { return Err(DistError::InvalidParameter); }
        let inv_sigma = 1.0 / sigma;
        let norm = INV_SQRT_2PI * inv_sigma;
        Ok(Self { mu, sigma, inv_sigma, norm })
    }
    #[inline] pub fn mean_param(&self) -> f64 { self.mu }
    #[inline] pub fn sigma(&self) -> f64 { self.sigma }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) * self.inv_sigma;
        self.norm * (-0.5 * z * z).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) * self.inv_sigma;
        math::standard_normal_cdf(z)
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        self.mu + self.sigma * math::standard_normal_inv_cdf(p)
    }
    fn mean(&self) -> f64 { self.mu }
    fn variance(&self) -> f64 { self.sigma * self.sigma }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        // Box-Muller polar (Marsaglia) without external dependencies.
        loop {
            let u1 = 2.0 * rng.next_f64() - 1.0; // (-1,1)
            let u2 = 2.0 * rng.next_f64() - 1.0;
            let s = u1 * u1 + u2 * u2;
            if s >= 1.0 || s == 0.0 { continue; }
            let factor = (-2.0 * s.ln() / s).sqrt();
            return self.mu + self.sigma * u1 * factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cdf_symmetry() {
        let n = Normal::new(0.0, 1.0).unwrap();
        let z = 0.7;
        let f = n.cdf(z);
        let f_neg = n.cdf(-z);
        assert!((f + f_neg - 1.0).abs() < 3e-15);
    }
}
