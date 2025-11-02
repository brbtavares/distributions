use super::gamma::Gamma;
use crate::dist::{Continuous, DistError, Distribution, Moments};
use crate::rng::RngCore;

#[derive(Debug, Clone, Copy)]
pub struct Beta {
    a: f64,
    b: f64,
    ln_beta: f64,
}

impl Beta {
    pub fn new(a: f64, b: f64) -> Result<Self, DistError> {
        if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
            return Err(DistError::InvalidParameter);
        }
        let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
        Ok(Self { a, b, ln_beta })
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

impl Distribution for Beta {
    type Value = f64;
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        reg_inc_beta(self.a, self.b, x)
    }
    fn in_support(&self, x: f64) -> bool {
        (0.0..=1.0).contains(&x) && x.is_finite()
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        let ga = Gamma::new(self.a, 1.0).unwrap().sample(rng);
        let gb = Gamma::new(self.b, 1.0).unwrap().sample(rng);
        ga / (ga + gb)
    }
}

impl Continuous for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if !self.in_support(x) {
            return 0.0;
        }
        ((self.a - 1.0) * x.ln() + (self.b - 1.0) * (1.0 - x).ln() - self.ln_beta).exp()
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        debug_assert!(p > 0.0 && p < 1.0);
        // Simple Newton with bracketing in [0,1]
        let mut lo = 0.0;
        let mut hi = 1.0;
        let mut x = p; // initial guess
        for _ in 0..60 {
            let fx = self.cdf(x) - p;
            if fx.abs() < 1e-10 {
                break;
            }
            if fx < 0.0 {
                lo = x;
            } else {
                hi = x;
            }
            let dfx = self.pdf(x).max(1e-300);
            let mut x_new = x - fx / dfx;
            if !(0.0..=1.0).contains(&x_new) {
                x_new = 0.5 * (lo + hi);
            }
            x = x_new;
        }
        x
    }
}

impl Moments for Beta {
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        (self.a * self.b) / ((self.a + self.b).powi(2) * (self.a + self.b + 1.0))
    }
    fn skewness(&self) -> f64 {
        let a = self.a;
        let b = self.b;
        let num = 2.0 * (b - a) * (a + b + 1.0).sqrt();
        let den = (a + b + 2.0) * (a * b).sqrt();
        num / den
    }
    fn kurtosis(&self) -> f64 {
        let a = self.a;
        let b = self.b;
        let num = 6.0 * ((a - b).powi(2) * (a + b + 1.0) - a * b * (a + b + 2.0));
        let den = a * b * (a + b + 2.0) * (a + b + 3.0);
        num / den
    }
    fn entropy(&self) -> f64 {
        // H = ln B(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b)
        let a = self.a;
        let b = self.b;
        let ln_beta =
            super::gamma::ln_gamma(a) + super::gamma::ln_gamma(b) - super::gamma::ln_gamma(a + b);
        ln_beta - (a - 1.0) * crate::num::digamma(a) - (b - 1.0) * crate::num::digamma(b)
            + (a + b - 2.0) * crate::num::digamma(a + b)
    }
}

// Helpers: ln_gamma and regularized incomplete beta (continued fractions)
fn ln_gamma(z: f64) -> f64 {
    super::gamma::ln_gamma(z)
}

fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    // Use symmetry to ensure x <= (a+1)/(a+b+2)
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = ((a + b).ln() - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    // Continued fraction for incomplete beta (Numerical Recipes style)
    let mut am = 1.0;
    let mut bm = 1.0;
    let mut az = 1.0;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut bz = 1.0 - qab * x / qap;
    let eps = 3e-14;
    let fpmin = 1e-300;
    for m in 1..=200 {
        let m2 = 2 * m;
        // even step
        let d = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        let ap = az + d * am;
        let bp = bz + d * bm;
        // odd step
        let d = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        let app = ap + d * az;
        let bpp = bp + d * bz;
        am = ap / bpp.max(fpmin);
        bm = bp / bpp.max(fpmin);
        az = app / bpp.max(fpmin);
        bz = 1.0;
        if (app - ap).abs() < eps * app.abs() {
            break;
        }
    }
    az
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moments() {
        let b = Beta::new(2.0, 5.0).unwrap();
        assert!((b.mean() - (2.0 / 7.0)).abs() < 1e-12);
    }
    #[test]
    fn moments_higher() {
        let b = Beta::new(2.0, 2.0).unwrap();
        assert!(b.skewness().abs() < 1e-15);
        assert!((b.kurtosis() - (-6.0 / 7.0)).abs() < 1e-12);
    }
}
