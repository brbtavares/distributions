use crate::dist::{Continuous, DistError, Distribution, Moments};
use crate::num;
use crate::rng::RngCore;

#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    shape: f64, // k > 0
    scale: f64, // theta > 0
    inv_scale: f64,
    ln_gamma_shape: f64,
}

impl Gamma {
    pub fn new(shape: f64, scale: f64) -> Result<Self, DistError> {
        if !(shape > 0.0 && scale > 0.0) || !shape.is_finite() || !scale.is_finite() {
            return Err(DistError::InvalidParameter);
        }
        let inv_scale = 1.0 / scale;
        let ln_gamma_shape = ln_gamma(shape);
        Ok(Self {
            shape,
            scale,
            inv_scale,
            ln_gamma_shape,
        })
    }
    #[inline]
    pub fn shape(&self) -> f64 {
        self.shape
    }
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    #[inline]
    fn x_to_z(&self, x: f64) -> f64 {
        x * self.inv_scale
    }
}

impl Distribution for Gamma {
    type Value = f64;
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 || !x.is_finite() {
            return 0.0;
        }
        let z = self.x_to_z(x);
        reg_lower_gamma(self.shape, z)
    }
    fn in_support(&self, x: f64) -> bool {
        x >= 0.0 && x.is_finite()
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> f64 {
        // Marsaglia & Tsang (2000) method
        let k = self.shape;
        if k >= 1.0 {
            // Algorithm for k >= 1
            let d = k - 1.0 / 3.0;
            let c = (1.0 / (9.0 * d)).sqrt();
            loop {
                let x = standard_normal(rng);
                let v = 1.0 + c * x;
                if v <= 0.0 {
                    continue;
                }
                let v3 = v * v * v;
                let u = rng.next_f64();
                // Squeeze and acceptance
                if u < 1.0 - 0.0331 * x * x * x * x {
                    return self.scale * d * v3;
                }
                if u.ln() < 0.5 * x * x + d * (1.0 - v3 + (v3).ln()) {
                    return self.scale * d * v3;
                }
            }
        } else {
            // Use boost: sample gamma(k+1) then scale by U^{1/k}
            let mut tmp = Gamma::new(k + 1.0, 1.0).unwrap().sample(rng);
            let u = rng.next_f64();
            tmp *= u.powf(1.0 / k);
            self.scale * tmp
        }
    }
}

impl Continuous for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if !self.in_support(x) {
            return 0.0;
        }
        let z = self.x_to_z(x);
        ((self.shape - 1.0) * z.ln() - z - self.ln_gamma_shape - self.shape * self.inv_scale.ln())
            .exp()
            * self.inv_scale
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        debug_assert!(p > 0.0 && p < 1.0);
        // Initial guess using normal approximation
        let mean = self.shape * self.scale;
        let std = (self.shape).sqrt() * self.scale;
        let mut x = mean + std * num::standard_normal_inv_cdf(p);
        if x <= 0.0 {
            x = mean.max(1e-12);
        }
        // Bracket and refine with safeguarded Newton
        let mut lo = 0.0_f64;
        let mut hi = mean.max(x) * 2.0 + 10.0 * self.scale;
        for _ in 0..50 {
            let fx = self.cdf(x) - p;
            if fx.abs() < 1e-10 {
                break;
            }
            // Update bracket
            if fx < 0.0 {
                lo = x;
            } else {
                hi = x;
            }
            // Newton step
            let dfx = self.pdf(x).max(1e-300);
            let mut x_new = x - fx / dfx;
            if x_new <= lo || x_new >= hi || !x_new.is_finite() {
                x_new = 0.5 * (lo + hi);
            }
            x = x_new;
        }
        x
    }
}

impl Moments for Gamma {
    fn mean(&self) -> f64 {
        self.shape * self.scale
    }
    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }
}

// --- helpers ---

fn standard_normal<R: RngCore>(rng: &mut R) -> f64 {
    // polar Box-Muller
    loop {
        let u1 = 2.0 * rng.next_f64() - 1.0;
        let u2 = 2.0 * rng.next_f64() - 1.0;
        let s = u1 * u1 + u2 * u2;
        if s == 0.0 || s >= 1.0 {
            continue;
        }
        let m = (-2.0 * s.ln() / s).sqrt();
        return u1 * m;
    }
}

// Lanczos approximation for ln Gamma
pub(crate) fn ln_gamma(z: f64) -> f64 {
    // Coefficients for g=7, n=9
    const COF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = COF[0];
    for (i, &c) in COF.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

// Regularized lower incomplete gamma P(a,x)
fn reg_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // series
        let mut sum = 1.0 / a;
        let mut del = sum;
        let mut ap = a;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        (sum * (-x + a * x.ln() - ln_gamma(a))).exp()
    } else {
        // continued fraction for Q, then P = 1 - Q
        let mut b0 = x + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = 1.0 / b0;
        let mut h = d;
        for i in 1..=1000 {
            let an = -(i as f64) * (i as f64 - a);
            b0 += 2.0;
            d = an * d + b0;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b0 + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-14 {
                break;
            }
        }
        1.0 - (h * (-x + a * x.ln() - ln_gamma(a))).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moments() {
        let g = Gamma::new(2.0, 3.0).unwrap();
        assert!((g.mean() - 6.0).abs() < 1e-12);
        assert!((g.variance() - 18.0).abs() < 1e-12);
    }
    #[test]
    fn cdf_monotone() {
        let g = Gamma::new(3.0, 2.0).unwrap();
        assert!(g.cdf(1.0) < g.cdf(5.0));
    }
}
