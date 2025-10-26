use crate::dist::{Discrete, DistError, Moments, Distribution};
use crate::rng::RngCore;

/// Poisson(λ) distribution over non-negative integers.
///
/// - Support: k = 0,1,2,...
/// - Mean = λ
/// - Var = λ
#[derive(Debug, Clone, Copy)]
pub struct Poisson { lambda: f64 }

impl Poisson {
    pub fn new(lambda: f64) -> Result<Self, DistError> {
        if !(lambda > 0.0 && lambda.is_finite()) { return Err(DistError::InvalidParameter); }
        Ok(Self { lambda })
    }
    #[inline] pub fn lambda(&self) -> f64 { self.lambda }

    #[inline]
    fn pmf_rec_start(&self) -> f64 {
        // p(0) = e^{-lambda}
        (-self.lambda).exp()
    }

    /// Compute pmf(k) using recurrence from 0..k: p(k) = p(k-1) * λ / k
    /// Cost O(k). Accurate and avoids factorial overflow.
    fn pmf_via_recurrence(&self, k: i64) -> f64 {
        if k < 0 { return 0.0; }
        let k = k as u64;
        let mut p = self.pmf_rec_start();
        for i in 1..=k { p *= self.lambda / (i as f64); }
        p
    }

    /// CDF up to k by summing recurrence.
    fn cdf_via_recurrence(&self, k: i64) -> f64 {
        if k < 0 { return 0.0; }
        let k = k as u64;
        let mut p = self.pmf_rec_start();
        let mut acc = p;
        for i in 1..=k { p *= self.lambda / (i as f64); acc += p; }
        acc
    }
}

impl Distribution for Poisson {
    type Value = i64;

    fn cdf(&self, x: Self::Value) -> f64 { self.cdf_via_recurrence(x) }

    fn in_support(&self, x: Self::Value) -> bool { x >= 0 }

    fn sample<R: RngCore>(&self, rng: &mut R) -> Self::Value {
        // Inversion method via cumulative sum of pmf, expected O(λ) iterations.
        // For small to moderate λ this is fine; can be improved later with PTRS.
        let mut k: i64 = 0;
        let mut p = self.pmf_rec_start();
        let mut c = p;
        let u = rng.next_f64();
        while u > c {
            k += 1;
            p *= self.lambda / (k as f64);
            c += p;
            // For very large λ this could be many iterations; left as future optimization.
        }
        k
    }
}

impl Discrete for Poisson {
    fn pmf(&self, x: Self::Value) -> f64 { self.pmf_via_recurrence(x) }

    fn inv_cdf(&self, p: f64) -> Self::Value {
        debug_assert!(p >= 0.0 && p <= 1.0);
        if p <= 0.0 { return 0; }
        if p >= 1.0 { return i64::MAX; }
        let mut k: i64 = 0;
        let mut pk = self.pmf_rec_start();
        let mut acc = pk;
        while acc < p {
            k += 1;
            pk *= self.lambda / (k as f64);
            acc += pk;
        }
        k
    }
}

impl Moments for Poisson {
    fn mean(&self) -> f64 { self.lambda }
    fn variance(&self) -> f64 { self.lambda }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pmf_values() {
        let p = Poisson::new(3.0).unwrap();
        let e3 = (-3.0f64).exp();
        assert!((p.pmf(0) - e3).abs() < 1e-15);
        // pmf(3) = e^-3 * 3^3/3! = e^-3 * 27/6 = 4.5 e^-3
        assert!((p.pmf(3) - 4.5 * e3).abs() < 1e-12);
    }

    #[test]
    fn cdf_bounds() {
        let p = Poisson::new(1.5).unwrap();
        assert_eq!(p.cdf(-1), 0.0);
        assert!(p.cdf(0) > 0.0);
        assert!(p.cdf(10) < 1.0);
    }

    #[test]
    fn inv_cdf_roundtrip() {
        let pois = Poisson::new(2.5).unwrap();
        let ps = [0.1, 0.3, 0.5, 0.8, 0.95];
        for &prob in &ps {
            let k = pois.inv_cdf(prob);
            assert!(pois.cdf(k) >= prob - 1e-15);
            if k > 0 { assert!(pois.cdf(k - 1) <= prob + 1e-15); }
        }
    }

    #[test]
    fn sampling_deterministic() {
        let pois = Poisson::new(4.0).unwrap();
        let mut r1 = crate::rng::SplitMix64::seed_from_u64(7);
        let mut r2 = crate::rng::SplitMix64::seed_from_u64(7);
        let x1 = pois.sample(&mut r1);
        let x2 = pois.sample(&mut r2);
        assert_eq!(x1, x2);
    }
}
