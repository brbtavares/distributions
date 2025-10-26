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
        // Hybrid sampler:
        // - For small λ use simple inversion from 0 (fast enough and exact).
        // - For large λ use exact inversion starting at the mode ("chop-down from the mode"),
        //   which takes O(|k-λ|) expected steps instead of O(λ).
        if self.lambda < 30.0 {
            // Small-λ inversion
            let mut k: i64 = 0;
            let mut p = self.pmf_rec_start();
            let mut c = p;
            let u = rng.next_f64();
            while u > c {
                k += 1;
                p *= self.lambda / (k as f64);
                c += p;
            }
            return k;
        }

        // Large-λ exact inversion from the mode.
        let lambda = self.lambda;
        let m = lambda.floor() as i64; // mode = floor(λ)
        let log_p_m = (m as f64) * lambda.ln() - lambda - ln_factorial_u64(m as u64);
    let p_m = log_p_m.exp(); // pmf at mode

        // In rare cases of extreme λ the exponent can underflow to 0; if so, fall back.
        if !(p_m > 0.0) {
            // Fallback to small-λ inversion (will be slow but safe for pathological params).
            let mut k: i64 = 0;
            let mut p = self.pmf_rec_start();
            let mut c = p;
            let u = rng.next_f64();
            while u > c {
                k += 1;
                p *= self.lambda / (k as f64);
                c += p;
            }
            return k;
        }

        let u = rng.next_f64();
        let mut c = p_m;
        if u <= c { return m; }

        // Grow symmetrically from the mode to both tails using recurrences:
        // p(k-1) = p(k) * k / λ,  p(k+1) = p(k) * λ / (k+1)
        let mut left = p_m;
        let mut right = p_m;
        let mut i: i64 = 1;
        loop {
            // Left side mass at m - i
            if i <= m {
                left *= (m - (i - 1)) as f64 / lambda; // from p(m-(i-1)) -> p(m-i)
                c += left;
                if u <= c { return m - i; }
            } else {
                // No more left mass once we pass zero
                left = 0.0;
            }

            // Right side mass at m + i
            right *= lambda / (m + i) as f64; // from p(m+(i-1)) -> p(m+i)
            c += right;
            if u <= c { return m + i; }

            // Guaranteed to terminate as c approaches 1.0
            i += 1;
        }
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

// -------- Internal helpers for large-λ sampling --------

#[inline]
fn ln_factorial_u64(n: u64) -> f64 {
    // Exact table for 0..=20
    const LN_FACT_SMALL: [f64; 21] = [
        0.0,
        0.0,
        0.6931471805599453,
        1.791759469228055,
        3.1780538303479458,
        4.787491742782046,
        6.579251212010101,
        8.525161361065415,
        10.60460290274525,
        12.80182748008147,
        15.104412573075516,
        17.502307845873887,
        19.98721449566189,
        22.552163853123425,
        25.19122118273868,
        27.899271383840894,
        30.671860106080675,
        33.50507345013689,
        36.39544520803305,
        39.339884187199495,
        42.335616460753485,
    ];
    if n <= 20 { return LN_FACT_SMALL[n as usize]; }
    // Stirling with 1/(12n) - 1/(360n^3) + 1/(1260 n^5) correction
    let x = n as f64;
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv5 = inv3 * inv2;
    x * x.ln() - x + 0.5 * ((2.0 * std::f64::consts::PI * x).ln()) + (inv / 12.0) - (inv3 / 360.0) + (inv5 / 1260.0)
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
