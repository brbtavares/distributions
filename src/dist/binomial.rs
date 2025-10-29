use crate::dist::{Discrete, DistError, Distribution, Moments};
use crate::rng::RngCore;

/// Binomial(n, p) over k=0..n.
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    pub fn new(n: u64, p: f64) -> Result<Self, DistError> {
        if !(0.0..=1.0).contains(&p) || !p.is_finite() { return Err(DistError::InvalidParameter); }
        Ok(Self { n, p })
    }
    #[inline] pub fn n(&self) -> u64 { self.n }
    #[inline] pub fn p(&self) -> f64 { self.p }

    fn pmf_recurrence(&self, k: u64) -> f64 {
        if k > self.n { return 0.0; }
        // Start at k=0: (1-p)^n
        let mut p0 = (1.0 - self.p).powi(self.n as i32);
        if k == 0 { return p0; }
        // p(k) = p(k-1) * (n-k+1)/k * (p/(1-p))
        let odds = self.p / (1.0 - self.p);
        for i in 1..=k {
            p0 *= (self.n - i + 1) as f64 / (i as f64) * odds;
        }
        p0
    }

    fn cdf_sum(&self, k: u64) -> f64 {
        if k >= self.n { return 1.0; }
        let mut sum = 0.0;
        for i in 0..=k { sum += self.pmf_recurrence(i); }
        sum
    }
}

impl Distribution for Binomial {
    type Value = i64;
    fn cdf(&self, x: i64) -> f64 {
        if x < 0 { 0.0 } else { self.cdf_sum(x as u64) }
    }
    fn in_support(&self, x: i64) -> bool { x >= 0 && (x as u64) <= self.n }
    fn sample<R: RngCore>(&self, rng: &mut R) -> i64 {
        // Inversion by summing pmf from 0.. until exceeds u
        let u = rng.next_f64();
        let mut acc = 0.0;
        for k in 0..=self.n {
            acc += self.pmf_recurrence(k);
            if u <= acc { return k as i64; }
        }
        self.n as i64
    }
}

impl Discrete for Binomial {
    fn pmf(&self, x: i64) -> f64 {
        if x < 0 { return 0.0; }
        self.pmf_recurrence(x as u64)
    }
    fn inv_cdf(&self, p: f64) -> i64 {
        debug_assert!((0.0..=1.0).contains(&p));
        if p <= 0.0 { return 0; }
        if p >= 1.0 { return self.n as i64; }
        let mut acc = 0.0;
        for k in 0..=self.n {
            acc += self.pmf_recurrence(k);
            if p <= acc { return k as i64; }
        }
        self.n as i64
    }
}

impl Moments for Binomial {
    fn mean(&self) -> f64 { (self.n as f64) * self.p }
    fn variance(&self) -> f64 { (self.n as f64) * self.p * (1.0 - self.p) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pmf_values() {
        let b = Binomial::new(5, 0.4).unwrap();
        // pmf(0) = (1-p)^n
        assert!((b.pmf(0) - 0.6f64.powi(5)).abs() < 1e-15);
        // pmf(5) = p^n
        assert!((b.pmf(5) - 0.4f64.powi(5)).abs() < 1e-15);
    }
    #[test]
    fn cdf_monotone() {
        let b = Binomial::new(10, 0.3).unwrap();
        assert!(b.cdf(3) <= b.cdf(4));
        assert!(b.cdf(9) <= 1.0);
    }
}
