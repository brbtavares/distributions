//! Frequently used numerical constants.

pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
pub const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
pub const SQRT_2PI: f64 = 2.506_628_274_631_000_2_f64; // sqrt(2*pi) using double precision
pub const INV_SQRT_2PI: f64 = 1.0 / SQRT_2PI; // 1 / sqrt(2*pi)
pub const LN_2: f64 = std::f64::consts::LN_2;

// Internal math helper functions.

/// Standard normal PDF.
#[inline]
pub fn standard_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() * INV_SQRT_2PI
}

/// Fast approximation of erf(x) (Abramowitz & Stegun 7.1.26).
pub fn erf(x: f64) -> f64 {
    // Preserve sign.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Standard normal CDF via erf.
pub fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z * INV_SQRT_2))
}

/// Standard normal inverse CDF (probit) using Peter J. Acklam's rational approximation.
/// Typical absolute error < 4.5e-4 in double precision.
#[allow(clippy::excessive_precision)]
pub fn standard_normal_inv_cdf(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "p must be in (0,1)");

    // Coefficients (Acklam 2003). See public documentation.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;
    if p < P_LOW {
        // Lower tail region
        let q = (-2.0 * p.ln()).sqrt();
        let x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
        return -x;
    }
    if p > P_HIGH {
        // Upper tail region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
        return x;
    }
    // Central region
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

/// Digamma function ψ(x) = d/dx ln Γ(x) for x > 0.
/// Implementation: recurrence to shift x >= 8, then asymptotic series.
pub fn digamma(mut x: f64) -> f64 {
    assert!(x > 0.0, "digamma requires x > 0");
    let mut result = 0.0;
    // Use recurrence ψ(x) = ψ(x+1) - 1/x, so move x up to a large value.
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion around infinity
    // References:
    // - NIST DLMF §5.11(ii), Eq. (5.11.2):
    //   ψ(z) ~ ln z − 1/(2z) − Σ_{n≥1} B_{2n}/(2n z^{2n})
    //   https://dlmf.nist.gov/5.11.E2
    // - Abramowitz & Stegun, Handbook of Mathematical Functions:
    //   6.3.18 (digamma asymptotic), 6.3.5 (recurrence ψ(x+1) = ψ(x) + 1/x)
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv4 = inv2 * inv2;
    let inv6 = inv4 * inv2;
    // Truncation uses Bernoulli numbers B2=1/6, B4=-1/30, B6=1/42:
    // ψ(x) ≈ ln x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    result + x.ln() - 0.5 * inv - (1.0 / 12.0) * inv2 + (1.0 / 120.0) * inv4 - (1.0 / 252.0) * inv6
}
