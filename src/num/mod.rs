/// Frequently used numerical constants.

pub const SQRT_2: f64 = 1.41421356237309504880168872420969808_f64;
pub const INV_SQRT_2: f64 = 1.0 / SQRT_2;
pub const SQRT_2PI: f64 = 2.50662827463100050241576528481104525_f64; // sqrt(2*pi)
pub const INV_SQRT_2PI: f64 = 1.0 / SQRT_2PI; // 1 / sqrt(2*pi)
pub const LN_2: f64 = 0.69314718055994530941723212145817657_f64;


/// Internal math helper functions.

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
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Standard normal inverse CDF (probit) using Peter J. Acklam's rational approximation.
/// Typical absolute error < 4.5e-4 in double precision.
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

    let q: f64;
    let r: f64;
    let x: f64;
if p < P_LOW { // Lower tail region
        q = (-2.0 * p.ln()).sqrt();
        x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
            ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
        return -x;
    }
if p > P_HIGH { // Upper tail region
        q = (-2.0 * (1.0 - p).ln()).sqrt();
        x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
            ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
        return x;
    }
// Central region
    q = p - 0.5;
    r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q /
        (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}
