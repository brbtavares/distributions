# probability-rs

[![CI](https://github.com/brbtavares/probability-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/brbtavares/probability-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/probability-rs.svg)](https://crates.io/crates/probability-rs)
[![docs.rs](https://docs.rs/probability-rs/badge.svg)](https://docs.rs/probability-rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.85%2B-informational)](Cargo.toml)
[![codecov](https://codecov.io/gh/brbtavares/probability-rs/graph/badge.svg?token=GTMW85PL6N)](https://codecov.io/gh/brbtavares/probability-rs)

A small, dependency-free Rust library for probability distributions focused on numerical clarity, clean APIs, and reproducible random sampling.

Current scope:
- Internal RNGs (non-cryptographic): SplitMix64, Xoroshiro128++, Xoshiro256**, PCG32
- Traits: `Distribution`, `Continuous`, `Discrete`, `Moments`
- Distributions:
  - Continuous: Uniform, Normal, Exponential
  - Discrete: Bernoulli, Poisson

## Why
- No external dependencies
- Deterministic sampling (seeded), useful for tests and teaching
- Simple and explicit math with careful domains and parameter checks

## Status
This is a work-in-progress library. APIs may evolve. Contributions and feedback are welcome.

## Quick start

Add to your workspace as a path dependency or use locally:

```toml
# Cargo.toml
[dependencies]
probability-rs = { path = "./probability-rs" }
```

Example: sampling and basic queries

```rust
use probability_rs::dist::{normal, uniform, exponential, bernoulli, poisson, Distribution, Continuous, Discrete, Moments};
use probability_rs::rng::SplitMix64;

fn main() {
    let normal = normal::Normal::new(0.0, 1.0).unwrap();
    let uniform = uniform::Uniform::new(-1.0, 1.0).unwrap();
    let expo = exponential::Exponential::new(2.0).unwrap();
    let bern = bernoulli::Bernoulli::new(0.4).unwrap();
    let pois = poisson::Poisson::new(3.0).unwrap();

    let mut rng = SplitMix64::seed_from_u64(2024);
    let x_n = normal.sample(&mut rng);
    let x_u = uniform.sample(&mut rng);
    let x_e = expo.sample(&mut rng);
    let x_b = bern.sample(&mut rng);
    let x_p = pois.sample(&mut rng);

    println!("Normal sample: {x_n:.6} pdf(0)={:.6}", normal.pdf(0.0));
    println!("Uniform sample: {x_u:.6} mean={:.3} var={:.3}", uniform.mean(), uniform.variance());
    println!("Exponential sample: {x_e:.6} CDF(1)={:.6}", expo.cdf(1.0));
    println!("Bernoulli sample: {x_b} p=0.4 var={:.3}", bern.variance());
    println!("Poisson sample: {x_p} lambda=3 pmf(3)={:.6}", pois.pmf(3));
}
```

Run tests:

```bash
cargo test --all
```

## API at a glance

- `Distribution` (common):
  - `cdf(x) -> f64`, `in_support(x) -> bool`, `sample(&mut Rng) -> Value`
- `Continuous` (f64): `pdf(x) -> f64`, `inv_cdf(p) -> f64`
- `Discrete` (i64): `pmf(k) -> f64`, `inv_cdf(p) -> i64`
- `Moments`: `mean() -> f64`, `variance() -> f64`
- RNG: `rng::RngCore`, `rng::SplitMix64`

## RNGs: picking the right generator

This crate ships a few small, non-cryptographic PRNGs with a common trait `rng::RngCore`.

- SplitMix64
  - Best for: seeding other RNGs, quick-and-simple deterministic tests.
  - Pros: tiny, very fast, good bit diffusion; great seed expander.
  - Cons: not the strongest statistical quality for long streams compared to xoshiro/pcg.
  - Use:
  - `use probability_rs::rng::SplitMix64;`
    - `let mut rng = SplitMix64::seed_from_u64(123);`

- Xoroshiro128++
  - Best for: fast simulations with small memory footprint (128-bit state).
  - Pros: excellent speed, good quality in practice for 64-bit outputs.
  - Cons: period 2^128−1; for massive parallel use, consider jump/long_jump to split streams.
  - Use:
  - `use probability_rs::rng::Xoroshiro128PlusPlus;`
    - `let mut rng = Xoroshiro128PlusPlus::seed_from_u64(123);`

- Xoshiro256**
  - Best for: general-purpose high-quality streams (256-bit state).
  - Pros: period 2^256−1, excellent statistical properties, jump/long_jump available.
  - Cons: slightly larger state than Xoroshiro128++.
  - Use:
  - `use probability_rs::rng::xoshiro256::Xoshiro256StarStar;`
    - `let mut rng = Xoshiro256StarStar::seed_from_u64(123);`

- PCG32 (XSH RR 64/32)
  - Best for: small-state RNG with good 32-bit outputs, reproducible parallel streams.
  - Pros: configurable streams via `from_seed_and_stream(seed, stream)`; great distribution.
  - Cons: 32-bit output per step (we combine two for 64-bit).
  - Use:
  - `use probability_rs::rng::Pcg32;`
    - `let mut rng = Pcg32::seed_from_u64(123);`
    - or `let mut rng = Pcg32::from_seed_and_stream(STATE, STREAM_ID);`

Guidelines by scenario:
- Reproducible tests, quick examples: SplitMix64
- High-throughput simulations (low memory): Xoroshiro128++
- High-quality general-purpose streams: Xoshiro256**
- Many independent parallel streams with small state: PCG32 (use different `stream`)

Note: none of these RNGs are cryptographic. For security-sensitive contexts, use a proper CSPRNG.

## Numerical notes

- Normal CDF/quantile use classic approximations (erf and Acklam’s probit). Tolerances in tests reflect expected approximation error.
- Poisson sampling uses a hybrid approach (inversion, mode-based, and quantile-anchored) depending on λ. PTRS may be added later for λ≫1.

## Benchmarks

We use Criterion for micro-benchmarks. To run:

```bash
cargo bench
```

The included benchmark compares Poisson sampling for small (λ=2.5) and large (λ=250) regimes.

## Roadmap

- Distributions and structure
  - More distributions: Gamma, Beta, Binomial, Geometric, Lognormal, Chi-squared, Dirichlet, Multivariate Normal
  - Truncation and affine transforms (shift/scale) as generic wrappers
  - Mixture models (finite mixtures) with EM fitting

- Inference and model assessment
  - Parameter estimation: MLE/MOM with uncertainty (Fisher information)
  - Model selection: AIC/BIC, automated “best fit” among candidates
  - Goodness-of-fit tests: Kolmogorov–Smirnov, Anderson–Darling, chi-squared
  - Robust statistics and empirical quantiles with confidence intervals

- Advanced sampling and performance
  - Faster samplers: Ziggurat or Ratio-of-Uniforms (Normal/Exponential), PTRS for Poisson (λ ≫ 1)
  - Alias method (Walker/Vose) for arbitrary categorical distributions
  - Variance reduction: antithetic variates, control variates, stratification
  - Vectorization/batching (std::simd where feasible), allocation-free sample_n and sample_iter

- Dependence and multivariate
  - Copulas (Gaussian, Student-t) to construct multivariate dependencies
  - Multivariate families: Multivariate Normal, Wishart/Inverse-Wishart, Dirichlet

- Stochastic processes and simulation
  - Poisson processes (homogeneous/inhomogeneous), renewal processes, simple Hawkes
  - Brownian motion, Ornstein–Uhlenbeck; SDE discretizations (Euler–Maruyama)
  - Time-series generators: AR(1), light ARMA components for simulations

- Practical statistics and summaries
  - Histograms, KDE, ECDF, descriptive summaries (median, MAD, etc.)
  - Streaming quantiles (P² algorithm, optional t-digest via feature flag)
  - Distances/divergences: KL, Jensen–Shannon, Wasserstein (1D)

- API ergonomics and safety
  - logpdf/logpmf/logcdf/logccdf for numerical stability; ccdf for tail work
  - Additional moments: entropy, skewness, kurtosis, cumulants
  - SeedableRng-style helper trait; domain types (Probability, Positive, Interval)
  - Feature flags: `serde`, `no_std` (where viable), `simd`, `special-fns`

- Numerics and special functions
  - Special functions: gamma/incomplete gamma, beta/incomplete beta, digamma/trigamma
  - Generic numerical inversion for CDFs (bracketing + Newton/Halley) with tolerances
  - Tail-accuracy improvements using log1p/expm1 and complemented functions

- Tooling and quality
  - Expanded benchmarks (Criterion) and lightweight statistical test harness
  - CI with lint/test/bench sanity; performance tracking
  - Rich documentation with runnable examples and optional notebooks

## License
MIT
