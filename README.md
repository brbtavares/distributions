# distributions

A small, dependency-free Rust library for probability distributions focused on numerical clarity, clean APIs, and reproducible random sampling.

Current scope:
- Internal RNG: SplitMix64 (non-cryptographic)
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
distributions = { path = "./distributions" }
```

Example: sampling and basic queries

```rust
use distributions::dist::{normal, uniform, exponential, bernoulli, poisson, Distribution, Continuous, Discrete, Moments};
use distributions::rng::SplitMix64;

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

## Numerical notes

- Normal CDF/quantile use classic approximations (erf and Acklam’s probit). Tolerances in tests reflect expected approximation error.
- Poisson sampling currently uses inversion by cumulative sum. For large λ, more efficient algorithms (PTRS) can be added.

## Roadmap

- More distributions: Gamma, Beta, Binomial, Geometric, Lognormal, Chi-squared
- Faster samplers: Ziggurat/ratio-of-uniforms for Normal; PTRS for Poisson(λ≫1)
- Benchmarks (Criterion) and CI
- Optional features: `serde`, potential `no_std` where feasible

## License
MIT
