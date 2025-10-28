//! Probability distributions library with no external dependencies.
//! Focus: numerical accuracy, clear API, and extensibility.
//!
//! Initial features:
//! - Internal pseudo-random number generator (SplitMix64)
//! - Generic trait `Continuous`
//! - Distributions: Uniform, Normal, Exponential
//! - Simple discrete distribution: Bernoulli
//! - PDF, CDF, inverse CDF (quantile), mean and variance
//! - Allocation-free sampling
//!
//! Quick examples:
//! ```
//! use probability_rs::{rng::SplitMix64, dist::normal::Normal, Continuous, Distribution};
//! let normal = Normal::new(0.0, 1.0).unwrap();
//! let mut rng = SplitMix64::seed_from_u64(123);
//! let x = normal.sample(&mut rng);
//! let p = normal.pdf(0.0);
//! let c = normal.cdf(0.0);
//! let q = normal.inv_cdf(0.975); // ~ 1.96
//! assert!((p - 0.39894228).abs() < 1e-7);
//! assert!((c - 0.5).abs() < 2e-6); // tolerance due to erf approximation
//! assert!((q - 1.95996).abs() < 5e-3);
//! ```

pub mod dist;
pub mod num;
pub mod rng;

// Re-export commonly used traits at crate root for ergonomic imports
pub use dist::{Continuous, Discrete, Distribution, Moments};
