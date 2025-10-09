//! Collection of probability distributions.
//! This module groups all distribution implementations under `dist`.

pub mod uniform;
pub mod normal;
pub mod exponential;
pub mod bernoulli;

// Optional convenient re-exports at `dist` root
pub use self::{
    uniform::Uniform,
    normal::Normal,
    exponential::Exponential,
    bernoulli::Bernoulli,
};
