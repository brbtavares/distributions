//! Internal pseudo-random number generators without external dependencies.
//! This module declares the core RNG trait and exposes concrete RNGs as submodules.

/// Minimal trait for random number generation.
pub trait RngCore {
    fn next_u64(&mut self) -> u64;
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // Use 53 bits of precision for f64 in [0,1).
        const DEN: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DEN
    }
}

pub mod splitmix64;
pub mod xoshiro256;
pub mod xoroshiro128;
pub mod pcg32;

// Re-export commonly used RNGs for ergonomic access: rng::SplitMix64
pub use splitmix64::SplitMix64;
pub use xoroshiro128::Xoroshiro128PlusPlus;
pub use pcg32::Pcg32;
