//! Internal pseudo-random number generators without external dependencies.
//! SplitMix64 (Sebastiano Vigna) implementation for seeding and simple
//! statistical sampling (non-cryptographic).

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

/// SplitMix64: fast, good bit diffusion, non-cryptographic.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn seed_from_u64(seed: u64) -> Self { Self { state: seed } }
}

impl RngCore for SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut z = { self.state = self.state.wrapping_add(0x9E3779B97F4A7C15); self.state };
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_sequence() {
        let mut r1 = SplitMix64::seed_from_u64(1);
        let mut r2 = SplitMix64::seed_from_u64(1);
        for _ in 0..10 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }
}
