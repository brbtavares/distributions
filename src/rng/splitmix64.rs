//! SplitMix64 (Sebastiano Vigna): fast, good bit diffusion, non-cryptographic RNG.

use super::RngCore;

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    pub(crate) state: u64,
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
