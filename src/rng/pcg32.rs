//! PCG32 (PCG XSH RR 64/32) by Melissa O'Neill.
//! Small-state, high-quality 32-bit output PRNG. Not cryptographic.

use super::RngCore;

#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
    inc: u64, // must be odd
}

impl Pcg32 {
    /// Seed with a 64-bit seed and a 64-bit stream selector (will be forced odd).
    pub fn from_seed_and_stream(seed: u64, stream: u64) -> Self {
        let mut pcg = Self { state: 0, inc: (stream << 1) | 1 };
        pcg.next_u64(); // advance once with inc set
        pcg.state = pcg.state.wrapping_add(seed);
        pcg.next_u64();
        pcg
    }

    /// Seed from a single seed using SplitMix64 to generate both state and stream.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = super::SplitMix64::seed_from_u64(seed);
        let init_state = sm.next_u64();
        let stream = sm.next_u64();
        Self::from_seed_and_stream(init_state, stream)
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
    }

    /// Produce the 32-bit output with XSH RR output function.
    #[inline]
    fn output32(state: u64) -> u32 {
        // xorshift and extract high bits
        let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
        let rot = (state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.step();
        Self::output32(old)
    }
}

impl RngCore for Pcg32 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        // Combine two 32-bit outputs to form a 64-bit value
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sequence_from_seed() {
        let mut r1 = Pcg32::seed_from_u64(123);
        let mut r2 = Pcg32::seed_from_u64(123);
        for _ in 0..32 { assert_eq!(r1.next_u32(), r2.next_u32()); }
    }

    #[test]
    fn different_streams_diverge() {
        let mut r1 = Pcg32::from_seed_and_stream(42, 1);
        let mut r2 = Pcg32::from_seed_and_stream(42, 2);
        // Not guaranteed first output differs, but within a few steps likely diverges
        let mut diff = false;
        for _ in 0..16 {
            if r1.next_u32() != r2.next_u32() { diff = true; break; }
        }
        assert!(diff);
    }

    #[test]
    fn next_f64_in_range() {
        let mut r = Pcg32::seed_from_u64(7);
        for _ in 0..1000 { let x = r.next_f64(); assert!(x >= 0.0 && x < 1.0); }
    }
}
