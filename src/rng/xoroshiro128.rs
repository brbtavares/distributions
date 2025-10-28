//! Xoroshiro128++ (Blackman & Vigna): fast, small-state RNG (128-bit).
//! Not suitable for cryptographic use. Period: 2^128 - 1.

use super::RngCore;

#[derive(Clone, Debug)]
pub struct Xoroshiro128PlusPlus {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128PlusPlus {
    pub fn seed_from_u64(seed: u64) -> Self {
        // Use SplitMix64 to expand into two 64-bit state values
        let mut sm = super::SplitMix64::seed_from_u64(seed);
        let mut s0 = sm.next_u64();
        let s1 = sm.next_u64();
        if s0 == 0 && s1 == 0 { s0 = 1; }
        Self { s0, s1 }
    }

    #[inline]
    fn rotl(x: u64, k: u32) -> u64 { (x << k) | (x >> (64 - k)) }

    /// Jump equivalent to 2^64 calls; can be used to generate non-overlapping sequences.
    pub fn jump(&mut self) {
        const JUMP: [u64; 2] = [0xbeac0467eba5facb, 0xd86b048b86aa9922];
        let mut s0 = 0u64;
        let mut s1 = 0u64;
        for &jump in &JUMP {
            let mut b = jump;
            while b != 0 {
                if (b & 1) != 0 { s0 ^= self.s0; s1 ^= self.s1; }
                let _ = self.next_u64();
                b >>= 1;
            }
        }
        self.s0 = s0; self.s1 = s1;
    }

    /// Long jump equivalent to 2^96 calls.
    pub fn long_jump(&mut self) {
        const LJUMP: [u64; 2] = [0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1];
        let mut s0 = 0u64;
        let mut s1 = 0u64;
        for &jump in &LJUMP {
            let mut b = jump;
            while b != 0 {
                if (b & 1) != 0 { s0 ^= self.s0; s1 ^= self.s1; }
                let _ = self.next_u64();
                b >>= 1;
            }
        }
        self.s0 = s0; self.s1 = s1;
    }
}

impl RngCore for Xoroshiro128PlusPlus {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = Self::rotl(s0.wrapping_add(s1), 17).wrapping_add(s0);
        s1 ^= s0;
        self.s0 = Self::rotl(s0, 49) ^ s1 ^ (s1 << 21);
        self.s1 = Self::rotl(s1, 28);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_sequence() {
        let mut r1 = Xoroshiro128PlusPlus::seed_from_u64(42);
        let mut r2 = Xoroshiro128PlusPlus::seed_from_u64(42);
        for _ in 0..16 { assert_eq!(r1.next_u64(), r2.next_u64()); }
    }

    #[test]
    fn next_f64_in_range() {
        let mut r = Xoroshiro128PlusPlus::seed_from_u64(7);
        for _ in 0..1000 {
            let x = r.next_f64();
            assert!(x >= 0.0 && x < 1.0);
        }
    }
}
