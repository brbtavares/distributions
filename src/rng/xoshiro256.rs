//! Xoshiro256** (Blackman & Vigna): fast, high-quality, non-cryptographic RNG.
//! State: 256-bit (4 x u64). Period: 2^256 - 1. Excellent statistical properties.
//! Not suitable for cryptographic use.

use super::RngCore;

#[derive(Clone, Debug)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Seed from a 64-bit value using SplitMix64 to expand into 256 bits.
    pub fn seed_from_u64(seed: u64) -> Self {
    let mut sm = super::SplitMix64::seed_from_u64(seed);
    let mut s = [0u64; 4];
    for slot in &mut s { *slot = sm.next_u64(); }
        // All-zero state is invalid; perturb if detected (extremely unlikely).
        if s == [0,0,0,0] { s[0] = 1; }
        Self { s }
    }

    #[inline]
    fn rotl(x: u64, k: u32) -> u64 { (x << k) | (x >> (64 - k)) }

    /// Jump equivalent to 2^128 calls; provides 2^128 non-overlapping subsequences.
    /// Useful for parallel streams.
    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [
            0x180ec6d33cfd0aba,
            0xd5a61266f0c9392c,
            0xa9582618e03fc9aa,
            0x39abdc4529b1661c,
        ];
        let mut t = [0u64; 4];
        for &jump in &JUMP {
            let mut b = jump;
            while b != 0 {
                if (b & 1) != 0 {
                    t[0] ^= self.s[0];
                    t[1] ^= self.s[1];
                    t[2] ^= self.s[2];
                    t[3] ^= self.s[3];
                }
                let _ = self.next_u64();
                b >>= 1;
            }
        }
        self.s = t;
    }

    /// Long jump equivalent to 2^192 calls.
    pub fn long_jump(&mut self) {
        const LJUMP: [u64; 4] = [
            0x76e15d3efefdcbbf,
            0xc5004e441c522fb3,
            0x77710069854ee241,
            0x39109bb02acbe635,
        ];
        let mut t = [0u64; 4];
        for &jump in &LJUMP {
            let mut b = jump;
            while b != 0 {
                if (b & 1) != 0 {
                    t[0] ^= self.s[0];
                    t[1] ^= self.s[1];
                    t[2] ^= self.s[2];
                    t[3] ^= self.s[3];
                }
                let _ = self.next_u64();
                b >>= 1;
            }
        }
        self.s = t;
    }
}

impl RngCore for Xoshiro256StarStar {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = Self::rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = Self::rotl(self.s[3], 45);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sequence() {
        let mut r1 = Xoshiro256StarStar::seed_from_u64(123);
        let mut r2 = Xoshiro256StarStar::seed_from_u64(123);
        for _ in 0..16 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn next_f64_in_range() {
        let mut r = Xoshiro256StarStar::seed_from_u64(1);
        for _ in 0..1000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
