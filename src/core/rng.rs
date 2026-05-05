//! Single seeded RNG type used throughout the crate.

use rand::SeedableRng;

/// The standard RNG used by `Initializer`, `Variation`, and built-in optimizers.
///
/// Fixed to a single concrete type so the public traits never need to be
/// generic over the RNG.
pub type Rng = rand::rngs::StdRng;

/// Build a deterministic [`Rng`] from a 64-bit seed.
pub fn rng_from_seed(seed: u64) -> Rng {
    Rng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng as _;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = rng_from_seed(42);
        let mut b = rng_from_seed(42);
        let av: u64 = a.random();
        let bv: u64 = b.random();
        assert_eq!(av, bv);
    }

    #[test]
    fn different_seed_different_sequence() {
        let mut a = rng_from_seed(1);
        let mut b = rng_from_seed(2);
        let av: u64 = a.random();
        let bv: u64 = b.random();
        assert_ne!(av, bv);
    }
}
