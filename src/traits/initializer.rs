//! Trait for sampling initial decisions.

use crate::core::rng::Rng;

/// Generates initial decisions for a population.
///
/// Implementations should return exactly `size` decisions; if that is
/// impossible, panic with a clear message in v1.
pub trait Initializer<D> {
    /// Produce `size` initial decisions using the supplied RNG.
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<D>;
}
