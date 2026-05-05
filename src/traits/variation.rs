//! Trait for producing child decisions from parents.

use crate::core::rng::Rng;

/// Generates child decisions from parent decisions.
///
/// The optimizer chooses how many parents to pass. Implementations may return
/// any number of children; algorithms that require a specific count should
/// panic with a clear message if they do not get it.
pub trait Variation<D> {
    /// Produce children from the given parents.
    fn vary(&mut self, parents: &[D], rng: &mut Rng) -> Vec<D>;
}
