//! Operators for real-valued (`Vec<f64>`) decisions.

use rand::Rng as _;
use rand_distr::{Distribution, Normal};

use crate::core::rng::Rng;
use crate::traits::{Initializer, Variation};

/// Uniformly initialize `Vec<f64>` decisions within per-variable bounds.
///
/// Bounds are inclusive `(lo, hi)` ranges per dimension. Panics if any bound
/// has `lo > hi` (spec §11.1).
#[derive(Debug, Clone)]
pub struct RealBounds {
    /// Per-variable inclusive bounds in decision order.
    pub bounds: Vec<(f64, f64)>,
}

impl RealBounds {
    /// Create a `RealBounds` initializer.
    ///
    /// # Panics
    /// If any `(lo, hi)` has `lo > hi`.
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            assert!(
                lo <= hi,
                "RealBounds bound at index {i} has lo > hi: ({lo}, {hi})",
            );
        }
        Self { bounds }
    }
}

impl Initializer<Vec<f64>> for RealBounds {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(size);
        for _ in 0..size {
            let mut decision = Vec::with_capacity(self.bounds.len());
            for &(lo, hi) in &self.bounds {
                let v = if lo == hi { lo } else { rng.random_range(lo..=hi) };
                decision.push(v);
            }
            out.push(decision);
        }
        out
    }
}

/// Add `Normal(0, sigma)` noise to every variable of the first parent.
///
/// Always returns exactly one child. Does not enforce bounds in v1 (spec §11.2).
#[derive(Debug, Clone)]
pub struct GaussianMutation {
    /// Standard deviation of the Gaussian noise. Must be positive.
    pub sigma: f64,
}

impl Variation<Vec<f64>> for GaussianMutation {
    fn vary(&mut self, parents: &[Vec<f64>], rng: &mut Rng) -> Vec<Vec<f64>> {
        assert!(self.sigma > 0.0, "GaussianMutation sigma must be positive");
        assert!(
            !parents.is_empty(),
            "GaussianMutation requires at least one parent",
        );
        let normal =
            Normal::new(0.0, self.sigma).expect("Normal distribution rejected sigma");
        let mut child = parents[0].clone();
        for x in child.iter_mut() {
            *x += normal.sample(rng);
        }
        vec![child]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rng::rng_from_seed;

    #[test]
    fn real_bounds_returns_correct_shape_and_range() {
        let mut init = RealBounds::new(vec![(-1.0, 1.0), (0.0, 10.0)]);
        let mut rng = rng_from_seed(7);
        let decisions = init.initialize(5, &mut rng);
        assert_eq!(decisions.len(), 5);
        for d in &decisions {
            assert_eq!(d.len(), 2);
            assert!(d[0] >= -1.0 && d[0] <= 1.0);
            assert!(d[1] >= 0.0 && d[1] <= 10.0);
        }
    }

    #[test]
    fn real_bounds_equal_bounds_yield_constant() {
        let mut init = RealBounds::new(vec![(2.5, 2.5)]);
        let mut rng = rng_from_seed(1);
        let d = init.initialize(3, &mut rng);
        assert!(d.iter().all(|v| v == &vec![2.5]));
    }

    #[test]
    #[should_panic(expected = "lo > hi")]
    fn real_bounds_invalid_panics() {
        RealBounds::new(vec![(1.0, 0.0)]);
    }

    #[test]
    fn gaussian_mutation_returns_one_child_same_length() {
        let mut m = GaussianMutation { sigma: 0.1 };
        let mut rng = rng_from_seed(99);
        let parents = vec![vec![0.0_f64, 1.0, 2.0]];
        let children = m.vary(&parents, &mut rng);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].len(), 3);
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn gaussian_mutation_zero_sigma_panics() {
        let mut m = GaussianMutation { sigma: 0.0 };
        let mut rng = rng_from_seed(1);
        m.vary(&[vec![0.0]], &mut rng);
    }

    #[test]
    #[should_panic(expected = "at least one parent")]
    fn gaussian_mutation_empty_parents_panics() {
        let mut m = GaussianMutation { sigma: 0.1 };
        let mut rng = rng_from_seed(1);
        m.vary(&[] as &[Vec<f64>], &mut rng);
    }
}
