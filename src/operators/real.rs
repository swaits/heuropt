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

/// Simulated Binary Crossover (Deb & Agrawal 1995): the standard real-valued
/// crossover used by NSGA-II.
///
/// Takes two parents, produces two children. Per dimension, with probability
/// `per_variable_probability`, mixes the parents using a polynomial spread
/// `β` drawn from a distribution controlled by `eta` (distribution index;
/// typical values 10–30, default 15: smaller `eta` → more spread, larger →
/// children stay closer to parents). Output is clamped to per-variable
/// bounds.
///
/// Panics on construction if any bound has `lo > hi`, or at run time if
/// `parents.len() < 2` or any parent length differs from `bounds.len()`.
#[derive(Debug, Clone)]
pub struct SimulatedBinaryCrossover {
    /// Per-variable inclusive bounds. Length must match the parent decisions.
    pub bounds: Vec<(f64, f64)>,
    /// Distribution index `η_c`. Must be `>= 0.0`. Default 15.0.
    pub eta: f64,
    /// Probability of mixing each variable. Typical: 0.5 or 1.0.
    pub per_variable_probability: f64,
}

impl SimulatedBinaryCrossover {
    /// Construct a `SimulatedBinaryCrossover`.
    ///
    /// # Panics
    /// If any bound has `lo > hi`, `eta < 0.0`, or
    /// `per_variable_probability` is outside `[0.0, 1.0]`.
    pub fn new(bounds: Vec<(f64, f64)>, eta: f64, per_variable_probability: f64) -> Self {
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            assert!(
                lo <= hi,
                "SimulatedBinaryCrossover bound at index {i} has lo > hi: ({lo}, {hi})",
            );
        }
        assert!(eta >= 0.0, "SimulatedBinaryCrossover eta must be >= 0.0");
        assert!(
            (0.0..=1.0).contains(&per_variable_probability),
            "SimulatedBinaryCrossover per_variable_probability must be in [0.0, 1.0]",
        );
        Self { bounds, eta, per_variable_probability }
    }
}

impl Variation<Vec<f64>> for SimulatedBinaryCrossover {
    fn vary(&mut self, parents: &[Vec<f64>], rng: &mut Rng) -> Vec<Vec<f64>> {
        assert!(
            parents.len() >= 2,
            "SimulatedBinaryCrossover requires at least two parents",
        );
        let p1 = &parents[0];
        let p2 = &parents[1];
        assert_eq!(
            p1.len(),
            self.bounds.len(),
            "SimulatedBinaryCrossover parent length must match bounds length",
        );
        assert_eq!(
            p2.len(),
            self.bounds.len(),
            "SimulatedBinaryCrossover parent length must match bounds length",
        );

        let mut c1 = p1.clone();
        let mut c2 = p2.clone();
        let exponent = 1.0 / (self.eta + 1.0);
        for j in 0..self.bounds.len() {
            if !rng.random_bool(self.per_variable_probability) {
                continue;
            }
            let u: f64 = rng.random();
            let beta = if u <= 0.5 {
                (2.0 * u).powf(exponent)
            } else {
                (1.0 / (2.0 * (1.0 - u))).powf(exponent)
            };
            let (lo, hi) = self.bounds[j];
            c1[j] = (0.5 * ((1.0 + beta) * p1[j] + (1.0 - beta) * p2[j])).clamp(lo, hi);
            c2[j] = (0.5 * ((1.0 - beta) * p1[j] + (1.0 + beta) * p2[j])).clamp(lo, hi);
        }
        vec![c1, c2]
    }
}

/// Bounded variant of [`GaussianMutation`]: add `Normal(0, sigma)` noise to
/// every variable of the first parent, then clamp each variable to its
/// per-dimension inclusive bound.
///
/// Always returns exactly one child. Use this when you want feasibility
/// maintained across generations without leaning on
/// clamp-inside-`Problem::evaluate`.
#[derive(Debug, Clone)]
pub struct BoundedGaussianMutation {
    /// Standard deviation of the Gaussian noise. Must be positive.
    pub sigma: f64,
    /// Per-variable inclusive bounds. Length must match the parent decision.
    pub bounds: Vec<(f64, f64)>,
}

impl BoundedGaussianMutation {
    /// Construct a `BoundedGaussianMutation`.
    ///
    /// # Panics
    /// If `sigma <= 0.0` or any bound has `lo > hi`.
    pub fn new(sigma: f64, bounds: Vec<(f64, f64)>) -> Self {
        assert!(sigma > 0.0, "BoundedGaussianMutation sigma must be positive");
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            assert!(
                lo <= hi,
                "BoundedGaussianMutation bound at index {i} has lo > hi: ({lo}, {hi})",
            );
        }
        Self { sigma, bounds }
    }
}

impl Variation<Vec<f64>> for BoundedGaussianMutation {
    fn vary(&mut self, parents: &[Vec<f64>], rng: &mut Rng) -> Vec<Vec<f64>> {
        assert!(
            !parents.is_empty(),
            "BoundedGaussianMutation requires at least one parent",
        );
        assert_eq!(
            parents[0].len(),
            self.bounds.len(),
            "BoundedGaussianMutation parent length must match bounds length",
        );
        let normal =
            Normal::new(0.0, self.sigma).expect("Normal distribution rejected sigma");
        let mut child = parents[0].clone();
        for (x, &(lo, hi)) in child.iter_mut().zip(self.bounds.iter()) {
            *x = (*x + normal.sample(rng)).clamp(lo, hi);
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

    #[test]
    fn bounded_gaussian_keeps_child_in_bounds() {
        let mut m = BoundedGaussianMutation::new(5.0, vec![(-1.0, 1.0); 4]);
        let mut rng = rng_from_seed(0);
        let parent = vec![0.0_f64; 4];
        // sigma=5 against bounds [-1, 1] guarantees clamping fires.
        for _ in 0..100 {
            let children = m.vary(std::slice::from_ref(&parent), &mut rng);
            assert_eq!(children.len(), 1);
            assert_eq!(children[0].len(), 4);
            for &x in &children[0] {
                assert!(x >= -1.0 && x <= 1.0, "out of bounds: {x}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn bounded_gaussian_zero_sigma_panics() {
        let _ = BoundedGaussianMutation::new(0.0, vec![(0.0, 1.0)]);
    }

    #[test]
    #[should_panic(expected = "lo > hi")]
    fn bounded_gaussian_invalid_bounds_panics() {
        let _ = BoundedGaussianMutation::new(0.1, vec![(1.0, 0.0)]);
    }

    #[test]
    #[should_panic(expected = "must match bounds length")]
    fn bounded_gaussian_mismatched_length_panics() {
        let mut m = BoundedGaussianMutation::new(0.1, vec![(0.0, 1.0); 3]);
        let mut rng = rng_from_seed(0);
        m.vary(&[vec![0.0; 2]], &mut rng);
    }

    #[test]
    fn sbx_returns_two_children_inside_bounds() {
        let mut x = SimulatedBinaryCrossover::new(vec![(-1.0, 1.0); 4], 15.0, 1.0);
        let mut rng = rng_from_seed(7);
        let p1 = vec![-0.5, 0.0, 0.25, -0.75];
        let p2 = vec![0.5, -0.25, -0.5, 0.75];
        let parents = vec![p1, p2];
        let children = x.vary(&parents, &mut rng);
        assert_eq!(children.len(), 2);
        for c in &children {
            assert_eq!(c.len(), 4);
            for &v in c {
                assert!(v >= -1.0 && v <= 1.0);
            }
        }
    }

    #[test]
    fn sbx_zero_per_variable_probability_returns_parents() {
        let mut x = SimulatedBinaryCrossover::new(vec![(-10.0, 10.0); 3], 15.0, 0.0);
        let mut rng = rng_from_seed(0);
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![-1.0, -2.0, -3.0];
        let parents = vec![p1.clone(), p2.clone()];
        let children = x.vary(&parents, &mut rng);
        assert_eq!(children[0], p1);
        assert_eq!(children[1], p2);
    }

    #[test]
    #[should_panic(expected = "at least two parents")]
    fn sbx_one_parent_panics() {
        let mut x = SimulatedBinaryCrossover::new(vec![(0.0, 1.0)], 15.0, 0.5);
        let mut rng = rng_from_seed(0);
        let _ = x.vary(&[vec![0.5]], &mut rng);
    }

    #[test]
    #[should_panic(expected = "eta must be >= 0.0")]
    fn sbx_negative_eta_panics() {
        let _ = SimulatedBinaryCrossover::new(vec![(0.0, 1.0)], -1.0, 0.5);
    }
}
