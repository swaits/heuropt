//! Per-operator property tests covering every Variation / Initializer /
//! Repair impl heuropt ships.

use proptest::prelude::*;

use heuropt::core::rng::rng_from_seed;
use heuropt::prelude::*;

/// Generate per-axis bounds whose width is at least 0.001 (avoid the
/// degenerate `lo == hi` case for properties that need a proper interval).
fn bounds(dim: usize) -> impl Strategy<Value = Vec<(f64, f64)>> {
    prop::collection::vec((-50.0_f64..50.0, 0.001_f64..50.0), dim..=dim)
        .prop_map(|pairs| pairs.into_iter().map(|(lo, span)| (lo, lo + span)).collect())
}

/// Generate a parent vector inside the given bounds.
fn parent_in_bounds(bounds: &[(f64, f64)]) -> Vec<f64> {
    bounds.iter().map(|&(lo, hi)| 0.5 * (lo + hi)).collect()
}

// -----------------------------------------------------------------------------
// Initializers
// -----------------------------------------------------------------------------

proptest! {
    #[test]
    fn real_bounds_returns_correct_shape(
        bounds in bounds(4),
        size in 1usize..30,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let mut init = RealBounds::new(bounds.clone());
        let decisions = init.initialize(size, &mut rng);
        prop_assert_eq!(decisions.len(), size);
        for d in &decisions {
            prop_assert_eq!(d.len(), 4);
            for (j, &v) in d.iter().enumerate() {
                let (lo, hi) = bounds[j];
                prop_assert!(v >= lo && v <= hi, "{v} out of [{lo}, {hi}]");
            }
        }
    }

    #[test]
    fn real_bounds_size_zero_returns_empty(
        bounds in bounds(3),
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let mut init = RealBounds::new(bounds);
        let decisions = init.initialize(0, &mut rng);
        prop_assert!(decisions.is_empty());
    }
}

// -----------------------------------------------------------------------------
// Real-valued Variation operators
// -----------------------------------------------------------------------------

proptest! {
    #[test]
    fn gaussian_mutation_preserves_length(
        sigma in 1e-6_f64..5.0,
        len in 1usize..10,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let parent: Vec<f64> = vec![0.0; len];
        let mut m = GaussianMutation { sigma };
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        prop_assert_eq!(children.len(), 1);
        prop_assert_eq!(children[0].len(), len);
    }

    #[test]
    fn bounded_gaussian_mutation_in_bounds(
        sigma in 1e-6_f64..5.0,
        bounds in bounds(4),
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let parent = parent_in_bounds(&bounds);
        let mut m = BoundedGaussianMutation::new(sigma, bounds.clone());
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        prop_assert_eq!(children.len(), 1);
        for (j, &v) in children[0].iter().enumerate() {
            let (lo, hi) = bounds[j];
            prop_assert!(v >= lo && v <= hi);
        }
    }

    #[test]
    fn bit_flip_mutation_preserves_length(
        probability in 0.0_f64..=1.0,
        len in 1usize..32,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let parent: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
        let mut m = BitFlipMutation { probability };
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        prop_assert_eq!(children.len(), 1);
        prop_assert_eq!(children[0].len(), len);
    }

    #[test]
    fn swap_mutation_is_a_permutation(
        len in 2usize..16,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let parent: Vec<usize> = (0..len).collect();
        let mut m = SwapMutation;
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        prop_assert_eq!(children.len(), 1);
        let mut sorted = children[0].clone();
        sorted.sort();
        let identity: Vec<usize> = (0..len).collect();
        prop_assert_eq!(sorted, identity);
    }

    #[test]
    fn sbx_in_bounds(
        bounds in bounds(3),
        eta in 1.0_f64..30.0,
        per_var_p in 0.0_f64..=1.0,
        a_frac in 0.0_f64..1.0,
        b_frac in 0.0_f64..1.0,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let p1: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + a_frac * (hi - lo)).collect();
        let p2: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + b_frac * (hi - lo)).collect();
        let mut sbx = SimulatedBinaryCrossover::new(bounds.clone(), eta, per_var_p);
        let children = sbx.vary(&[p1, p2], &mut rng);
        prop_assert_eq!(children.len(), 2);
        for c in &children {
            for (j, &v) in c.iter().enumerate() {
                let (lo, hi) = bounds[j];
                prop_assert!(v >= lo && v <= hi);
            }
        }
    }

    #[test]
    fn polymut_in_bounds(
        bounds in bounds(3),
        eta in 1.0_f64..40.0,
        per_var_p in 0.0_f64..=1.0,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let parent = parent_in_bounds(&bounds);
        let mut pm = PolynomialMutation::new(bounds.clone(), eta, per_var_p);
        let children = pm.vary(std::slice::from_ref(&parent), &mut rng);
        prop_assert_eq!(children.len(), 1);
        for (j, &v) in children[0].iter().enumerate() {
            let (lo, hi) = bounds[j];
            prop_assert!(v >= lo && v <= hi);
        }
    }

    #[test]
    fn levy_mutation_in_bounds(
        bounds in bounds(3),
        alpha in 0.5_f64..2.0,
        scale in 0.01_f64..1.0,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let parent = parent_in_bounds(&bounds);
        let mut m = LevyMutation::new(alpha, scale, bounds.clone());
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        prop_assert_eq!(children.len(), 1);
        for (j, &v) in children[0].iter().enumerate() {
            let (lo, hi) = bounds[j];
            prop_assert!(v >= lo && v <= hi);
        }
    }

    #[test]
    fn composite_variation_preserves_count(
        bounds in bounds(3),
        a_frac in 0.0_f64..1.0,
        b_frac in 0.0_f64..1.0,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let p1: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + a_frac * (hi - lo)).collect();
        let p2: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + b_frac * (hi - lo)).collect();
        // SBX produces 2 children, PolyMut produces 1 each → expect 2.
        let mut v = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 0.5),
        };
        let children = v.vary(&[p1, p2], &mut rng);
        prop_assert_eq!(children.len(), 2);
    }
}

// -----------------------------------------------------------------------------
// Repair operators
// -----------------------------------------------------------------------------

proptest! {
    #[test]
    fn clamp_to_bounds_lands_in_bounds(
        bounds in bounds(5),
        seed in any::<u64>(),
    ) {
        use rand::Rng as _;
        let mut rng = rng_from_seed(seed);
        let mut x: Vec<f64> = (0..5).map(|_| rng.random_range(-1000.0..=1000.0)).collect();
        let mut r = ClampToBounds::new(bounds.clone());
        r.repair(&mut x);
        for (j, &v) in x.iter().enumerate() {
            let (lo, hi) = bounds[j];
            prop_assert!(v >= lo && v <= hi);
        }
    }

    #[test]
    fn clamp_to_bounds_idempotent(
        bounds in bounds(5),
        seed in any::<u64>(),
    ) {
        use rand::Rng as _;
        let mut rng = rng_from_seed(seed);
        let mut x: Vec<f64> = (0..5).map(|_| rng.random_range(-1000.0..=1000.0)).collect();
        let mut r = ClampToBounds::new(bounds);
        r.repair(&mut x);
        let after_one = x.clone();
        r.repair(&mut x);
        prop_assert_eq!(x, after_one);
    }

    #[test]
    fn project_to_simplex_lands_in_simplex(
        n in 2usize..8,
        total in 0.5_f64..10.0,
        seed in any::<u64>(),
    ) {
        use rand::Rng as _;
        let mut rng = rng_from_seed(seed);
        let mut x: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
        let mut r = ProjectToSimplex::new(total);
        r.repair(&mut x);
        for &v in &x {
            prop_assert!(v >= 0.0);
        }
        let s: f64 = x.iter().sum();
        prop_assert!((s - total).abs() < 1e-9);
    }

    #[test]
    fn project_to_simplex_idempotent(
        n in 2usize..8,
        total in 0.5_f64..5.0,
        seed in any::<u64>(),
    ) {
        use rand::Rng as _;
        let mut rng = rng_from_seed(seed);
        let mut x: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
        let mut r = ProjectToSimplex::new(total);
        r.repair(&mut x);
        let after_one = x.clone();
        r.repair(&mut x);
        for (a, b) in after_one.iter().zip(x.iter()) {
            prop_assert!((a - b).abs() < 1e-9);
        }
    }
}
