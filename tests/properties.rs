//! Property-based tests for heuropt invariants.
//!
//! Where the unit-test suite checks specific cases, this suite checks
//! invariants that should hold for *any* well-formed input. proptest
//! generates random instances and shrinks failures.

use proptest::prelude::*;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::pareto::dominance::{Dominance, pareto_compare};
use heuropt::pareto::front::pareto_front;
use heuropt::pareto::sort::non_dominated_sort;
use heuropt::prelude::*;

// -----------------------------------------------------------------------------
// Strategies
// -----------------------------------------------------------------------------

/// Generate a 2-objective minimize ObjectiveSpace.
fn space_2d() -> ObjectiveSpace {
    ObjectiveSpace::new(vec![
        Objective::minimize("f1"),
        Objective::minimize("f2"),
    ])
}

/// Generate a candidate with a 2-D objective vector in `[lo, hi]`.
fn candidate_2d(lo: f64, hi: f64) -> impl Strategy<Value = Candidate<()>> {
    (lo..hi, lo..hi).prop_map(|(a, b)| {
        Candidate::new((), Evaluation::new(vec![a, b]))
    })
}

/// Generate a small 2-D population.
fn population_2d() -> impl Strategy<Value = Vec<Candidate<()>>> {
    prop::collection::vec(candidate_2d(-100.0, 100.0), 1..=15)
}

/// Generate per-axis bounds.
fn bounds(dim: usize) -> impl Strategy<Value = Vec<(f64, f64)>> {
    prop::collection::vec((-50.0_f64..50.0, 0.001_f64..50.0), dim..=dim)
        .prop_map(|pairs| pairs.into_iter().map(|(lo, span)| (lo, lo + span)).collect())
}

// -----------------------------------------------------------------------------
// Pareto invariants
// -----------------------------------------------------------------------------

proptest! {
    /// `pareto_compare` is anti-symmetric on Dominates / DominatedBy.
    #[test]
    fn pareto_compare_is_antisymmetric(
        a in candidate_2d(-100.0, 100.0),
        b in candidate_2d(-100.0, 100.0),
    ) {
        let s = space_2d();
        let ab = pareto_compare(&a.evaluation, &b.evaluation, &s);
        let ba = pareto_compare(&b.evaluation, &a.evaluation, &s);
        match (ab, ba) {
            (Dominance::Dominates, Dominance::DominatedBy) => {}
            (Dominance::DominatedBy, Dominance::Dominates) => {}
            (Dominance::Equal, Dominance::Equal) => {}
            (Dominance::NonDominated, Dominance::NonDominated) => {}
            (l, r) => prop_assert!(false, "asymmetric result: ab={l:?}, ba={r:?}"),
        }
    }

    /// Comparing a candidate with itself returns Equal.
    #[test]
    fn pareto_compare_reflexive(a in candidate_2d(-100.0, 100.0)) {
        let s = space_2d();
        let r = pareto_compare(&a.evaluation, &a.evaluation, &s);
        prop_assert_eq!(r, Dominance::Equal);
    }

    /// `pareto_front` output members are pairwise non-dominated.
    #[test]
    fn pareto_front_is_internally_nondominated(pop in population_2d()) {
        let s = space_2d();
        let front = pareto_front(&pop, &s);
        for i in 0..front.len() {
            for j in 0..front.len() {
                if i == j {
                    continue;
                }
                let r = pareto_compare(&front[i].evaluation, &front[j].evaluation, &s);
                prop_assert!(
                    !matches!(r, Dominance::DominatedBy),
                    "front member {i} dominated by {j}",
                );
            }
        }
    }

    /// `non_dominated_sort` partitions every population member into exactly
    /// one front (no missing or duplicate indices).
    #[test]
    fn non_dominated_sort_partitions_population(pop in population_2d()) {
        let s = space_2d();
        let fronts = non_dominated_sort(&pop, &s);
        let mut seen = vec![false; pop.len()];
        for front in &fronts {
            for &idx in front {
                prop_assert!(!seen[idx], "index {idx} appears in multiple fronts");
                seen[idx] = true;
            }
        }
        for (i, &was_seen) in seen.iter().enumerate() {
            prop_assert!(was_seen, "index {i} is missing from all fronts");
        }
    }
}

// -----------------------------------------------------------------------------
// Operator invariants
// -----------------------------------------------------------------------------

proptest! {
    /// SBX returns exactly 2 children, both in bounds when parents are in
    /// bounds. (SBX only clamps variables it actually mixes — when
    /// `per_variable_probability < 1` the rest pass through from the
    /// parents, so out-of-bounds parents would yield out-of-bounds children
    /// by design. We're checking the in-bounds-parent contract.)
    #[test]
    fn sbx_children_in_bounds_when_parents_in_bounds(
        bounds in bounds(3),
        eta in 1.0_f64..30.0,
        per_var_p in 0.0_f64..=1.0,
        a_frac in 0.0_f64..1.0,
        b_frac in 0.0_f64..1.0,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        // Parents are convex combinations of bounds — strictly in box.
        let p1: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + a_frac * (hi - lo)).collect();
        let p2: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + b_frac * (hi - lo)).collect();
        let mut sbx = SimulatedBinaryCrossover::new(bounds.clone(), eta, per_var_p);
        let children = sbx.vary(&[p1, p2], &mut rng);
        prop_assert_eq!(children.len(), 2);
        for c in &children {
            prop_assert_eq!(c.len(), 3);
            for (j, &v) in c.iter().enumerate() {
                let (lo, hi) = bounds[j];
                prop_assert!(v >= lo && v <= hi, "SBX child[{j}] = {v} out of [{lo}, {hi}]");
            }
        }
    }

    /// PolynomialMutation returns 1 child in bounds.
    #[test]
    fn polymut_child_in_bounds(
        bounds in bounds(4),
        eta in 1.0_f64..40.0,
        per_var_p in 0.0_f64..=1.0,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        let parent: Vec<f64> = bounds.iter().map(|&(lo, hi)| 0.5 * (lo + hi)).collect();
        let mut pm = PolynomialMutation::new(bounds.clone(), eta, per_var_p);
        let children = pm.vary(std::slice::from_ref(&parent), &mut rng);
        prop_assert_eq!(children.len(), 1);
        for (j, &v) in children[0].iter().enumerate() {
            let (lo, hi) = bounds[j];
            prop_assert!(v >= lo && v <= hi);
        }
    }

    /// ClampToBounds always lands every variable in bounds.
    #[test]
    fn clamp_to_bounds_lands_in_bounds(
        bounds in bounds(5),
        x_seed in any::<u64>(),
    ) {
        // Sample a "before-repair" vector that may be wildly out of bounds.
        let mut rng = rng_from_seed(x_seed);
        use rand::Rng as _;
        let mut x: Vec<f64> = (0..5).map(|_| rng.random_range(-1000.0..=1000.0)).collect();
        let mut r = ClampToBounds::new(bounds.clone());
        r.repair(&mut x);
        for (j, &v) in x.iter().enumerate() {
            let (lo, hi) = bounds[j];
            prop_assert!(v >= lo && v <= hi);
        }
    }

    /// ProjectToSimplex always lands in the simplex { x ≥ 0, Σ x = total }.
    #[test]
    fn project_to_simplex_lands_in_simplex(
        n in 2usize..8,
        total in 0.5_f64..10.0,
        seed in any::<u64>(),
    ) {
        let mut rng = rng_from_seed(seed);
        use rand::Rng as _;
        let mut x: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
        let mut r = ProjectToSimplex::new(total);
        r.repair(&mut x);
        for &v in &x {
            prop_assert!(v >= 0.0, "negative entry: {v}");
        }
        let s: f64 = x.iter().sum();
        prop_assert!(
            (s - total).abs() < 1e-9,
            "sum {s} != total {total}",
        );
    }
}

// -----------------------------------------------------------------------------
// Optimizer determinism
// -----------------------------------------------------------------------------

/// Tiny single-objective sphere problem reused across determinism props.
struct Sphere1D;
impl Problem for Sphere1D {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }
    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        Evaluation::new(vec![x[0] * x[0]])
    }
}

proptest! {
    #[test]
    fn de_deterministic_with_seed(seed in any::<u64>()) {
        let mut a = DifferentialEvolution::new(
            DifferentialEvolutionConfig {
                population_size: 10,
                generations: 5,
                differential_weight: 0.5,
                crossover_probability: 0.9,
                seed,
            },
            RealBounds::new(vec![(-3.0, 3.0)]),
        );
        let mut b = DifferentialEvolution::new(
            DifferentialEvolutionConfig {
                population_size: 10,
                generations: 5,
                differential_weight: 0.5,
                crossover_probability: 0.9,
                seed,
            },
            RealBounds::new(vec![(-3.0, 3.0)]),
        );
        let ra = a.run(&Sphere1D);
        let rb = b.run(&Sphere1D);
        prop_assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn cmaes_deterministic_with_seed(seed in any::<u64>()) {
        let cfg = CmaEsConfig {
            population_size: 8,
            generations: 5,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            initial_mean: None,
            seed,
        };
        let mut a = CmaEs::new(cfg.clone(), RealBounds::new(vec![(-3.0, 3.0)]));
        let mut b = CmaEs::new(cfg, RealBounds::new(vec![(-3.0, 3.0)]));
        let ra = a.run(&Sphere1D);
        let rb = b.run(&Sphere1D);
        prop_assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }
}
