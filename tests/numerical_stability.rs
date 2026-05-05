//! Stress tests for numerical edge cases.
//!
//! These don't check correctness in detail — they check that algorithms
//! and helpers don't panic, return NaN, or produce nonsensical sizes on
//! pathological inputs. The kind of failures these surface are typically
//! division-by-zero, log/sqrt of negatives, empty-collection .min(),
//! etc. — all the things property tests on "ordinary" inputs would miss.

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::metrics::hypervolume::{hypervolume_2d, hypervolume_nd};
use heuropt::metrics::spacing::spacing;
use heuropt::pareto::crowding::crowding_distance;
use heuropt::pareto::dominance::pareto_compare;
use heuropt::pareto::front::{best_candidate, pareto_front};
use heuropt::pareto::sort::non_dominated_sort;
use heuropt::prelude::*;

fn space_2d() -> ObjectiveSpace {
    ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
}

fn cand2(a: f64, b: f64) -> Candidate<()> {
    Candidate::new((), Evaluation::new(vec![a, b]))
}

// -----------------------------------------------------------------------------
// Pareto utilities — empty / singleton / duplicate populations
// -----------------------------------------------------------------------------

#[test]
fn pareto_front_on_empty_population() {
    let s = space_2d();
    let front = pareto_front::<()>(&[], &s);
    assert!(front.is_empty());
}

#[test]
fn pareto_front_on_singleton() {
    let s = space_2d();
    let pop = vec![cand2(1.0, 2.0)];
    let front = pareto_front(&pop, &s);
    assert_eq!(front.len(), 1);
}

#[test]
fn pareto_front_on_all_duplicates() {
    let s = space_2d();
    let pop: Vec<_> = (0..5).map(|_| cand2(1.0, 1.0)).collect();
    let front = pareto_front(&pop, &s);
    // All members are mutually Equal — every one is non-dominated.
    assert_eq!(front.len(), 5);
}

#[test]
fn non_dominated_sort_on_empty() {
    let s = space_2d();
    let fronts = non_dominated_sort::<()>(&[], &s);
    assert!(fronts.is_empty());
}

#[test]
fn non_dominated_sort_on_all_duplicates() {
    let s = space_2d();
    let pop: Vec<_> = (0..6).map(|_| cand2(1.0, 1.0)).collect();
    let fronts = non_dominated_sort(&pop, &s);
    // Every member is "Equal" with every other member — should be one
    // front containing all of them.
    assert_eq!(fronts.len(), 1);
    assert_eq!(fronts[0].len(), 6);
}

#[test]
fn crowding_distance_on_empty_front_is_empty() {
    let s = space_2d();
    let pop: Vec<Candidate<()>> = vec![];
    let d = crowding_distance(&pop, &[], &s);
    assert!(d.is_empty());
}

#[test]
fn crowding_distance_on_two_point_front_is_infinity() {
    let s = space_2d();
    let pop = vec![cand2(0.0, 1.0), cand2(1.0, 0.0)];
    let d = crowding_distance(&pop, &[0, 1], &s);
    assert!(d[0].is_infinite());
    assert!(d[1].is_infinite());
}

#[test]
fn crowding_distance_on_collinear_points_finite_or_inf() {
    let s = space_2d();
    // All points have f2 = 5; f1 axis varies but f2 doesn't.
    let pop = vec![cand2(0.0, 5.0), cand2(1.0, 5.0), cand2(2.0, 5.0)];
    let d = crowding_distance(&pop, &[0, 1, 2], &s);
    // f2 axis has zero span so it contributes nothing; f1 axis gives the
    // boundaries infinity, the interior finite.
    assert!(d[0].is_infinite());
    assert!(d[2].is_infinite());
    assert!(d[1].is_finite());
}

#[test]
fn pareto_compare_with_zero_constraint_violations() {
    let s = space_2d();
    let a = Evaluation::constrained(vec![1.0, 1.0], 0.0);
    let b = Evaluation::constrained(vec![2.0, 2.0], 0.0);
    let r = pareto_compare(&a, &b, &s);
    use heuropt::pareto::dominance::Dominance;
    assert_eq!(r, Dominance::Dominates);
}

#[test]
fn best_candidate_on_empty_returns_none() {
    let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
    let pop: Vec<Candidate<()>> = vec![];
    let best = best_candidate(&pop, &s);
    assert!(best.is_none());
}

#[test]
fn best_candidate_all_infeasible_returns_none() {
    let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
    let pop = vec![
        Candidate::new((), Evaluation::constrained(vec![1.0], 0.5)),
        Candidate::new((), Evaluation::constrained(vec![2.0], 0.7)),
    ];
    let best = best_candidate(&pop, &s);
    assert!(best.is_none());
}

// -----------------------------------------------------------------------------
// Metrics — degenerate inputs
// -----------------------------------------------------------------------------

#[test]
fn hv2_on_empty_is_zero() {
    let s = space_2d();
    let front: Vec<Candidate<()>> = vec![];
    assert_eq!(hypervolume_2d(&front, &s, [1.0, 1.0]), 0.0);
}

#[test]
fn hv2_when_no_point_dominates_reference_is_zero() {
    let s = space_2d();
    let front = vec![cand2(5.0, 5.0)]; // worse than reference (1, 1)
    assert_eq!(hypervolume_2d(&front, &s, [1.0, 1.0]), 0.0);
}

#[test]
fn hv_nd_on_empty_is_zero() {
    let s = ObjectiveSpace::new(vec![
        Objective::minimize("a"),
        Objective::minimize("b"),
        Objective::minimize("c"),
    ]);
    let front: Vec<Candidate<()>> = vec![];
    assert_eq!(hypervolume_nd(&front, &s, &[1.0, 1.0, 1.0]), 0.0);
}

#[test]
fn spacing_on_empty_is_zero() {
    let s = space_2d();
    let pop: Vec<Candidate<()>> = vec![];
    assert_eq!(spacing(&pop, &s), 0.0);
}

#[test]
fn spacing_on_singleton_is_zero() {
    let s = space_2d();
    let pop = vec![cand2(0.5, 0.5)];
    assert_eq!(spacing(&pop, &s), 0.0);
}

// -----------------------------------------------------------------------------
// Algorithms — extreme inputs
// -----------------------------------------------------------------------------

struct ConstantFn;
impl heuropt::core::problem::Problem for ConstantFn {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }
    fn evaluate(&self, _: &Vec<f64>) -> Evaluation {
        // Flat fitness — every point is equally good.
        Evaluation::new(vec![0.0])
    }
}

#[test]
fn de_handles_flat_fitness() {
    // No gradient, no signal. DE should still run to completion and
    // return a valid result (every point ties for best).
    let mut opt = DifferentialEvolution::new(
        DifferentialEvolutionConfig {
            population_size: 10,
            generations: 5,
            differential_weight: 0.5,
            crossover_probability: 0.9,
            seed: 0,
        },
        RealBounds::new(vec![(-1.0, 1.0); 3]),
    );
    let r = opt.run(&ConstantFn);
    let best = r.best.unwrap();
    assert_eq!(best.evaluation.objectives, vec![0.0]);
    assert!(r.evaluations > 0);
}

#[test]
fn cma_es_handles_flat_fitness() {
    let mut opt = CmaEs::new(
        CmaEsConfig {
            population_size: 8,
            generations: 5,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            initial_mean: None,
            seed: 0,
        },
        RealBounds::new(vec![(-1.0, 1.0); 3]),
    );
    let r = opt.run(&ConstantFn);
    assert!(r.best.is_some());
}

#[test]
fn nelder_mead_handles_flat_fitness() {
    let mut opt = NelderMead::new(
        NelderMeadConfig::default(),
        RealBounds::new(vec![(-1.0, 1.0); 3]),
    );
    let r = opt.run(&ConstantFn);
    assert!(r.best.is_some());
}

#[test]
fn bayesian_opt_handles_flat_fitness() {
    let mut opt = BayesianOpt::new(
        BayesianOptConfig {
            initial_samples: 4,
            iterations: 6,
            length_scales: None,
            signal_variance: 1.0,
            noise_variance: 1e-3,
            acquisition_samples: 50,
            seed: 0,
        },
        RealBounds::new(vec![(-1.0, 1.0); 2]),
    );
    let r = opt.run(&ConstantFn);
    assert!(r.best.is_some());
}

#[test]
fn de_handles_zero_width_bounds() {
    // lo == hi on every axis — search space is a single point.
    let mut opt = DifferentialEvolution::new(
        DifferentialEvolutionConfig {
            population_size: 4,
            generations: 3,
            differential_weight: 0.5,
            crossover_probability: 0.9,
            seed: 0,
        },
        RealBounds::new(vec![(0.5, 0.5); 2]),
    );
    let r = opt.run(&ConstantFn);
    let best = r.best.unwrap();
    // Every decision must be exactly (0.5, 0.5).
    for d in &r.population.candidates {
        for &v in &d.decision {
            assert_eq!(v, 0.5);
        }
    }
    let _ = best;
}
