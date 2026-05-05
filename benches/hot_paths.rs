//! Instruction-count benchmarks for heuropt's algorithmic hot paths.
//!
//! Run with `cargo bench`. Requires `valgrind` installed.
//!
//! The benchmarks here are *not* end-to-end optimizer runs; those are
//! covered by `examples/compare`. These are the inner-loop primitives
//! that every algorithm depends on, so a regression here lights up
//! across the whole crate.

use std::hint::black_box;

use gungraun::prelude::*;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::metrics::hypervolume::{hypervolume_2d, hypervolume_nd};
use heuropt::pareto::crowding::crowding_distance;
use heuropt::pareto::sort::non_dominated_sort;
use heuropt::core::problem::Problem;
use heuropt::prelude::*;

// -----------------------------------------------------------------------------
// Pareto utilities
// -----------------------------------------------------------------------------

fn make_2d_population(n: usize) -> Vec<Candidate<()>> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            Candidate::new((), Evaluation::new(vec![t, 1.0 - t.sqrt()]))
        })
        .collect()
}

fn space_2d() -> ObjectiveSpace {
    ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
}

#[library_benchmark]
#[bench::n_50(50)]
#[bench::n_200(200)]
fn non_dominated_sort_2d(n: usize) -> Vec<Vec<usize>> {
    let pop = make_2d_population(n);
    let s = space_2d();
    black_box(non_dominated_sort(black_box(&pop), black_box(&s)))
}

#[library_benchmark]
#[bench::n_50(50)]
#[bench::n_200(200)]
fn crowding_distance_2d(n: usize) -> Vec<f64> {
    let pop = make_2d_population(n);
    let s = space_2d();
    let front: Vec<usize> = (0..pop.len()).collect();
    black_box(crowding_distance(black_box(&pop), black_box(&front), black_box(&s)))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn hypervolume_2d_bench(n: usize) -> f64 {
    let pop = make_2d_population(n);
    let s = space_2d();
    black_box(hypervolume_2d(black_box(&pop), black_box(&s), black_box([1.1, 1.1])))
}

fn make_3d_population(n: usize) -> (Vec<Candidate<()>>, ObjectiveSpace) {
    let s = ObjectiveSpace::new(vec![
        Objective::minimize("f1"),
        Objective::minimize("f2"),
        Objective::minimize("f3"),
    ]);
    let pop = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let theta = 0.5 * std::f64::consts::PI * t;
            Candidate::new(
                (),
                Evaluation::new(vec![theta.cos(), theta.sin(), 1.0 - t]),
            )
        })
        .collect();
    (pop, s)
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn hypervolume_nd_bench_3d(n: usize) -> f64 {
    let (pop, s) = make_3d_population(n);
    black_box(hypervolume_nd(black_box(&pop), black_box(&s), black_box(&[2.0, 2.0, 2.0])))
}

library_benchmark_group!(
    name = pareto_group;
    benchmarks =
        non_dominated_sort_2d,
        crowding_distance_2d,
        hypervolume_2d_bench,
        hypervolume_nd_bench_3d
);

// -----------------------------------------------------------------------------
// End-to-end algorithm smoke benches (single-generation cost)
// -----------------------------------------------------------------------------

/// Schaffer N.1 (2-objective). Inlined here so the bench doesn't need
/// to reach into the crate's `cfg(test)` test support.
struct SchafferN1;
impl Problem for SchafferN1 {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }
    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let v = x[0];
        Evaluation::new(vec![v * v, (v - 2.0).powi(2)])
    }
}

#[library_benchmark]
fn nsga2_one_generation() -> usize {
    let bounds = vec![(-5.0, 5.0)];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
    };
    let mut opt = Nsga2::new(
        Nsga2Config { population_size: 50, generations: 1, seed: 0 },
        initializer,
        variation,
    );
    let result = opt.run(black_box(&SchafferN1));
    black_box(result.evaluations)
}

#[library_benchmark]
fn cma_es_one_generation() -> usize {
    let bounds = RealBounds::new(vec![(-5.0, 5.0); 5]);
    let mut opt = CmaEs::new(
        CmaEsConfig {
            population_size: 16,
            generations: 1,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            initial_mean: None,
            seed: 0,
        },
        bounds,
    );
    struct Sphere5D;
    impl Problem for Sphere5D {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }
        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            Evaluation::new(vec![x.iter().map(|v| v * v).sum()])
        }
    }
    let result = opt.run(black_box(&Sphere5D));
    black_box(result.evaluations)
}

library_benchmark_group!(
    name = algorithm_group;
    benchmarks = nsga2_one_generation, cma_es_one_generation
);

main!(library_benchmark_groups = pareto_group, algorithm_group);
