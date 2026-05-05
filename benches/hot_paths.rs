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
use heuropt::core::problem::Problem;
use heuropt::metrics::hypervolume::{hypervolume_2d, hypervolume_nd};
use heuropt::pareto::crowding::crowding_distance;
use heuropt::pareto::sort::non_dominated_sort;
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
    black_box(crowding_distance(
        black_box(&pop),
        black_box(&front),
        black_box(&s),
    ))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn hypervolume_2d_bench(n: usize) -> f64 {
    let pop = make_2d_population(n);
    let s = space_2d();
    black_box(hypervolume_2d(
        black_box(&pop),
        black_box(&s),
        black_box([1.1, 1.1]),
    ))
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
            Candidate::new((), Evaluation::new(vec![theta.cos(), theta.sin(), 1.0 - t]))
        })
        .collect();
    (pop, s)
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn hypervolume_nd_bench_3d(n: usize) -> f64 {
    let (pop, s) = make_3d_population(n);
    black_box(hypervolume_nd(
        black_box(&pop),
        black_box(&s),
        black_box(&[2.0, 2.0, 2.0]),
    ))
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
        Nsga2Config {
            population_size: 50,
            generations: 1,
            seed: 0,
        },
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

// -----------------------------------------------------------------------------
// Wider single-objective sweep
// -----------------------------------------------------------------------------

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

fn so_bounds() -> RealBounds {
    RealBounds::new(vec![(-3.0, 3.0)])
}

#[library_benchmark]
fn random_search_short() -> usize {
    let mut o = RandomSearch::new(
        RandomSearchConfig {
            iterations: 50,
            batch_size: 1,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn hill_climber_short() -> usize {
    let mut o = HillClimber::new(
        HillClimberConfig {
            iterations: 50,
            seed: 0,
        },
        so_bounds(),
        GaussianMutation { sigma: 0.1 },
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn one_plus_one_es_short() -> usize {
    let mut o = OnePlusOneEs::new(
        OnePlusOneEsConfig {
            iterations: 50,
            initial_sigma: 0.5,
            adaptation_period: 10,
            step_increase: 1.22,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn simulated_annealing_short() -> usize {
    let mut o = SimulatedAnnealing::new(
        SimulatedAnnealingConfig {
            iterations: 50,
            initial_temperature: 1.0,
            final_temperature: 1e-3,
            seed: 0,
        },
        so_bounds(),
        GaussianMutation { sigma: 0.1 },
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn genetic_algorithm_short() -> usize {
    let bounds = vec![(-3.0, 3.0)];
    let mut o = GeneticAlgorithm::new(
        GeneticAlgorithmConfig {
            population_size: 10,
            generations: 5,
            tournament_size: 2,
            elitism: 1,
            seed: 0,
        },
        RealBounds::new(bounds.clone()),
        CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        },
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn particle_swarm_short() -> usize {
    let mut o = ParticleSwarm::new(
        ParticleSwarmConfig {
            swarm_size: 10,
            generations: 5,
            inertia: 0.7,
            cognitive: 1.5,
            social: 1.5,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn differential_evolution_short() -> usize {
    let mut o = DifferentialEvolution::new(
        DifferentialEvolutionConfig {
            population_size: 10,
            generations: 5,
            differential_weight: 0.5,
            crossover_probability: 0.9,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn tlbo_short() -> usize {
    let mut o = Tlbo::new(
        TlboConfig {
            population_size: 10,
            generations: 5,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn separable_nes_short() -> usize {
    let mut o = SeparableNes::new(
        SeparableNesConfig {
            population_size: 8,
            generations: 5,
            initial_sigma: 0.5,
            mean_learning_rate: 1.0,
            sigma_learning_rate: None,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn nelder_mead_short() -> usize {
    let mut o = NelderMead::new(
        NelderMeadConfig {
            iterations: 50,
            ..NelderMeadConfig::default()
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn bayesian_opt_short() -> usize {
    let mut o = BayesianOpt::new(
        BayesianOptConfig {
            initial_samples: 5,
            iterations: 10,
            length_scales: None,
            signal_variance: 1.0,
            noise_variance: 1e-6,
            acquisition_samples: 100,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn tpe_short() -> usize {
    let mut o = Tpe::new(
        TpeConfig {
            initial_samples: 5,
            iterations: 10,
            good_fraction: 0.25,
            candidate_samples: 12,
            bandwidth_factor: 1.0,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

#[library_benchmark]
fn ipop_cma_es_short() -> usize {
    let mut o = IpopCmaEs::new(
        IpopCmaEsConfig {
            initial_population_size: 8,
            total_generations: 30,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            stall_generations: None,
            seed: 0,
        },
        so_bounds(),
    );
    black_box(o.run(black_box(&Sphere1D)).evaluations)
}

library_benchmark_group!(
    name = single_objective_group;
    benchmarks =
        random_search_short, hill_climber_short, one_plus_one_es_short,
        simulated_annealing_short, genetic_algorithm_short,
        particle_swarm_short, differential_evolution_short, tlbo_short,
        separable_nes_short, nelder_mead_short,
        bayesian_opt_short, tpe_short, ipop_cma_es_short
);

// -----------------------------------------------------------------------------
// Multi-objective sweep
// -----------------------------------------------------------------------------

fn schaffer_bounds() -> Vec<(f64, f64)> {
    vec![(-3.0, 3.0)]
}
fn mo_variation() -> CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation> {
    let bounds = schaffer_bounds();
    CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
    }
}

#[library_benchmark]
fn nsga3_short() -> usize {
    let mut o = Nsga3::new(
        Nsga3Config {
            population_size: 12,
            generations: 1,
            reference_divisions: 11,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn spea2_short() -> usize {
    let mut o = Spea2::new(
        Spea2Config {
            population_size: 10,
            archive_size: 10,
            generations: 1,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn moead_short() -> usize {
    let mut o = Moead::new(
        MoeadConfig {
            generations: 1,
            reference_divisions: 9,
            neighborhood_size: 4,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn mopso_short() -> usize {
    let mut o = Mopso::new(
        MopsoConfig {
            swarm_size: 10,
            generations: 1,
            archive_size: 10,
            inertia: 0.7,
            cognitive: 1.5,
            social: 1.5,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn ibea_short() -> usize {
    let mut o = Ibea::new(
        IbeaConfig {
            population_size: 10,
            generations: 1,
            kappa: 0.05,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn sms_emoa_short() -> usize {
    let mut o = SmsEmoa::new(
        SmsEmoaConfig {
            population_size: 8,
            generations: 5,
            reference_point: vec![10.0, 10.0],
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn hype_short() -> usize {
    let mut o = Hype::new(
        HypeConfig {
            population_size: 10,
            generations: 1,
            reference_point: vec![10.0, 10.0],
            mc_samples: 100,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn pesa2_short() -> usize {
    let mut o = PesaII::new(
        PesaIIConfig {
            population_size: 10,
            archive_size: 10,
            generations: 1,
            grid_divisions: 4,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn epsilon_moea_short() -> usize {
    let mut o = EpsilonMoea::new(
        EpsilonMoeaConfig {
            population_size: 10,
            evaluations: 30,
            epsilon: vec![0.05, 0.05],
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn age_moea_short() -> usize {
    let mut o = AgeMoea::new(
        AgeMoeaConfig {
            population_size: 10,
            generations: 1,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn grea_short() -> usize {
    let mut o = Grea::new(
        GreaConfig {
            population_size: 10,
            generations: 1,
            grid_divisions: 4,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn knea_short() -> usize {
    let mut o = Knea::new(
        KneaConfig {
            population_size: 10,
            generations: 1,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn rvea_short() -> usize {
    let mut o = Rvea::new(
        RveaConfig {
            population_size: 10,
            generations: 1,
            reference_divisions: 9,
            alpha: 2.0,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        mo_variation(),
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

#[library_benchmark]
fn paes_short() -> usize {
    let mut o = Paes::new(
        PaesConfig {
            iterations: 30,
            archive_size: 10,
            seed: 0,
        },
        RealBounds::new(schaffer_bounds()),
        GaussianMutation { sigma: 0.1 },
    );
    black_box(o.run(black_box(&SchafferN1)).evaluations)
}

library_benchmark_group!(
    name = multi_objective_group;
    benchmarks =
        nsga3_short, spea2_short, moead_short, mopso_short, ibea_short,
        sms_emoa_short, hype_short, pesa2_short, epsilon_moea_short,
        age_moea_short, grea_short, knea_short, rvea_short, paes_short
);

main!(
    library_benchmark_groups = pareto_group,
    algorithm_group,
    single_objective_group,
    multi_objective_group
);
