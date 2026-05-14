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
use rand::Rng as _;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::core::partial_problem::PartialProblem;
use heuropt::core::problem::Problem;
use heuropt::core::rng::{Rng, rng_from_seed};
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

/// 1-D integer parabola: minimize `(x - 5)^2`. `Vec<i32>` decision so it
/// satisfies `TabuSearch`'s `Hash + Eq` decision bound (`f64` is neither).
struct IntParabola;
impl Problem for IntParabola {
    type Decision = Vec<i32>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }
    fn evaluate(&self, x: &Vec<i32>) -> Evaluation {
        let v = (x[0] - 5) as f64;
        Evaluation::new(vec![v * v])
    }
}

/// Start every 1-D integer decision at 0.
struct IntStartAtZero;
impl Initializer<Vec<i32>> for IntStartAtZero {
    fn initialize(&mut self, size: usize, _rng: &mut Rng) -> Vec<Vec<i32>> {
        (0..size).map(|_| vec![0]).collect()
    }
}

#[library_benchmark]
fn tabu_search_short() -> usize {
    let neighbors = |x: &Vec<i32>, _rng: &mut Rng| {
        vec![
            vec![x[0] - 2],
            vec![x[0] - 1],
            vec![x[0] + 1],
            vec![x[0] + 2],
        ]
    };
    let mut o = TabuSearch::new(
        TabuSearchConfig {
            iterations: 50,
            tabu_tenure: 8,
            seed: 0,
        },
        IntStartAtZero,
        neighbors,
    );
    black_box(o.run(black_box(&IntParabola)).evaluations)
}

/// OneMax over 16 bits: maximize the count of `true` bits.
struct OneMax16;
impl Problem for OneMax16 {
    type Decision = Vec<bool>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::maximize("ones")])
    }
    fn evaluate(&self, x: &Vec<bool>) -> Evaluation {
        Evaluation::new(vec![x.iter().filter(|b| **b).count() as f64])
    }
}

#[library_benchmark]
fn umda_short() -> usize {
    let mut o = Umda::new(UmdaConfig {
        population_size: 20,
        selected_size: 8,
        generations: 5,
        bits: 16,
        seed: 0,
    });
    black_box(o.run(black_box(&OneMax16)).evaluations)
}

library_benchmark_group!(
    name = single_objective_group;
    benchmarks =
        random_search_short, hill_climber_short, one_plus_one_es_short,
        simulated_annealing_short, genetic_algorithm_short,
        particle_swarm_short, differential_evolution_short, tlbo_short,
        separable_nes_short, nelder_mead_short,
        bayesian_opt_short, tpe_short, ipop_cma_es_short,
        tabu_search_short, umda_short
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

// -----------------------------------------------------------------------------
// Permutation operator micro-benchmarks
// -----------------------------------------------------------------------------

fn perm_parent(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// Reversed `[0..n)`: same value multiset as `perm_parent`, shares no oriented
/// edges with it — a stress input for the edge-based crossovers.
fn perm_parent_rev(n: usize) -> Vec<usize> {
    (0..n).rev().collect()
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn shuffled_permutation_init(n: usize) -> Vec<Vec<usize>> {
    let mut rng = rng_from_seed(0);
    let mut init = ShuffledPermutation { n };
    black_box(init.initialize(black_box(16), black_box(&mut rng)))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn shuffled_multiset_permutation_init(n: usize) -> Vec<Vec<usize>> {
    let mut rng = rng_from_seed(0);
    let mut init = ShuffledMultisetPermutation::new(vec![5; n]);
    black_box(init.initialize(black_box(16), black_box(&mut rng)))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn swap_mutation_vary(n: usize) -> Vec<Vec<usize>> {
    let parent = perm_parent(n);
    let mut rng = rng_from_seed(1);
    let mut op = SwapMutation;
    black_box(op.vary(
        black_box(std::slice::from_ref(&parent)),
        black_box(&mut rng),
    ))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn inversion_mutation_vary(n: usize) -> Vec<Vec<usize>> {
    let parent = perm_parent(n);
    let mut rng = rng_from_seed(1);
    let mut op = InversionMutation;
    black_box(op.vary(
        black_box(std::slice::from_ref(&parent)),
        black_box(&mut rng),
    ))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn insertion_mutation_vary(n: usize) -> Vec<Vec<usize>> {
    let parent = perm_parent(n);
    let mut rng = rng_from_seed(1);
    let mut op = InsertionMutation;
    black_box(op.vary(
        black_box(std::slice::from_ref(&parent)),
        black_box(&mut rng),
    ))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn scramble_mutation_vary(n: usize) -> Vec<Vec<usize>> {
    let parent = perm_parent(n);
    let mut rng = rng_from_seed(1);
    let mut op = ScrambleMutation;
    black_box(op.vary(
        black_box(std::slice::from_ref(&parent)),
        black_box(&mut rng),
    ))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn order_crossover_vary(n: usize) -> Vec<Vec<usize>> {
    let parents = [perm_parent(n), perm_parent_rev(n)];
    let mut rng = rng_from_seed(2);
    let mut op = OrderCrossover;
    black_box(op.vary(black_box(&parents), black_box(&mut rng)))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn pmx_crossover_vary(n: usize) -> Vec<Vec<usize>> {
    let parents = [perm_parent(n), perm_parent_rev(n)];
    let mut rng = rng_from_seed(2);
    let mut op = PartiallyMappedCrossover;
    black_box(op.vary(black_box(&parents), black_box(&mut rng)))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn cycle_crossover_vary(n: usize) -> Vec<Vec<usize>> {
    let parents = [perm_parent(n), perm_parent_rev(n)];
    let mut rng = rng_from_seed(2);
    let mut op = CycleCrossover;
    black_box(op.vary(black_box(&parents), black_box(&mut rng)))
}

#[library_benchmark]
#[bench::n_30(30)]
#[bench::n_100(100)]
fn edge_recombination_crossover_vary(n: usize) -> Vec<Vec<usize>> {
    let parents = [perm_parent(n), perm_parent_rev(n)];
    let mut rng = rng_from_seed(2);
    let mut op = EdgeRecombinationCrossover;
    black_box(op.vary(black_box(&parents), black_box(&mut rng)))
}

library_benchmark_group!(
    name = permutation_ops_group;
    benchmarks =
        shuffled_permutation_init, shuffled_multiset_permutation_init,
        swap_mutation_vary, inversion_mutation_vary, insertion_mutation_vary,
        scramble_mutation_vary, order_crossover_vary, pmx_crossover_vary,
        cycle_crossover_vary, edge_recombination_crossover_vary
);

// -----------------------------------------------------------------------------
// Un-benchmarked operators from the binary / real / repair families
// -----------------------------------------------------------------------------

#[library_benchmark]
fn bit_flip_mutation_vary() -> Vec<Vec<bool>> {
    let parent: Vec<bool> = (0..64).map(|i| i % 2 == 0).collect();
    let mut rng = rng_from_seed(3);
    let mut op = BitFlipMutation {
        probability: 1.0 / 64.0,
    };
    black_box(op.vary(
        black_box(std::slice::from_ref(&parent)),
        black_box(&mut rng),
    ))
}

#[library_benchmark]
fn levy_mutation_vary() -> Vec<Vec<f64>> {
    let parent = vec![0.0_f64; 16];
    let mut rng = rng_from_seed(3);
    let mut op = LevyMutation::new(1.5, 0.1, vec![(-5.0, 5.0); 16]);
    black_box(op.vary(
        black_box(std::slice::from_ref(&parent)),
        black_box(&mut rng),
    ))
}

#[library_benchmark]
fn bounded_gaussian_mutation_vary() -> Vec<Vec<f64>> {
    let parent = vec![0.0_f64; 16];
    let mut rng = rng_from_seed(3);
    let mut op = BoundedGaussianMutation::new(0.3, vec![(-1.0, 1.0); 16]);
    black_box(op.vary(
        black_box(std::slice::from_ref(&parent)),
        black_box(&mut rng),
    ))
}

#[library_benchmark]
fn clamp_to_bounds_repair() -> Vec<f64> {
    let mut x: Vec<f64> = (0..32).map(|i| (i as f64) - 16.0).collect();
    let mut op = ClampToBounds::new(vec![(-1.0, 1.0); 32]);
    op.repair(black_box(&mut x));
    black_box(x)
}

#[library_benchmark]
fn project_to_simplex_repair() -> Vec<f64> {
    // 32-dim mixed-sign vector; exercises the sort-based projection path.
    let mut x: Vec<f64> = (0..32).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();
    let mut op = ProjectToSimplex::new(1.0);
    op.repair(black_box(&mut x));
    black_box(x)
}

library_benchmark_group!(
    name = variation_ops_group;
    benchmarks =
        bit_flip_mutation_vary, levy_mutation_vary, bounded_gaussian_mutation_vary,
        clamp_to_bounds_repair, project_to_simplex_repair
);

// -----------------------------------------------------------------------------
// Combinatorial / sequencing end-to-end benches
// -----------------------------------------------------------------------------

const TSP_N: usize = 15;

/// Deterministic pseudo-scattered city coordinates. The bench only needs a
/// stable distance matrix, not a known optimum.
fn tsp_coords() -> Vec<(f64, f64)> {
    (0..TSP_N)
        .map(|i| {
            let x = ((i * 37) % 100) as f64;
            let y = ((i * 53 + 11) % 100) as f64;
            (x, y)
        })
        .collect()
}

fn tsp_distance_matrix() -> Vec<Vec<f64>> {
    let c = tsp_coords();
    let n = c.len();
    let mut d = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let dx = c[i].0 - c[j].0;
                let dy = c[i].1 - c[j].1;
                d[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }
    }
    d
}

/// Single-objective TSP over a precomputed distance matrix.
struct TspProblem {
    distances: Vec<Vec<f64>>,
}
impl Problem for TspProblem {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("length")])
    }
    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        let n = tour.len();
        let mut len = 0.0;
        for i in 0..n {
            len += self.distances[tour[i]][tour[(i + 1) % n]];
        }
        Evaluation::new(vec![len])
    }
}

/// Bi-objective TSP: two distance matrices over the same city set.
struct BiTspProblem {
    dist_a: Vec<Vec<f64>>,
    dist_b: Vec<Vec<f64>>,
}
impl Problem for BiTspProblem {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("length_a"),
            Objective::minimize("length_b"),
        ])
    }
    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        let n = tour.len();
        let (mut la, mut lb) = (0.0, 0.0);
        for i in 0..n {
            let (u, v) = (tour[i], tour[(i + 1) % n]);
            la += self.dist_a[u][v];
            lb += self.dist_b[u][v];
        }
        Evaluation::new(vec![la, lb])
    }
}

#[library_benchmark]
fn tsp_nsga2_short() -> usize {
    let dist_a = tsp_distance_matrix();
    // Second objective: a distinct symmetric matrix with a zero diagonal.
    let dist_b: Vec<Vec<f64>> = dist_a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, &d)| if i == j { 0.0 } else { d * 0.5 + 3.0 })
                .collect()
        })
        .collect();
    let problem = BiTspProblem { dist_a, dist_b };
    let mut o = Nsga2::new(
        Nsga2Config {
            population_size: 20,
            generations: 3,
            seed: 0,
        },
        ShuffledPermutation { n: TSP_N },
        CompositeVariation {
            crossover: OrderCrossover,
            mutation: InversionMutation,
        },
    );
    black_box(o.run(black_box(&problem)).evaluations)
}

#[library_benchmark]
fn ant_colony_tsp_short() -> usize {
    let distances = tsp_distance_matrix();
    let problem = TspProblem {
        distances: distances.clone(),
    };
    let mut o = AntColonyTsp::new(
        AntColonyTspConfig {
            ants: 8,
            generations: 3,
            alpha: 1.0,
            beta: 2.0,
            evaporation: 0.5,
            deposit: 1.0,
            initial_pheromone: 1.0,
            seed: 0,
        },
        distances,
    );
    black_box(o.run(black_box(&problem)).evaluations)
}

const JSS_JOBS: usize = 6;
const JSS_MACHINES: usize = 6;

/// FT06 (Fisher & Thompson 1963) routing — machine id of the k-th operation
/// of job j.
const FT06_MACHINE: [[usize; JSS_MACHINES]; JSS_JOBS] = [
    [2, 0, 1, 3, 5, 4],
    [1, 2, 4, 5, 0, 3],
    [2, 3, 5, 0, 1, 4],
    [1, 0, 2, 3, 4, 5],
    [2, 1, 4, 5, 0, 3],
    [1, 3, 5, 0, 4, 2],
];

/// FT06 processing times — duration of the k-th operation of job j.
const FT06_TIME: [[f64; JSS_MACHINES]; JSS_JOBS] = [
    [1.0, 3.0, 6.0, 7.0, 3.0, 6.0],
    [8.0, 5.0, 10.0, 10.0, 10.0, 4.0],
    [5.0, 4.0, 8.0, 9.0, 1.0, 7.0],
    [5.0, 5.0, 5.0, 3.0, 8.0, 9.0],
    [9.0, 3.0, 5.0, 4.0, 3.0, 1.0],
    [3.0, 3.0, 9.0, 10.0, 4.0, 1.0],
];

/// Bi-objective FT06 job-shop scheduling: f1 = makespan, f2 = total flow time.
struct Ft06Problem;
impl Problem for Ft06Problem {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("makespan"),
            Objective::minimize("total_flow_time"),
        ])
    }
    fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
        let mut job_next = [0_usize; JSS_JOBS];
        let mut job_clock = [0.0_f64; JSS_JOBS];
        let mut machine_clock = [0.0_f64; JSS_MACHINES];
        for &job in schedule {
            let k = job_next[job];
            let m = FT06_MACHINE[job][k];
            let t = FT06_TIME[job][k];
            let start = job_clock[job].max(machine_clock[m]);
            let end = start + t;
            job_clock[job] = end;
            machine_clock[m] = end;
            job_next[job] = k + 1;
        }
        let makespan = machine_clock.iter().cloned().fold(0.0_f64, f64::max);
        let flow_time: f64 = job_clock.iter().sum();
        Evaluation::new(vec![makespan, flow_time])
    }
}

/// Precedence-Order Crossover — multiset-preserving crossover for the
/// operation-string JSS encoding. Trimmed from `examples/mo_jss_la01.rs`;
/// the strict-permutation crossovers cannot be used on multiset encodings.
#[derive(Debug, Clone, Copy, Default)]
struct PrecedenceOrderCrossover;
impl Variation<Vec<usize>> for PrecedenceOrderCrossover {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(parents.len() >= 2, "POX requires 2 parents");
        let (p1, p2) = (&parents[0], &parents[1]);
        let mut in_j1 = [false; JSS_JOBS];
        loop {
            for slot in &mut in_j1 {
                *slot = rng.random_bool(0.5);
            }
            let c = in_j1.iter().filter(|&&b| b).count();
            if c > 0 && c < JSS_JOBS {
                break;
            }
        }
        vec![pox_child(p1, p2, &in_j1), pox_child(p2, p1, &in_j1)]
    }
}
fn pox_child(donor: &[usize], filler: &[usize], in_donor_set: &[bool]) -> Vec<usize> {
    let n = donor.len();
    let mut child = vec![usize::MAX; n];
    for k in 0..n {
        if in_donor_set[donor[k]] {
            child[k] = donor[k];
        }
    }
    let mut fill_idx = 0;
    for &v in filler {
        if !in_donor_set[v] {
            while fill_idx < n && child[fill_idx] != usize::MAX {
                fill_idx += 1;
            }
            child[fill_idx] = v;
            fill_idx += 1;
        }
    }
    child
}

#[library_benchmark]
fn jss_nsga2_short() -> usize {
    let mut o = Nsga2::new(
        Nsga2Config {
            population_size: 20,
            generations: 3,
            seed: 0,
        },
        ShuffledMultisetPermutation::new(vec![JSS_MACHINES; JSS_JOBS]),
        CompositeVariation {
            crossover: PrecedenceOrderCrossover,
            mutation: InsertionMutation,
        },
    );
    black_box(o.run(black_box(&Ft06Problem)).evaluations)
}

const KNAPSACK_N: usize = 20;

const KP_PROFIT_A: [f64; KNAPSACK_N] = [
    61.0, 17.0, 92.0, 49.0, 73.0, 28.0, 84.0, 36.0, 55.0, 78.0, 23.0, 91.0, 12.0, 67.0, 45.0, 58.0,
    33.0, 71.0, 14.0, 26.0,
];
const KP_PROFIT_B: [f64; KNAPSACK_N] = [
    24.0, 81.0, 16.0, 67.0, 29.0, 73.0, 41.0, 60.0, 52.0, 19.0, 77.0, 34.0, 95.0, 22.0, 71.0, 88.0,
    56.0, 27.0, 64.0, 90.0,
];
const KP_WEIGHT: [f64; KNAPSACK_N] = [
    35.0, 58.0, 22.0, 71.0, 14.0, 86.0, 31.0, 53.0, 78.0, 19.0, 44.0, 16.0, 67.0, 88.0, 25.0, 51.0,
    33.0, 74.0, 12.0, 47.0,
];

/// Bi-objective 0/1 knapsack with a penalty-based capacity constraint.
struct KnapsackProblem {
    capacity: f64,
}
impl Problem for KnapsackProblem {
    type Decision = Vec<bool>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::maximize("profit_a"),
            Objective::maximize("profit_b"),
        ])
    }
    fn evaluate(&self, take: &Vec<bool>) -> Evaluation {
        let (mut pa, mut pb, mut w) = (0.0, 0.0, 0.0);
        for (i, &t) in take.iter().enumerate() {
            if t {
                pa += KP_PROFIT_A[i];
                pb += KP_PROFIT_B[i];
                w += KP_WEIGHT[i];
            }
        }
        let penalty = 1000.0 * (w - self.capacity).max(0.0);
        Evaluation::new(vec![pa - penalty, pb - penalty])
    }
}

/// Random binary initializer — each bit 50/50 independently.
#[derive(Debug, Clone, Copy)]
struct RandomBinary {
    n: usize,
}
impl Initializer<Vec<bool>> for RandomBinary {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<Vec<bool>> {
        (0..size)
            .map(|_| (0..self.n).map(|_| rng.random_bool(0.5)).collect())
            .collect()
    }
}

/// One-point crossover for binary chromosomes. Trimmed from
/// `examples/mo_knapsack.rs`.
#[derive(Debug, Clone, Copy, Default)]
struct OnePointCrossoverBool;
impl Variation<Vec<bool>> for OnePointCrossoverBool {
    fn vary(&mut self, parents: &[Vec<bool>], rng: &mut Rng) -> Vec<Vec<bool>> {
        assert!(
            parents.len() >= 2,
            "OnePointCrossoverBool requires 2 parents"
        );
        let (p1, p2) = (&parents[0], &parents[1]);
        let n = p1.len();
        if n < 2 {
            return vec![p1.clone(), p2.clone()];
        }
        let cut = rng.random_range(1..n);
        let mut c1 = Vec::with_capacity(n);
        let mut c2 = Vec::with_capacity(n);
        c1.extend_from_slice(&p1[..cut]);
        c1.extend_from_slice(&p2[cut..]);
        c2.extend_from_slice(&p2[..cut]);
        c2.extend_from_slice(&p1[cut..]);
        vec![c1, c2]
    }
}

#[library_benchmark]
fn knapsack_nsga2_short() -> usize {
    let capacity = 0.5 * KP_WEIGHT.iter().sum::<f64>();
    let problem = KnapsackProblem { capacity };
    let mut o = Nsga2::new(
        Nsga2Config {
            population_size: 20,
            generations: 3,
            seed: 0,
        },
        RandomBinary { n: KNAPSACK_N },
        CompositeVariation {
            crossover: OnePointCrossoverBool,
            mutation: BitFlipMutation {
                probability: 1.0 / KNAPSACK_N as f64,
            },
        },
    );
    black_box(o.run(black_box(&problem)).evaluations)
}

library_benchmark_group!(
    name = combinatorial_group;
    benchmarks =
        tsp_nsga2_short, ant_colony_tsp_short, jss_nsga2_short, knapsack_nsga2_short
);

// -----------------------------------------------------------------------------
// Multi-fidelity (Hyperband)
// -----------------------------------------------------------------------------

/// Multi-fidelity 2-D sphere: higher budget shrinks an additive residual, so
/// the loss is budget-monotone the way Hyperband expects. Deterministic.
struct MultiFidelitySphere;
impl PartialProblem for MultiFidelitySphere {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("loss")])
    }
    fn evaluate_at_budget(&self, x: &Vec<f64>, budget: f64) -> Evaluation {
        let true_f: f64 = x.iter().map(|v| v * v).sum();
        let residual = 1.0 / (budget + 1.0);
        Evaluation::new(vec![true_f + residual])
    }
}

#[library_benchmark]
fn hyperband_short() -> usize {
    let mut o = Hyperband::new(
        HyperbandConfig {
            max_budget: 27.0,
            eta: 3.0,
            max_brackets: 3,
            seed: 0,
        },
        RealBounds::new(vec![(-5.0, 5.0); 2]),
    );
    black_box(o.run(black_box(&MultiFidelitySphere)).evaluations)
}

library_benchmark_group!(
    name = multi_fidelity_group;
    benchmarks = hyperband_short
);

main!(
    library_benchmark_groups = pareto_group,
    algorithm_group,
    single_objective_group,
    multi_objective_group,
    permutation_ops_group,
    variation_ops_group,
    combinatorial_group,
    multi_fidelity_group
);
