//! Per-algorithm property tests.
//!
//! For every `Optimizer` impl in heuropt we check the same three properties:
//!   1. **Deterministic-with-seed**: two runs with the same seed produce
//!      the same `best.evaluation.objectives`.
//!   2. **No panic on random valid inputs**: random seeds, random tiny
//!      problems, random bounds — the algorithm runs to completion.
//!   3. **Population-size invariant** (where the algorithm documents one):
//!      the final population has the configured size.

use proptest::prelude::*;

use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::core::problem::Problem;
use heuropt::prelude::*;

// -----------------------------------------------------------------------------
// Tiny problems
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

struct OneMax {
    #[allow(dead_code)]
    bits: usize,
}
impl Problem for OneMax {
    type Decision = Vec<bool>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::maximize("count")])
    }
    fn evaluate(&self, x: &Vec<bool>) -> Evaluation {
        Evaluation::new(vec![x.iter().filter(|b| **b).count() as f64])
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn so_bounds() -> RealBounds {
    RealBounds::new(vec![(-3.0, 3.0)])
}
fn so_bounds_2d() -> RealBounds {
    RealBounds::new(vec![(-3.0, 3.0); 2])
}
fn mo_bounds() -> Vec<(f64, f64)> {
    vec![(-3.0, 3.0)]
}

fn mo_variation() -> CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation> {
    let bounds = mo_bounds();
    CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
    }
}

// -----------------------------------------------------------------------------
// Single-objective continuous
// -----------------------------------------------------------------------------

proptest! {
    #[test]
    fn random_search_deterministic(seed in any::<u64>()) {
        let make = || RandomSearch::new(
            RandomSearchConfig { iterations: 20, batch_size: 1, seed },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn hill_climber_deterministic(seed in any::<u64>()) {
        let make = || HillClimber::new(
            HillClimberConfig { iterations: 20, seed },
            so_bounds(),
            GaussianMutation { sigma: 0.1 },
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn one_plus_one_es_deterministic(seed in any::<u64>()) {
        let make = || OnePlusOneEs::new(
            OnePlusOneEsConfig {
                iterations: 50,
                initial_sigma: 0.5,
                adaptation_period: 10,
                step_increase: 1.22,
                seed,
            },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn simulated_annealing_deterministic(seed in any::<u64>()) {
        let make = || SimulatedAnnealing::new(
            SimulatedAnnealingConfig {
                iterations: 50,
                initial_temperature: 1.0,
                final_temperature: 1e-3,
                seed,
            },
            so_bounds(),
            GaussianMutation { sigma: 0.1 },
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn ga_deterministic(seed in any::<u64>()) {
        let bounds = mo_bounds();
        let make = || GeneticAlgorithm::new(
            GeneticAlgorithmConfig {
                population_size: 10,
                generations: 5,
                tournament_size: 2,
                elitism: 1,
                seed,
            },
            RealBounds::new(bounds.clone()),
            CompositeVariation {
                crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
                mutation: PolynomialMutation::new(bounds.clone(), 20.0, 1.0),
            },
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn pso_deterministic(seed in any::<u64>()) {
        let make = || ParticleSwarm::new(
            ParticleSwarmConfig {
                swarm_size: 10,
                generations: 5,
                inertia: 0.7,
                cognitive: 1.5,
                social: 1.5,
                seed,
            },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn de_deterministic(seed in any::<u64>()) {
        let make = || DifferentialEvolution::new(
            DifferentialEvolutionConfig {
                population_size: 10,
                generations: 5,
                differential_weight: 0.5,
                crossover_probability: 0.9,
                seed,
            },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn cmaes_deterministic(seed in any::<u64>()) {
        let cfg = CmaEsConfig {
            population_size: 8,
            generations: 5,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            initial_mean: None,
            seed,
        };
        let mut a = CmaEs::new(cfg.clone(), so_bounds());
        let mut b = CmaEs::new(cfg, so_bounds());
        let r1 = a.run(&Sphere1D);
        let r2 = b.run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn ipop_cmaes_deterministic(seed in any::<u64>()) {
        let cfg = IpopCmaEsConfig {
            initial_population_size: 8,
            total_generations: 30,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            stall_generations: None,
            seed,
        };
        let mut a = IpopCmaEs::new(cfg.clone(), so_bounds());
        let mut b = IpopCmaEs::new(cfg, so_bounds());
        let r1 = a.run(&Sphere1D);
        let r2 = b.run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn snes_deterministic(seed in any::<u64>()) {
        let make = || SeparableNes::new(
            SeparableNesConfig {
                population_size: 8,
                generations: 5,
                initial_sigma: 0.5,
                mean_learning_rate: 1.0,
                sigma_learning_rate: None,
                seed,
            },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn tlbo_deterministic(seed in any::<u64>()) {
        let make = || Tlbo::new(
            TlboConfig { population_size: 10, generations: 5, seed },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn nelder_mead_deterministic(_dummy in any::<bool>()) {
        // Nelder-Mead is purely deterministic; no seed.
        let make = || NelderMead::new(
            NelderMeadConfig { iterations: 50, ..NelderMeadConfig::default() },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn bayesian_opt_deterministic(seed in any::<u64>()) {
        let make = || BayesianOpt::new(
            BayesianOptConfig {
                initial_samples: 5,
                iterations: 10,
                length_scales: None,
                signal_variance: 1.0,
                noise_variance: 1e-6,
                acquisition_samples: 100,
                seed,
            },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn tpe_deterministic(seed in any::<u64>()) {
        let make = || Tpe::new(
            TpeConfig {
                initial_samples: 5,
                iterations: 10,
                good_fraction: 0.25,
                candidate_samples: 12,
                bandwidth_factor: 1.0,
                seed,
            },
            so_bounds(),
        );
        let r1 = make().run(&Sphere1D);
        let r2 = make().run(&Sphere1D);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }
}

// -----------------------------------------------------------------------------
// Multi-objective
// -----------------------------------------------------------------------------

proptest! {
    #[test]
    fn nsga2_deterministic_and_pop_size(seed in any::<u64>()) {
        let make = || Nsga2::new(
            Nsga2Config { population_size: 10, generations: 3, seed },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
        prop_assert_eq!(r1.population.len(), 10);
    }

    #[test]
    fn nsga3_deterministic_and_pop_size(seed in any::<u64>()) {
        let make = || Nsga3::new(
            Nsga3Config {
                population_size: 12,
                generations: 3,
                reference_divisions: 11,
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
        prop_assert_eq!(r1.population.len(), 12);
    }

    #[test]
    fn spea2_deterministic(seed in any::<u64>()) {
        let make = || Spea2::new(
            Spea2Config {
                population_size: 10,
                archive_size: 10,
                generations: 3,
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn moead_deterministic(seed in any::<u64>()) {
        let make = || Moead::new(
            MoeadConfig {
                generations: 3,
                reference_divisions: 9,
                neighborhood_size: 4,
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.population.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.population.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn mopso_deterministic(seed in any::<u64>()) {
        let make = || Mopso::new(
            MopsoConfig {
                swarm_size: 10,
                generations: 3,
                archive_size: 10,
                inertia: 0.7,
                cognitive: 1.5,
                social: 1.5,
                seed,
            },
            RealBounds::new(mo_bounds()),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn ibea_deterministic(seed in any::<u64>()) {
        let make = || Ibea::new(
            IbeaConfig {
                population_size: 10,
                generations: 3,
                kappa: 0.05,
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn sms_emoa_deterministic(seed in any::<u64>()) {
        let make = || SmsEmoa::new(
            SmsEmoaConfig {
                population_size: 8,
                generations: 5,
                reference_point: vec![10.0, 10.0],
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn hype_deterministic(seed in any::<u64>()) {
        let make = || Hype::new(
            HypeConfig {
                population_size: 10,
                generations: 3,
                reference_point: vec![10.0, 10.0],
                mc_samples: 100,
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn pesa2_deterministic(seed in any::<u64>()) {
        let make = || PesaII::new(
            PesaIIConfig {
                population_size: 10,
                archive_size: 10,
                generations: 3,
                grid_divisions: 4,
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn epsilon_moea_deterministic(seed in any::<u64>()) {
        let make = || EpsilonMoea::new(
            EpsilonMoeaConfig {
                population_size: 10,
                evaluations: 30,
                epsilon: vec![0.05, 0.05],
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn age_moea_deterministic(seed in any::<u64>()) {
        let make = || AgeMoea::new(
            AgeMoeaConfig { population_size: 10, generations: 3, seed },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn grea_deterministic(seed in any::<u64>()) {
        let make = || Grea::new(
            GreaConfig {
                population_size: 10,
                generations: 3,
                grid_divisions: 4,
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn knea_deterministic(seed in any::<u64>()) {
        let make = || Knea::new(
            KneaConfig { population_size: 10, generations: 3, seed },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn rvea_deterministic(seed in any::<u64>()) {
        let make = || Rvea::new(
            RveaConfig {
                population_size: 10,
                generations: 3,
                reference_divisions: 9,
                alpha: 2.0,
                seed,
            },
            RealBounds::new(mo_bounds()),
            mo_variation(),
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }

    #[test]
    fn paes_deterministic(seed in any::<u64>()) {
        let make = || Paes::new(
            PaesConfig { iterations: 30, archive_size: 10, seed },
            RealBounds::new(mo_bounds()),
            GaussianMutation { sigma: 0.1 },
        );
        let r1 = make().run(&SchafferN1);
        let r2 = make().run(&SchafferN1);
        let oa: Vec<Vec<f64>> = r1.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> = r2.pareto_front.iter()
            .map(|c| c.evaluation.objectives.clone()).collect();
        prop_assert_eq!(oa, ob);
    }
}

// -----------------------------------------------------------------------------
// Other decision types
// -----------------------------------------------------------------------------

proptest! {
    #[test]
    fn umda_deterministic(seed in any::<u64>(), bits in 4usize..16) {
        let problem = OneMax { bits };
        let make = || Umda::new(UmdaConfig {
            population_size: 10,
            selected_size: 5,
            generations: 3,
            bits,
            seed,
        });
        let r1 = make().run(&problem);
        let r2 = make().run(&problem);
        prop_assert_eq!(
            r1.best.unwrap().evaluation.objectives,
            r2.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    fn random_search_evaluation_count_invariant(
        iterations in 1usize..30,
        batch_size in 1usize..5,
        seed in any::<u64>(),
    ) {
        let mut opt = RandomSearch::new(
            RandomSearchConfig { iterations, batch_size, seed },
            so_bounds(),
        );
        let r = opt.run(&Sphere1D);
        prop_assert_eq!(r.evaluations, iterations * batch_size);
        prop_assert_eq!(r.population.len(), iterations * batch_size);
        prop_assert_eq!(r.generations, iterations);
    }
}

// -----------------------------------------------------------------------------
// Cross-cutting: best is at least as good as any front member
// -----------------------------------------------------------------------------

proptest! {
    #[test]
    fn so_optimizer_best_beats_initial(seed in any::<u64>()) {
        // After running an SO optimizer, the result's best.evaluation
        // should be at least as good as the worst point sampled — i.e.
        // the optimizer doesn't return None or some random non-best.
        let mut opt = DifferentialEvolution::new(
            DifferentialEvolutionConfig {
                population_size: 10,
                generations: 5,
                differential_weight: 0.5,
                crossover_probability: 0.9,
                seed,
            },
            so_bounds_2d(),
        );
        let r = opt.run(&Sphere1D);
        let best_f = r.best.unwrap().evaluation.objectives[0];
        let pop_min = r.population.iter()
            .map(|c| c.evaluation.objectives[0])
            .fold(f64::INFINITY, f64::min);
        prop_assert!(
            best_f <= pop_min + 1e-12,
            "best f = {best_f}, pop min = {pop_min}",
        );
    }
}

// -----------------------------------------------------------------------------
// AlgorithmInfo sweep — exact name / full_name / seed per algorithm
// -----------------------------------------------------------------------------
//
// Why this exists: every algorithm has three trivial trait methods returning
// `&'static str` and `Option<u64>`. A `cargo mutants` run discovers that
// these are unconstrained — replacing `"NSGA-II"` with `""` or `"xyzzy"`
// survives because no test reads the string. The constants below pin every
// algorithm's identifying strings exactly. Updating an algorithm's name
// requires updating its test, by design.

#[test]
fn age_moea_algorithm_info_is_correct() {
    let opt = AgeMoea::new(
        AgeMoeaConfig { population_size: 4, generations: 1, seed: 42 },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "AGE-MOEA");
    assert_eq!(
        opt.full_name(),
        "Adaptive Geometry Estimation Multi-Objective Evolutionary Algorithm",
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn ant_colony_tsp_algorithm_info_is_correct() {
    let opt = AntColonyTsp::new(
        AntColonyTspConfig {
            ants: 2,
            generations: 1,
            alpha: 1.0,
            beta: 2.0,
            evaporation: 0.5,
            deposit: 1.0,
            initial_pheromone: 1.0,
            seed: 42,
        },
        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
    );
    assert_eq!(opt.name(), "Ant Colony");
    assert_eq!(opt.full_name(), "Ant Colony System for TSP");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn bayesian_opt_algorithm_info_is_correct() {
    let opt = BayesianOpt::new(
        BayesianOptConfig {
            initial_samples: 2,
            iterations: 1,
            length_scales: None,
            signal_variance: 1.0,
            noise_variance: 1e-3,
            acquisition_samples: 4,
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "Bayesian Optimization");
    assert_eq!(
        opt.full_name(),
        "Gaussian Process Bayesian Optimization with Expected Improvement",
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn cma_es_algorithm_info_is_correct() {
    let opt = CmaEs::new(
        CmaEsConfig {
            population_size: 4,
            generations: 1,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            initial_mean: None,
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "CMA-ES");
    assert_eq!(opt.full_name(), "Covariance Matrix Adaptation Evolution Strategy");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn differential_evolution_algorithm_info_is_correct() {
    let opt = DifferentialEvolution::new(
        DifferentialEvolutionConfig {
            population_size: 4,
            generations: 1,
            differential_weight: 0.5,
            crossover_probability: 0.9,
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "DE");
    assert_eq!(opt.full_name(), "Differential Evolution");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn epsilon_moea_algorithm_info_is_correct() {
    let opt = EpsilonMoea::new(
        EpsilonMoeaConfig {
            population_size: 4,
            evaluations: 4,
            epsilon: vec![0.1, 0.1],
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "ε-MOEA");
    assert_eq!(
        opt.full_name(),
        "ε-dominance Multi-Objective Evolutionary Algorithm",
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn genetic_algorithm_algorithm_info_is_correct() {
    let bounds = mo_bounds();
    let opt = GeneticAlgorithm::new(
        GeneticAlgorithmConfig {
            population_size: 4,
            generations: 1,
            tournament_size: 2,
            elitism: 1,
            seed: 42,
        },
        RealBounds::new(bounds.clone()),
        CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        },
    );
    assert_eq!(opt.name(), "GA");
    assert_eq!(opt.full_name(), "Genetic Algorithm");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn grea_algorithm_info_is_correct() {
    let opt = Grea::new(
        GreaConfig {
            population_size: 4,
            generations: 1,
            grid_divisions: 4,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "GrEA");
    assert_eq!(opt.full_name(), "Grid-based Evolutionary Algorithm");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn hill_climber_algorithm_info_is_correct() {
    let opt = HillClimber::new(
        HillClimberConfig { iterations: 1, seed: 42 },
        so_bounds(),
        GaussianMutation { sigma: 0.1 },
    );
    assert_eq!(opt.name(), "Hill Climber");
    assert_eq!(opt.full_name(), "Hill Climbing");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn hyperband_algorithm_info_is_correct() {
    let opt: Hyperband<RealBounds, Vec<f64>> = Hyperband::new(
        HyperbandConfig {
            max_budget: 8.0,
            eta: 2.0,
            max_brackets: 2,
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "Hyperband");
    assert_eq!(opt.full_name(), "Hyperband multi-fidelity bandit search");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn hype_algorithm_info_is_correct() {
    let opt = Hype::new(
        HypeConfig {
            population_size: 4,
            generations: 1,
            reference_point: vec![10.0, 10.0],
            mc_samples: 4,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "HypE");
    assert_eq!(opt.full_name(), "Hypervolume Estimation Algorithm");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn ibea_algorithm_info_is_correct() {
    let opt = Ibea::new(
        IbeaConfig {
            population_size: 4,
            generations: 1,
            kappa: 0.05,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "IBEA");
    assert_eq!(opt.full_name(), "Indicator-Based Evolutionary Algorithm");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn ipop_cma_es_algorithm_info_is_correct() {
    let opt = IpopCmaEs::new(
        IpopCmaEsConfig {
            initial_population_size: 4,
            total_generations: 1,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            stall_generations: None,
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "IPOP-CMA-ES");
    assert_eq!(opt.full_name(), "Increasing-Population CMA-ES with Restarts");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn knea_algorithm_info_is_correct() {
    let opt = Knea::new(
        KneaConfig { population_size: 4, generations: 1, seed: 42 },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "KnEA");
    assert_eq!(opt.full_name(), "Knee point-driven Evolutionary Algorithm");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn moead_algorithm_info_is_correct() {
    let opt = Moead::new(
        MoeadConfig {
            generations: 1,
            reference_divisions: 3,
            neighborhood_size: 2,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "MOEA/D");
    assert_eq!(
        opt.full_name(),
        "Multi-Objective Evolutionary Algorithm based on Decomposition",
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn mopso_algorithm_info_is_correct() {
    let opt = Mopso::new(
        MopsoConfig {
            swarm_size: 4,
            generations: 1,
            archive_size: 4,
            inertia: 0.5,
            cognitive: 1.0,
            social: 1.0,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
    );
    assert_eq!(opt.name(), "MOPSO");
    assert_eq!(opt.full_name(), "Multi-Objective Particle Swarm Optimization");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn nelder_mead_algorithm_info_is_correct() {
    let opt = NelderMead::new(
        NelderMeadConfig { iterations: 1, ..NelderMeadConfig::default() },
        so_bounds(),
    );
    assert_eq!(opt.name(), "Nelder-Mead");
    assert_eq!(opt.full_name(), "Nelder-Mead simplex direct search");
    // NelderMead is deterministic — no seed. Matches the default AlgorithmInfo
    // impl which returns None.
    assert_eq!(opt.seed(), None);
}

#[test]
fn nsga2_algorithm_info_is_correct() {
    let opt = Nsga2::new(
        Nsga2Config { population_size: 4, generations: 1, seed: 42 },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "NSGA-II");
    assert_eq!(opt.full_name(), "Non-dominated Sorting Genetic Algorithm II");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn nsga3_algorithm_info_is_correct() {
    let opt = Nsga3::new(
        Nsga3Config {
            population_size: 4,
            generations: 1,
            reference_divisions: 4,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "NSGA-III");
    assert_eq!(opt.full_name(), "Non-dominated Sorting Genetic Algorithm III");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn one_plus_one_es_algorithm_info_is_correct() {
    let opt = OnePlusOneEs::new(
        OnePlusOneEsConfig {
            iterations: 1,
            initial_sigma: 0.5,
            adaptation_period: 4,
            step_increase: 1.5,
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "(1+1)-ES");
    assert_eq!(
        opt.full_name(),
        "(1+1) Evolution Strategy with one-fifth success rule",
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn paes_algorithm_info_is_correct() {
    let opt = Paes::new(
        PaesConfig { iterations: 1, archive_size: 4, seed: 42 },
        RealBounds::new(mo_bounds()),
        GaussianMutation { sigma: 0.1 },
    );
    assert_eq!(opt.name(), "PAES");
    assert_eq!(opt.full_name(), "Pareto Archived Evolution Strategy");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn particle_swarm_algorithm_info_is_correct() {
    let opt = ParticleSwarm::new(
        ParticleSwarmConfig {
            swarm_size: 4,
            generations: 1,
            inertia: 0.5,
            cognitive: 1.0,
            social: 1.0,
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "PSO");
    assert_eq!(opt.full_name(), "Particle Swarm Optimization");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn pesa_ii_algorithm_info_is_correct() {
    let opt = PesaII::new(
        PesaIIConfig {
            population_size: 4,
            archive_size: 4,
            generations: 1,
            grid_divisions: 4,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "PESA-II");
    assert_eq!(opt.full_name(), "Pareto Envelope-based Selection Algorithm II");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn random_search_algorithm_info_is_correct() {
    let opt = RandomSearch::new(
        RandomSearchConfig { iterations: 1, batch_size: 1, seed: 42 },
        so_bounds(),
    );
    assert_eq!(opt.name(), "Random Search");
    // No `full_name` override — defaults to `name`.
    assert_eq!(opt.full_name(), "Random Search");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn rvea_algorithm_info_is_correct() {
    let opt = Rvea::new(
        RveaConfig {
            population_size: 4,
            generations: 1,
            reference_divisions: 4,
            alpha: 2.0,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "RVEA");
    assert_eq!(opt.full_name(), "Reference Vector-guided Evolutionary Algorithm");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn simulated_annealing_algorithm_info_is_correct() {
    let opt = SimulatedAnnealing::new(
        SimulatedAnnealingConfig {
            iterations: 1,
            initial_temperature: 1.0,
            final_temperature: 0.1,
            seed: 42,
        },
        so_bounds(),
        GaussianMutation { sigma: 0.1 },
    );
    assert_eq!(opt.name(), "Simulated Annealing");
    // No `full_name` override — defaults to `name`.
    assert_eq!(opt.full_name(), "Simulated Annealing");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn sms_emoa_algorithm_info_is_correct() {
    let opt = SmsEmoa::new(
        SmsEmoaConfig {
            population_size: 4,
            generations: 1,
            reference_point: vec![100.0, 100.0],
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "SMS-EMOA");
    assert_eq!(
        opt.full_name(),
        "S-Metric Selection Evolutionary Multi-Objective Algorithm",
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn separable_nes_algorithm_info_is_correct() {
    let opt = SeparableNes::new(
        SeparableNesConfig {
            population_size: 4,
            generations: 1,
            initial_sigma: 0.5,
            mean_learning_rate: 1.0,
            sigma_learning_rate: Some(0.1),
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "sNES");
    assert_eq!(opt.full_name(), "Separable Natural Evolution Strategy");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn spea2_algorithm_info_is_correct() {
    let opt = Spea2::new(
        Spea2Config {
            population_size: 4,
            archive_size: 4,
            generations: 1,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "SPEA2");
    assert_eq!(opt.full_name(), "Strength Pareto Evolutionary Algorithm 2");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn tabu_search_algorithm_info_is_correct() {
    struct StartAtZero;
    impl Initializer<Vec<i32>> for StartAtZero {
        fn initialize(
            &mut self,
            _size: usize,
            _rng: &mut heuropt::core::rng::Rng,
        ) -> Vec<Vec<i32>> {
            vec![vec![0]]
        }
    }
    let neighbors = |x: &Vec<i32>, _rng: &mut heuropt::core::rng::Rng| {
        vec![vec![x[0] - 1], vec![x[0] + 1]]
    };
    let opt = TabuSearch::new(
        TabuSearchConfig { iterations: 1, tabu_tenure: 4, seed: 42 },
        StartAtZero,
        neighbors,
    );
    assert_eq!(opt.name(), "Tabu Search");
    // No `full_name` override — defaults to `name`.
    assert_eq!(opt.full_name(), "Tabu Search");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn tlbo_algorithm_info_is_correct() {
    let opt = Tlbo::new(
        TlboConfig { population_size: 4, generations: 1, seed: 42 },
        so_bounds(),
    );
    assert_eq!(opt.name(), "TLBO");
    assert_eq!(opt.full_name(), "Teaching-Learning-Based Optimization");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn tpe_algorithm_info_is_correct() {
    let opt = Tpe::new(
        TpeConfig {
            initial_samples: 2,
            iterations: 1,
            good_fraction: 0.25,
            candidate_samples: 4,
            bandwidth_factor: 0.1,
            seed: 42,
        },
        so_bounds(),
    );
    assert_eq!(opt.name(), "TPE");
    assert_eq!(opt.full_name(), "Tree-structured Parzen Estimator");
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn umda_algorithm_info_is_correct() {
    let opt = Umda::new(UmdaConfig {
        bits: 4,
        population_size: 4,
        selected_size: 2,
        generations: 1,
        seed: 42,
    });
    assert_eq!(opt.name(), "UMDA");
    assert_eq!(opt.full_name(), "Univariate Marginal Distribution Algorithm");
    assert_eq!(opt.seed(), Some(42));
}
