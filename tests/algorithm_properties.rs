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
        AgeMoeaConfig {
            population_size: 4,
            generations: 1,
            seed: 42,
        },
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
    assert_eq!(
        opt.full_name(),
        "Covariance Matrix Adaptation Evolution Strategy"
    );
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
        HillClimberConfig {
            iterations: 1,
            seed: 42,
        },
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
    assert_eq!(
        opt.full_name(),
        "Increasing-Population CMA-ES with Restarts"
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn knea_algorithm_info_is_correct() {
    let opt = Knea::new(
        KneaConfig {
            population_size: 4,
            generations: 1,
            seed: 42,
        },
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
    assert_eq!(
        opt.full_name(),
        "Multi-Objective Particle Swarm Optimization"
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn nelder_mead_algorithm_info_is_correct() {
    let opt = NelderMead::new(
        NelderMeadConfig {
            iterations: 1,
            ..NelderMeadConfig::default()
        },
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
        Nsga2Config {
            population_size: 4,
            generations: 1,
            seed: 42,
        },
        RealBounds::new(mo_bounds()),
        mo_variation(),
    );
    assert_eq!(opt.name(), "NSGA-II");
    assert_eq!(
        opt.full_name(),
        "Non-dominated Sorting Genetic Algorithm II"
    );
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
    assert_eq!(
        opt.full_name(),
        "Non-dominated Sorting Genetic Algorithm III"
    );
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
        PaesConfig {
            iterations: 1,
            archive_size: 4,
            seed: 42,
        },
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
    assert_eq!(
        opt.full_name(),
        "Pareto Envelope-based Selection Algorithm II"
    );
    assert_eq!(opt.seed(), Some(42));
}

#[test]
fn random_search_algorithm_info_is_correct() {
    let opt = RandomSearch::new(
        RandomSearchConfig {
            iterations: 1,
            batch_size: 1,
            seed: 42,
        },
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
    assert_eq!(
        opt.full_name(),
        "Reference Vector-guided Evolutionary Algorithm"
    );
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
    let neighbors =
        |x: &Vec<i32>, _rng: &mut heuropt::core::rng::Rng| vec![vec![x[0] - 1], vec![x[0] + 1]];
    let opt = TabuSearch::new(
        TabuSearchConfig {
            iterations: 1,
            tabu_tenure: 4,
            seed: 42,
        },
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
        TlboConfig {
            population_size: 4,
            generations: 1,
            seed: 42,
        },
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
    assert_eq!(
        opt.full_name(),
        "Univariate Marginal Distribution Algorithm"
    );
    assert_eq!(opt.seed(), Some(42));
}

// -----------------------------------------------------------------------------
// run_async ↔ run parity sweep
// -----------------------------------------------------------------------------
//
// Every algorithm exposes both `run` and `run_async` (the latter behind the
// `async` feature). Before this sweep, nothing exercised `run_async`, so a
// `cargo mutants` run survived essentially every mutation to its body —
// "replace run_async with OptimizationResult::new()", every comparison flip,
// every += → -= inside the async loop. This sweep asserts that with the
// same Config + seed + problem, the async runner produces *identical*
// best.evaluation.objectives as the sync runner. The two implementations
// share the algorithmic logic; only the evaluation dispatch differs.
//
// Hyperband uses AsyncPartialProblem (multi-fidelity) instead of
// AsyncProblem, so it gets its own test fixture below.

#[cfg(feature = "async")]
mod async_parity {
    use super::*;
    use heuropt::core::async_problem::{AsyncPartialProblem, AsyncProblem};
    use heuropt::core::partial_problem::PartialProblem;

    // ---- Async-capable test fixtures ----------------------------------------
    //
    // These are multi-dimensional on purpose: the snapshot assertions below
    // pin each algorithm's exact run() output, and a 1-D problem leaves the
    // per-axis / covariance-matrix / simplex machinery degenerate, so
    // arithmetic mutations there wouldn't change the result. 3-D problems
    // exercise the full loop body.

    /// 3-D Rosenbrock: f(x) = Σ [100·(x_{i+1} − x_i²)² + (1 − x_i)²].
    /// Single objective. Deliberately *hard*: a curved, non-convex valley
    /// that none of these algorithms fully solve in a modest budget. That
    /// matters for the snapshot assertions — on a convex sphere the
    /// optimizers converge to the exact optimum regardless of small
    /// arithmetic perturbations, so a mutated run() still lands on 0.0 and
    /// the snapshot can't tell the difference. On Rosenbrock the result
    /// always reflects the exact trajectory.
    struct SnapSphere;
    impl Problem for SnapSphere {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }
        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            let mut f = 0.0;
            for i in 0..x.len() - 1 {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                f += 100.0 * a * a + b * b;
            }
            Evaluation::new(vec![f])
        }
    }
    impl AsyncProblem for SnapSphere {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            <Self as Problem>::objectives(self)
        }
        async fn evaluate_async(&self, x: &Vec<f64>) -> Evaluation {
            <Self as Problem>::evaluate(self, x)
        }
    }

    /// 3-variable, 2-objective problem: f1 = Σ xᵢ², f2 = Σ (xᵢ − 2)².
    /// A genuine multi-variable Pareto front, unlike the 1-variable
    /// SchafferN1 used elsewhere in this file.
    struct SnapMo;
    impl Problem for SnapMo {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
        }
        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            let f1: f64 = x.iter().map(|v| v * v).sum();
            let f2: f64 = x.iter().map(|v| (v - 2.0) * (v - 2.0)).sum();
            Evaluation::new(vec![f1, f2])
        }
    }
    impl AsyncProblem for SnapMo {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            <Self as Problem>::objectives(self)
        }
        async fn evaluate_async(&self, x: &Vec<f64>) -> Evaluation {
            <Self as Problem>::evaluate(self, x)
        }
    }

    impl AsyncProblem for OneMax {
        type Decision = Vec<bool>;
        fn objectives(&self) -> ObjectiveSpace {
            <Self as Problem>::objectives(self)
        }
        async fn evaluate_async(&self, x: &Vec<bool>) -> Evaluation {
            <Self as Problem>::evaluate(self, x)
        }
    }

    fn snap_so_bounds() -> RealBounds {
        RealBounds::new(vec![(-3.0, 3.0); 3])
    }
    fn snap_mo_bounds() -> Vec<(f64, f64)> {
        vec![(-3.0, 3.0); 3]
    }
    fn snap_mo_variation() -> CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation> {
        let bounds = snap_mo_bounds();
        CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        }
    }

    /// 8-city scattered 2-D TSP fixture for AntColonyTsp parity. The cities
    /// are deliberately *not* on a line: a nearest-neighbour / distance-
    /// heuristic-only tour is **not** optimal here, so the pheromone-update
    /// arithmetic in `run()` genuinely determines which tour is found.
    /// (A collinear instance is solved by the heuristic alone, which makes
    /// the pheromone math inert and its mutations undetectable.)
    struct TinyTsp {
        dist: Vec<Vec<f64>>,
    }
    impl TinyTsp {
        fn new() -> Self {
            // 8 scattered cities — irregular 2-D layout with several
            // near-equal competing edges so pheromone reinforcement is
            // load-bearing for the recovered tour.
            let pts = [
                (0.0_f64, 0.0_f64),
                (4.0, 1.0),
                (1.0, 3.0),
                (5.0, 4.0),
                (2.0, 5.0),
                (6.0, 2.0),
                (3.0, 6.0),
                (0.5, 4.5),
            ];
            let n = pts.len();
            let mut dist = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let dx = pts[i].0 - pts[j].0;
                    let dy = pts[i].1 - pts[j].1;
                    dist[i][j] = (dx * dx + dy * dy).sqrt();
                }
            }
            Self { dist }
        }
        fn length(&self, tour: &[usize]) -> f64 {
            let n = tour.len();
            let mut total = 0.0;
            for i in 0..n {
                total += self.dist[tour[i]][tour[(i + 1) % n]];
            }
            total
        }
    }
    impl Problem for TinyTsp {
        type Decision = Vec<usize>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("length")])
        }
        fn evaluate(&self, t: &Vec<usize>) -> Evaluation {
            Evaluation::new(vec![self.length(t)])
        }
    }
    impl AsyncProblem for TinyTsp {
        type Decision = Vec<usize>;
        fn objectives(&self) -> ObjectiveSpace {
            <Self as Problem>::objectives(self)
        }
        async fn evaluate_async(&self, t: &Vec<usize>) -> Evaluation {
            <Self as Problem>::evaluate(self, t)
        }
    }

    /// Trivial integer problem for TabuSearch — minimize |x|.
    struct AbsInt;
    impl Problem for AbsInt {
        type Decision = Vec<i32>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("absx")])
        }
        fn evaluate(&self, x: &Vec<i32>) -> Evaluation {
            Evaluation::new(vec![x[0].unsigned_abs() as f64])
        }
    }
    impl AsyncProblem for AbsInt {
        type Decision = Vec<i32>;
        fn objectives(&self) -> ObjectiveSpace {
            <Self as Problem>::objectives(self)
        }
        async fn evaluate_async(&self, x: &Vec<i32>) -> Evaluation {
            <Self as Problem>::evaluate(self, x)
        }
    }

    /// 3-D multi-fidelity wrapper for Hyperband — the underlying objective
    /// is 3-D Rosenbrock, but evaluations are **budget-sensitive**: a
    /// low-budget evaluation is biased high by a deterministic per-x
    /// penalty that decays as `1 / budget`. This matters for the snapshot
    /// assertion: a budget-*insensitive* problem makes Hyperband's bracket
    /// / rung / budget arithmetic inert (every config is judged the same
    /// regardless of allocation), so mutations there can't be detected.
    /// With a budget-sensitive problem the recovered best reflects exactly
    /// which configs Hyperband promoted to which budgets.
    struct SnapSpherePartial;
    impl SnapSpherePartial {
        fn rosenbrock(x: &[f64]) -> f64 {
            let mut f = 0.0;
            for i in 0..x.len() - 1 {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                f += 100.0 * a * a + b * b;
            }
            f
        }
        /// Deterministic per-x bias, scaled by 1/budget. Higher budget →
        /// smaller bias → more accurate estimate (the PartialProblem
        /// monotonicity contract).
        fn budgeted(x: &[f64], budget: f64) -> f64 {
            let bias: f64 = x.iter().map(|v| v.abs()).sum();
            Self::rosenbrock(x) + 50.0 * bias / budget.max(1.0)
        }
    }
    impl PartialProblem for SnapSpherePartial {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }
        fn evaluate_at_budget(&self, x: &Vec<f64>, budget: f64) -> Evaluation {
            Evaluation::new(vec![Self::budgeted(x, budget)])
        }
    }
    impl AsyncPartialProblem for SnapSpherePartial {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            <Self as PartialProblem>::objectives(self)
        }
        async fn evaluate_at_budget_async(&self, x: &Vec<f64>, budget: f64) -> Evaluation {
            <Self as PartialProblem>::evaluate_at_budget(self, x, budget)
        }
    }

    fn objectives_of<D>(r: &OptimizationResult<D>) -> Vec<f64> {
        r.best
            .as_ref()
            .map(|c| c.evaluation.objectives.clone())
            .unwrap_or_default()
    }

    // ---- Per-algorithm parity tests -----------------------------------------

    #[tokio::test]
    async fn random_search_async_matches_sync() {
        let cfg = RandomSearchConfig {
            iterations: 8,
            batch_size: 1,
            seed: 42,
        };
        let mut a = RandomSearch::new(cfg.clone(), snap_so_bounds());
        let mut b = RandomSearch::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![63.306134606086815],
                vec![704.5027621820259],
                vec![2710.1445719354297],
                vec![4282.489345404012],
                vec![4362.422716839316],
                vec![7901.9472922577515],
                vec![8445.708398022229],
                vec![10473.450683629166]
            ]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn hill_climber_async_matches_sync() {
        let cfg = HillClimberConfig {
            iterations: 40,
            seed: 42,
        };
        let mut a = HillClimber::new(
            cfg.clone(),
            snap_so_bounds(),
            GaussianMutation { sigma: 0.1 },
        );
        let mut b = HillClimber::new(cfg, snap_so_bounds(), GaussianMutation { sigma: 0.1 });
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![0.8713282461232607]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn one_plus_one_es_async_matches_sync() {
        let cfg = OnePlusOneEsConfig {
            iterations: 40,
            initial_sigma: 0.5,
            adaptation_period: 4,
            step_increase: 1.5,
            seed: 42,
        };
        let mut a = OnePlusOneEs::new(cfg.clone(), snap_so_bounds());
        let mut b = OnePlusOneEs::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![1.448688485637438]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn simulated_annealing_async_matches_sync() {
        let cfg = SimulatedAnnealingConfig {
            iterations: 40,
            initial_temperature: 1.0,
            final_temperature: 0.1,
            seed: 42,
        };
        let mut a = SimulatedAnnealing::new(
            cfg.clone(),
            snap_so_bounds(),
            GaussianMutation { sigma: 0.1 },
        );
        let mut b = SimulatedAnnealing::new(cfg, snap_so_bounds(), GaussianMutation { sigma: 0.1 });
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![1.8936764528185597]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn genetic_algorithm_async_matches_sync() {
        let bounds = vec![(-3.0_f64, 3.0); 3];
        let cfg = GeneticAlgorithmConfig {
            population_size: 6,
            generations: 25,
            tournament_size: 2,
            elitism: 1,
            seed: 42,
        };
        let make = || {
            GeneticAlgorithm::new(
                cfg.clone(),
                RealBounds::new(bounds.clone()),
                CompositeVariation {
                    crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
                    mutation: PolynomialMutation::new(bounds.clone(), 20.0, 1.0),
                },
            )
        };
        let r_sync = make().run(&SnapSphere);
        let r_async = make().run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![1.473670536153147],
                vec![3.5514321351551894],
                vec![10.006935556353785],
                vec![13.100758990828941],
                vec![84.24433280566845],
                vec![102.49327038081938]
            ]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn particle_swarm_async_matches_sync() {
        let cfg = ParticleSwarmConfig {
            swarm_size: 6,
            generations: 25,
            inertia: 0.5,
            cognitive: 1.0,
            social: 1.0,
            seed: 42,
        };
        let mut a = ParticleSwarm::new(cfg.clone(), snap_so_bounds());
        let mut b = ParticleSwarm::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.0014273534965355294],
                vec![0.0016092898775173475],
                vec![0.0016360280965425223],
                vec![0.0028941361029547383],
                vec![0.0033691650194037533],
                vec![0.3357310584047361]
            ]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn differential_evolution_async_matches_sync() {
        let cfg = DifferentialEvolutionConfig {
            population_size: 6,
            generations: 25,
            differential_weight: 0.5,
            crossover_probability: 0.9,
            seed: 42,
        };
        let mut a = DifferentialEvolution::new(cfg.clone(), snap_so_bounds());
        let mut b = DifferentialEvolution::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![1.0209899358667007],
                vec![1.177250863812577],
                vec![1.1849640927940035],
                vec![1.2565307288266827],
                vec![1.2651999986995113],
                vec![1.296215885697605]
            ]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn cma_es_async_matches_sync() {
        let cfg = CmaEsConfig {
            population_size: 6,
            generations: 25,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            initial_mean: None,
            seed: 42,
        };
        let mut a = CmaEs::new(cfg.clone(), snap_so_bounds());
        let mut b = CmaEs::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![0.6700664542332742]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn ipop_cma_es_async_matches_sync() {
        let cfg = IpopCmaEsConfig {
            initial_population_size: 4,
            total_generations: 40,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            stall_generations: None,
            seed: 42,
        };
        let mut a = IpopCmaEs::new(cfg.clone(), snap_so_bounds());
        let mut b = IpopCmaEs::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![0.9074496962843028]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn separable_nes_async_matches_sync() {
        let cfg = SeparableNesConfig {
            population_size: 6,
            generations: 25,
            initial_sigma: 0.5,
            mean_learning_rate: 1.0,
            sigma_learning_rate: Some(0.1),
            seed: 42,
        };
        let mut a = SeparableNes::new(cfg.clone(), snap_so_bounds());
        let mut b = SeparableNes::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![2.3348936837614187]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn tlbo_async_matches_sync() {
        let cfg = TlboConfig {
            population_size: 6,
            generations: 25,
            seed: 42,
        };
        let mut a = Tlbo::new(cfg.clone(), snap_so_bounds());
        let mut b = Tlbo::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.22584258931975904],
                vec![0.287744793845985],
                vec![0.2909810385734612],
                vec![0.3033740027429419],
                vec![0.32232945704973176],
                vec![0.32269000540663206]
            ]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn nelder_mead_async_matches_sync() {
        let cfg = NelderMeadConfig {
            iterations: 40,
            ..NelderMeadConfig::default()
        };
        let mut a = NelderMead::new(cfg.clone(), snap_so_bounds());
        let mut b = NelderMead::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![0.37482713384688104]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn bayesian_opt_async_matches_sync() {
        let cfg = BayesianOptConfig {
            initial_samples: 3,
            iterations: 12,
            length_scales: None,
            signal_variance: 1.0,
            noise_variance: 1e-3,
            acquisition_samples: 8,
            seed: 42,
        };
        let mut a = BayesianOpt::new(cfg.clone(), snap_so_bounds());
        let mut b = BayesianOpt::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![37.636221207678574],
                vec![63.306134606086815],
                vec![330.7204672154741],
                vec![420.2872728247263],
                vec![469.01167857530334],
                vec![683.620651365178],
                vec![1296.331547643922],
                vec![1975.425113659962],
                vec![2308.580211846731],
                vec![4362.422716839316],
                vec![4781.49796690649],
                vec![5246.461124345242],
                vec![5726.51240444759],
                vec![7901.947292257745],
                vec![10608.203503373621]
            ]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn tpe_async_matches_sync() {
        let cfg = TpeConfig {
            initial_samples: 3,
            iterations: 12,
            good_fraction: 0.25,
            candidate_samples: 8,
            bandwidth_factor: 0.1,
            seed: 42,
        };
        let mut a = Tpe::new(cfg.clone(), snap_so_bounds());
        let mut b = Tpe::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSphere);
        let r_async = b.run_async(&SnapSphere, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![63.306109614392106],
                vec![63.306116949436515],
                vec![63.306117012658824],
                vec![63.30611956234049],
                vec![63.30612002408215],
                vec![63.30612002731914],
                vec![63.30612144250236],
                vec![63.30613258185937],
                vec![63.30613285480626],
                vec![63.306134606086815],
                vec![63.306138944169454],
                vec![63.306140428140985],
                vec![63.30614044375412],
                vec![4362.422716839316],
                vec![7901.947292257745]
            ]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    // --- Multi-objective: best-comparison falls back to pareto front size ----
    //
    // For multi-objective algorithms, `best` is only meaningful as
    // `best_by_some_scalarization`. We compare the sorted Pareto-front
    // objective tuples instead.

    fn front_objectives<D>(r: &OptimizationResult<D>) -> Vec<Vec<f64>> {
        let mut front: Vec<Vec<f64>> = r
            .pareto_front
            .iter()
            .map(|c| c.evaluation.objectives.clone())
            .collect();
        front.sort_by(|a, b| {
            for (x, y) in a.iter().zip(b.iter()) {
                match x.partial_cmp(y) {
                    Some(std::cmp::Ordering::Equal) => continue,
                    Some(ord) => return ord,
                    None => return std::cmp::Ordering::Equal,
                }
            }
            std::cmp::Ordering::Equal
        });
        front
    }

    /// Sorted objective tuples of the *entire final population* — far more
    /// mutation-sensitive than `best` alone, since a mutated update rule
    /// changes the search distribution (and hence the sampled population)
    /// even when the single best-ever point happens to be unchanged.
    /// Falls back to the best candidate if the algorithm leaves
    /// `population` empty.
    fn population_objectives<D>(r: &OptimizationResult<D>) -> Vec<Vec<f64>> {
        let mut pop: Vec<Vec<f64>> = r
            .population
            .iter()
            .map(|c| c.evaluation.objectives.clone())
            .collect();
        if pop.is_empty() {
            if let Some(best) = r.best.as_ref() {
                pop.push(best.evaluation.objectives.clone());
            }
        }
        pop.sort_by(|a, b| {
            for (x, y) in a.iter().zip(b.iter()) {
                match x.partial_cmp(y) {
                    Some(std::cmp::Ordering::Equal) => continue,
                    Some(ord) => return ord,
                    None => return std::cmp::Ordering::Equal,
                }
            }
            std::cmp::Ordering::Equal
        });
        pop
    }

    #[tokio::test]
    async fn nsga2_async_matches_sync() {
        let make = || {
            Nsga2::new(
                Nsga2Config {
                    population_size: 8,
                    generations: 25,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.013894099789814499, 12.396569748512157],
                vec![0.9986044047366547, 10.04928885630924],
                vec![1.4179339198509142, 5.938538752276063],
                vec![1.9377635550108065, 5.175062400323034],
                vec![4.287814676408594, 2.072533399238466],
                vec![6.144590989246883, 0.9919485019724119],
                vec![7.568542905344836, 0.8775124508515268],
                vec![11.162223470843461, 0.06736833759926654]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn nsga3_async_matches_sync() {
        let make = || {
            Nsga3::new(
                Nsga3Config {
                    population_size: 8,
                    generations: 25,
                    reference_divisions: 4,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.04207557391436843, 11.452614567675102],
                vec![0.5483942154144871, 8.561837256536368],
                vec![1.547355561053402, 5.037243897511284],
                vec![2.1897000988710995, 4.570866266616154],
                vec![2.6617298789502133, 3.4774928534855687],
                vec![4.557616972017328, 2.413271056308534],
                vec![8.18734188319979, 0.40562353775794924],
                vec![11.538740749611467, 0.20265277704364293]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn spea2_async_matches_sync() {
        let make = || {
            Spea2::new(
                Spea2Config {
                    population_size: 8,
                    archive_size: 4,
                    generations: 25,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.010459871966689165, 11.383115832165583],
                vec![1.00406197192257, 6.710772782810457],
                vec![4.617743017065397, 2.193494053600796],
                vec![11.716970274948123, 0.016460995225913946]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn moead_async_matches_sync() {
        let make = || {
            Moead::new(
                MoeadConfig {
                    generations: 25,
                    reference_divisions: 4,
                    neighborhood_size: 2,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.0018965941331487578, 11.808507055951427],
                vec![1.4874217300175174, 5.27715535815546],
                vec![2.947894875106906, 3.446066071999487],
                vec![4.30232389012548, 1.949958003829106],
                vec![9.43090543273526, 0.6038381317627364]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn mopso_async_matches_sync() {
        let make = || {
            Mopso::new(
                MopsoConfig {
                    swarm_size: 6,
                    generations: 25,
                    archive_size: 4,
                    inertia: 0.5,
                    cognitive: 1.0,
                    social: 1.0,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![1.209299108345119, 5.907064121476177],
                vec![1.5729555505269657, 4.991703224346624],
                vec![1.9490201701027443, 4.316356698004658],
                vec![1.9596051383622117, 4.292838266270067]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn ibea_async_matches_sync() {
        let make = || {
            Ibea::new(
                IbeaConfig {
                    population_size: 8,
                    generations: 25,
                    kappa: 0.05,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![1.102989174772048, 5.927013751828334],
                vec![1.6270095299640162, 4.827939086661083],
                vec![2.1660395399724406, 4.144479169637154],
                vec![2.75551550490187, 3.2592048793171986],
                vec![3.4190662252024984, 2.638232122470969],
                vec![5.096097851324554, 1.4897953804615525],
                vec![6.191588790986529, 1.076799546289505],
                vec![10.471191707237805, 0.19856457625474933]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn sms_emoa_async_matches_sync() {
        let make = || {
            SmsEmoa::new(
                SmsEmoaConfig {
                    population_size: 8,
                    generations: 25,
                    reference_point: vec![100.0, 100.0],
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.48629723752715737, 9.65467157344917],
                vec![0.76152466146821, 7.823579428272707],
                vec![1.013464442017104, 6.402048672524341],
                vec![1.2493755385572065, 6.77907250568554],
                vec![2.23189833205296, 5.88957515209449],
                vec![3.8980878113466164, 5.570985049586769],
                vec![4.442278547014425, 3.495601181208072],
                vec![5.04184251314055, 4.67520303050857]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn hype_async_matches_sync() {
        let make = || {
            Hype::new(
                HypeConfig {
                    population_size: 8,
                    generations: 25,
                    reference_point: vec![10.0, 10.0],
                    mc_samples: 4,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![2.1468356594601867, 4.082170172701482],
                vec![2.258243590202526, 3.9253070443424924],
                vec![2.438385974767159, 3.681728006983862],
                vec![3.3195313427544137, 3.197540830543744],
                vec![3.471491609011644, 2.6174420196371186],
                vec![3.8982204071014666, 2.61118163669461],
                vec![3.9359534122250466, 2.515067124016713],
                vec![4.040911931124477, 2.1294731723321547]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn pesa_ii_async_matches_sync() {
        let make = || {
            PesaII::new(
                PesaIIConfig {
                    population_size: 8,
                    archive_size: 4,
                    generations: 25,
                    grid_divisions: 4,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![3.6245079224196353, 5.073627020565048],
                vec![5.546147318944607, 1.8534963745959785],
                vec![6.526086710933148, 1.0660738392669853],
                vec![10.515670253117545, 0.12279280481335361]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn epsilon_moea_async_matches_sync() {
        let make = || {
            EpsilonMoea::new(
                EpsilonMoeaConfig {
                    population_size: 8,
                    evaluations: 12,
                    epsilon: vec![0.1, 0.1],
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.76152466146821, 7.823579428272707],
                vec![2.2218156464658056, 5.784958465320774]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn age_moea_async_matches_sync() {
        let make = || {
            AgeMoea::new(
                AgeMoeaConfig {
                    population_size: 8,
                    generations: 25,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.04111219105652014, 13.074348360880695],
                vec![0.4346731920812637, 9.242155513828948],
                vec![1.358331428580249, 5.683016286040829],
                vec![2.5755814019595404, 3.60622785688025],
                vec![3.837138107106994, 2.3180546082918916],
                vec![5.286102901854588, 1.4045244018265168],
                vec![7.638613726361841, 0.9440692528451459],
                vec![11.019174430211159, 0.15681647211812566]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn grea_async_matches_sync() {
        let make = || {
            Grea::new(
                GreaConfig {
                    population_size: 8,
                    generations: 25,
                    grid_divisions: 4,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.5552666375678257, 7.47469613619204],
                vec![1.4056896475138974, 5.293501721710313],
                vec![1.5852083174630922, 4.882658343892137],
                vec![1.7395860023245642, 4.82943787827538],
                vec![1.7409421319942333, 4.813466888123339],
                vec![2.055412878673668, 4.334431622012692],
                vec![2.7400972765938705, 3.784963007905067],
                vec![2.968002937322503, 3.3418351793574135]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn knea_async_matches_sync() {
        let make = || {
            Knea::new(
                KneaConfig {
                    population_size: 8,
                    generations: 25,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.0018483545960612325, 12.22673649645838],
                vec![0.017237011688299664, 11.387998693142336],
                vec![0.027973368060834943, 10.968779947026622],
                vec![0.04220440957898664, 10.726923433631885],
                vec![0.04349788464967149, 10.665580898567551],
                vec![0.08092072017224045, 10.438020089521329],
                vec![0.12927891629102406, 10.135723096330521],
                vec![0.13777903061778596, 9.820391893329669]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn rvea_async_matches_sync() {
        let make = || {
            Rvea::new(
                RveaConfig {
                    population_size: 8,
                    generations: 25,
                    reference_divisions: 4,
                    alpha: 2.0,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                snap_mo_variation(),
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.7822578730650152, 6.840627682227026],
                vec![1.886520215889041, 4.4221143878379205],
                vec![2.925708852333473, 3.0764597322421308],
                vec![2.925708852333473, 3.0764597322421308],
                vec![2.925708852333473, 3.0764597322421308],
                vec![2.925708852333473, 3.0764597322421308],
                vec![4.2172159905416375, 2.2535573427617255],
                vec![5.629611931678675, 1.2046715353568256]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    #[tokio::test]
    async fn paes_async_matches_sync() {
        let make = || {
            Paes::new(
                PaesConfig {
                    iterations: 40,
                    archive_size: 4,
                    seed: 42,
                },
                RealBounds::new(snap_mo_bounds()),
                GaussianMutation { sigma: 0.1 },
            )
        };
        let r_sync = make().run(&SnapMo);
        let r_async = make().run_async(&SnapMo, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![
                vec![0.40915759624698217, 8.374880286293996],
                vec![0.5346610020355553, 8.30481689014011],
                vec![0.5822402093594947, 7.942141169439722],
                vec![0.6766093633299375, 7.383787211092805]
            ]
        );
        assert_eq!(front_objectives(&r_sync), front_objectives(&r_async));
    }

    // --- Binary, integer, permutation, multi-fidelity ----------------------

    #[tokio::test]
    async fn umda_async_matches_sync() {
        let cfg = UmdaConfig {
            bits: 4,
            population_size: 6,
            selected_size: 3,
            generations: 25,
            seed: 42,
        };
        let mut a = Umda::new(cfg.clone());
        let mut b = Umda::new(cfg);
        let problem = OneMax { bits: 4 };
        let r_sync = a.run(&problem);
        let r_async = b.run_async(&problem, 2).await;
        assert_eq!(population_objectives(&r_sync), vec![vec![4.0]]);
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn ant_colony_tsp_async_matches_sync() {
        let problem = TinyTsp::new();
        let cfg = AntColonyTspConfig {
            ants: 4,
            generations: 25,
            alpha: 1.0,
            beta: 2.0,
            evaporation: 0.5,
            deposit: 1.0,
            initial_pheromone: 1.0,
            seed: 42,
        };
        let mut a = AntColonyTsp::new(cfg.clone(), problem.dist.clone());
        let mut b = AntColonyTsp::new(cfg, problem.dist.clone());
        let r_sync = a.run(&problem);
        let r_async = b.run_async(&problem, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![19.16243758807328]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn tabu_search_async_matches_sync() {
        struct StartAt5;
        impl Initializer<Vec<i32>> for StartAt5 {
            fn initialize(
                &mut self,
                _size: usize,
                _rng: &mut heuropt::core::rng::Rng,
            ) -> Vec<Vec<i32>> {
                vec![vec![5]]
            }
        }
        let neighbors =
            |x: &Vec<i32>, _rng: &mut heuropt::core::rng::Rng| vec![vec![x[0] - 1], vec![x[0] + 1]];
        let cfg = TabuSearchConfig {
            iterations: 40,
            tabu_tenure: 3,
            seed: 42,
        };
        let mut a = TabuSearch::new(cfg.clone(), StartAt5, neighbors);
        let mut b = TabuSearch::new(cfg, StartAt5, neighbors);
        let r_sync = a.run(&AbsInt);
        let r_async = b.run_async(&AbsInt, 2).await;
        assert_eq!(population_objectives(&r_sync), vec![vec![0.0]]);
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }

    #[tokio::test]
    async fn hyperband_async_matches_sync() {
        let cfg = HyperbandConfig {
            max_budget: 27.0,
            eta: 3.0,
            max_brackets: 2,
            seed: 42,
        };
        let mut a: Hyperband<RealBounds, Vec<f64>> = Hyperband::new(cfg.clone(), snap_so_bounds());
        let mut b: Hyperband<RealBounds, Vec<f64>> = Hyperband::new(cfg, snap_so_bounds());
        let r_sync = a.run(&SnapSpherePartial);
        let r_async = b.run_async(&SnapSpherePartial, 2).await;
        assert_eq!(
            population_objectives(&r_sync),
            vec![vec![65.59222036219585]]
        );
        assert_eq!(objectives_of(&r_sync), objectives_of(&r_async));
    }
}
