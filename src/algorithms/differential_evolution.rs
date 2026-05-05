//! Differential Evolution (DE/rand/1/bin) for single-objective real-valued problems.

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::operators::real::RealBounds;
use crate::pareto::front::{best_candidate, pareto_front};
use crate::traits::Optimizer;

/// Configuration for [`DifferentialEvolution`].
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionConfig {
    /// Number of agents in the population.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Differential weight `F`. Typical values are in `[0.4, 1.0]`.
    pub differential_weight: f64,
    /// Per-dimension crossover probability `CR`. Typical values are in `[0.5, 0.95]`.
    pub crossover_probability: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for DifferentialEvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 200,
            differential_weight: 0.7,
            crossover_probability: 0.9,
            seed: 42,
        }
    }
}

/// Single-objective DE/rand/1/bin (spec §12.4).
///
/// `Vec<f64>` decisions only; single-objective problems only. Bounds come from
/// the embedded `RealBounds`, and mutant vectors are clamped to those bounds.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// struct Sphere;
/// impl Problem for Sphere {
///     type Decision = Vec<f64>;
///     fn objectives(&self) -> ObjectiveSpace {
///         ObjectiveSpace::new(vec![Objective::minimize("f")])
///     }
///     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
///         Evaluation::new(vec![x.iter().map(|v| v * v).sum::<f64>()])
///     }
/// }
///
/// let mut opt = DifferentialEvolution::new(
///     DifferentialEvolutionConfig {
///         population_size: 20,
///         generations: 50,
///         differential_weight: 0.5,
///         crossover_probability: 0.9,
///         seed: 42,
///     },
///     RealBounds::new(vec![(-5.0, 5.0); 5]),
/// );
/// let r = opt.run(&Sphere);
/// // DE crushes Sphere; expect very small objective.
/// assert!(r.best.unwrap().evaluation.objectives[0] < 1e-3);
/// ```
#[derive(Debug, Clone)]
pub struct DifferentialEvolution {
    /// Algorithm configuration.
    pub config: DifferentialEvolutionConfig,
    /// Per-variable bounds — used both to seed the population and to clamp mutants.
    pub bounds: RealBounds,
}

impl DifferentialEvolution {
    /// Construct a `DifferentialEvolution` optimizer.
    pub fn new(config: DifferentialEvolutionConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for DifferentialEvolution
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        self.run_with(problem, &mut ())
    }

    fn run_with<O>(&mut self, problem: &P, observer: &mut O) -> OptimizationResult<P::Decision>
    where
        O: crate::observer::Observer<P::Decision>,
    {
        use crate::observer::Snapshot;
        use std::ops::ControlFlow;

        assert!(
            self.config.population_size >= 4,
            "DifferentialEvolution requires population_size >= 4 (DE/rand/1 needs three distinct donors plus the target)",
        );
        assert!(
            (0.0..=1.0).contains(&self.config.crossover_probability),
            "DifferentialEvolution crossover_probability must be in [0.0, 1.0]",
        );

        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "DifferentialEvolution only supports single-objective problems",
        );
        let direction = objectives.objectives[0].direction;
        let started = std::time::Instant::now();

        let dim = self.bounds.bounds.len();
        let n = self.config.population_size;
        let mut rng = rng_from_seed(self.config.seed);

        // Seed the population using the bounds as a sampler.
        let mut decisions: Vec<Vec<f64>> = {
            use crate::traits::Initializer as _;
            self.bounds.initialize(n, &mut rng)
        };
        let initial_pop = evaluate_batch(problem, decisions.clone());
        let mut evaluations = initial_pop.len();
        let mut current_pop = initial_pop;
        let mut evals: Vec<f64> = current_pop
            .iter()
            .map(|c| c.evaluation.objectives[0])
            .collect();
        let mut completed_generations: usize = 0;

        // Initial snapshot.
        {
            let best = best_candidate(&current_pop, &objectives);
            let snap = Snapshot {
                iteration: 0,
                evaluations,
                elapsed: started.elapsed(),
                population: &current_pop,
                pareto_front: None,
                best: best.as_ref(),
                objectives: &objectives,
            };
            if let ControlFlow::Break(()) = observer.observe(&snap) {
                let front = pareto_front(&current_pop, &objectives);
                let best = best_candidate(&current_pop, &objectives);
                return OptimizationResult::new(
                    Population::new(current_pop),
                    front,
                    best,
                    evaluations,
                    completed_generations,
                );
            }
        }

        for generation in 1..=self.config.generations {
            // Phase 1 (serial): construct one trial per target. RNG state is
            // consumed in deterministic order so seeded runs reproduce
            // exactly regardless of the `parallel` feature.
            let trials: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let (r1, r2, r3) = pick_three_distinct(n, i, &mut rng);
                    let j_rand = rng.random_range(0..dim);
                    let mut trial = decisions[i].clone();
                    for j in 0..dim {
                        let take_donor =
                            rng.random_bool(self.config.crossover_probability) || j == j_rand;
                        if take_donor {
                            let mutant = decisions[r1][j]
                                + self.config.differential_weight
                                    * (decisions[r2][j] - decisions[r3][j]);
                            let (lo, hi) = self.bounds.bounds[j];
                            trial[j] = mutant.clamp(lo, hi);
                        }
                    }
                    trial
                })
                .collect();

            // Phase 2 (parallel-friendly): evaluate every trial.
            let trial_cands = evaluate_batch(problem, trials);
            evaluations += trial_cands.len();

            // Phase 3 (serial): greedy replacement.
            for (i, trial_cand) in trial_cands.into_iter().enumerate() {
                let trial_obj = trial_cand.evaluation.objectives[0];
                let target_obj = evals[i];
                let trial_better = match direction {
                    Direction::Minimize => trial_obj <= target_obj,
                    Direction::Maximize => trial_obj >= target_obj,
                };
                if trial_better {
                    decisions[i] = trial_cand.decision.clone();
                    evals[i] = trial_obj;
                    current_pop[i] = trial_cand;
                }
            }
            completed_generations = generation;

            // Per-generation snapshot.
            let best = best_candidate(&current_pop, &objectives);
            let snap = Snapshot {
                iteration: generation,
                evaluations,
                elapsed: started.elapsed(),
                population: &current_pop,
                pareto_front: None,
                best: best.as_ref(),
                objectives: &objectives,
            };
            if let ControlFlow::Break(()) = observer.observe(&snap) {
                break;
            }
        }

        // Re-evaluate to make sure final population is consistent (current_pop is already current).
        let front = pareto_front(&current_pop, &objectives);
        let best = best_candidate(&current_pop, &objectives);
        OptimizationResult::new(
            Population::new(current_pop),
            front,
            best,
            evaluations,
            completed_generations,
        )
    }
}

#[cfg(feature = "async")]
impl DifferentialEvolution {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations per batch (initial
    /// population and per-generation trials).
    pub async fn run_async<P>(
        &mut self,
        problem: &P,
        concurrency: usize,
    ) -> OptimizationResult<Vec<f64>>
    where
        P: crate::core::async_problem::AsyncProblem<Decision = Vec<f64>>,
    {
        use rand::Rng as _;

        use crate::algorithms::parallel_eval_async::evaluate_batch_async;
        use crate::core::candidate::Candidate;
        use crate::traits::Initializer as _;

        assert!(
            self.config.population_size >= 4,
            "DifferentialEvolution requires population_size >= 4",
        );
        assert!(
            (0.0..=1.0).contains(&self.config.crossover_probability),
            "DifferentialEvolution crossover_probability must be in [0.0, 1.0]",
        );

        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "DifferentialEvolution only supports single-objective problems",
        );
        let direction = objectives.objectives[0].direction;

        let dim = self.bounds.bounds.len();
        let n = self.config.population_size;
        let mut rng = rng_from_seed(self.config.seed);

        let mut decisions: Vec<Vec<f64>> = self.bounds.initialize(n, &mut rng);
        let initial_pop = evaluate_batch_async(problem, decisions.clone(), concurrency).await;
        let mut evaluations = initial_pop.len();
        let mut current_pop = initial_pop;
        let mut evals: Vec<f64> = current_pop
            .iter()
            .map(|c| c.evaluation.objectives[0])
            .collect();

        for _generation in 0..self.config.generations {
            let trials: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let (r1, r2, r3) = pick_three_distinct(n, i, &mut rng);
                    let j_rand = rng.random_range(0..dim);
                    let mut trial = decisions[i].clone();
                    for j in 0..dim {
                        let take_donor =
                            rng.random_bool(self.config.crossover_probability) || j == j_rand;
                        if take_donor {
                            let mutant = decisions[r1][j]
                                + self.config.differential_weight
                                    * (decisions[r2][j] - decisions[r3][j]);
                            let (lo, hi) = self.bounds.bounds[j];
                            trial[j] = mutant.clamp(lo, hi);
                        }
                    }
                    trial
                })
                .collect();
            let trial_cands: Vec<Candidate<Vec<f64>>> =
                evaluate_batch_async(problem, trials, concurrency).await;
            evaluations += trial_cands.len();
            for (i, trial_cand) in trial_cands.into_iter().enumerate() {
                let trial_obj = trial_cand.evaluation.objectives[0];
                let target_obj = evals[i];
                let trial_better = match direction {
                    crate::core::objective::Direction::Minimize => trial_obj <= target_obj,
                    crate::core::objective::Direction::Maximize => trial_obj >= target_obj,
                };
                if trial_better {
                    decisions[i] = trial_cand.decision.clone();
                    evals[i] = trial_obj;
                    current_pop[i] = trial_cand;
                }
            }
        }

        let front = pareto_front(&current_pop, &objectives);
        let best = best_candidate(&current_pop, &objectives);
        OptimizationResult::new(
            Population::new(current_pop),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

fn pick_three_distinct(
    n: usize,
    exclude: usize,
    rng: &mut crate::core::rng::Rng,
) -> (usize, usize, usize) {
    let pick = |rng: &mut crate::core::rng::Rng, taken: &[usize]| -> usize {
        loop {
            let v = rng.random_range(0..n);
            if v != exclude && !taken.contains(&v) {
                return v;
            }
        }
    };
    let a = pick(rng, &[]);
    let b = pick(rng, &[a]);
    let c = pick(rng, &[a, b]);
    (a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{SchafferN1, Sphere1D};

    #[test]
    fn finds_minimum_of_sphere() {
        let mut opt = DifferentialEvolution::new(
            DifferentialEvolutionConfig {
                population_size: 30,
                generations: 100,
                differential_weight: 0.7,
                crossover_probability: 0.9,
                seed: 1,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
        );
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-3,
            "DE should converge near 0"
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let cfg = DifferentialEvolutionConfig {
            population_size: 20,
            generations: 30,
            differential_weight: 0.5,
            crossover_probability: 0.7,
            seed: 99,
        };
        let mut a = DifferentialEvolution::new(cfg.clone(), RealBounds::new(vec![(-5.0, 5.0)]));
        let mut b = DifferentialEvolution::new(cfg, RealBounds::new(vec![(-5.0, 5.0)]));
        let ra = a.run(&Sphere1D);
        let rb = b.run(&Sphere1D);
        assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives
        );
    }

    #[test]
    #[should_panic(expected = "single-objective")]
    fn multi_objective_panics() {
        let mut opt = DifferentialEvolution::new(
            DifferentialEvolutionConfig::default(),
            RealBounds::new(vec![(-5.0, 5.0)]),
        );
        let _ = opt.run(&SchafferN1);
    }

    #[test]
    #[should_panic(expected = "population_size >= 4")]
    fn too_small_population_panics() {
        let mut opt = DifferentialEvolution::new(
            DifferentialEvolutionConfig {
                population_size: 3,
                generations: 1,
                differential_weight: 0.5,
                crossover_probability: 0.5,
                seed: 0,
            },
            RealBounds::new(vec![(-1.0, 1.0)]),
        );
        let _ = opt.run(&Sphere1D);
    }
}
