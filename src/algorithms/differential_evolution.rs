//! Differential Evolution (DE/rand/1/bin) for single-objective real-valued problems.

use rand::Rng as _;

use crate::core::candidate::Candidate;
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
    P: Problem<Decision = Vec<f64>>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
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

        let dim = self.bounds.bounds.len();
        let n = self.config.population_size;
        let mut rng = rng_from_seed(self.config.seed);

        // Seed the population using the bounds as a sampler.
        let mut decisions: Vec<Vec<f64>> = {
            use crate::traits::Initializer as _;
            self.bounds.initialize(n, &mut rng)
        };
        let mut evaluations = 0usize;
        let mut evals: Vec<f64> = decisions
            .iter()
            .map(|d| {
                let e = problem.evaluate(d);
                evaluations += 1;
                e.objectives[0]
            })
            .collect();

        for _gen in 0..self.config.generations {
            for i in 0..n {
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
                let trial_obj = {
                    let e = problem.evaluate(&trial);
                    evaluations += 1;
                    e.objectives[0]
                };
                let target_obj = evals[i];
                let trial_better = match direction {
                    Direction::Minimize => trial_obj <= target_obj,
                    Direction::Maximize => trial_obj >= target_obj,
                };
                if trial_better {
                    decisions[i] = trial;
                    evals[i] = trial_obj;
                }
            }
        }

        let final_pop: Vec<Candidate<Vec<f64>>> = decisions
            .into_iter()
            .map(|d| {
                let e = problem.evaluate(&d);
                evaluations += 1;
                Candidate::new(d, e)
            })
            .collect();
        let front = pareto_front(&final_pop, &objectives);
        let best = best_candidate(&final_pop, &objectives);
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

fn pick_three_distinct(n: usize, exclude: usize, rng: &mut crate::core::rng::Rng) -> (usize, usize, usize) {
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
        assert!(best.evaluation.objectives[0] < 1e-3, "DE should converge near 0");
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
        let mut a =
            DifferentialEvolution::new(cfg.clone(), RealBounds::new(vec![(-5.0, 5.0)]));
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
