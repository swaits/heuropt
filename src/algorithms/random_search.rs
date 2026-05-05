//! Baseline `RandomSearch` optimizer.
//!
//! This is the reference example for spec §2.4 / §12.1: read this file before
//! writing your own optimizer.

use crate::core::candidate::Candidate;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::pareto::{best_candidate, pareto_front};
use crate::traits::{Initializer, Optimizer};

/// Configuration for [`RandomSearch`].
#[derive(Debug, Clone)]
pub struct RandomSearchConfig {
    /// Number of iterations (equals `generations` in the result).
    pub iterations: usize,
    /// Decisions sampled per iteration.
    pub batch_size: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for RandomSearchConfig {
    fn default() -> Self {
        Self { iterations: 100, batch_size: 1, seed: 42 }
    }
}

/// Sample-evaluate-keep baseline optimizer.
///
/// Each iteration the configured `Initializer` produces `batch_size` decisions
/// which are evaluated and pushed into the population. Cheap, parallelism-free,
/// and useful as a sanity-check baseline.
#[derive(Debug, Clone)]
pub struct RandomSearch<I> {
    /// Algorithm configuration.
    pub config: RandomSearchConfig,
    /// Decision-sampling strategy.
    pub initializer: I,
}

impl<I> RandomSearch<I> {
    /// Construct a `RandomSearch` from its config and initializer.
    pub fn new(config: RandomSearchConfig, initializer: I) -> Self {
        Self { config, initializer }
    }
}

impl<P, I> Optimizer<P> for RandomSearch<I>
where
    P: Problem,
    I: Initializer<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);
        let mut all: Vec<Candidate<P::Decision>> = Vec::new();
        let mut evaluations = 0usize;

        for _ in 0..self.config.iterations {
            let decisions = self.initializer.initialize(self.config.batch_size, &mut rng);
            for decision in decisions {
                let eval = problem.evaluate(&decision);
                evaluations += 1;
                all.push(Candidate::new(decision, eval));
            }
        }

        let front = pareto_front(&all, &objectives);
        let best = best_candidate(&all, &objectives);
        OptimizationResult::new(
            Population::new(all),
            front,
            best,
            evaluations,
            self.config.iterations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::RealBounds;
    use crate::tests_support::{SchafferN1, Sphere1D};

    #[test]
    fn evaluation_count_matches_iterations_times_batch() {
        let mut opt = RandomSearch::new(
            RandomSearchConfig { iterations: 30, batch_size: 4, seed: 1 },
            RealBounds::new(vec![(-2.0, 2.0)]),
        );
        let r = opt.run(&Sphere1D);
        assert_eq!(r.evaluations, 30 * 4);
        assert_eq!(r.population.len(), 30 * 4);
        assert_eq!(r.generations, 30);
    }

    #[test]
    fn pareto_front_non_empty_for_multi_objective() {
        let mut opt = RandomSearch::new(
            RandomSearchConfig { iterations: 50, batch_size: 1, seed: 42 },
            RealBounds::new(vec![(-5.0, 5.0)]),
        );
        let r = opt.run(&SchafferN1);
        assert!(!r.pareto_front.is_empty());
        // Multi-objective ⇒ best is None.
        assert!(r.best.is_none());
    }

    #[test]
    fn single_objective_returns_best() {
        let mut opt = RandomSearch::new(
            RandomSearchConfig { iterations: 100, batch_size: 1, seed: 7 },
            RealBounds::new(vec![(-1.0, 1.0)]),
        );
        let r = opt.run(&Sphere1D);
        assert!(r.best.is_some());
    }
}
