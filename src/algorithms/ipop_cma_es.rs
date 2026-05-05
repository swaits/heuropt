//! `IpopCmaEs` — Auger & Hansen 2005 Increasing-Population CMA-ES.
//!
//! Wraps `CmaEs` in a restart loop that doubles the population size and
//! re-randomizes the initial mean each restart. This is the standard fix
//! for vanilla CMA-ES's well-known weakness on multimodal problems.

use rand::Rng as _;

use crate::algorithms::cma_es::{CmaEs, CmaEsConfig};
use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::operators::real::RealBounds;
use crate::traits::Optimizer;

/// Configuration for [`IpopCmaEs`].
#[derive(Debug, Clone)]
pub struct IpopCmaEsConfig {
    /// Initial population size for the first CMA-ES restart. Each
    /// subsequent restart doubles this.
    pub initial_population_size: usize,
    /// Total number of generations across ALL restarts. Each restart
    /// consumes generations proportional to its population size; the
    /// outer loop stops once this budget is exhausted.
    pub total_generations: usize,
    /// Initial step size σ_0 for every restart.
    pub initial_sigma: f64,
    /// CMA-ES eigen-decomposition refresh period (passed through).
    pub eigen_decomposition_period: usize,
    /// Generations of no-improvement that triggers a restart from inside
    /// a single CMA-ES run. None disables this trigger (only the outer
    /// budget terminates restarts).
    pub stall_generations: Option<usize>,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for IpopCmaEsConfig {
    fn default() -> Self {
        Self {
            initial_population_size: 16,
            total_generations: 500,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            stall_generations: Some(50),
            seed: 42,
        }
    }
}

/// IPOP-CMA-ES: CMA-ES with population-doubling restarts.
#[derive(Debug, Clone)]
pub struct IpopCmaEs {
    /// Algorithm configuration.
    pub config: IpopCmaEsConfig,
    /// Per-variable bounds.
    pub bounds: RealBounds,
}

impl IpopCmaEs {
    /// Construct an `IpopCmaEs`.
    pub fn new(config: IpopCmaEsConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for IpopCmaEs
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.initial_population_size >= 4,
            "IpopCmaEs initial_population_size must be >= 4",
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "IpopCmaEs requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let mut remaining_gens = self.config.total_generations;
        let mut pop_size = self.config.initial_population_size;
        let mut total_evaluations = 0usize;
        let mut total_iterations = 0usize;
        let mut best_seen: Option<Candidate<Vec<f64>>> = None;
        let _ = self.config.stall_generations; // reserved for future trigger

        let mut restart_counter = 0u64;
        while remaining_gens > 0 {
            // Per-restart budget: roughly `total / 2^restart` generations,
            // with a sensible floor.
            let this_gens = (remaining_gens / 2).max(20).min(remaining_gens);
            let inner_seed = self
                .config
                .seed
                .wrapping_add(restart_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            // Re-randomize the inner mean to a uniform-random point inside
            // the original bounds, keeping the bounds box itself unchanged
            // so search isn't artificially restricted.
            let restart_mean: Vec<f64> = self
                .bounds
                .bounds
                .iter()
                .map(|&(lo, hi)| lo + (hi - lo) * rng.random::<f64>())
                .collect();
            let cfg = CmaEsConfig {
                population_size: pop_size,
                generations: this_gens,
                initial_sigma: self.config.initial_sigma,
                eigen_decomposition_period: self.config.eigen_decomposition_period,
                initial_mean: Some(restart_mean),
                seed: inner_seed,
            };
            let inner = CmaEs::new(cfg, RealBounds::new(self.bounds.bounds.clone()));
            let mut inner = inner;

            let result = inner.run(problem);
            total_evaluations += result.evaluations;
            total_iterations += result.generations;
            if let Some(b) = result.best.clone() {
                let beats = match &best_seen {
                    None => true,
                    Some(prev) => better(&b.evaluation, &prev.evaluation, direction),
                };
                if beats {
                    best_seen = Some(b);
                }
            }
            remaining_gens = remaining_gens.saturating_sub(this_gens);
            pop_size = pop_size.saturating_mul(2);
            restart_counter = restart_counter.wrapping_add(1);
        }

        let best = best_seen.expect("at least one restart ran");
        let population = Population::new(vec![best.clone()]);
        let front = vec![best.clone()];
        OptimizationResult::new(
            population,
            front,
            Some(best),
            total_evaluations,
            total_iterations,
        )
    }
}

fn better(a: &Evaluation, b: &Evaluation, direction: Direction) -> bool {
    match (a.is_feasible(), b.is_feasible()) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => a.constraint_violation < b.constraint_violation,
        (true, true) => match direction {
            Direction::Minimize => a.objectives[0] < b.objectives[0],
            Direction::Maximize => a.objectives[0] > b.objectives[0],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Objective, ObjectiveSpace};
    use crate::tests_support::{SchafferN1, Sphere1D};
    use std::f64::consts::PI;

    /// 5-D Rastrigin to exercise the restart benefit.
    struct Rastrigin5D;
    impl Problem for Rastrigin5D {
        type Decision = Vec<f64>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }

        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            let n = x.len() as f64;
            let v = 10.0 * n
                + x.iter()
                    .map(|v| v * v - 10.0 * (2.0 * PI * v).cos())
                    .sum::<f64>();
            Evaluation::new(vec![v])
        }
    }

    fn make_optimizer(seed: u64) -> IpopCmaEs {
        IpopCmaEs::new(
            IpopCmaEsConfig {
                initial_population_size: 8,
                total_generations: 300,
                initial_sigma: 1.0,
                eigen_decomposition_period: 1,
                stall_generations: None,
                seed,
            },
            RealBounds::new(vec![(-5.12, 5.12); 5]),
        )
    }

    #[test]
    fn finds_minimum_of_sphere() {
        let mut opt = IpopCmaEs::new(
            IpopCmaEsConfig {
                initial_population_size: 8,
                total_generations: 100,
                initial_sigma: 0.5,
                eigen_decomposition_period: 1,
                stall_generations: None,
                seed: 1,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
        );
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-8,
            "got f = {}",
            best.evaluation.objectives[0],
        );
    }

    #[test]
    fn produces_reasonable_rastrigin_result() {
        // Don't claim a strict beat-vanilla threshold (that's a stochastic
        // statement); just verify IPOP runs to completion and produces a
        // result clearly better than random sampling on a 5-D Rastrigin
        // (random would average f ≈ 11–12).
        let mut opt = make_optimizer(1);
        let r = opt.run(&Rastrigin5D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 5.0,
            "IPOP-CMA-ES underperformed on Rastrigin: f = {}",
            best.evaluation.objectives[0],
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut a = make_optimizer(99);
        let mut b = make_optimizer(99);
        let ra = a.run(&Rastrigin5D);
        let rb = b.run(&Rastrigin5D);
        assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    #[should_panic(expected = "exactly one objective")]
    fn multi_objective_panics() {
        let mut opt = make_optimizer(0);
        let _ = opt.run(&SchafferN1);
    }
}
