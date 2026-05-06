//! `HillClimber` — single-objective greedy local search.

use crate::core::candidate::Candidate;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`HillClimber`].
#[derive(Debug, Clone)]
pub struct HillClimberConfig {
    /// Number of mutation iterations.
    pub iterations: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for HillClimberConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            seed: 42,
        }
    }
}

/// Single-objective greedy hill climber.
///
/// Starts from one initializer-sampled decision, repeatedly mutates it via
/// the variation operator, and keeps the child only when it is strictly
/// better than the current incumbent. Standard feasibility tiebreaks apply:
/// feasible beats infeasible, smaller violation wins among infeasibles.
///
/// Single-objective only.
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
/// let mut opt = HillClimber::new(
///     HillClimberConfig { iterations: 500, seed: 42 },
///     RealBounds::new(vec![(-5.0, 5.0); 3]),
///     GaussianMutation { sigma: 0.3 },
/// );
/// let r = opt.run(&Sphere);
/// assert!(r.best.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct HillClimber<I, V> {
    /// Algorithm configuration.
    pub config: HillClimberConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Mutation operator.
    pub variation: V,
}

impl<I, V> HillClimber<I, V> {
    /// Construct a `HillClimber`.
    pub fn new(config: HillClimberConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for HillClimber<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "HillClimber requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let mut initial = self.initializer.initialize(1, &mut rng);
        assert!(
            !initial.is_empty(),
            "HillClimber initializer returned no decisions"
        );
        let mut current_decision = initial.remove(0);
        let mut current_eval = problem.evaluate(&current_decision);
        let mut evaluations = 1usize;

        for _ in 0..self.config.iterations {
            let parents = vec![current_decision.clone()];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(
                !children.is_empty(),
                "HillClimber variation returned no children"
            );
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate(&child_decision);
            evaluations += 1;

            let child_better = match (child_eval.is_feasible(), current_eval.is_feasible()) {
                (true, false) => true,
                (false, true) => false,
                (false, false) => {
                    child_eval.constraint_violation < current_eval.constraint_violation
                }
                (true, true) => match direction {
                    Direction::Minimize => child_eval.objectives[0] < current_eval.objectives[0],
                    Direction::Maximize => child_eval.objectives[0] > current_eval.objectives[0],
                },
            };
            if child_better {
                current_decision = child_decision;
                current_eval = child_eval;
            }
        }

        let best = Candidate::new(current_decision, current_eval);
        let population = Population::new(vec![best.clone()]);
        let front = vec![best.clone()];
        OptimizationResult::new(
            population,
            front,
            Some(best),
            evaluations,
            self.config.iterations,
        )
    }
}

#[cfg(feature = "async")]
impl<I, V> HillClimber<I, V> {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` is mostly inert here because HillClimber evaluates
    /// one child per iteration; it's accepted for API parity with other
    /// algorithms.
    pub async fn run_async<P>(
        &mut self,
        problem: &P,
        concurrency: usize,
    ) -> OptimizationResult<P::Decision>
    where
        P: crate::core::async_problem::AsyncProblem,
        I: Initializer<P::Decision>,
        V: Variation<P::Decision>,
    {
        let _ = concurrency;
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "HillClimber requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let mut initial = self.initializer.initialize(1, &mut rng);
        assert!(
            !initial.is_empty(),
            "HillClimber initializer returned no decisions"
        );
        let mut current_decision = initial.remove(0);
        let mut current_eval = problem.evaluate_async(&current_decision).await;
        let mut evaluations = 1usize;

        for _ in 0..self.config.iterations {
            let parents = vec![current_decision.clone()];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(
                !children.is_empty(),
                "HillClimber variation returned no children"
            );
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate_async(&child_decision).await;
            evaluations += 1;

            let child_better = match (child_eval.is_feasible(), current_eval.is_feasible()) {
                (true, false) => true,
                (false, true) => false,
                (false, false) => {
                    child_eval.constraint_violation < current_eval.constraint_violation
                }
                (true, true) => match direction {
                    Direction::Minimize => child_eval.objectives[0] < current_eval.objectives[0],
                    Direction::Maximize => child_eval.objectives[0] > current_eval.objectives[0],
                },
            };
            if child_better {
                current_decision = child_decision;
                current_eval = child_eval;
            }
        }

        let best = Candidate::new(current_decision, current_eval);
        let population = Population::new(vec![best.clone()]);
        let front = vec![best.clone()];
        OptimizationResult::new(
            population,
            front,
            Some(best),
            evaluations,
            self.config.iterations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{GaussianMutation, RealBounds};
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(seed: u64) -> HillClimber<RealBounds, GaussianMutation> {
        HillClimber::new(
            HillClimberConfig {
                iterations: 500,
                seed,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.3 },
        )
    }

    #[test]
    fn finds_minimum_of_sphere() {
        let mut opt = make_optimizer(1);
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-2,
            "got f = {}",
            best.evaluation.objectives[0]
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut a = make_optimizer(99);
        let mut b = make_optimizer(99);
        let ra = a.run(&Sphere1D);
        let rb = b.run(&Sphere1D);
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
