//! Pareto Archived Evolution Strategy — a small (1+1)-with-archive optimizer.

use crate::core::candidate::Candidate;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::pareto::archive::ParetoArchive;
use crate::pareto::dominance::{Dominance, pareto_compare};
use crate::pareto::front::{best_candidate, pareto_front};
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Paes`].
#[derive(Debug, Clone)]
pub struct PaesConfig {
    /// Number of mutation iterations.
    pub iterations: usize,
    /// Maximum size of the Pareto archive.
    pub archive_size: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for PaesConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            archive_size: 100,
            seed: 42,
        }
    }
}

/// A simple Pareto Archived Evolution Strategy.
///
/// One current candidate, one mutation per iteration, one bounded archive.
/// Intentionally a readable baseline rather than a research-perfect PAES
/// (spec §12.2).
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// struct Schaffer;
/// impl Problem for Schaffer {
///     type Decision = Vec<f64>;
///     fn objectives(&self) -> ObjectiveSpace {
///         ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
///     }
///     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
///         Evaluation::new(vec![x[0] * x[0], (x[0] - 2.0).powi(2)])
///     }
/// }
///
/// let mut opt = Paes::new(
///     PaesConfig { iterations: 200, archive_size: 30, seed: 42 },
///     RealBounds::new(vec![(-5.0, 5.0)]),
///     GaussianMutation { sigma: 0.3 },
/// );
/// let r = opt.run(&Schaffer);
/// assert!(!r.pareto_front.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct Paes<I, V> {
    /// Algorithm configuration.
    pub config: PaesConfig,
    /// How the initial decision is sampled.
    pub initializer: I,
    /// How children are produced from the current decision.
    pub variation: V,
}

impl<I, V> Paes<I, V> {
    /// Construct a `Paes` optimizer.
    pub fn new(config: PaesConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for Paes<I, V>
where
    P: Problem,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.archive_size > 0,
            "PAES archive_size must be greater than 0",
        );

        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let mut initial = self.initializer.initialize(1, &mut rng);
        assert!(
            !initial.is_empty(),
            "PAES initializer returned no decisions",
        );
        let mut current_decision = initial.remove(0);
        let mut current_eval = problem.evaluate(&current_decision);
        let mut evaluations = 1usize;

        let mut archive = ParetoArchive::new(objectives.clone());
        archive.insert(Candidate::new(
            current_decision.clone(),
            current_eval.clone(),
        ));

        for _ in 0..self.config.iterations {
            let parents = vec![current_decision.clone()];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(!children.is_empty(), "PAES variation returned no children",);
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate(&child_decision);
            evaluations += 1;

            match pareto_compare(&child_eval, &current_eval, &objectives) {
                Dominance::Dominates => {
                    current_decision = child_decision.clone();
                    current_eval = child_eval.clone();
                }
                Dominance::DominatedBy => {
                    // Stay at current.
                }
                Dominance::NonDominated | Dominance::Equal => {
                    // v1: move to child on non-dominated comparison.
                    current_decision = child_decision.clone();
                    current_eval = child_eval.clone();
                }
            }

            archive.insert(Candidate::new(child_decision, child_eval));
            archive.insert(Candidate::new(
                current_decision.clone(),
                current_eval.clone(),
            ));
            archive.truncate(self.config.archive_size);
        }

        let members = archive.into_vec();
        let front = pareto_front(&members, &objectives);
        let best = best_candidate(&members, &objectives);
        OptimizationResult::new(
            Population::new(members),
            front,
            best,
            evaluations,
            self.config.iterations,
        )
    }
}

#[cfg(feature = "async")]
impl<I, V> Paes<I, V> {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` is mostly inert here because PAES evaluates one
    /// child per iteration; it's accepted for API parity with other
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
        assert!(
            self.config.archive_size > 0,
            "PAES archive_size must be greater than 0",
        );

        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let mut initial = self.initializer.initialize(1, &mut rng);
        assert!(
            !initial.is_empty(),
            "PAES initializer returned no decisions",
        );
        let mut current_decision = initial.remove(0);
        let mut current_eval = problem.evaluate_async(&current_decision).await;
        let mut evaluations = 1usize;

        let mut archive = ParetoArchive::new(objectives.clone());
        archive.insert(Candidate::new(
            current_decision.clone(),
            current_eval.clone(),
        ));

        for _ in 0..self.config.iterations {
            let parents = vec![current_decision.clone()];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(!children.is_empty(), "PAES variation returned no children",);
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate_async(&child_decision).await;
            evaluations += 1;

            match pareto_compare(&child_eval, &current_eval, &objectives) {
                Dominance::Dominates => {
                    current_decision = child_decision.clone();
                    current_eval = child_eval.clone();
                }
                Dominance::DominatedBy => {
                    // Stay at current.
                }
                Dominance::NonDominated | Dominance::Equal => {
                    current_decision = child_decision.clone();
                    current_eval = child_eval.clone();
                }
            }

            archive.insert(Candidate::new(child_decision, child_eval));
            archive.insert(Candidate::new(
                current_decision.clone(),
                current_eval.clone(),
            ));
            archive.truncate(self.config.archive_size);
        }

        let members = archive.into_vec();
        let front = pareto_front(&members, &objectives);
        let best = best_candidate(&members, &objectives);
        OptimizationResult::new(
            Population::new(members),
            front,
            best,
            evaluations,
            self.config.iterations,
        )
    }
}

impl<I, V> crate::traits::AlgorithmInfo for Paes<I, V> {
    fn name(&self) -> &'static str {
        "Paes"
    }
    fn seed(&self) -> Option<u64> {
        Some(self.config.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{GaussianMutation, RealBounds};
    use crate::tests_support::{SchafferN1, Sphere1D};

    #[test]
    fn produces_at_least_one_candidate() {
        let mut opt = Paes::new(
            PaesConfig {
                iterations: 50,
                archive_size: 16,
                seed: 1,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.3 },
        );
        let r = opt.run(&SchafferN1);
        assert!(!r.population.is_empty());
        assert!(!r.pareto_front.is_empty());
    }

    #[test]
    fn archive_size_respected() {
        let mut opt = Paes::new(
            PaesConfig {
                iterations: 200,
                archive_size: 8,
                seed: 2,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.2 },
        );
        let r = opt.run(&SchafferN1);
        assert!(r.population.len() <= 8);
    }

    #[test]
    fn single_objective_returns_best() {
        let mut opt = Paes::new(
            PaesConfig {
                iterations: 200,
                archive_size: 8,
                seed: 3,
            },
            RealBounds::new(vec![(-2.0, 2.0)]),
            GaussianMutation { sigma: 0.1 },
        );
        let r = opt.run(&Sphere1D);
        assert!(r.best.is_some());
    }
}
