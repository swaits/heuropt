//! `SimulatedAnnealing` — Kirkpatrick et al. 1983 SA for single-objective problems.

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`SimulatedAnnealing`].
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingConfig {
    /// Number of mutation iterations.
    pub iterations: usize,
    /// Starting temperature `T_0`. Must be positive.
    pub initial_temperature: f64,
    /// Ending temperature `T_n`. Must be positive and `<= initial_temperature`.
    pub final_temperature: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for SimulatedAnnealingConfig {
    fn default() -> Self {
        Self {
            iterations: 5_000,
            initial_temperature: 1.0,
            final_temperature: 1e-3,
            seed: 42,
        }
    }
}

/// Single-objective Simulated Annealing.
///
/// Like a hill climber, but worse moves are accepted with probability
/// `exp(-Δ / T)` where `Δ` is the (direction-aware) objective degradation
/// and `T` anneals geometrically from `initial_temperature` to
/// `final_temperature` over the iteration count. Generic over decision
/// type — pair with any `Variation` impl that returns one child per call.
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
/// let mut opt = SimulatedAnnealing::new(
///     SimulatedAnnealingConfig {
///         iterations: 2_000,
///         initial_temperature: 1.0,
///         final_temperature: 1e-3,
///         seed: 42,
///     },
///     RealBounds::new(vec![(-5.0, 5.0); 3]),
///     GaussianMutation { sigma: 0.3 },
/// );
/// let r = opt.run(&Sphere);
/// assert!(r.best.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct SimulatedAnnealing<I, V> {
    /// Algorithm configuration.
    pub config: SimulatedAnnealingConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Mutation operator.
    pub variation: V,
}

impl<I, V> SimulatedAnnealing<I, V> {
    /// Construct a `SimulatedAnnealing`.
    pub fn new(config: SimulatedAnnealingConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for SimulatedAnnealing<I, V>
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
            "SimulatedAnnealing requires exactly one objective",
        );
        assert!(
            self.config.initial_temperature > 0.0,
            "SimulatedAnnealing initial_temperature must be positive",
        );
        assert!(
            self.config.final_temperature > 0.0,
            "SimulatedAnnealing final_temperature must be positive",
        );
        assert!(
            self.config.final_temperature <= self.config.initial_temperature,
            "SimulatedAnnealing final_temperature must be <= initial_temperature",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let mut initial = self.initializer.initialize(1, &mut rng);
        assert!(
            !initial.is_empty(),
            "SimulatedAnnealing initializer returned no decisions",
        );
        let mut current_decision = initial.remove(0);
        let mut current_eval = problem.evaluate(&current_decision);
        let mut best_decision = current_decision.clone();
        let mut best_eval = current_eval.clone();
        let mut evaluations = 1usize;

        // Geometric cooling: T(k) = T_0 * (T_n / T_0)^(k / (N - 1))
        let cooling = if self.config.iterations <= 1 {
            1.0
        } else {
            (self.config.final_temperature / self.config.initial_temperature)
                .powf(1.0 / (self.config.iterations as f64 - 1.0))
        };
        let mut temperature = self.config.initial_temperature;

        for _ in 0..self.config.iterations {
            let parents = vec![current_decision.clone()];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(
                !children.is_empty(),
                "SimulatedAnnealing variation returned no children"
            );
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate(&child_decision);
            evaluations += 1;

            let accept = match (child_eval.is_feasible(), current_eval.is_feasible()) {
                (true, false) => true,
                (false, true) => false,
                (false, false) => {
                    child_eval.constraint_violation <= current_eval.constraint_violation
                }
                (true, true) => {
                    let delta = match direction {
                        Direction::Minimize => {
                            child_eval.objectives[0] - current_eval.objectives[0]
                        }
                        Direction::Maximize => {
                            current_eval.objectives[0] - child_eval.objectives[0]
                        }
                    };
                    if delta <= 0.0 {
                        true
                    } else {
                        let prob = (-delta / temperature).exp();
                        rng.random::<f64>() < prob
                    }
                }
            };

            if accept {
                current_decision = child_decision;
                current_eval = child_eval;
                if better_than(&current_eval, &best_eval, direction) {
                    best_decision = current_decision.clone();
                    best_eval = current_eval.clone();
                }
            }
            temperature *= cooling;
        }

        let best = Candidate::new(best_decision, best_eval);
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

fn better_than(
    a: &crate::core::evaluation::Evaluation,
    b: &crate::core::evaluation::Evaluation,
    direction: Direction,
) -> bool {
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

#[cfg(feature = "async")]
impl<I, V> SimulatedAnnealing<I, V> {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` is mostly inert here because SA evaluates one
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
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "SimulatedAnnealing requires exactly one objective",
        );
        assert!(
            self.config.initial_temperature > 0.0,
            "SimulatedAnnealing initial_temperature must be positive",
        );
        assert!(
            self.config.final_temperature > 0.0,
            "SimulatedAnnealing final_temperature must be positive",
        );
        assert!(
            self.config.final_temperature <= self.config.initial_temperature,
            "SimulatedAnnealing final_temperature must be <= initial_temperature",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let mut initial = self.initializer.initialize(1, &mut rng);
        assert!(
            !initial.is_empty(),
            "SimulatedAnnealing initializer returned no decisions",
        );
        let mut current_decision = initial.remove(0);
        let mut current_eval = problem.evaluate_async(&current_decision).await;
        let mut best_decision = current_decision.clone();
        let mut best_eval = current_eval.clone();
        let mut evaluations = 1usize;

        let cooling = if self.config.iterations <= 1 {
            1.0
        } else {
            (self.config.final_temperature / self.config.initial_temperature)
                .powf(1.0 / (self.config.iterations as f64 - 1.0))
        };
        let mut temperature = self.config.initial_temperature;

        for _ in 0..self.config.iterations {
            let parents = vec![current_decision.clone()];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(
                !children.is_empty(),
                "SimulatedAnnealing variation returned no children"
            );
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate_async(&child_decision).await;
            evaluations += 1;

            let accept = match (child_eval.is_feasible(), current_eval.is_feasible()) {
                (true, false) => true,
                (false, true) => false,
                (false, false) => {
                    child_eval.constraint_violation <= current_eval.constraint_violation
                }
                (true, true) => {
                    let delta = match direction {
                        Direction::Minimize => {
                            child_eval.objectives[0] - current_eval.objectives[0]
                        }
                        Direction::Maximize => {
                            current_eval.objectives[0] - child_eval.objectives[0]
                        }
                    };
                    if delta <= 0.0 {
                        true
                    } else {
                        let prob = (-delta / temperature).exp();
                        rng.random::<f64>() < prob
                    }
                }
            };

            if accept {
                current_decision = child_decision;
                current_eval = child_eval;
                if better_than(&current_eval, &best_eval, direction) {
                    best_decision = current_decision.clone();
                    best_eval = current_eval.clone();
                }
            }
            temperature *= cooling;
        }

        let best = Candidate::new(best_decision, best_eval);
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

impl<I, V> crate::traits::AlgorithmInfo for SimulatedAnnealing<I, V> {
    fn name(&self) -> &'static str {
        "Simulated Annealing"
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

    fn make_optimizer(seed: u64) -> SimulatedAnnealing<RealBounds, GaussianMutation> {
        SimulatedAnnealing::new(
            SimulatedAnnealingConfig {
                iterations: 2_000,
                initial_temperature: 1.0,
                final_temperature: 1e-4,
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
            best.evaluation.objectives[0],
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

    #[test]
    #[should_panic(expected = "initial_temperature must be positive")]
    fn zero_initial_temperature_panics() {
        let mut opt = SimulatedAnnealing::new(
            SimulatedAnnealingConfig {
                iterations: 10,
                initial_temperature: 0.0,
                final_temperature: 1e-3,
                seed: 0,
            },
            RealBounds::new(vec![(-1.0, 1.0)]),
            GaussianMutation { sigma: 0.1 },
        );
        let _ = opt.run(&Sphere1D);
    }

    // ---- Mutation-test pinned helpers --------------------------------------

    use crate::core::evaluation::Evaluation;
    use crate::core::objective::Direction;

    #[test]
    fn better_than_feasibility_first_and_direction() {
        let feasible = Evaluation::new(vec![100.0]);
        let infeasible = Evaluation::constrained(vec![0.0], 1.0);
        assert!(better_than(&feasible, &infeasible, Direction::Minimize));
        assert!(!better_than(&infeasible, &feasible, Direction::Minimize));
        let lo = Evaluation::new(vec![1.0]);
        let hi = Evaluation::new(vec![2.0]);
        assert!(better_than(&lo, &hi, Direction::Minimize));
        assert!(better_than(&hi, &lo, Direction::Maximize));
        let eq = Evaluation::new(vec![1.0]);
        assert!(!better_than(&lo, &eq, Direction::Minimize));
        let v_lo = Evaluation::constrained(vec![0.0], 0.2);
        let v_hi = Evaluation::constrained(vec![0.0], 0.8);
        assert!(better_than(&v_lo, &v_hi, Direction::Minimize));
    }
}
