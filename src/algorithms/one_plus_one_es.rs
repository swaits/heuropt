//! `OnePlusOneEs` — the (1+1) evolution strategy with Rechenberg's
//! one-fifth success rule for σ adaptation.

use rand_distr::{Distribution, Normal};

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::operators::real::RealBounds;
use crate::traits::Optimizer;

/// Configuration for [`OnePlusOneEs`].
#[derive(Debug, Clone)]
pub struct OnePlusOneEsConfig {
    /// Number of mutation iterations.
    pub iterations: usize,
    /// Initial mutation step size (`σ_0`).
    pub initial_sigma: f64,
    /// Number of recent iterations the success-rate is computed over.
    /// The classic value is 10·dim; 50 is a fine default for low-dim
    /// problems.
    pub adaptation_period: usize,
    /// Step-size multiplier when the success rate exceeds 1/5. Reciprocal
    /// is applied when the rate is below 1/5. Rechenberg's analytical
    /// derivation gives ≈ `0.817^(-1/n)` for dim n; 1.22 is a popular
    /// dimension-agnostic value.
    pub step_increase: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for OnePlusOneEsConfig {
    fn default() -> Self {
        Self {
            iterations: 5_000,
            initial_sigma: 0.5,
            adaptation_period: 50,
            step_increase: 1.22,
            seed: 42,
        }
    }
}

/// (1+1)-ES with the one-fifth rule: tiny, parameter-light continuous
/// optimizer. `Vec<f64>` decisions only; single-objective only.
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
/// let mut opt = OnePlusOneEs::new(
///     OnePlusOneEsConfig {
///         iterations: 1_000,
///         initial_sigma: 0.5,
///         adaptation_period: 50,
///         step_increase: 1.22,
///         seed: 42,
///     },
///     RealBounds::new(vec![(-5.0, 5.0); 3]),
/// );
/// let r = opt.run(&Sphere);
/// assert!(r.best.unwrap().evaluation.objectives[0] < 1e-3);
/// ```
#[derive(Debug, Clone)]
pub struct OnePlusOneEs {
    /// Algorithm configuration.
    pub config: OnePlusOneEsConfig,
    /// Per-variable bounds — used to seed the parent at the box midpoint
    /// and clamp every mutated child.
    pub bounds: RealBounds,
}

impl OnePlusOneEs {
    /// Construct a `OnePlusOneEs`.
    pub fn new(config: OnePlusOneEsConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for OnePlusOneEs
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.initial_sigma > 0.0,
            "OnePlusOneEs initial_sigma must be > 0"
        );
        assert!(
            self.config.step_increase > 1.0,
            "OnePlusOneEs step_increase must be > 1",
        );
        assert!(
            self.config.adaptation_period >= 1,
            "OnePlusOneEs adaptation_period must be >= 1",
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "OnePlusOneEs requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        // Seed parent at midpoint of bounds.
        let mut parent: Vec<f64> = self
            .bounds
            .bounds
            .iter()
            .map(|&(lo, hi)| 0.5 * (lo + hi))
            .collect();
        let mut parent_eval = problem.evaluate(&parent);
        let mut evaluations = 1usize;

        let mut sigma = self.config.initial_sigma;
        let mut window = std::collections::VecDeque::with_capacity(self.config.adaptation_period);

        for _ in 0..self.config.iterations {
            let normal = Normal::new(0.0, sigma).expect("Normal::new(0, sigma)");
            let mut child = parent.clone();
            for (j, x) in child.iter_mut().enumerate() {
                let (lo, hi) = self.bounds.bounds[j];
                *x = (*x + normal.sample(&mut rng)).clamp(lo, hi);
            }
            let child_eval = problem.evaluate(&child);
            evaluations += 1;

            // Accept if not strictly worse (so neutral moves are kept and
            // can drive σ up when on a plateau).
            let accepted = !worse_than(&child_eval, &parent_eval, direction);
            if accepted {
                parent = child;
                parent_eval = child_eval;
            }

            // Update success window.
            window.push_back(if accepted { 1u8 } else { 0u8 });
            if window.len() > self.config.adaptation_period {
                window.pop_front();
            }
            // Apply one-fifth rule once we have a full window.
            if window.len() == self.config.adaptation_period {
                let success_count: usize = window.iter().map(|&b| b as usize).sum();
                let rate = success_count as f64 / window.len() as f64;
                if rate > 0.2 {
                    sigma *= self.config.step_increase;
                } else if rate < 0.2 {
                    sigma /= self.config.step_increase;
                }
            }
        }

        let best = Candidate::new(parent, parent_eval);
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

fn worse_than(a: &Evaluation, b: &Evaluation, direction: Direction) -> bool {
    match (a.is_feasible(), b.is_feasible()) {
        (false, true) => true,
        (true, false) => false,
        (false, false) => a.constraint_violation > b.constraint_violation,
        (true, true) => match direction {
            Direction::Minimize => a.objectives[0] > b.objectives[0],
            Direction::Maximize => a.objectives[0] < b.objectives[0],
        },
    }
}

#[cfg(feature = "async")]
impl OnePlusOneEs {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` is mostly inert here because (1+1)-ES evaluates
    /// one child per iteration; it's accepted for API parity with
    /// other algorithms.
    pub async fn run_async<P>(
        &mut self,
        problem: &P,
        concurrency: usize,
    ) -> OptimizationResult<Vec<f64>>
    where
        P: crate::core::async_problem::AsyncProblem<Decision = Vec<f64>>,
    {
        let _ = concurrency;
        assert!(
            self.config.initial_sigma > 0.0,
            "OnePlusOneEs initial_sigma must be > 0"
        );
        assert!(
            self.config.step_increase > 1.0,
            "OnePlusOneEs step_increase must be > 1",
        );
        assert!(
            self.config.adaptation_period >= 1,
            "OnePlusOneEs adaptation_period must be >= 1",
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "OnePlusOneEs requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let mut parent: Vec<f64> = self
            .bounds
            .bounds
            .iter()
            .map(|&(lo, hi)| 0.5 * (lo + hi))
            .collect();
        let mut parent_eval = problem.evaluate_async(&parent).await;
        let mut evaluations = 1usize;

        let mut sigma = self.config.initial_sigma;
        let mut window = std::collections::VecDeque::with_capacity(self.config.adaptation_period);

        for _ in 0..self.config.iterations {
            let normal = Normal::new(0.0, sigma).expect("Normal::new(0, sigma)");
            let mut child = parent.clone();
            for (j, x) in child.iter_mut().enumerate() {
                let (lo, hi) = self.bounds.bounds[j];
                *x = (*x + normal.sample(&mut rng)).clamp(lo, hi);
            }
            let child_eval = problem.evaluate_async(&child).await;
            evaluations += 1;

            let accepted = !worse_than(&child_eval, &parent_eval, direction);
            if accepted {
                parent = child;
                parent_eval = child_eval;
            }

            window.push_back(if accepted { 1u8 } else { 0u8 });
            if window.len() > self.config.adaptation_period {
                window.pop_front();
            }
            if window.len() == self.config.adaptation_period {
                let success_count: usize = window.iter().map(|&b| b as usize).sum();
                let rate = success_count as f64 / window.len() as f64;
                if rate > 0.2 {
                    sigma *= self.config.step_increase;
                } else if rate < 0.2 {
                    sigma /= self.config.step_increase;
                }
            }
        }

        let best = Candidate::new(parent, parent_eval);
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

impl crate::traits::AlgorithmInfo for OnePlusOneEs {
    fn name(&self) -> &'static str {
        "OnePlusOneEs"
    }
    fn seed(&self) -> Option<u64> {
        Some(self.config.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(seed: u64) -> OnePlusOneEs {
        OnePlusOneEs::new(
            OnePlusOneEsConfig {
                iterations: 2_000,
                initial_sigma: 1.0,
                adaptation_period: 30,
                step_increase: 1.22,
                seed,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
        )
    }

    #[test]
    fn finds_minimum_of_sphere() {
        let mut opt = make_optimizer(1);
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-6,
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
}
