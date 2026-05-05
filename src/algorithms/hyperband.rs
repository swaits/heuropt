//! `Hyperband` — Li et al. 2017 multi-fidelity hyperparameter optimizer
//! built on Successive Halving (Karnin et al. 2013).

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::Direction;
use crate::core::partial_problem::PartialProblem;
use crate::core::population::Population;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::traits::Initializer;

/// Configuration for [`Hyperband`].
#[derive(Debug, Clone)]
pub struct HyperbandConfig {
    /// Maximum fidelity budget per configuration. Common units: epochs,
    /// timesteps, simulation iterations.
    pub max_budget: f64,
    /// Reduction factor `η`. Each Successive-Halving round survives
    /// `1/η` of configurations and promotes them to `η×` budget. Li
    /// et al. recommend 3 (which gives smin=1) or 4 (slightly more
    /// aggressive promotion).
    pub eta: f64,
    /// Maximum number of brackets. The standard formula is
    /// `floor(log_η(max_budget)) + 1`; pass a larger value to allow
    /// it, smaller to truncate.
    pub max_brackets: usize,
    /// Seed for the deterministic RNG used to sample configurations.
    pub seed: u64,
}

impl Default for HyperbandConfig {
    fn default() -> Self {
        Self {
            max_budget: 81.0,
            eta: 3.0,
            max_brackets: 5,
            seed: 42,
        }
    }
}

/// Hyperband: a budget-aware single-objective optimizer for problems
/// where each evaluation can be performed at a tunable *fidelity*
/// (e.g. an ML training run for `budget` epochs).
///
/// Each "bracket" is a Successive-Halving sweep that starts with many
/// configurations at low budget and progressively promotes the top
/// `1/η` fraction to higher budgets, eliminating the rest. Hyperband
/// runs several brackets with different (configurations, budget)
/// trade-offs — early brackets favor exploration (many configs at
/// low budget), later brackets favor exploitation (fewer configs run
/// near the max budget). The single best result across all brackets
/// is returned.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
/// use heuropt::core::partial_problem::PartialProblem;
///
/// struct Tuning;
/// impl PartialProblem for Tuning {
///     type Decision = Vec<f64>;
///     fn objectives(&self) -> ObjectiveSpace {
///         ObjectiveSpace::new(vec![Objective::minimize("loss")])
///     }
///     fn evaluate_at_budget(&self, x: &Vec<f64>, budget: f64) -> Evaluation {
///         // Pretend a model where more budget = lower loss.
///         let loss = x[0].powi(2) + x[1].powi(2) + 1.0 / (budget + 1.0);
///         Evaluation::new(vec![loss])
///     }
/// }
///
/// let mut opt = Hyperband::new(
///     HyperbandConfig {
///         max_budget: 27.0,
///         eta: 3.0,
///         max_brackets: 4,
///         seed: 42,
///     },
///     RealBounds::new(vec![(-1.0, 1.0); 2]),
/// );
/// let r = opt.run(&Tuning);
/// assert!(r.best.is_some());
/// ```
pub struct Hyperband<I, D>
where
    D: Clone,
    I: Initializer<D>,
{
    /// Algorithm configuration.
    pub config: HyperbandConfig,
    /// Random configuration sampler (same trait used everywhere else).
    pub initializer: I,
    _marker: std::marker::PhantomData<D>,
}

impl<I, D> Hyperband<I, D>
where
    D: Clone,
    I: Initializer<D>,
{
    /// Construct a `Hyperband`.
    pub fn new(config: HyperbandConfig, initializer: I) -> Self {
        Self {
            config,
            initializer,
            _marker: std::marker::PhantomData,
        }
    }

    /// Run Hyperband on a multi-fidelity problem, returning the standard
    /// `OptimizationResult`. Single-objective only.
    pub fn run<P>(&mut self, problem: &P) -> OptimizationResult<D>
    where
        P: PartialProblem<Decision = D>,
    {
        assert!(
            self.config.max_budget > 0.0,
            "Hyperband max_budget must be > 0"
        );
        assert!(self.config.eta > 1.0, "Hyperband eta must be > 1");
        assert!(
            self.config.max_brackets >= 1,
            "Hyperband max_brackets must be >= 1"
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "Hyperband requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        // Number of brackets s_max = floor(log_η(max_budget)).
        let s_max = (self.config.max_budget.ln() / self.config.eta.ln()).floor() as i64;
        let s_max = (s_max as usize).min(self.config.max_brackets);

        let mut total_evaluations = 0usize;
        let mut total_iterations = 0usize;
        let mut best_seen: Option<Candidate<D>> = None;

        // Brackets are indexed s = s_max, s_max - 1, ..., 0.
        for s in (0..=s_max).rev() {
            let s_f = s as f64;
            let n =
                ((s_max as f64 + 1.0) / (s_f + 1.0) * self.config.eta.powf(s_f)).ceil() as usize;
            let r = self.config.max_budget / self.config.eta.powf(s_f);

            // Sample n configurations.
            let mut configs: Vec<D> = self.initializer.initialize(n, &mut rng);
            // SH inner loop.
            for i in 0..=s {
                let n_i = (n as f64 / self.config.eta.powi(i as i32)).floor() as usize;
                let r_i = r * self.config.eta.powi(i as i32);
                if configs.is_empty() {
                    break;
                }
                let evals: Vec<Evaluation> = configs
                    .iter()
                    .map(|c| problem.evaluate_at_budget(c, r_i))
                    .collect();
                total_evaluations += configs.len();

                // Track best.
                for (cfg, e) in configs.iter().zip(evals.iter()) {
                    let beats = match &best_seen {
                        None => true,
                        Some(b) => better(e, &b.evaluation, direction),
                    };
                    if beats {
                        best_seen = Some(Candidate::new(cfg.clone(), e.clone()));
                    }
                }
                total_iterations += 1;

                // Top n_{i+1} survive.
                let next_size = (n_i / self.config.eta as usize).max(1);
                if next_size >= configs.len() {
                    continue;
                }
                let mut order: Vec<usize> = (0..configs.len()).collect();
                order.sort_by(|&a, &b| compare(&evals[a], &evals[b], direction));
                let keep: std::collections::HashSet<usize> =
                    order.into_iter().take(next_size).collect();
                let new_configs: Vec<D> = configs
                    .into_iter()
                    .enumerate()
                    .filter_map(|(idx, c)| if keep.contains(&idx) { Some(c) } else { None })
                    .collect();
                configs = new_configs;
            }
        }

        let best = best_seen.expect("at least one bracket ran");
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

fn compare(a: &Evaluation, b: &Evaluation, direction: Direction) -> std::cmp::Ordering {
    match (a.is_feasible(), b.is_feasible()) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a
            .constraint_violation
            .partial_cmp(&b.constraint_violation)
            .unwrap_or(std::cmp::Ordering::Equal),
        (true, true) => match direction {
            Direction::Minimize => a.objectives[0]
                .partial_cmp(&b.objectives[0])
                .unwrap_or(std::cmp::Ordering::Equal),
            Direction::Maximize => b.objectives[0]
                .partial_cmp(&a.objectives[0])
                .unwrap_or(std::cmp::Ordering::Equal),
        },
    }
}

fn better(a: &Evaluation, b: &Evaluation, direction: Direction) -> bool {
    compare(a, b, direction) == std::cmp::Ordering::Less
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Objective, ObjectiveSpace};
    use crate::operators::real::RealBounds;

    /// A multi-fidelity Sphere1D where higher budgets give a less noisy
    /// estimate of `f(x) = x[0]²`.
    struct NoisySphere {
        noise_decay: f64, // higher noise_decay = less noise per unit budget
    }
    impl PartialProblem for NoisySphere {
        type Decision = Vec<f64>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }

        fn evaluate_at_budget(&self, x: &Vec<f64>, budget: f64) -> Evaluation {
            // Pure Sphere; the budget controls how much "noise" we add
            // (deterministic — no RNG so the test is reproducible).
            // Higher budget → smaller residual.
            let true_f = x[0] * x[0];
            let residual = (1.0 / (budget * self.noise_decay)).min(10.0);
            Evaluation::new(vec![true_f + residual])
        }
    }

    #[test]
    fn hyperband_finds_minimum() {
        let problem = NoisySphere { noise_decay: 1.0 };
        let mut opt = Hyperband::new(
            HyperbandConfig {
                max_budget: 81.0,
                eta: 3.0,
                max_brackets: 4,
                seed: 1,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
        );
        let r = opt.run(&problem);
        let best = r.best.unwrap();
        // The "true" minimum of Sphere is 0; but at finite budget the
        // residual term keeps it from being zero. A good run should at
        // least clearly beat random.
        assert!(
            best.evaluation.objectives[0] < 0.5,
            "got f = {}",
            best.evaluation.objectives[0],
        );
        assert!(r.evaluations > 0);
    }

    #[test]
    fn hyperband_deterministic_with_same_seed() {
        let make = || {
            Hyperband::new(
                HyperbandConfig {
                    max_budget: 27.0,
                    eta: 3.0,
                    max_brackets: 3,
                    seed: 99,
                },
                RealBounds::new(vec![(-5.0, 5.0)]),
            )
        };
        let problem = NoisySphere { noise_decay: 1.0 };
        let mut a = make();
        let mut b = make();
        let ra = a.run(&problem);
        let rb = b.run(&problem);
        assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    #[should_panic(expected = "exactly one objective")]
    fn hyperband_multi_objective_panics() {
        struct MultiObj;
        impl PartialProblem for MultiObj {
            type Decision = Vec<f64>;
            fn objectives(&self) -> ObjectiveSpace {
                ObjectiveSpace::new(vec![Objective::minimize("a"), Objective::minimize("b")])
            }
            fn evaluate_at_budget(&self, _: &Vec<f64>, _: f64) -> Evaluation {
                Evaluation::new(vec![0.0, 0.0])
            }
        }
        let mut opt = Hyperband::new(
            HyperbandConfig::default(),
            RealBounds::new(vec![(0.0, 1.0)]),
        );
        let _ = opt.run(&MultiObj);
    }
}
