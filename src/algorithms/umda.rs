//! `Umda` — Mühlenbein 1997 Univariate Marginal Distribution Algorithm for
//! binary (`Vec<bool>`) decisions.

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::pareto::front::best_candidate;
use crate::traits::Optimizer;

/// Configuration for [`Umda`].
#[derive(Debug, Clone)]
pub struct UmdaConfig {
    /// Sample size per generation.
    pub population_size: usize,
    /// Number of top members to use for the marginal estimate.
    pub selected_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Number of bits in each decision.
    pub bits: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for UmdaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            selected_size: 50,
            generations: 50,
            bits: 32,
            seed: 42,
        }
    }
}

/// Univariate Marginal Distribution Algorithm.
///
/// `Vec<bool>` decisions only; single-objective only. Each generation
/// estimates per-bit marginal probabilities from the top `selected_size`
/// members and samples the next population from the resulting independent
/// Bernoulli vector. Probabilities are clamped to
/// `[1 / (2·selected_size), 1 - 1 / (2·selected_size)]` (Laplace-style
/// smoothing) so the population never collapses to a single deterministic
/// string.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// struct OneMax;
/// impl Problem for OneMax {
///     type Decision = Vec<bool>;
///     fn objectives(&self) -> ObjectiveSpace {
///         ObjectiveSpace::new(vec![Objective::maximize("ones")])
///     }
///     fn evaluate(&self, x: &Vec<bool>) -> Evaluation {
///         Evaluation::new(vec![x.iter().filter(|b| **b).count() as f64])
///     }
/// }
///
/// let mut opt = Umda::new(UmdaConfig {
///     population_size: 50,
///     selected_size: 20,
///     generations: 30,
///     bits: 16,
///     seed: 42,
/// });
/// let r = opt.run(&OneMax);
/// // OneMax with 16 bits: optimum is 16. UMDA should be very close.
/// assert!(r.best.unwrap().evaluation.objectives[0] >= 14.0);
/// ```
#[derive(Debug, Clone)]
pub struct Umda {
    /// Algorithm configuration.
    pub config: UmdaConfig,
}

impl Umda {
    /// Construct a `Umda`.
    pub fn new(config: UmdaConfig) -> Self {
        Self { config }
    }
}

impl<P> Optimizer<P> for Umda
where
    P: Problem<Decision = Vec<bool>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size >= 2,
            "Umda population_size must be >= 2"
        );
        assert!(
            self.config.selected_size >= 1,
            "Umda selected_size must be >= 1",
        );
        assert!(
            self.config.selected_size <= self.config.population_size,
            "Umda selected_size must be <= population_size",
        );
        assert!(self.config.bits >= 1, "Umda bits must be >= 1");
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "Umda requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let n = self.config.population_size;
        let bits = self.config.bits;
        let mu = self.config.selected_size;
        let mut rng = rng_from_seed(self.config.seed);

        // Initial sample: uniform Bernoulli(0.5) across all bits.
        let mut decisions: Vec<Vec<bool>> = (0..n)
            .map(|_| (0..bits).map(|_| rng.random_bool(0.5)).collect())
            .collect();
        let mut population = evaluate_batch(problem, decisions.clone());
        let mut evaluations = population.len();

        let smoothing = 1.0 / (2.0 * mu as f64);
        let prob_min = smoothing;
        let prob_max = 1.0 - smoothing;

        let mut best_seen: Option<Candidate<Vec<bool>>> = None;
        for c in &population {
            let beats = match &best_seen {
                None => true,
                Some(b) => better_than_so(&c.evaluation, &b.evaluation, direction),
            };
            if beats {
                best_seen = Some(c.clone());
            }
        }

        for _ in 0..self.config.generations {
            // --- Phase 1: select top μ members ---
            let mut order: Vec<usize> = (0..population.len()).collect();
            order.sort_by(|&a, &b| {
                compare_so(
                    &population[a].evaluation,
                    &population[b].evaluation,
                    direction,
                )
            });
            let selected: Vec<&Candidate<Vec<bool>>> =
                order.iter().take(mu).map(|&i| &population[i]).collect();

            // --- Phase 2: estimate per-bit marginals ---
            let mut probs = vec![0.0_f64; bits];
            for c in &selected {
                for (i, b) in c.decision.iter().enumerate() {
                    if *b {
                        probs[i] += 1.0;
                    }
                }
            }
            for p in probs.iter_mut() {
                *p = (*p / mu as f64).clamp(prob_min, prob_max);
            }

            // --- Phase 3: sample a new population (uses RNG serially) ---
            decisions = (0..n)
                .map(|_| probs.iter().map(|&p| rng.random_bool(p)).collect())
                .collect();

            // --- Phase 4: evaluate (parallel-friendly) ---
            population = evaluate_batch(problem, decisions.clone());
            evaluations += population.len();

            // Track best.
            for c in &population {
                let beats = match &best_seen {
                    None => true,
                    Some(b) => better_than_so(&c.evaluation, &b.evaluation, direction),
                };
                if beats {
                    best_seen = Some(c.clone());
                }
            }
        }

        let best = best_seen.expect("at least one generation evaluated");
        let final_pop = vec![best.clone()];
        let front = vec![best.clone()];
        let best_opt = best_candidate(&final_pop, &objectives);
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best_opt,
            evaluations,
            self.config.generations,
        )
    }
}

#[cfg(feature = "async")]
impl Umda {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations per batch (initial
    /// population and per-generation samples).
    pub async fn run_async<P>(
        &mut self,
        problem: &P,
        concurrency: usize,
    ) -> OptimizationResult<Vec<bool>>
    where
        P: crate::core::async_problem::AsyncProblem<Decision = Vec<bool>>,
    {
        use crate::algorithms::parallel_eval_async::evaluate_batch_async;

        assert!(
            self.config.population_size >= 2,
            "Umda population_size must be >= 2"
        );
        assert!(
            self.config.selected_size >= 1,
            "Umda selected_size must be >= 1",
        );
        assert!(
            self.config.selected_size <= self.config.population_size,
            "Umda selected_size must be <= population_size",
        );
        assert!(self.config.bits >= 1, "Umda bits must be >= 1");
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "Umda requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let n = self.config.population_size;
        let bits = self.config.bits;
        let mu = self.config.selected_size;
        let mut rng = rng_from_seed(self.config.seed);

        let mut decisions: Vec<Vec<bool>> = (0..n)
            .map(|_| (0..bits).map(|_| rng.random_bool(0.5)).collect())
            .collect();
        let mut population = evaluate_batch_async(problem, decisions.clone(), concurrency).await;
        let mut evaluations = population.len();

        let smoothing = 1.0 / (2.0 * mu as f64);
        let prob_min = smoothing;
        let prob_max = 1.0 - smoothing;

        let mut best_seen: Option<Candidate<Vec<bool>>> = None;
        for c in &population {
            let beats = match &best_seen {
                None => true,
                Some(b) => better_than_so(&c.evaluation, &b.evaluation, direction),
            };
            if beats {
                best_seen = Some(c.clone());
            }
        }

        for _ in 0..self.config.generations {
            let mut order: Vec<usize> = (0..population.len()).collect();
            order.sort_by(|&a, &b| {
                compare_so(
                    &population[a].evaluation,
                    &population[b].evaluation,
                    direction,
                )
            });
            let selected: Vec<&Candidate<Vec<bool>>> =
                order.iter().take(mu).map(|&i| &population[i]).collect();

            let mut probs = vec![0.0_f64; bits];
            for c in &selected {
                for (i, b) in c.decision.iter().enumerate() {
                    if *b {
                        probs[i] += 1.0;
                    }
                }
            }
            for p in probs.iter_mut() {
                *p = (*p / mu as f64).clamp(prob_min, prob_max);
            }

            decisions = (0..n)
                .map(|_| probs.iter().map(|&p| rng.random_bool(p)).collect())
                .collect();

            population = evaluate_batch_async(problem, decisions.clone(), concurrency).await;
            evaluations += population.len();

            for c in &population {
                let beats = match &best_seen {
                    None => true,
                    Some(b) => better_than_so(&c.evaluation, &b.evaluation, direction),
                };
                if beats {
                    best_seen = Some(c.clone());
                }
            }
        }

        let best = best_seen.expect("at least one generation evaluated");
        let final_pop = vec![best.clone()];
        let front = vec![best.clone()];
        let best_opt = best_candidate(&final_pop, &objectives);
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best_opt,
            evaluations,
            self.config.generations,
        )
    }
}

fn compare_so(
    a: &crate::core::evaluation::Evaluation,
    b: &crate::core::evaluation::Evaluation,
    direction: Direction,
) -> std::cmp::Ordering {
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

fn better_than_so(
    a: &crate::core::evaluation::Evaluation,
    b: &crate::core::evaluation::Evaluation,
    direction: Direction,
) -> bool {
    compare_so(a, b, direction) == std::cmp::Ordering::Less
}

impl crate::traits::AlgorithmInfo for Umda {
    fn name(&self) -> &'static str {
        "UMDA"
    }
    fn full_name(&self) -> &'static str {
        "Univariate Marginal Distribution Algorithm"
    }
    fn seed(&self) -> Option<u64> {
        Some(self.config.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Objective, ObjectiveSpace};

    /// OneMax: maximize the sum of true bits.
    struct OneMax {
        #[allow(dead_code)]
        bits: usize,
    }
    impl Problem for OneMax {
        type Decision = Vec<bool>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::maximize("bits")])
        }

        fn evaluate(&self, x: &Vec<bool>) -> Evaluation {
            let count = x.iter().filter(|b| **b).count();
            Evaluation::new(vec![count as f64])
        }
    }

    /// Trivial multi-objective problem to exercise the panic.
    struct DummyMo;
    impl Problem for DummyMo {
        type Decision = Vec<bool>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("a"), Objective::minimize("b")])
        }

        fn evaluate(&self, _x: &Vec<bool>) -> Evaluation {
            Evaluation::new(vec![0.0, 0.0])
        }
    }

    #[test]
    fn solves_onemax_20() {
        let problem = OneMax { bits: 20 };
        let mut opt = Umda::new(UmdaConfig {
            population_size: 50,
            selected_size: 20,
            generations: 30,
            bits: 20,
            seed: 1,
        });
        let r = opt.run(&problem);
        let best = r.best.unwrap();
        assert_eq!(best.evaluation.objectives[0], 20.0);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let problem = OneMax { bits: 16 };
        let cfg = UmdaConfig {
            population_size: 30,
            selected_size: 10,
            generations: 10,
            bits: 16,
            seed: 99,
        };
        let mut a = Umda::new(cfg.clone());
        let mut b = Umda::new(cfg);
        let ra = a.run(&problem);
        let rb = b.run(&problem);
        assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    #[should_panic(expected = "exactly one objective")]
    fn multi_objective_panics() {
        let mut opt = Umda::new(UmdaConfig {
            population_size: 10,
            selected_size: 5,
            generations: 1,
            bits: 4,
            seed: 0,
        });
        let _ = opt.run(&DummyMo);
    }

    // ---- Mutation-test pinned helpers --------------------------------------

    #[test]
    fn compare_so_feasibility_first_and_direction() {
        let feasible = Evaluation::new(vec![100.0]);
        let infeasible = Evaluation::constrained(vec![0.0], 1.0);
        assert_eq!(compare_so(&feasible, &infeasible, Direction::Minimize), std::cmp::Ordering::Less);
        let lo = Evaluation::new(vec![1.0]);
        let hi = Evaluation::new(vec![2.0]);
        assert_eq!(compare_so(&lo, &hi, Direction::Minimize), std::cmp::Ordering::Less);
        assert_eq!(compare_so(&lo, &hi, Direction::Maximize), std::cmp::Ordering::Greater);
        let v_lo = Evaluation::constrained(vec![0.0], 0.2);
        let v_hi = Evaluation::constrained(vec![0.0], 0.8);
        assert_eq!(compare_so(&v_lo, &v_hi, Direction::Minimize), std::cmp::Ordering::Less);
    }

    #[test]
    fn better_than_so_is_strict_less() {
        let lo = Evaluation::new(vec![1.0]);
        let hi = Evaluation::new(vec![2.0]);
        assert!(better_than_so(&lo, &hi, Direction::Minimize));
        assert!(!better_than_so(&hi, &lo, Direction::Minimize));
        let eq = Evaluation::new(vec![1.0]);
        assert!(!better_than_so(&lo, &eq, Direction::Minimize));
    }
}
