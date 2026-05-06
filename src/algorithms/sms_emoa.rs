//! `SmsEmoa` — Beume, Naujoks & Emmerich 2007 S-Metric Selection EMOA.

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::metrics::hypervolume::hypervolume_nd_from_evaluations;
use crate::pareto::front::{best_candidate, pareto_front};
use crate::pareto::sort::non_dominated_sort;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`SmsEmoa`].
#[derive(Debug, Clone)]
pub struct SmsEmoaConfig {
    /// Constant population size carried across generations.
    pub population_size: usize,
    /// Number of generations. SMS-EMOA is steady-state — each generation
    /// produces and evaluates exactly one child.
    pub generations: usize,
    /// Reference point used for hypervolume contribution computations.
    /// Must have one entry per objective; should be worse than every
    /// realistic objective value.
    pub reference_point: Vec<f64>,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for SmsEmoaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 1_000,
            reference_point: vec![11.0, 11.0],
            seed: 42,
        }
    }
}

/// SMS-EMOA: a steady-state MOEA that selects survivors by hypervolume
/// contribution.
///
/// Each generation produces a single offspring via the user's variation
/// operator and replaces the worst-contribution member of the worst
/// non-dominated front. Excellent convergence quality at the price of
/// quadratic-in-N hypervolume evaluations per generation, so practical
/// up to ~4 objectives at population sizes ≤ 200.
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
/// let bounds = vec![(-5.0_f64, 5.0_f64)];
/// let mut opt = SmsEmoa::new(
///     SmsEmoaConfig {
///         population_size: 20,
///         generations: 100,
///         reference_point: vec![30.0, 30.0],
///         seed: 42,
///     },
///     RealBounds::new(bounds.clone()),
///     CompositeVariation {
///         crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
///         mutation:  PolynomialMutation::new(bounds, 20.0, 1.0),
///     },
/// );
/// let r = opt.run(&Schaffer);
/// assert!(!r.pareto_front.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct SmsEmoa<I, V> {
    /// Algorithm configuration.
    pub config: SmsEmoaConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> SmsEmoa<I, V> {
    /// Construct a `SmsEmoa`.
    pub fn new(config: SmsEmoaConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for SmsEmoa<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "SmsEmoa population_size must be > 0"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        assert_eq!(
            self.config.reference_point.len(),
            objectives.len(),
            "SmsEmoa reference_point.len() must equal number of objectives",
        );
        let reference = self.config.reference_point.clone();
        let mut rng = rng_from_seed(self.config.seed);

        // Initial population.
        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> = initial_decisions
            .into_iter()
            .map(|d| {
                let e = problem.evaluate(&d);
                Candidate::new(d, e)
            })
            .collect();
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            // --- One offspring (steady-state) ---
            let p1 = rng.random_range(0..population.len());
            let p2 = rng.random_range(0..population.len());
            let parents = vec![
                population[p1].decision.clone(),
                population[p2].decision.clone(),
            ];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(
                !children.is_empty(),
                "SmsEmoa variation returned no children"
            );
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate(&child_decision);
            evaluations += 1;
            let child = Candidate::new(child_decision, child_eval);

            // --- Combine and decide who to drop ---
            population.push(child);
            let drop_idx = pick_drop_index(&population, &objectives, &reference);
            population.swap_remove(drop_idx);
        }

        let front = pareto_front(&population, &objectives);
        let best = best_candidate(&population, &objectives);
        OptimizationResult::new(
            Population::new(population),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

#[cfg(feature = "async")]
impl<I, V> SmsEmoa<I, V> {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations of the initial
    /// population. Per-generation evaluations are sequential because
    /// SMS-EMOA is a steady-state algorithm (one child per generation).
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
        use crate::algorithms::parallel_eval_async::evaluate_batch_async;

        assert!(
            self.config.population_size > 0,
            "SmsEmoa population_size must be > 0"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        assert_eq!(
            self.config.reference_point.len(),
            objectives.len(),
            "SmsEmoa reference_point.len() must equal number of objectives",
        );
        let reference = self.config.reference_point.clone();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch_async(problem, initial_decisions, concurrency).await;
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            let p1 = rng.random_range(0..population.len());
            let p2 = rng.random_range(0..population.len());
            let parents = vec![
                population[p1].decision.clone(),
                population[p2].decision.clone(),
            ];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(
                !children.is_empty(),
                "SmsEmoa variation returned no children"
            );
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate_async(&child_decision).await;
            evaluations += 1;
            let child = Candidate::new(child_decision, child_eval);

            population.push(child);
            let drop_idx = pick_drop_index(&population, &objectives, &reference);
            population.swap_remove(drop_idx);
        }

        let front = pareto_front(&population, &objectives);
        let best = best_candidate(&population, &objectives);
        OptimizationResult::new(
            Population::new(population),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

/// Choose the index in `pool` whose removal is preferred per SMS-EMOA's
/// rules: drop from the worst non-dominated front; within that front,
/// drop the member whose removal increases hypervolume the most (= the
/// one with the smallest hypervolume contribution).
fn pick_drop_index<D>(
    pool: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    reference: &[f64],
) -> usize {
    let fronts = non_dominated_sort(pool, objectives);
    let worst_front = fronts
        .last()
        .expect("non_dominated_sort must return at least one front for non-empty pool");

    if worst_front.len() == 1 {
        return worst_front[0];
    }

    // Compute each candidate's hypervolume contribution = HV(front) -
    // HV(front \ {member}). Smallest contribution = drop.
    let evals: Vec<&Evaluation> = worst_front.iter().map(|&i| &pool[i].evaluation).collect();
    let total_hv = hypervolume_nd_from_evaluations(&evals, objectives, reference);

    let mut worst_idx_in_front = 0;
    let mut min_contrib = f64::INFINITY;
    for k in 0..worst_front.len() {
        let mut without: Vec<&Evaluation> = Vec::with_capacity(worst_front.len() - 1);
        for (j, &gi) in worst_front.iter().enumerate() {
            if j != k {
                without.push(&pool[gi].evaluation);
            }
        }
        let hv_without = hypervolume_nd_from_evaluations(&without, objectives, reference);
        let contrib = total_hv - hv_without;
        if contrib < min_contrib {
            min_contrib = contrib;
            worst_idx_in_front = k;
        }
    }
    worst_front[worst_idx_in_front]
}

impl<I, V> crate::traits::AlgorithmInfo for SmsEmoa<I, V> {
    fn name(&self) -> &'static str {
        "SmsEmoa"
    }
    fn seed(&self) -> Option<u64> {
        Some(self.config.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{
        CompositeVariation, PolynomialMutation, RealBounds, SimulatedBinaryCrossover,
    };
    use crate::tests_support::SchafferN1;

    fn make_optimizer(
        seed: u64,
    ) -> SmsEmoa<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        SmsEmoa::new(
            SmsEmoaConfig {
                population_size: 20,
                generations: 100,
                reference_point: vec![30.0, 30.0],
                seed,
            },
            initializer,
            variation,
        )
    }

    #[test]
    fn produces_pareto_front() {
        let mut opt = make_optimizer(1);
        let r = opt.run(&SchafferN1);
        assert_eq!(r.population.len(), 20);
        assert!(!r.pareto_front.is_empty());
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut a = make_optimizer(99);
        let mut b = make_optimizer(99);
        let ra = a.run(&SchafferN1);
        let rb = b.run(&SchafferN1);
        let oa: Vec<Vec<f64>> = ra
            .pareto_front
            .iter()
            .map(|c| c.evaluation.objectives.clone())
            .collect();
        let ob: Vec<Vec<f64>> = rb
            .pareto_front
            .iter()
            .map(|c| c.evaluation.objectives.clone())
            .collect();
        assert_eq!(oa, ob);
    }

    #[test]
    #[should_panic(expected = "population_size must be > 0")]
    fn zero_population_size_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = SmsEmoa::new(
            SmsEmoaConfig {
                population_size: 0,
                generations: 1,
                reference_point: vec![1.0, 1.0],
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }

    #[test]
    #[should_panic(expected = "reference_point.len() must equal number of objectives")]
    fn dim_mismatch_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = SmsEmoa::new(
            SmsEmoaConfig {
                population_size: 4,
                generations: 1,
                reference_point: vec![1.0, 1.0, 1.0],
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }
}
