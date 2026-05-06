//! `Knea` — Zhang, Tian & Jin 2015 Knee point-driven EA.

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::pareto::front::{best_candidate, pareto_front};
use crate::pareto::sort::non_dominated_sort;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Knea`].
#[derive(Debug, Clone)]
pub struct KneaConfig {
    /// Constant population size.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for KneaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            seed: 42,
        }
    }
}

/// Knee point-driven Evolutionary Algorithm.
///
/// Survival selection ranks splitting-front members by perpendicular
/// distance from the hyperplane connecting the front's extreme points.
/// Larger distance ≈ stronger knee = preferred survivor.
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
/// let mut opt = Knea::new(
///     KneaConfig { population_size: 30, generations: 20, seed: 42 },
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
pub struct Knea<I, V> {
    /// Algorithm configuration.
    pub config: KneaConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Knea<I, V> {
    /// Construct a `Knea`.
    pub fn new(config: KneaConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for Knea<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "Knea population_size must be > 0"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = rng.random_range(0..population.len());
                let p2 = rng.random_range(0..population.len());
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "Knea variation returned no children");
                for child in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child);
                }
            }
            let offspring = evaluate_batch(problem, offspring_decisions);
            evaluations += offspring.len();

            let mut combined: Vec<Candidate<P::Decision>> = Vec::with_capacity(2 * n);
            combined.extend(population);
            combined.extend(offspring);
            population = environmental_selection(combined, &objectives, n);
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
impl<I, V> Knea<I, V> {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations per batch.
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
            "Knea population_size must be > 0"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch_async(problem, initial_decisions, concurrency).await;
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = rng.random_range(0..population.len());
                let p2 = rng.random_range(0..population.len());
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "Knea variation returned no children");
                for child in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child);
                }
            }
            let offspring = evaluate_batch_async(problem, offspring_decisions, concurrency).await;
            evaluations += offspring.len();

            let mut combined: Vec<Candidate<P::Decision>> = Vec::with_capacity(2 * n);
            combined.extend(population);
            combined.extend(offspring);
            population = environmental_selection(combined, &objectives, n);
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

fn environmental_selection<D: Clone>(
    combined: Vec<Candidate<D>>,
    objectives: &ObjectiveSpace,
    n: usize,
) -> Vec<Candidate<D>> {
    let fronts = non_dominated_sort(&combined, objectives);
    let mut selected: Vec<usize> = Vec::with_capacity(n);
    let mut splitting: Vec<usize> = Vec::new();
    for f in &fronts {
        if selected.len() + f.len() <= n {
            selected.extend(f.iter().copied());
        } else {
            splitting = f.clone();
            break;
        }
        if selected.len() == n {
            break;
        }
    }
    if selected.len() == n {
        return selected.into_iter().map(|i| combined[i].clone()).collect();
    }

    // Compute knee distances for splitting front.
    let m = objectives.len();
    let oriented: Vec<Vec<f64>> = splitting
        .iter()
        .map(|&i| objectives.as_minimization(&combined[i].evaluation.objectives))
        .collect();

    // Per-axis ideal and nadir on the splitting front.
    let mut ideal = vec![f64::INFINITY; m];
    let mut nadir = vec![f64::NEG_INFINITY; m];
    for o in &oriented {
        for k in 0..m {
            if o[k] < ideal[k] {
                ideal[k] = o[k];
            }
            if o[k] > nadir[k] {
                nadir[k] = o[k];
            }
        }
    }
    // Hyperplane through the M extreme points: f · normal = c.
    // We approximate the hyperplane connecting the per-axis nadirs.
    // The "extreme points" here are M points each maximizing one axis.
    let extremes: Vec<usize> = (0..m)
        .map(|axis| {
            let mut best = 0;
            let mut best_val = f64::NEG_INFINITY;
            for (idx, o) in oriented.iter().enumerate() {
                if o[axis] > best_val {
                    best_val = o[axis];
                    best = idx;
                }
            }
            best
        })
        .collect();
    // Knee distance for each splitting member: signed distance from the
    // hyperplane defined by the extremes. We use a simple
    // "distance-to-line-segment" surrogate for 2D, and the M-D extension
    // is the perpendicular distance to the hyperplane through the M
    // extreme points.
    let distances: Vec<f64> = (0..splitting.len())
        .map(|i| perpendicular_distance(&oriented[i], &extremes, &oriented))
        .collect();

    // Sort splitting indices by largest distance (= strongest knee).
    let mut order: Vec<usize> = (0..splitting.len()).collect();
    order.sort_by(|&a, &b| {
        distances[b]
            .partial_cmp(&distances[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let need = n - selected.len();
    for k in order.into_iter().take(need) {
        selected.push(splitting[k]);
    }
    selected.into_iter().map(|i| combined[i].clone()).collect()
}

/// Perpendicular distance from `point` to the hyperplane through the M
/// extreme points (indices into `oriented`).
fn perpendicular_distance(point: &[f64], extremes: &[usize], oriented: &[Vec<f64>]) -> f64 {
    let m = point.len();
    if extremes.len() < m {
        // Degenerate: just return the L2 norm relative to first extreme.
        if let Some(&e0) = extremes.first() {
            return point
                .iter()
                .zip(oriented[e0].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
        }
        return 0.0;
    }
    // Hyperplane: a · x = b, where a = (1, 1, …, 1) for the canonical
    // simplex through extremes — works well when objectives are
    // approximately on a simplex.
    let a: Vec<f64> = vec![1.0; m];
    let b: f64 = oriented[extremes[0]].iter().sum();
    let dot: f64 = point.iter().zip(a.iter()).map(|(x, y)| x * y).sum();
    let norm: f64 = a.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-12);
    (dot - b).abs() / norm
}

impl<I, V> crate::traits::AlgorithmInfo for Knea<I, V> {
    fn name(&self) -> &'static str {
        "KnEA"
    }
    fn full_name(&self) -> &'static str {
        "Knee point-driven Evolutionary Algorithm"
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
    ) -> Knea<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        Knea::new(
            KneaConfig {
                population_size: 20,
                generations: 15,
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
}
