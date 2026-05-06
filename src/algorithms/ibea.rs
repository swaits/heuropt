//! `Ibea` — Zitzler & Künzli 2004 Indicator-Based Evolutionary Algorithm.

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::pareto::front::{best_candidate, pareto_front};
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Ibea`].
#[derive(Debug, Clone)]
pub struct IbeaConfig {
    /// Constant population size carried across generations.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Indicator scaling factor `κ`. Default 0.05 (Zitzler & Künzli §3.2).
    pub kappa: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for IbeaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            kappa: 0.05,
            seed: 42,
        }
    }
}

/// IBEA (Indicator-Based EA) using the additive ε-indicator.
///
/// Selects survivors by their contribution to a quality indicator
/// (additive ε) rather than by dominance + crowding. On the comparison
/// harness it consistently produces the best convergence of the dominance-
/// alternative methods on smooth and disconnected fronts alike.
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
/// let mut opt = Ibea::new(
///     IbeaConfig { population_size: 30, generations: 20, kappa: 0.05, seed: 42 },
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
pub struct Ibea<I, V> {
    /// Algorithm configuration.
    pub config: IbeaConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Ibea<I, V> {
    /// Construct an `Ibea` optimizer.
    pub fn new(config: IbeaConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for Ibea<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "Ibea population_size must be > 0"
        );
        assert!(self.config.kappa > 0.0, "Ibea kappa must be > 0");
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        // Initial population.
        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            // --- Phase 1: parent selection (binary tournament on fitness) ---
            let fitness = compute_fitness(&population, &objectives, self.config.kappa);
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = binary_tournament(&fitness, &mut rng);
                let p2 = binary_tournament(&fitness, &mut rng);
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "Ibea variation returned no children");
                for child in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child);
                }
            }

            // --- Phase 2: parallel-friendly batch evaluation ---
            let offspring = evaluate_batch(problem, offspring_decisions);
            evaluations += offspring.len();

            // --- Phase 3: combine + indicator-based survival ---
            let mut combined: Vec<Candidate<P::Decision>> = Vec::with_capacity(2 * n);
            combined.extend(population);
            combined.extend(offspring);
            population = environmental_selection(combined, &objectives, n, self.config.kappa);
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
impl<I, V> Ibea<I, V> {
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
            "Ibea population_size must be > 0"
        );
        assert!(self.config.kappa > 0.0, "Ibea kappa must be > 0");
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch_async(problem, initial_decisions, concurrency).await;
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            let fitness = compute_fitness(&population, &objectives, self.config.kappa);
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = binary_tournament(&fitness, &mut rng);
                let p2 = binary_tournament(&fitness, &mut rng);
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "Ibea variation returned no children");
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
            population = environmental_selection(combined, &objectives, n, self.config.kappa);
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

/// Iteratively remove the worst-fitness member from `pool` until `n` remain.
///
/// IBEA's standard "subtract the dropped member's contribution from every
/// survivor's fitness" recomputation is implemented here so we don't have
/// to rebuild the full O(N²·M) indicator matrix each removal.
fn environmental_selection<D: Clone>(
    mut pool: Vec<Candidate<D>>,
    objectives: &ObjectiveSpace,
    n: usize,
    kappa: f64,
) -> Vec<Candidate<D>> {
    if pool.len() <= n {
        return pool;
    }
    let oriented: Vec<Vec<f64>> = pool
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();

    // Indicator matrix: indicator[i][j] = max_k (oriented[i][k] - oriented[j][k]).
    let indicator: Vec<Vec<f64>> = (0..pool.len())
        .map(|i| {
            (0..pool.len())
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        oriented[i]
                            .iter()
                            .zip(oriented[j].iter())
                            .map(|(a, b)| a - b)
                            .fold(f64::NEG_INFINITY, f64::max)
                    }
                })
                .collect()
        })
        .collect();
    // Normalize indicator by its global magnitude to keep exp() sane.
    let mut max_abs = 1e-12_f64;
    for row in &indicator {
        for &v in row {
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }
    }

    // Fitness F(i) = -Σ_{j≠i} exp(-indicator[j][i] / (max_abs · kappa)).
    // (Higher is better — so a candidate dominated by many is heavily negative.)
    let scale = max_abs * kappa;
    let mut fitness: Vec<f64> = (0..pool.len())
        .map(|i| {
            (0..pool.len())
                .filter(|&j| j != i)
                .map(|j| -(-indicator[j][i] / scale).exp())
                .sum()
        })
        .collect();

    let mut alive: Vec<bool> = vec![true; pool.len()];
    let mut alive_count = pool.len();
    while alive_count > n {
        // Find the lowest-fitness alive member.
        let mut worst = usize::MAX;
        for i in 0..pool.len() {
            if !alive[i] {
                continue;
            }
            if worst == usize::MAX || fitness[i] < fitness[worst] {
                worst = i;
            }
        }
        // Remove its contribution from every other survivor's fitness.
        for i in 0..pool.len() {
            if !alive[i] || i == worst {
                continue;
            }
            fitness[i] += (-indicator[worst][i] / scale).exp();
        }
        alive[worst] = false;
        alive_count -= 1;
    }

    // Materialize survivors, in original order.
    let mut survivors = Vec::with_capacity(n);
    for (i, c) in pool.drain(..).enumerate() {
        if alive[i] {
            survivors.push(c);
        }
    }
    survivors
}

/// Compute IBEA fitness without mutating, for use in tournament selection.
fn compute_fitness<D>(pool: &[Candidate<D>], objectives: &ObjectiveSpace, kappa: f64) -> Vec<f64> {
    if pool.is_empty() {
        return Vec::new();
    }
    let oriented: Vec<Vec<f64>> = pool
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();
    let indicator: Vec<Vec<f64>> = (0..pool.len())
        .map(|i| {
            (0..pool.len())
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        oriented[i]
                            .iter()
                            .zip(oriented[j].iter())
                            .map(|(a, b)| a - b)
                            .fold(f64::NEG_INFINITY, f64::max)
                    }
                })
                .collect()
        })
        .collect();
    let mut max_abs = 1e-12_f64;
    for row in &indicator {
        for &v in row {
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }
    }
    let scale = max_abs * kappa;
    (0..pool.len())
        .map(|i| {
            (0..pool.len())
                .filter(|&j| j != i)
                .map(|j| -(-indicator[j][i] / scale).exp())
                .sum()
        })
        .collect()
}

fn binary_tournament(fitness: &[f64], rng: &mut Rng) -> usize {
    let a = rng.random_range(0..fitness.len());
    let b = rng.random_range(0..fitness.len());
    if fitness[a] > fitness[b] {
        a
    } else if fitness[a] < fitness[b] {
        b
    } else if rng.random_bool(0.5) {
        a
    } else {
        b
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
    ) -> Ibea<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        Ibea::new(
            IbeaConfig {
                population_size: 20,
                generations: 15,
                kappa: 0.05,
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
        assert_eq!(r.population.len(), 20);
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
        let mut opt = Ibea::new(
            IbeaConfig {
                population_size: 0,
                generations: 1,
                kappa: 0.05,
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }
}
