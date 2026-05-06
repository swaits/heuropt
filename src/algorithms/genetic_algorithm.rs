//! `GeneticAlgorithm` — single-objective generational GA with elitism.

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::pareto::front::best_candidate;
use crate::selection::tournament::tournament_select_single_objective;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`GeneticAlgorithm`].
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmConfig {
    /// Constant population size.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Tournament size for parent selection (typical: 2).
    pub tournament_size: usize,
    /// Number of elite members to carry over each generation (must be
    /// `< population_size`).
    pub elitism: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for GeneticAlgorithmConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 200,
            tournament_size: 2,
            elitism: 2,
            seed: 42,
        }
    }
}

/// Single-objective generational genetic algorithm with elitism.
///
/// Each generation: binary tournament selection (on the configured
/// `tournament_size`) chooses parent pairs, the variation operator
/// produces offspring, those are evaluated, and the next population is
/// the top `elitism` from the previous generation plus the best
/// `population_size - elitism` offspring (by fitness).
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
/// let bounds = vec![(-5.0_f64, 5.0_f64); 3];
/// let mut opt = GeneticAlgorithm::new(
///     GeneticAlgorithmConfig {
///         population_size: 30,
///         generations: 50,
///         tournament_size: 2,
///         elitism: 2,
///         seed: 42,
///     },
///     RealBounds::new(bounds.clone()),
///     CompositeVariation {
///         crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
///         mutation:  PolynomialMutation::new(bounds, 20.0, 1.0),
///     },
/// );
/// let r = opt.run(&Sphere);
/// assert!(r.best.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct GeneticAlgorithm<I, V> {
    /// Algorithm configuration.
    pub config: GeneticAlgorithmConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> GeneticAlgorithm<I, V> {
    /// Construct a `GeneticAlgorithm`.
    pub fn new(config: GeneticAlgorithmConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for GeneticAlgorithm<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size >= 2,
            "GeneticAlgorithm population_size must be >= 2",
        );
        assert!(
            self.config.tournament_size >= 1,
            "GeneticAlgorithm tournament_size must be >= 1",
        );
        assert!(
            self.config.elitism < self.config.population_size,
            "GeneticAlgorithm elitism must be < population_size",
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "GeneticAlgorithm requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            // --- Phase 1: parent selection + variation (serial RNG) ---
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let parents_decisions = tournament_select_single_objective(
                    &population,
                    &objectives,
                    self.config.tournament_size,
                    2,
                    &mut rng,
                );
                let children = self.variation.vary(&parents_decisions, &mut rng);
                assert!(
                    !children.is_empty(),
                    "GeneticAlgorithm variation returned no children"
                );
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

            // --- Phase 3: survival = elites + best offspring ---
            population =
                survival_selection(&population, offspring, direction, n, self.config.elitism);
        }

        let best = best_candidate(&population, &objectives);
        let front: Vec<Candidate<P::Decision>> = best.iter().cloned().collect();
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
impl<I, V> GeneticAlgorithm<I, V> {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations per batch (initial
    /// population and per-generation offspring).
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
            self.config.population_size >= 2,
            "GeneticAlgorithm population_size must be >= 2",
        );
        assert!(
            self.config.tournament_size >= 1,
            "GeneticAlgorithm tournament_size must be >= 1",
        );
        assert!(
            self.config.elitism < self.config.population_size,
            "GeneticAlgorithm elitism must be < population_size",
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "GeneticAlgorithm requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch_async(problem, initial_decisions, concurrency).await;
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let parents_decisions = tournament_select_single_objective(
                    &population,
                    &objectives,
                    self.config.tournament_size,
                    2,
                    &mut rng,
                );
                let children = self.variation.vary(&parents_decisions, &mut rng);
                assert!(
                    !children.is_empty(),
                    "GeneticAlgorithm variation returned no children"
                );
                for child in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child);
                }
            }

            let offspring = evaluate_batch_async(problem, offspring_decisions, concurrency).await;
            evaluations += offspring.len();

            population =
                survival_selection(&population, offspring, direction, n, self.config.elitism);
        }

        let best = best_candidate(&population, &objectives);
        let front: Vec<Candidate<P::Decision>> = best.iter().cloned().collect();
        OptimizationResult::new(
            Population::new(population),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

fn survival_selection<D: Clone>(
    parents: &[Candidate<D>],
    offspring: Vec<Candidate<D>>,
    direction: Direction,
    n: usize,
    elitism: usize,
) -> Vec<Candidate<D>> {
    // Sort the parents by fitness descending (best first).
    let mut sorted_parents: Vec<Candidate<D>> = parents.to_vec();
    sorted_parents.sort_by(|a, b| compare_for_fitness(a, b, direction));

    // Sort the offspring the same way.
    let mut sorted_offspring = offspring;
    sorted_offspring.sort_by(|a, b| compare_for_fitness(a, b, direction));

    let mut next: Vec<Candidate<D>> = Vec::with_capacity(n);
    next.extend(sorted_parents.into_iter().take(elitism));
    next.extend(sorted_offspring.into_iter().take(n - elitism));
    next
}

/// Order such that "best" comes first. Feasible beats infeasible; among
/// infeasibles, lower violation wins; among feasibles, direction-aware
/// objective comparison.
fn compare_for_fitness<D>(
    a: &Candidate<D>,
    b: &Candidate<D>,
    direction: Direction,
) -> std::cmp::Ordering {
    match (a.evaluation.is_feasible(), b.evaluation.is_feasible()) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a
            .evaluation
            .constraint_violation
            .partial_cmp(&b.evaluation.constraint_violation)
            .unwrap_or(std::cmp::Ordering::Equal),
        (true, true) => match direction {
            Direction::Minimize => a.evaluation.objectives[0]
                .partial_cmp(&b.evaluation.objectives[0])
                .unwrap_or(std::cmp::Ordering::Equal),
            Direction::Maximize => b.evaluation.objectives[0]
                .partial_cmp(&a.evaluation.objectives[0])
                .unwrap_or(std::cmp::Ordering::Equal),
        },
    }
}

impl<I, V> crate::traits::AlgorithmInfo for GeneticAlgorithm<I, V> {
    fn name(&self) -> &'static str {
        "GeneticAlgorithm"
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
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(
        seed: u64,
    ) -> GeneticAlgorithm<
        RealBounds,
        CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>,
    > {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        GeneticAlgorithm::new(
            GeneticAlgorithmConfig {
                population_size: 30,
                generations: 50,
                tournament_size: 2,
                elitism: 2,
                seed,
            },
            initializer,
            variation,
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
    #[should_panic(expected = "elitism must be < population_size")]
    fn elitism_too_large_panics() {
        let bounds = vec![(-1.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = GeneticAlgorithm::new(
            GeneticAlgorithmConfig {
                population_size: 4,
                generations: 1,
                tournament_size: 2,
                elitism: 4,
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&Sphere1D);
    }
}
