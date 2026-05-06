//! NSGA-II — the canonical Pareto-based evolutionary algorithm.

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::pareto::crowding::crowding_distance;
use crate::pareto::front::{best_candidate, pareto_front};
use crate::pareto::sort::non_dominated_sort;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Nsga2`].
#[derive(Debug, Clone)]
pub struct Nsga2Config {
    /// Constant population size carried across generations.
    pub population_size: usize,
    /// Number of generations to run.
    pub generations: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for Nsga2Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            seed: 42,
        }
    }
}

/// NSGA-II optimizer (spec §12.3).
///
/// The canonical Pareto-based EA: combines non-dominated sorting with
/// crowding-distance secondary ranking. A strong default for 2- or
/// 3-objective problems.
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
///         ObjectiveSpace::new(vec![
///             Objective::minimize("f1"),
///             Objective::minimize("f2"),
///         ])
///     }
///     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
///         Evaluation::new(vec![x[0] * x[0], (x[0] - 2.0).powi(2)])
///     }
/// }
///
/// let bounds = vec![(-5.0_f64, 5.0_f64)];
/// let mut opt = Nsga2::new(
///     Nsga2Config { population_size: 30, generations: 20, seed: 42 },
///     RealBounds::new(bounds.clone()),
///     CompositeVariation {
///         crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
///         mutation:  PolynomialMutation::new(bounds, 20.0, 1.0),
///     },
/// );
/// let r = opt.run(&Schaffer);
/// assert_eq!(r.population.len(), 30);
/// assert!(!r.pareto_front.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct Nsga2<I, V> {
    /// Algorithm configuration.
    pub config: Nsga2Config,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Nsga2<I, V> {
    /// Construct an `Nsga2` optimizer.
    pub fn new(config: Nsga2Config, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

/// Private bookkeeping for NSGA-II survival selection.
struct Nsga2Entry<D> {
    candidate: Candidate<D>,
    rank: usize,
    crowding_distance: f64,
}

impl<P, I, V> Optimizer<P> for Nsga2<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "Nsga2 population_size must be greater than 0",
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        // Initial population.
        let initial_decisions = self.initializer.initialize(n, &mut rng);
        assert_eq!(
            initial_decisions.len(),
            n,
            "NSGA-II initializer must return exactly population_size decisions",
        );
        let population: Vec<Candidate<P::Decision>> = evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        // Annotate the starting population with rank and crowding so the first
        // round of tournament selection has data to compare on.
        let mut annotated = annotate(population, &objectives);

        for _ in 0..self.config.generations {
            // --- Phase 1: serial parent selection + variation ---
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = binary_tournament(&annotated, &mut rng);
                let p2 = binary_tournament(&annotated, &mut rng);
                let parents = vec![
                    annotated[p1].candidate.decision.clone(),
                    annotated[p2].candidate.decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(
                    !children.is_empty(),
                    "NSGA-II variation returned no children",
                );
                for child_decision in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child_decision);
                }
            }
            // --- Phase 2: parallel-friendly batch evaluation ---
            let offspring: Vec<Candidate<P::Decision>> =
                evaluate_batch(problem, offspring_decisions);
            evaluations += offspring.len();

            // --- Combine + survival selection ---
            let mut combined: Vec<Candidate<P::Decision>> = Vec::with_capacity(2 * n);
            combined.extend(annotated.into_iter().map(|e| e.candidate));
            combined.extend(offspring);

            let fronts = non_dominated_sort(&combined, &objectives);
            let mut next: Vec<Candidate<P::Decision>> = Vec::with_capacity(n);
            for front in &fronts {
                if next.len() + front.len() <= n {
                    for &idx in front {
                        next.push(combined[idx].clone());
                    }
                } else {
                    // Partial last front: keep the most diverse by crowding.
                    let dist = crowding_distance(&combined, front, &objectives);
                    let mut order: Vec<usize> = (0..front.len()).collect();
                    order.sort_by(|&a, &b| {
                        dist[b]
                            .partial_cmp(&dist[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let needed = n - next.len();
                    for &k in order.iter().take(needed) {
                        next.push(combined[front[k]].clone());
                    }
                    break;
                }
                if next.len() == n {
                    break;
                }
            }
            annotated = annotate(next, &objectives);
        }

        // Return final state.
        let final_pop: Vec<Candidate<P::Decision>> =
            annotated.into_iter().map(|e| e.candidate).collect();
        let front = pareto_front(&final_pop, &objectives);
        let best = best_candidate(&final_pop, &objectives);
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

fn annotate<D: Clone>(
    population: Vec<Candidate<D>>,
    objectives: &crate::core::objective::ObjectiveSpace,
) -> Vec<Nsga2Entry<D>> {
    let n = population.len();
    let fronts = non_dominated_sort(&population, objectives);
    let mut rank = vec![0usize; n];
    let mut dist = vec![0.0_f64; n];
    for (r, front) in fronts.iter().enumerate() {
        let d = crowding_distance(&population, front, objectives);
        for (k, &idx) in front.iter().enumerate() {
            rank[idx] = r;
            dist[idx] = d[k];
        }
    }
    population
        .into_iter()
        .enumerate()
        .map(|(i, c)| Nsga2Entry {
            candidate: c,
            rank: rank[i],
            crowding_distance: dist[i],
        })
        .collect()
}

#[cfg(feature = "async")]
impl<I, V> Nsga2<I, V> {
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
            self.config.population_size > 0,
            "Nsga2 population_size must be greater than 0",
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        assert_eq!(
            initial_decisions.len(),
            n,
            "NSGA-II initializer must return exactly population_size decisions",
        );
        let population: Vec<Candidate<P::Decision>> =
            evaluate_batch_async(problem, initial_decisions, concurrency).await;
        let mut evaluations = population.len();

        let mut annotated = annotate(population, &objectives);

        for _ in 0..self.config.generations {
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = binary_tournament(&annotated, &mut rng);
                let p2 = binary_tournament(&annotated, &mut rng);
                let parents = vec![
                    annotated[p1].candidate.decision.clone(),
                    annotated[p2].candidate.decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(
                    !children.is_empty(),
                    "NSGA-II variation returned no children",
                );
                for child_decision in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child_decision);
                }
            }
            let offspring: Vec<Candidate<P::Decision>> =
                evaluate_batch_async(problem, offspring_decisions, concurrency).await;
            evaluations += offspring.len();

            let mut combined: Vec<Candidate<P::Decision>> = Vec::with_capacity(2 * n);
            combined.extend(annotated.into_iter().map(|e| e.candidate));
            combined.extend(offspring);

            let fronts = non_dominated_sort(&combined, &objectives);
            let mut next: Vec<Candidate<P::Decision>> = Vec::with_capacity(n);
            for front in &fronts {
                if next.len() + front.len() <= n {
                    for &idx in front {
                        next.push(combined[idx].clone());
                    }
                } else {
                    let dist = crowding_distance(&combined, front, &objectives);
                    let mut order: Vec<usize> = (0..front.len()).collect();
                    order.sort_by(|&a, &b| {
                        dist[b]
                            .partial_cmp(&dist[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let needed = n - next.len();
                    for &k in order.iter().take(needed) {
                        next.push(combined[front[k]].clone());
                    }
                    break;
                }
                if next.len() == n {
                    break;
                }
            }
            annotated = annotate(next, &objectives);
        }

        let final_pop: Vec<Candidate<P::Decision>> =
            annotated.into_iter().map(|e| e.candidate).collect();
        let front = pareto_front(&final_pop, &objectives);
        let best = best_candidate(&final_pop, &objectives);
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

fn binary_tournament<D>(entries: &[Nsga2Entry<D>], rng: &mut Rng) -> usize {
    let n = entries.len();
    let a = rng.random_range(0..n);
    let b = rng.random_range(0..n);
    let ea = &entries[a];
    let eb = &entries[b];
    if ea.rank < eb.rank {
        a
    } else if ea.rank > eb.rank {
        b
    } else if ea.crowding_distance > eb.crowding_distance {
        a
    } else if ea.crowding_distance < eb.crowding_distance {
        b
    } else if rng.random_bool(0.5) {
        a
    } else {
        b
    }
}

impl<I, V> crate::traits::AlgorithmInfo for Nsga2<I, V> {
    fn name(&self) -> &'static str {
        "NSGA-II"
    }
    fn full_name(&self) -> &'static str {
        "Non-dominated Sorting Genetic Algorithm II"
    }
    fn seed(&self) -> Option<u64> {
        Some(self.config.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{GaussianMutation, RealBounds};
    use crate::tests_support::SchafferN1;

    #[test]
    fn final_population_has_expected_size() {
        let mut opt = Nsga2::new(
            Nsga2Config {
                population_size: 20,
                generations: 5,
                seed: 1,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.3 },
        );
        let r = opt.run(&SchafferN1);
        assert_eq!(r.population.len(), 20);
        assert!(!r.pareto_front.is_empty());
    }

    #[test]
    fn evaluation_count_at_least_initial_population() {
        let mut opt = Nsga2::new(
            Nsga2Config {
                population_size: 16,
                generations: 3,
                seed: 2,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.3 },
        );
        let r = opt.run(&SchafferN1);
        assert!(r.evaluations >= 16);
        assert_eq!(r.generations, 3);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut a = Nsga2::new(
            Nsga2Config {
                population_size: 16,
                generations: 5,
                seed: 99,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.2 },
        );
        let mut b = Nsga2::new(
            Nsga2Config {
                population_size: 16,
                generations: 5,
                seed: 99,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.2 },
        );
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
    #[should_panic(expected = "population_size must be greater than 0")]
    fn zero_population_size_panics() {
        let mut opt = Nsga2::new(
            Nsga2Config {
                population_size: 0,
                generations: 1,
                seed: 0,
            },
            RealBounds::new(vec![(-1.0, 1.0)]),
            GaussianMutation { sigma: 0.1 },
        );
        let _ = opt.run(&SchafferN1);
    }
}
