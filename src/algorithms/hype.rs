//! `Hype` — Bader & Zitzler 2011 Hypervolume Estimation Algorithm.
//!
//! HypE replaces the exact hypervolume contribution used in SMS-EMOA with
//! a Monte Carlo estimate, so it scales to arbitrary objective counts at
//! the cost of stochastic noise on the contribution estimate.

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::pareto::front::{best_candidate, pareto_front};
use crate::pareto::sort::non_dominated_sort;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Hype`].
#[derive(Debug, Clone)]
pub struct HypeConfig {
    /// Constant population size.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Reference point used to bound the Monte Carlo integration box.
    /// Must have one entry per objective; should be worse than every
    /// realistic objective value.
    pub reference_point: Vec<f64>,
    /// Number of Monte Carlo samples per HV estimation step.
    pub mc_samples: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for HypeConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            reference_point: vec![11.0, 11.0],
            mc_samples: 10_000,
            seed: 42,
        }
    }
}

/// Hypervolume Estimation Algorithm: many-objective MOEA that selects via
/// Monte Carlo–estimated hypervolume contributions.
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
/// let mut opt = Hype::new(
///     HypeConfig {
///         population_size: 20,
///         generations: 20,
///         reference_point: vec![30.0, 30.0],
///         mc_samples: 100,
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
pub struct Hype<I, V> {
    /// Algorithm configuration.
    pub config: HypeConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Hype<I, V> {
    /// Construct a `Hype`.
    pub fn new(config: HypeConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for Hype<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "Hype population_size must be > 0"
        );
        assert!(self.config.mc_samples > 0, "Hype mc_samples must be > 0");
        let n = self.config.population_size;
        let objectives = problem.objectives();
        assert_eq!(
            self.config.reference_point.len(),
            objectives.len(),
            "Hype reference_point.len() must equal number of objectives",
        );
        let reference = self.config.reference_point.clone();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            // Phase 1: parent selection + variation (random tournament on
            // a fitness-by-HV-estimate proxy).
            let fitness = hype_fitness(
                &population,
                &objectives,
                &reference,
                self.config.mc_samples,
                &mut rng,
            );
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = binary_tournament(&fitness, &mut rng);
                let p2 = binary_tournament(&fitness, &mut rng);
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "Hype variation returned no children");
                for child in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child);
                }
            }

            // Phase 2: parallel-friendly batch evaluation.
            let offspring = evaluate_batch(problem, offspring_decisions);
            evaluations += offspring.len();

            // Phase 3: combine + survival via front-by-front fill plus
            // estimated-contribution truncation on the splitting front.
            let mut combined: Vec<Candidate<P::Decision>> = Vec::with_capacity(2 * n);
            combined.extend(population);
            combined.extend(offspring);

            let fronts = non_dominated_sort(&combined, &objectives);
            let mut keep_indices: Vec<usize> = Vec::with_capacity(n);
            let mut splitting: &[usize] = &[];
            for f in &fronts {
                if keep_indices.len() + f.len() <= n {
                    keep_indices.extend(f.iter().copied());
                } else {
                    splitting = f;
                    break;
                }
                if keep_indices.len() == n {
                    break;
                }
            }
            if keep_indices.len() < n {
                // Need to choose `n - keep_indices.len()` from `splitting`
                // by largest HV contribution.
                let pool: Vec<&Candidate<P::Decision>> =
                    splitting.iter().map(|&i| &combined[i]).collect();
                let contributions = estimate_contributions(
                    &pool,
                    &objectives,
                    &reference,
                    self.config.mc_samples,
                    &mut rng,
                );
                let mut order: Vec<usize> = (0..splitting.len()).collect();
                order.sort_by(|&a, &b| {
                    contributions[b]
                        .partial_cmp(&contributions[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for k in order.into_iter().take(n - keep_indices.len()) {
                    keep_indices.push(splitting[k]);
                }
            }

            // Materialize the next generation.
            population = keep_indices
                .into_iter()
                .map(|i| combined[i].clone())
                .collect();
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

fn hype_fitness<D>(
    pool: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    reference: &[f64],
    samples: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    if pool.is_empty() {
        return Vec::new();
    }
    let pool_refs: Vec<&Candidate<D>> = pool.iter().collect();
    estimate_contributions(&pool_refs, objectives, reference, samples, rng)
}

/// Estimate each candidate's expected unique hypervolume contribution by
/// Monte Carlo sampling uniformly inside the [ideal, reference] box and
/// counting per-sample which candidates dominate it. A sample dominated
/// by exactly one candidate contributes 1/samples × box_volume to that
/// candidate; samples dominated by k candidates contribute proportionally
/// less, weighted by HypE's "weighted hypervolume" rule (1 / k).
fn estimate_contributions<D>(
    pool: &[&Candidate<D>],
    objectives: &ObjectiveSpace,
    reference: &[f64],
    samples: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let n = pool.len();
    if n == 0 {
        return Vec::new();
    }
    let m = reference.len();

    // Cache minimization-oriented objective values.
    let oriented: Vec<Vec<f64>> = pool
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();

    // Compute the lower bound (ideal) of the integration box: per-axis min
    // across the population, capped at the reference (so the box has
    // non-negative width even if no point dominates the reference).
    let mut lower = vec![f64::INFINITY; m];
    for o in &oriented {
        for (k, &v) in o.iter().enumerate() {
            if v < lower[k] {
                lower[k] = v;
            }
        }
    }
    for k in 0..m {
        if !lower[k].is_finite() || lower[k] >= reference[k] {
            // No point on this axis dominates the reference → zero
            // contribution everywhere.
            return vec![0.0; n];
        }
    }

    let box_volume: f64 = (0..m).map(|k| reference[k] - lower[k]).product();
    if box_volume <= 0.0 {
        return vec![0.0; n];
    }

    let mut contrib = vec![0.0_f64; n];
    let mut sample = vec![0.0_f64; m];
    for _ in 0..samples {
        for k in 0..m {
            let u: f64 = rng.random();
            sample[k] = lower[k] + u * (reference[k] - lower[k]);
        }
        // Count and identify candidates that dominate this sample (point
        // in the box).
        let mut dominators: Vec<usize> = Vec::with_capacity(n);
        for (i, o) in oriented.iter().enumerate() {
            if o.iter().zip(sample.iter()).all(|(p, s)| *p <= *s) {
                dominators.push(i);
            }
        }
        if dominators.is_empty() {
            continue;
        }
        // HypE weighting: each sample contributes 1/k to each of its k
        // dominators. (This generalizes "exactly-one dominator" to
        // arbitrary multiplicities.)
        let weight = 1.0 / dominators.len() as f64;
        for i in dominators {
            contrib[i] += weight;
        }
    }

    let scale = box_volume / samples as f64;
    contrib.into_iter().map(|c| c * scale).collect()
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
    ) -> Hype<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        Hype::new(
            HypeConfig {
                population_size: 20,
                generations: 15,
                reference_point: vec![30.0, 30.0],
                mc_samples: 1_000,
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
    #[should_panic(expected = "reference_point.len() must equal number of objectives")]
    fn dim_mismatch_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = Hype::new(
            HypeConfig {
                population_size: 4,
                generations: 1,
                reference_point: vec![1.0, 1.0, 1.0],
                mc_samples: 100,
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }
}
