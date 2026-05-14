//! `Rvea` — Cheng, Jin, Olhofer & Sendhoff 2016 Reference Vector-guided EA.

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::pareto::front::{best_candidate, pareto_front};
use crate::pareto::reference_points::das_dennis;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Rvea`].
#[derive(Debug, Clone)]
pub struct RveaConfig {
    /// Constant population size.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Number of divisions `H` for Das–Dennis reference vectors. Pop size
    /// should be roughly `binomial(H + M − 1, M − 1)`.
    pub reference_divisions: usize,
    /// Penalty exponent `α`. The paper recommends 2.0.
    pub alpha: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for RveaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            reference_divisions: 12,
            alpha: 2.0,
            seed: 42,
        }
    }
}

/// Reference Vector-guided Evolutionary Algorithm.
///
/// Many-objective EA that uses Das–Dennis reference vectors with an
/// adaptive penalty term to balance convergence and diversity as
/// generations progress.
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
/// let mut opt = Rvea::new(
///     RveaConfig {
///         population_size: 30,
///         generations: 20,
///         reference_divisions: 19,
///         alpha: 2.0,
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
pub struct Rvea<I, V> {
    /// Algorithm configuration.
    pub config: RveaConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Rvea<I, V> {
    /// Construct an `Rvea`.
    pub fn new(config: RveaConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for Rvea<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "Rvea population_size must be > 0"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let m = objectives.len();
        // Reference vectors normalized to unit norm.
        let raw_refs = das_dennis(m, self.config.reference_divisions);
        let references: Vec<Vec<f64>> = raw_refs.into_iter().map(unit_normalize).collect();
        assert!(
            !references.is_empty(),
            "Rvea: no reference vectors generated"
        );

        // Smallest angle between any two reference vectors — used to scale
        // the APD penalty term.
        let theta_max = smallest_neighbor_angle(&references);
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        for gen_idx in 0..self.config.generations {
            // Phase 1: random parent selection + variation.
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = rng.random_range(0..population.len());
                let p2 = rng.random_range(0..population.len());
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "Rvea variation returned no children");
                for child in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child);
                }
            }
            let offspring = evaluate_batch(problem, offspring_decisions);
            evaluations += offspring.len();

            // Combine + APD-based survival.
            let mut combined: Vec<Candidate<P::Decision>> = Vec::with_capacity(2 * n);
            combined.extend(population);
            combined.extend(offspring);

            // Ideal point z*.
            let m_dim = m;
            let mut ideal = vec![f64::INFINITY; m_dim];
            for c in &combined {
                let oriented = objectives.as_minimization(&c.evaluation.objectives);
                for (k, v) in oriented.iter().enumerate() {
                    if *v < ideal[k] {
                        ideal[k] = *v;
                    }
                }
            }
            // Translate.
            let translated: Vec<Vec<f64>> = combined
                .iter()
                .map(|c| {
                    let oriented = objectives.as_minimization(&c.evaluation.objectives);
                    oriented
                        .iter()
                        .enumerate()
                        .map(|(k, v)| v - ideal[k])
                        .collect()
                })
                .collect();

            // Associate each member with its closest-angle reference vector.
            let mut assoc: Vec<usize> = vec![0; combined.len()];
            let mut angles: Vec<f64> = vec![0.0; combined.len()];
            for (i, t) in translated.iter().enumerate() {
                let (best_ref, best_angle) = closest_reference(t, &references);
                assoc[i] = best_ref;
                angles[i] = best_angle;
            }

            // For each occupied reference vector, keep the member with the
            // smallest APD score.
            let alpha_t = (gen_idx as f64 / (self.config.generations as f64).max(1.0))
                .powf(self.config.alpha);
            let mut keep: Vec<Option<(usize, f64)>> = vec![None; references.len()];
            for i in 0..combined.len() {
                let r = assoc[i];
                let length: f64 = translated[i].iter().map(|v| v * v).sum::<f64>().sqrt();
                let theta_max_safe = theta_max.max(1e-12);
                let penalty = 1.0 + (m_dim as f64) * alpha_t * (angles[i] / theta_max_safe);
                let apd = penalty * length;
                match keep[r] {
                    None => keep[r] = Some((i, apd)),
                    Some((_, current)) if apd < current => keep[r] = Some((i, apd)),
                    _ => {}
                }
            }

            let mut next: Vec<Candidate<P::Decision>> = keep
                .into_iter()
                .flatten()
                .map(|(i, _)| combined[i].clone())
                .collect();
            // If we ended up with fewer than n (some references unfilled),
            // backfill with the lowest-APD remaining candidates.
            if next.len() < n {
                let mut all_apds: Vec<(usize, f64)> = (0..combined.len())
                    .map(|i| {
                        let length: f64 = translated[i].iter().map(|v| v * v).sum::<f64>().sqrt();
                        let theta_max_safe = theta_max.max(1e-12);
                        let penalty = 1.0 + (m_dim as f64) * alpha_t * (angles[i] / theta_max_safe);
                        (i, penalty * length)
                    })
                    .collect();
                all_apds.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for (i, _) in all_apds {
                    if next.len() >= n {
                        break;
                    }
                    if !next
                        .iter()
                        .any(|c| std::ptr::eq(c as *const _, &combined[i] as *const _))
                    {
                        next.push(combined[i].clone());
                    }
                }
            }
            // If too many (only possible if the reference set has > n
            // vectors), truncate by APD.
            if next.len() > n {
                next.truncate(n);
            }
            population = next;
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
impl<I, V> Rvea<I, V> {
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
            "Rvea population_size must be > 0"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let m = objectives.len();
        let raw_refs = das_dennis(m, self.config.reference_divisions);
        let references: Vec<Vec<f64>> = raw_refs.into_iter().map(unit_normalize).collect();
        assert!(
            !references.is_empty(),
            "Rvea: no reference vectors generated"
        );

        let theta_max = smallest_neighbor_angle(&references);
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch_async(problem, initial_decisions, concurrency).await;
        let mut evaluations = population.len();

        for gen_idx in 0..self.config.generations {
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = rng.random_range(0..population.len());
                let p2 = rng.random_range(0..population.len());
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "Rvea variation returned no children");
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

            let m_dim = m;
            let mut ideal = vec![f64::INFINITY; m_dim];
            for c in &combined {
                let oriented = objectives.as_minimization(&c.evaluation.objectives);
                for (k, v) in oriented.iter().enumerate() {
                    if *v < ideal[k] {
                        ideal[k] = *v;
                    }
                }
            }
            let translated: Vec<Vec<f64>> = combined
                .iter()
                .map(|c| {
                    let oriented = objectives.as_minimization(&c.evaluation.objectives);
                    oriented
                        .iter()
                        .enumerate()
                        .map(|(k, v)| v - ideal[k])
                        .collect()
                })
                .collect();

            let mut assoc: Vec<usize> = vec![0; combined.len()];
            let mut angles: Vec<f64> = vec![0.0; combined.len()];
            for (i, t) in translated.iter().enumerate() {
                let (best_ref, best_angle) = closest_reference(t, &references);
                assoc[i] = best_ref;
                angles[i] = best_angle;
            }

            let alpha_t = (gen_idx as f64 / (self.config.generations as f64).max(1.0))
                .powf(self.config.alpha);
            let mut keep: Vec<Option<(usize, f64)>> = vec![None; references.len()];
            for i in 0..combined.len() {
                let r = assoc[i];
                let length: f64 = translated[i].iter().map(|v| v * v).sum::<f64>().sqrt();
                let theta_max_safe = theta_max.max(1e-12);
                let penalty = 1.0 + (m_dim as f64) * alpha_t * (angles[i] / theta_max_safe);
                let apd = penalty * length;
                match keep[r] {
                    None => keep[r] = Some((i, apd)),
                    Some((_, current)) if apd < current => keep[r] = Some((i, apd)),
                    _ => {}
                }
            }

            let mut next: Vec<Candidate<P::Decision>> = keep
                .into_iter()
                .flatten()
                .map(|(i, _)| combined[i].clone())
                .collect();
            if next.len() < n {
                let mut all_apds: Vec<(usize, f64)> = (0..combined.len())
                    .map(|i| {
                        let length: f64 = translated[i].iter().map(|v| v * v).sum::<f64>().sqrt();
                        let theta_max_safe = theta_max.max(1e-12);
                        let penalty = 1.0 + (m_dim as f64) * alpha_t * (angles[i] / theta_max_safe);
                        (i, penalty * length)
                    })
                    .collect();
                all_apds.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for (i, _) in all_apds {
                    if next.len() >= n {
                        break;
                    }
                    if !next
                        .iter()
                        .any(|c| std::ptr::eq(c as *const _, &combined[i] as *const _))
                    {
                        next.push(combined[i].clone());
                    }
                }
            }
            if next.len() > n {
                next.truncate(n);
            }
            population = next;
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

fn unit_normalize(mut v: Vec<f64>) -> Vec<f64> {
    let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v
}

fn closest_reference(point: &[f64], references: &[Vec<f64>]) -> (usize, f64) {
    let length: f64 = point.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    let mut best = 0;
    let mut best_angle = f64::INFINITY;
    for (i, r) in references.iter().enumerate() {
        let dot: f64 = point.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
        let cosine = (dot / length).clamp(-1.0, 1.0);
        let angle = cosine.acos();
        if angle < best_angle {
            best_angle = angle;
            best = i;
        }
    }
    (best, best_angle)
}

fn smallest_neighbor_angle(references: &[Vec<f64>]) -> f64 {
    let mut min_angle = f64::INFINITY;
    for i in 0..references.len() {
        for j in (i + 1)..references.len() {
            let dot: f64 = references[i]
                .iter()
                .zip(references[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            let angle = dot.clamp(-1.0, 1.0).acos();
            if angle < min_angle {
                min_angle = angle;
            }
        }
    }
    if !min_angle.is_finite() {
        std::f64::consts::FRAC_PI_4
    } else {
        min_angle
    }
}

impl<I, V> crate::traits::AlgorithmInfo for Rvea<I, V> {
    fn name(&self) -> &'static str {
        "RVEA"
    }
    fn full_name(&self) -> &'static str {
        "Reference Vector-guided Evolutionary Algorithm"
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
    ) -> Rvea<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        Rvea::new(
            RveaConfig {
                population_size: 20,
                generations: 15,
                reference_divisions: 19,
                alpha: 2.0,
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

    #[test]
    #[should_panic(expected = "population_size must be > 0")]
    fn zero_population_size_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = Rvea::new(
            RveaConfig {
                population_size: 0,
                generations: 1,
                reference_divisions: 5,
                alpha: 2.0,
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }

    // ---- Mutation-test pinned helpers --------------------------------------

    #[test]
    fn unit_normalize_produces_unit_vector() {
        let v = unit_normalize(vec![3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
        assert!((v[0] - 0.6).abs() < 1e-12);
        assert!((v[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn unit_normalize_zero_vector_unchanged() {
        // A (near-)zero vector is left as-is (no division by ~0).
        let v = unit_normalize(vec![0.0, 0.0]);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn closest_reference_picks_smallest_angle() {
        // References along the two axes; a point near the x-axis associates
        // with reference 0 at a small angle.
        let refs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (idx, angle) = closest_reference(&[1.0, 0.0], &refs);
        assert_eq!(idx, 0);
        assert!(angle.abs() < 1e-9, "angle = {angle}");
        let (idx2, _) = closest_reference(&[0.1, 1.0], &refs);
        assert_eq!(idx2, 1);
    }

    #[test]
    fn smallest_neighbor_angle_of_orthogonal_refs_is_pi_over_2() {
        let refs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let a = smallest_neighbor_angle(&refs);
        assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-9, "angle = {a}");
    }
}
