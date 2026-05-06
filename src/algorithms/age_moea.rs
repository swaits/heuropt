//! `AgeMoea` — Panichella 2019 Adaptive Geometry Estimation MOEA.

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

/// Configuration for [`AgeMoea`].
#[derive(Debug, Clone)]
pub struct AgeMoeaConfig {
    /// Constant population size.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for AgeMoeaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            seed: 42,
        }
    }
}

/// Adaptive Geometry Estimation MOEA.
///
/// Estimates the current front's L_p geometry parameter and uses it to
/// score survivors by a combination of proximity (distance to the
/// translated origin in the L_p frame) and diversity (distance to the
/// nearest survivor in the same frame).
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
/// let mut opt = AgeMoea::new(
///     AgeMoeaConfig { population_size: 30, generations: 20, seed: 42 },
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
pub struct AgeMoea<I, V> {
    /// Algorithm configuration.
    pub config: AgeMoeaConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> AgeMoea<I, V> {
    /// Construct an `AgeMoea`.
    pub fn new(config: AgeMoeaConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for AgeMoea<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "AgeMoea population_size must be > 0"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
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
                assert!(
                    !children.is_empty(),
                    "AgeMoea variation returned no children"
                );
                for child in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child);
                }
            }
            let offspring = evaluate_batch(problem, offspring_decisions);
            evaluations += offspring.len();

            // Phase 3: combine + age-moea survival selection.
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
impl<I, V> AgeMoea<I, V> {
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
            "AgeMoea population_size must be > 0"
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
                assert!(
                    !children.is_empty(),
                    "AgeMoea variation returned no children"
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

    // Translate by ideal point z*.
    let m = objectives.len();
    let n0_oriented: Vec<Vec<f64>> = fronts[0]
        .iter()
        .map(|&i| objectives.as_minimization(&combined[i].evaluation.objectives))
        .collect();
    let mut ideal = vec![f64::INFINITY; m];
    for o in &n0_oriented {
        for (k, v) in o.iter().enumerate() {
            if *v < ideal[k] {
                ideal[k] = *v;
            }
        }
    }

    // Translate every combined member.
    let translated: Vec<Vec<f64>> = combined
        .iter()
        .map(|c| {
            let oriented = objectives.as_minimization(&c.evaluation.objectives);
            oriented
                .iter()
                .enumerate()
                .map(|(k, v)| (v - ideal[k]).max(0.0))
                .collect()
        })
        .collect();

    // Estimate p (geometry parameter) from the *first* front's
    // extreme points: find the point with the largest single-axis value
    // for each axis, then solve for p such that all extreme points have
    // unit L_p norm after normalizing by the per-axis maximum.
    let p = estimate_p(&fronts[0], &translated, m);

    // Score every member of the splitting front by:
    //   proximity = ||translated||_p
    //   diversity = nearest-neighbor distance in the same L_p frame
    //               among already-selected + splitting members.
    //
    // Two caches make this much cheaper than the textbook formulation:
    //   * `prox[i]` — `lp_norm(translated[i], p)` is constant across
    //     iterations, so compute it once per splitting-front member.
    //   * `nearest[i]` — the nearest-keep distance only ever decreases
    //     when a new candidate is picked, so we maintain it
    //     incrementally: seed it from `selected`, then on every pick
    //     update each remaining `i`'s nearest by taking
    //     `min(nearest[i], lp_distance(translated[i], translated[pick], p))`.
    //
    // That cuts the score loop from O(R · K · M) per iteration (where
    // R = remaining count, K = current keep count) to O(R · M) per
    // iteration, with the dominant `powf` calls in lp_distance counted
    // once per (remaining, pick) pair instead of per (remaining, all-keep).
    let mut keep = selected.clone();
    let mut remaining: Vec<usize> = splitting.clone();
    let prox: Vec<f64> = (0..combined.len())
        .map(|i| lp_norm(&translated[i], p))
        .collect();
    let mut nearest: Vec<f64> = (0..combined.len())
        .map(|i| nearest_neighbor_distance(i, &translated, &keep, p))
        .collect();
    while keep.len() < n {
        // Pick the remaining candidate with the largest score.
        let mut best_idx: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for &i in &remaining {
            let score = nearest[i] / (prox[i].max(1e-12));
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }
        match best_idx {
            None => break,
            Some(pick) => {
                keep.push(pick);
                remaining.retain(|&i| i != pick);
                // Update each surviving remaining's nearest-keep using
                // just the distance to the new pick.
                for &i in &remaining {
                    let d = lp_distance(&translated[i], &translated[pick], p);
                    if d < nearest[i] {
                        nearest[i] = d;
                    }
                }
            }
        }
    }
    keep.into_iter().map(|i| combined[i].clone()).collect()
}

fn lp_norm(v: &[f64], p: f64) -> f64 {
    v.iter().map(|x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

fn lp_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

fn nearest_neighbor_distance(i: usize, translated: &[Vec<f64>], selected: &[usize], p: f64) -> f64 {
    if selected.is_empty() {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for &j in selected {
        if j == i {
            continue;
        }
        let d = lp_distance(&translated[i], &translated[j], p);
        if d < best {
            best = d;
        }
    }
    best
}

/// Estimate the L_p geometry parameter from the front's extreme points.
///
/// Find the extreme point on each axis (the front member maximizing that
/// objective relative to its own L_∞ norm), then choose p such that all
/// extreme points have approximately the same L_p magnitude. Falls back
/// to p = 2 (spherical) if anything degenerates.
fn estimate_p(front_indices: &[usize], translated: &[Vec<f64>], m: usize) -> f64 {
    if front_indices.is_empty() || m == 0 {
        return 2.0;
    }
    // For each axis, find the extreme: the front member with the largest
    // ratio of its k-th coordinate to its own L1 norm (i.e., the most
    // "k-aligned" member).
    let extremes: Vec<usize> = (0..m)
        .map(|axis| {
            let mut best = front_indices[0];
            let mut best_ratio = f64::NEG_INFINITY;
            for &idx in front_indices {
                let l1: f64 = translated[idx].iter().sum::<f64>().max(1e-12);
                let ratio = translated[idx][axis] / l1;
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best = idx;
                }
            }
            best
        })
        .collect();

    // Solve for p ∈ [0.1, 10.0] that minimizes std-dev of L_p norms across
    // extremes (a coarse sweep is fine — full Brent isn't needed for this
    // shape estimate).
    let candidates: Vec<f64> = (1..=40).map(|i| (i as f64) * 0.25).collect();
    let mut best_p = 2.0;
    let mut best_loss = f64::INFINITY;
    for &p in &candidates {
        let norms: Vec<f64> = extremes
            .iter()
            .map(|&i| lp_norm(&translated[i], p))
            .collect();
        let mean = norms.iter().sum::<f64>() / norms.len() as f64;
        if mean.is_finite() && mean > 0.0 {
            let var = norms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / norms.len() as f64;
            let loss = var.sqrt() / mean;
            if loss < best_loss {
                best_loss = loss;
                best_p = p;
            }
        }
    }
    best_p
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
    ) -> AgeMoea<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        AgeMoea::new(
            AgeMoeaConfig {
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
    fn zero_pop_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = AgeMoea::new(
            AgeMoeaConfig {
                population_size: 0,
                generations: 1,
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }
}
