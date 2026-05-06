//! `Grea` — Yang, Li, Liu & Zheng 2013 Grid-based Evolutionary Algorithm.

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

/// Configuration for [`Grea`].
#[derive(Debug, Clone)]
pub struct GreaConfig {
    /// Constant population size.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Grid divisions per objective axis.
    pub grid_divisions: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for GreaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            grid_divisions: 8,
            seed: 42,
        }
    }
}

/// Grid-based Evolutionary Algorithm (GrEA).
///
/// Many-objective EA that uses three grid-based metrics — grid rank,
/// grid crowding distance, and grid coordinate point distance — to
/// select survivors. Particularly strong on linear / simplex-shaped
/// fronts (e.g. DTLZ1).
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
/// let mut opt = Grea::new(
///     GreaConfig { population_size: 30, generations: 20, grid_divisions: 8, seed: 42 },
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
pub struct Grea<I, V> {
    /// Algorithm configuration.
    pub config: GreaConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Grea<I, V> {
    /// Construct a `Grea`.
    pub fn new(config: GreaConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for Grea<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "Grea population_size must be > 0"
        );
        assert!(
            self.config.grid_divisions >= 1,
            "Grea grid_divisions must be >= 1"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            // Random parent selection + variation.
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = rng.random_range(0..population.len());
                let p2 = rng.random_range(0..population.len());
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "Grea variation returned no children");
                for child in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child);
                }
            }
            let offspring = evaluate_batch(problem, offspring_decisions);
            evaluations += offspring.len();

            // Survival.
            let mut combined: Vec<Candidate<P::Decision>> = Vec::with_capacity(2 * n);
            combined.extend(population);
            combined.extend(offspring);
            population =
                environmental_selection(combined, &objectives, n, self.config.grid_divisions);
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
impl<I, V> Grea<I, V> {
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
            "Grea population_size must be > 0"
        );
        assert!(
            self.config.grid_divisions >= 1,
            "Grea grid_divisions must be >= 1"
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
                assert!(!children.is_empty(), "Grea variation returned no children");
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
            population =
                environmental_selection(combined, &objectives, n, self.config.grid_divisions);
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
    divisions: usize,
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

    // Build grid + per-member coordinates on the splitting front (using
    // its own min/max per axis to define the grid box).
    let m = objectives.len();
    let oriented: Vec<Vec<f64>> = splitting
        .iter()
        .map(|&i| objectives.as_minimization(&combined[i].evaluation.objectives))
        .collect();
    let mut lo = vec![f64::INFINITY; m];
    let mut hi = vec![f64::NEG_INFINITY; m];
    for o in &oriented {
        for k in 0..m {
            if o[k] < lo[k] {
                lo[k] = o[k];
            }
            if o[k] > hi[k] {
                hi[k] = o[k];
            }
        }
    }
    let grid_coords: Vec<Vec<usize>> = oriented
        .iter()
        .map(|o| {
            (0..m)
                .map(|k| {
                    let span = (hi[k] - lo[k]).max(1e-12);
                    let frac = ((o[k] - lo[k]) / span).clamp(0.0, 1.0 - 1e-9);
                    (frac * divisions as f64) as usize
                })
                .collect()
        })
        .collect();

    let scores: Vec<(usize, usize, isize, isize)> = (0..splitting.len())
        .map(|local_idx| {
            let gr: usize = grid_coords[local_idx].iter().sum();
            // GCD: count of other splitting members in adjacent grid cells.
            let mut gcd = 0_isize;
            for j in 0..splitting.len() {
                if j == local_idx {
                    continue;
                }
                let max_diff: usize = (0..m)
                    .map(|k| grid_coords[local_idx][k].abs_diff(grid_coords[j][k]))
                    .max()
                    .unwrap_or(0);
                if max_diff < 1 {
                    gcd += 1;
                }
            }
            // GCPD: grid coordinate point distance to that cell's "ideal"
            // origin. We negate to keep "smaller is better" through the
            // sort key.
            let gcpd: isize = grid_coords[local_idx]
                .iter()
                .map(|&c| (c as isize).pow(2))
                .sum::<isize>();
            (local_idx, gr, gcd, gcpd)
        })
        .collect();

    // Sort by (GR ascending, GCD ascending, GCPD ascending).
    let mut sorted_scores = scores;
    sorted_scores.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
    });
    let need = n - selected.len();
    for (local_idx, _, _, _) in sorted_scores.into_iter().take(need) {
        selected.push(splitting[local_idx]);
    }
    selected.into_iter().map(|i| combined[i].clone()).collect()
}

impl<I, V> crate::traits::AlgorithmInfo for Grea<I, V> {
    fn name(&self) -> &'static str {
        "GrEA"
    }
    fn full_name(&self) -> &'static str {
        "Grid-based Evolutionary Algorithm"
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
    ) -> Grea<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        Grea::new(
            GreaConfig {
                population_size: 20,
                generations: 15,
                grid_divisions: 8,
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
