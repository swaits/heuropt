//! MOEA/D — Multi-Objective Evolutionary Algorithm by Decomposition
//! (Zhang & Li 2007), with the Tchebycheff scalarizing function.

use rand::seq::IndexedRandom;

use crate::core::candidate::Candidate;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::pareto::front::{best_candidate, pareto_front};
use crate::pareto::reference_points::das_dennis;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Moead`].
#[derive(Debug, Clone)]
pub struct MoeadConfig {
    /// Number of generations (passes over the weight set).
    pub generations: usize,
    /// Das–Dennis divisions `H`. The number of weight vectors (= the
    /// population size) is `binomial(H + M - 1, M - 1)`.
    pub reference_divisions: usize,
    /// Neighborhood size `T`: each subproblem mates within and updates
    /// at most this many neighbors.
    pub neighborhood_size: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for MoeadConfig {
    fn default() -> Self {
        Self {
            generations: 250,
            reference_divisions: 99, // 100 weights for 2 objectives
            neighborhood_size: 20,
            seed: 42,
        }
    }
}

/// MOEA/D optimizer using the Tchebycheff scalarizing function.
#[derive(Debug, Clone)]
pub struct Moead<I, V> {
    /// Algorithm configuration.
    pub config: MoeadConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Moead<I, V> {
    /// Construct a `Moead` optimizer.
    pub fn new(config: MoeadConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for Moead<I, V>
where
    P: Problem,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        let objectives = problem.objectives();
        let m = objectives.len();
        let weights = das_dennis(m, self.config.reference_divisions);
        assert!(
            !weights.is_empty(),
            "Moead weight set is empty — increase reference_divisions",
        );
        let n = weights.len();
        let t = self.config.neighborhood_size.min(n);
        assert!(t >= 2, "Moead neighborhood_size must be >= 2");

        let mut rng = rng_from_seed(self.config.seed);

        // Initial population: one decision per weight vector.
        let initial_decisions = self.initializer.initialize(n, &mut rng);
        assert_eq!(
            initial_decisions.len(),
            n,
            "MOEA/D initializer must return exactly {n} decisions",
        );
        let mut population: Vec<Candidate<P::Decision>> = initial_decisions
            .into_iter()
            .map(|d| {
                let e = problem.evaluate(&d);
                Candidate::new(d, e)
            })
            .collect();
        let mut evaluations = population.len();

        // Ideal point z*: per-axis min in oriented space, seeded from the
        // initial population.
        let mut ideal = vec![f64::INFINITY; m];
        for c in &population {
            let oriented = objectives.as_minimization(&c.evaluation.objectives);
            for (k, v) in oriented.iter().enumerate() {
                if *v < ideal[k] {
                    ideal[k] = *v;
                }
            }
        }

        // Neighborhoods B[i] = T closest weight vectors to weights[i] by
        // Euclidean distance, including i itself.
        let neighborhoods: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| {
                    let da = weight_distance(&weights[i], &weights[a]);
                    let db = weight_distance(&weights[i], &weights[b]);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
                idx.into_iter().take(t).collect()
            })
            .collect();

        for _ in 0..self.config.generations {
            #[allow(clippy::needless_range_loop)]
            // Body indexes both `neighborhoods[i]` and `population[j]` via `nbh`.
            for i in 0..n {
                // Pick two distinct parents from the neighborhood.
                let nbh = &neighborhoods[i];
                let p1 = *nbh.choose(&mut rng).unwrap();
                let mut p2 = *nbh.choose(&mut rng).unwrap();
                while p2 == p1 && nbh.len() > 1 {
                    p2 = *nbh.choose(&mut rng).unwrap();
                }
                let parents = vec![
                    population[p1].decision.clone(),
                    population[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(
                    !children.is_empty(),
                    "MOEA/D variation returned no children"
                );
                let child_decision = children.into_iter().next().unwrap();
                let child_eval = problem.evaluate(&child_decision);
                evaluations += 1;

                // Update ideal point.
                let oriented_child = objectives.as_minimization(&child_eval.objectives);
                for (k, v) in oriented_child.iter().enumerate() {
                    if *v < ideal[k] {
                        ideal[k] = *v;
                    }
                }

                // Walk the neighborhood; replace current members where the
                // child improves the Tchebycheff scalar.
                for &j in nbh {
                    let cur_oriented =
                        objectives.as_minimization(&population[j].evaluation.objectives);
                    let g_cur = tchebycheff(&cur_oriented, &weights[j], &ideal);
                    let g_new = tchebycheff(&oriented_child, &weights[j], &ideal);
                    if g_new <= g_cur {
                        population[j] = Candidate::new(child_decision.clone(), child_eval.clone());
                    }
                }
            }
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

/// Tchebycheff scalarization: `max_k w_k * |f_k - z*_k|`.
///
/// `weight` components that are zero are floored to `1e-6` so every axis
/// contributes (matches the convention used in the original paper).
fn tchebycheff(oriented_objectives: &[f64], weight: &[f64], ideal: &[f64]) -> f64 {
    let mut g: f64 = 0.0;
    for (k, &f) in oriented_objectives.iter().enumerate() {
        let w = weight[k].max(1e-6);
        let term = w * (f - ideal[k]).abs();
        if term > g {
            g = term;
        }
    }
    g
}

fn weight_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
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
    ) -> Moead<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        Moead::new(
            MoeadConfig {
                generations: 30,
                reference_divisions: 19, // 20 weights for 2-obj
                neighborhood_size: 5,
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
        assert_eq!(r.population.len(), 20); // 19 divisions + 1 → 20 weights
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut a = make_optimizer(99);
        let mut b = make_optimizer(99);
        let ra = a.run(&SchafferN1);
        let rb = b.run(&SchafferN1);
        let oa: Vec<Vec<f64>> = ra
            .population
            .iter()
            .map(|c| c.evaluation.objectives.clone())
            .collect();
        let ob: Vec<Vec<f64>> = rb
            .population
            .iter()
            .map(|c| c.evaluation.objectives.clone())
            .collect();
        assert_eq!(oa, ob);
    }

    #[test]
    #[should_panic(expected = "neighborhood_size must be >= 2")]
    fn neighborhood_size_one_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = Moead::new(
            MoeadConfig {
                generations: 1,
                reference_divisions: 4,
                neighborhood_size: 1,
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }
}
