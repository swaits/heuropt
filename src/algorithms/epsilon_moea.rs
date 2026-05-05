//! `EpsilonMoea` — Deb, Mohan & Mishra 2003 ε-dominance MOEA.

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::pareto::dominance::{Dominance, pareto_compare};
use crate::pareto::front::{best_candidate, pareto_front};
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`EpsilonMoea`].
#[derive(Debug, Clone)]
pub struct EpsilonMoeaConfig {
    /// Internal population size.
    pub population_size: usize,
    /// Number of evaluations to perform (steady-state: one offspring per gen).
    pub evaluations: usize,
    /// ε for each objective. Must have one entry per objective; controls
    /// the resolution of the regular box-grid the archive lives on.
    pub epsilon: Vec<f64>,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for EpsilonMoeaConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            evaluations: 25_000,
            epsilon: vec![0.05, 0.05],
            seed: 42,
        }
    }
}

/// ε-dominance MOEA.
///
/// Steady-state EA with an ε-grid archive: every member that lands in
/// the same ε-box as an existing one is replaced by the closer point
/// to the box's grid corner. Auto-bounds the front size by the choice
/// of `epsilon`.
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
/// let mut opt = EpsilonMoea::new(
///     EpsilonMoeaConfig {
///         population_size: 20,
///         evaluations: 1_000,
///         epsilon: vec![0.1, 0.1],
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
pub struct EpsilonMoea<I, V> {
    /// Algorithm configuration.
    pub config: EpsilonMoeaConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> EpsilonMoea<I, V> {
    /// Construct an `EpsilonMoea`.
    pub fn new(config: EpsilonMoeaConfig, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for EpsilonMoea<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "EpsilonMoea population_size must be > 0"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        assert_eq!(
            self.config.epsilon.len(),
            objectives.len(),
            "EpsilonMoea epsilon.len() must equal number of objectives",
        );
        for (i, &e) in self.config.epsilon.iter().enumerate() {
            assert!(e > 0.0, "EpsilonMoea epsilon[{i}] must be > 0.0");
        }
        let epsilon = self.config.epsilon.clone();
        let mut rng = rng_from_seed(self.config.seed);

        // Internal population.
        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut population: Vec<Candidate<P::Decision>> = initial_decisions
            .into_iter()
            .map(|d| {
                let e = problem.evaluate(&d);
                Candidate::new(d, e)
            })
            .collect();
        let mut evaluations = population.len();

        // ε-archive.
        let mut archive: Vec<Candidate<P::Decision>> = Vec::new();
        for c in &population {
            insert_into_epsilon_archive(&mut archive, c.clone(), &objectives, &epsilon);
        }

        let total_evals = self.config.evaluations.max(evaluations);
        while evaluations < total_evals {
            // Pick one parent from the population, one from the archive
            // (when non-empty; else two from the population).
            let p1_idx = rng.random_range(0..population.len());
            let parent_a = population[p1_idx].decision.clone();
            let parent_b = if !archive.is_empty() {
                let j = rng.random_range(0..archive.len());
                archive[j].decision.clone()
            } else {
                let j = rng.random_range(0..population.len());
                population[j].decision.clone()
            };
            let parents = vec![parent_a, parent_b];
            let children = self.variation.vary(&parents, &mut rng);
            assert!(
                !children.is_empty(),
                "EpsilonMoea variation returned no children"
            );
            let child_decision = children.into_iter().next().unwrap();
            let child_eval = problem.evaluate(&child_decision);
            evaluations += 1;
            let child = Candidate::new(child_decision, child_eval);

            // Update population: child replaces a Pareto-dominated random member,
            // or any random member if non-dominated wrt every population member.
            update_population(&mut population, &child, &objectives, &mut rng);

            // Update ε-archive.
            insert_into_epsilon_archive(&mut archive, child, &objectives, &epsilon);
        }

        let final_pop: Vec<Candidate<P::Decision>> = if !archive.is_empty() {
            archive.clone()
        } else {
            population
        };
        let front = pareto_front(&final_pop, &objectives);
        let best = best_candidate(&final_pop, &objectives);
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best,
            evaluations,
            self.config.evaluations,
        )
    }
}

/// Standard ε-MOEA population update: if the child is dominated by some
/// member, drop it; if it dominates a member, replace that member; if
/// non-dominated wrt all, replace a random member.
fn update_population<D: Clone>(
    population: &mut [Candidate<D>],
    child: &Candidate<D>,
    objectives: &ObjectiveSpace,
    rng: &mut crate::core::rng::Rng,
) {
    let mut dominated_indices: Vec<usize> = Vec::new();
    for (i, c) in population.iter().enumerate() {
        match pareto_compare(&child.evaluation, &c.evaluation, objectives) {
            Dominance::DominatedBy => return, // child dominated → discard
            Dominance::Dominates => dominated_indices.push(i),
            _ => {}
        }
    }
    if !dominated_indices.is_empty() {
        let pick = dominated_indices[rng.random_range(0..dominated_indices.len())];
        population[pick] = child.clone();
    } else {
        let pick = rng.random_range(0..population.len());
        population[pick] = child.clone();
    }
}

/// Insert `child` into the ε-archive following Deb's standard rule:
///
/// - Translate every objective vector into ε-box coordinates
///   `b_i = floor(o_i / ε_i)` (in minimization frame).
/// - If `child`'s box is ε-dominated by an existing member → drop child.
/// - Else, drop existing members whose box is ε-dominated by `child`'s.
/// - Among members in the SAME box as `child`, keep the one closer to its
///   box's "ideal corner" (smallest L2 distance from box origin).
fn insert_into_epsilon_archive<D: Clone>(
    archive: &mut Vec<Candidate<D>>,
    child: Candidate<D>,
    objectives: &ObjectiveSpace,
    epsilon: &[f64],
) {
    let child_box = box_coords(&child.evaluation, objectives, epsilon);
    let child_corner_dist = corner_distance(&child.evaluation, objectives, epsilon, &child_box);

    let mut to_drop: Vec<usize> = Vec::new();
    let mut child_box_index: Option<usize> = None;
    for (i, member) in archive.iter().enumerate() {
        let member_box = box_coords(&member.evaluation, objectives, epsilon);
        if box_dominates(&member_box, &child_box) {
            // Child's box is ε-dominated; ignore the child.
            return;
        }
        if box_dominates(&child_box, &member_box) {
            to_drop.push(i);
        } else if member_box == child_box {
            child_box_index = Some(i);
        }
    }
    // Drop ε-dominated members (in reverse order to keep indices valid).
    to_drop.sort_unstable();
    for i in to_drop.into_iter().rev() {
        archive.swap_remove(i);
    }
    if let Some(idx) = child_box_index {
        // Same box: keep whichever is closer to box's ideal corner.
        let member_corner_dist =
            corner_distance(&archive[idx].evaluation, objectives, epsilon, &child_box);
        if child_corner_dist < member_corner_dist {
            archive[idx] = child;
        }
    } else {
        archive.push(child);
    }
}

fn box_coords(eval: &Evaluation, objectives: &ObjectiveSpace, epsilon: &[f64]) -> Vec<i64> {
    let oriented = objectives.as_minimization(&eval.objectives);
    oriented
        .iter()
        .zip(epsilon.iter())
        .map(|(v, e)| (v / e).floor() as i64)
        .collect()
}

fn corner_distance(
    eval: &Evaluation,
    objectives: &ObjectiveSpace,
    epsilon: &[f64],
    box_idx: &[i64],
) -> f64 {
    let oriented = objectives.as_minimization(&eval.objectives);
    let mut sq = 0.0;
    for k in 0..oriented.len() {
        let corner = box_idx[k] as f64 * epsilon[k];
        let d = oriented[k] - corner;
        sq += d * d;
    }
    sq.sqrt()
}

/// Box-A ε-dominates box-B iff every coordinate of A is ≤ B and at least
/// one is strictly less.
fn box_dominates(a: &[i64], b: &[i64]) -> bool {
    let mut strictly_less = false;
    for (x, y) in a.iter().zip(b.iter()) {
        if x > y {
            return false;
        }
        if x < y {
            strictly_less = true;
        }
    }
    strictly_less
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
    ) -> EpsilonMoea<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>>
    {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        EpsilonMoea::new(
            EpsilonMoeaConfig {
                population_size: 20,
                evaluations: 1_000,
                epsilon: vec![0.05, 0.05],
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
    #[should_panic(expected = "epsilon.len() must equal number of objectives")]
    fn dim_mismatch_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = EpsilonMoea::new(
            EpsilonMoeaConfig {
                population_size: 4,
                evaluations: 100,
                epsilon: vec![0.1, 0.1, 0.1],
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }

    #[test]
    #[should_panic(expected = "must be > 0.0")]
    fn zero_epsilon_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = EpsilonMoea::new(
            EpsilonMoeaConfig {
                population_size: 4,
                evaluations: 100,
                epsilon: vec![0.0, 0.1],
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }
}
