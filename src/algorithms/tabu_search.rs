//! `TabuSearch` — Glover 1986 tabu search with a user-supplied neighbor
//! generator and decision-level FIFO tabu list.

use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

use crate::core::candidate::Candidate;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::traits::{Initializer, Optimizer};

/// Configuration for [`TabuSearch`].
#[derive(Debug, Clone)]
pub struct TabuSearchConfig {
    /// Number of iterations.
    pub iterations: usize,
    /// Maximum size of the FIFO tabu list (older entries are evicted).
    pub tabu_tenure: usize,
    /// Seed for the deterministic RNG used by the neighbor generator.
    pub seed: u64,
}

impl Default for TabuSearchConfig {
    fn default() -> Self {
        Self { iterations: 500, tabu_tenure: 16, seed: 42 }
    }
}

/// Single-objective tabu search.
///
/// Each iteration the user-supplied `neighbors` closure produces a finite
/// list of candidate moves from the current incumbent. The best non-tabu
/// neighbor (or any tabu neighbor that improves the best-seen-ever
/// incumbent — the standard "aspiration" override) is accepted as the new
/// incumbent and its decision is appended to a FIFO tabu list of size
/// `tabu_tenure`. Tabu matches the full decision; users wanting move-based
/// tabu can wrap moves into a custom decision type.
pub struct TabuSearch<D, I, N>
where
    D: Clone + Hash + Eq,
    I: Initializer<D>,
    N: FnMut(&D, &mut Rng) -> Vec<D>,
{
    /// Algorithm configuration.
    pub config: TabuSearchConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Neighbor generator: produces a finite list of candidate moves from
    /// the current incumbent.
    pub neighbors: N,
    _marker: std::marker::PhantomData<D>,
}

impl<D, I, N> TabuSearch<D, I, N>
where
    D: Clone + Hash + Eq,
    I: Initializer<D>,
    N: FnMut(&D, &mut Rng) -> Vec<D>,
{
    /// Construct a `TabuSearch`.
    pub fn new(config: TabuSearchConfig, initializer: I, neighbors: N) -> Self {
        Self { config, initializer, neighbors, _marker: std::marker::PhantomData }
    }
}

impl<P, I, N> Optimizer<P> for TabuSearch<P::Decision, I, N>
where
    P: Problem + Sync,
    P::Decision: Clone + Hash + Eq + Send,
    I: Initializer<P::Decision>,
    N: FnMut(&P::Decision, &mut Rng) -> Vec<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "TabuSearch requires exactly one objective",
        );
        assert!(
            self.config.tabu_tenure >= 1,
            "TabuSearch tabu_tenure must be >= 1",
        );
        let direction = objectives.objectives[0].direction;
        let mut rng = rng_from_seed(self.config.seed);

        let mut initial = self.initializer.initialize(1, &mut rng);
        assert!(!initial.is_empty(), "TabuSearch initializer returned no decisions");
        let mut current_decision = initial.remove(0);
        let mut current_eval = problem.evaluate(&current_decision);
        let mut best_decision = current_decision.clone();
        let mut best_eval = current_eval.clone();
        let mut evaluations = 1usize;

        let mut tabu_queue: VecDeque<P::Decision> = VecDeque::with_capacity(self.config.tabu_tenure);
        let mut tabu_set: HashSet<P::Decision> = HashSet::new();

        for _ in 0..self.config.iterations {
            let candidates = (self.neighbors)(&current_decision, &mut rng);
            if candidates.is_empty() {
                break;
            }

            // Best non-tabu candidate, OR best tabu candidate that beats the
            // best-seen-ever (aspiration).
            let mut best_idx: Option<usize> = None;
            let mut best_cand_eval: Option<crate::core::evaluation::Evaluation> = None;
            let evaluations_before = evaluations;
            let mut cand_evals: Vec<crate::core::evaluation::Evaluation> =
                Vec::with_capacity(candidates.len());
            for c in &candidates {
                cand_evals.push(problem.evaluate(c));
            }
            evaluations += candidates.len();
            let _ = evaluations_before;

            for (i, c) in candidates.iter().enumerate() {
                let is_tabu = tabu_set.contains(c);
                let aspires = is_tabu
                    && better_than(&cand_evals[i], &best_eval, direction);
                if is_tabu && !aspires {
                    continue;
                }
                let eligible = match &best_cand_eval {
                    None => true,
                    Some(b) => better_than(&cand_evals[i], b, direction),
                };
                if eligible {
                    best_idx = Some(i);
                    best_cand_eval = Some(cand_evals[i].clone());
                }
            }

            // If everything is tabu and nothing aspires, fall back to the
            // best tabu candidate (avoid getting stuck).
            if best_idx.is_none() {
                for (i, _) in candidates.iter().enumerate() {
                    let eligible = match &best_cand_eval {
                        None => true,
                        Some(b) => better_than(&cand_evals[i], b, direction),
                    };
                    if eligible {
                        best_idx = Some(i);
                        best_cand_eval = Some(cand_evals[i].clone());
                    }
                }
            }

            let chosen_idx = best_idx.expect("non-empty candidate list");
            let chosen_decision = candidates[chosen_idx].clone();
            current_eval = cand_evals.remove(chosen_idx);
            current_decision = chosen_decision.clone();

            if better_than(&current_eval, &best_eval, direction) {
                best_decision = current_decision.clone();
                best_eval = current_eval.clone();
            }

            // Update FIFO tabu list.
            tabu_queue.push_back(chosen_decision.clone());
            tabu_set.insert(chosen_decision);
            if tabu_queue.len() > self.config.tabu_tenure {
                if let Some(old) = tabu_queue.pop_front() {
                    tabu_set.remove(&old);
                }
            }
        }

        let best = Candidate::new(best_decision, best_eval);
        let population = Population::new(vec![best.clone()]);
        let front = vec![best.clone()];
        OptimizationResult::new(
            population,
            front,
            Some(best),
            evaluations,
            self.config.iterations,
        )
    }
}

fn better_than(
    a: &crate::core::evaluation::Evaluation,
    b: &crate::core::evaluation::Evaluation,
    direction: Direction,
) -> bool {
    match (a.is_feasible(), b.is_feasible()) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => a.constraint_violation < b.constraint_violation,
        (true, true) => match direction {
            Direction::Minimize => a.objectives[0] < b.objectives[0],
            Direction::Maximize => a.objectives[0] > b.objectives[0],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Objective, ObjectiveSpace};
    use rand::Rng as _;

    /// Trivial integer-grid problem: minimize `(x - 7)^2`.
    struct GridProblem;
    impl Problem for GridProblem {
        type Decision = Vec<i32>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }

        fn evaluate(&self, x: &Vec<i32>) -> Evaluation {
            let v = (x[0] - 7) as f64;
            Evaluation::new(vec![v * v])
        }
    }

    /// Initialize a single 1-D integer at 0.
    struct StartAtZero;
    impl Initializer<Vec<i32>> for StartAtZero {
        fn initialize(&mut self, size: usize, _rng: &mut Rng) -> Vec<Vec<i32>> {
            (0..size).map(|_| vec![0]).collect()
        }
    }

    fn make_optimizer<F>(
        seed: u64,
        neighbors: F,
    ) -> TabuSearch<Vec<i32>, StartAtZero, F>
    where
        F: FnMut(&Vec<i32>, &mut Rng) -> Vec<Vec<i32>>,
    {
        TabuSearch::new(
            TabuSearchConfig { iterations: 50, tabu_tenure: 4, seed },
            StartAtZero,
            neighbors,
        )
    }

    #[test]
    fn finds_optimum_on_grid() {
        // Neighbors: ±1 of current value.
        let neighbors = |x: &Vec<i32>, _rng: &mut Rng| {
            vec![vec![x[0] - 1], vec![x[0] + 1]]
        };
        let mut opt = make_optimizer(1, neighbors);
        let r = opt.run(&GridProblem);
        let best = r.best.unwrap();
        assert_eq!(best.decision, vec![7]);
        assert_eq!(best.evaluation.objectives, vec![0.0]);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let neighbors = |x: &Vec<i32>, rng: &mut Rng| {
            (0..5)
                .map(|_| vec![x[0] + rng.random_range(-3..=3)])
                .collect::<Vec<_>>()
        };
        let mut a = make_optimizer(99, neighbors);
        let mut b = make_optimizer(99, |x: &Vec<i32>, rng: &mut Rng| {
            (0..5)
                .map(|_| vec![x[0] + rng.random_range(-3..=3)])
                .collect::<Vec<_>>()
        });
        let ra = a.run(&GridProblem);
        let rb = b.run(&GridProblem);
        assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }
}
