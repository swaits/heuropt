//! `Tlbo` — Rao 2011 Teaching-Learning-Based Optimization, parameter-free
//! single-objective optimizer for `Vec<f64>` decisions.

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::operators::real::RealBounds;
use crate::pareto::front::best_candidate;
use crate::traits::Optimizer;

/// Configuration for [`Tlbo`].
#[derive(Debug, Clone)]
pub struct TlboConfig {
    /// Population size (= number of "learners").
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for TlboConfig {
    fn default() -> Self {
        Self { population_size: 30, generations: 200, seed: 42 }
    }
}

/// Teaching-Learning-Based Optimization.
///
/// The standout feature: NO algorithm-specific hyperparameters. Just
/// population_size and generations. Compared with the rest of heuropt's
/// SO toolkit (DE has F+CR, PSO has w+c1+c2, CMA-ES has σ, GA needs
/// crossover+mutation operators), TLBO works out of the box.
#[derive(Debug, Clone)]
pub struct Tlbo {
    /// Algorithm configuration.
    pub config: TlboConfig,
    /// Per-variable bounds — used both to seed the population and to clamp
    /// every learner's position.
    pub bounds: RealBounds,
}

impl Tlbo {
    /// Construct a `Tlbo`.
    pub fn new(config: TlboConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for Tlbo
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(self.config.population_size >= 2, "Tlbo population_size must be >= 2");
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "Tlbo requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let dim = self.bounds.bounds.len();
        let n = self.config.population_size;
        let mut rng = rng_from_seed(self.config.seed);

        let mut decisions: Vec<Vec<f64>> = {
            use crate::traits::Initializer as _;
            self.bounds.initialize(n, &mut rng)
        };
        let mut evals: Vec<Evaluation> =
            decisions.iter().map(|d| problem.evaluate(d)).collect();
        let mut evaluations = decisions.len();

        for _ in 0..self.config.generations {
            // Identify teacher (best learner).
            let teacher_idx = best_index(&evals, direction);
            let teacher = decisions[teacher_idx].clone();
            // Compute the population mean per dimension.
            let mut mean = vec![0.0_f64; dim];
            for d in &decisions {
                for j in 0..dim {
                    mean[j] += d[j];
                }
            }
            for v in mean.iter_mut() {
                *v /= n as f64;
            }
            // Teaching factor.
            let tf = if rng.random_bool(0.5) { 1.0 } else { 2.0 };

            // Teacher phase.
            for i in 0..n {
                let mut candidate = decisions[i].clone();
                for j in 0..dim {
                    let r: f64 = rng.random();
                    candidate[j] += r * (teacher[j] - tf * mean[j]);
                    let (lo, hi) = self.bounds.bounds[j];
                    candidate[j] = candidate[j].clamp(lo, hi);
                }
                let cand_eval = problem.evaluate(&candidate);
                evaluations += 1;
                if better(&cand_eval, &evals[i], direction) {
                    decisions[i] = candidate;
                    evals[i] = cand_eval;
                }
            }

            // Learner phase: each learner mates with a random different
            // partner and accepts a move toward the better one.
            for i in 0..n {
                let mut k = rng.random_range(0..n);
                while k == i && n > 1 {
                    k = rng.random_range(0..n);
                }
                let partner_better = better(&evals[k], &evals[i], direction);
                let mut candidate = decisions[i].clone();
                for j in 0..dim {
                    let r: f64 = rng.random();
                    let delta = if partner_better {
                        r * (decisions[k][j] - decisions[i][j])
                    } else {
                        r * (decisions[i][j] - decisions[k][j])
                    };
                    candidate[j] += delta;
                    let (lo, hi) = self.bounds.bounds[j];
                    candidate[j] = candidate[j].clamp(lo, hi);
                }
                let cand_eval = problem.evaluate(&candidate);
                evaluations += 1;
                if better(&cand_eval, &evals[i], direction) {
                    decisions[i] = candidate;
                    evals[i] = cand_eval;
                }
            }
        }

        let final_pop: Vec<Candidate<Vec<f64>>> = decisions
            .into_iter()
            .zip(evals)
            .map(|(d, e)| Candidate::new(d, e))
            .collect();
        let best = best_candidate(&final_pop, &objectives);
        let front: Vec<Candidate<Vec<f64>>> = best.iter().cloned().collect();
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

fn best_index(evals: &[Evaluation], direction: Direction) -> usize {
    let mut idx = 0;
    for i in 1..evals.len() {
        if better(&evals[i], &evals[idx], direction) {
            idx = i;
        }
    }
    idx
}

fn better(a: &Evaluation, b: &Evaluation, direction: Direction) -> bool {
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
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(seed: u64) -> Tlbo {
        Tlbo::new(
            TlboConfig {
                population_size: 30,
                generations: 100,
                seed,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
        )
    }

    #[test]
    fn finds_minimum_of_sphere() {
        let mut opt = make_optimizer(1);
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-3,
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
}
