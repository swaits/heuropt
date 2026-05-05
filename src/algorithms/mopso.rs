//! `Mopso` — Coello, Pulido & Lechuga 2004 Multi-Objective Particle Swarm.

use rand::Rng as _;
use rand::seq::IndexedRandom;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::operators::real::RealBounds;
use crate::pareto::archive::ParetoArchive;
use crate::pareto::dominance::{Dominance, pareto_compare};
use crate::pareto::front::{best_candidate, pareto_front};
use crate::traits::Optimizer;

/// Configuration for [`Mopso`].
#[derive(Debug, Clone)]
pub struct MopsoConfig {
    /// Number of particles in the swarm.
    pub swarm_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// External Pareto archive size cap (simple-tail truncation).
    pub archive_size: usize,
    /// Inertia weight `w`.
    pub inertia: f64,
    /// Cognitive coefficient `c_1` (toward personal best).
    pub cognitive: f64,
    /// Social coefficient `c_2` (toward archive leader).
    pub social: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for MopsoConfig {
    fn default() -> Self {
        Self {
            swarm_size: 40,
            generations: 200,
            archive_size: 100,
            inertia: 0.7,
            cognitive: 1.5,
            social: 1.5,
            seed: 42,
        }
    }
}

/// Multi-objective particle swarm with an external Pareto archive.
///
/// `Vec<f64>` decisions only. Each particle maintains a personal best (the
/// last position that was Pareto-non-dominated by any later position). The
/// social leader is sampled uniformly from the external archive each step.
#[derive(Debug, Clone)]
pub struct Mopso {
    /// Algorithm configuration.
    pub config: MopsoConfig,
    /// Per-variable bounds — used both to seed the swarm and to clamp positions.
    pub bounds: RealBounds,
}

impl Mopso {
    /// Construct a `Mopso`.
    pub fn new(config: MopsoConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for Mopso
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(self.config.swarm_size >= 1, "Mopso swarm_size must be >= 1");
        assert!(self.config.archive_size >= 1, "Mopso archive_size must be >= 1");
        let objectives = problem.objectives();
        assert!(
            objectives.is_multi_objective(),
            "Mopso requires multi-objective problems (use ParticleSwarm for single-objective)",
        );
        let dim = self.bounds.bounds.len();
        let n = self.config.swarm_size;
        let mut rng = rng_from_seed(self.config.seed);

        let mut positions: Vec<Vec<f64>> = {
            use crate::traits::Initializer as _;
            self.bounds.initialize(n, &mut rng)
        };
        let mut velocities: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                self.bounds
                    .bounds
                    .iter()
                    .map(|&(lo, hi)| 0.1 * (hi - lo) * (rng.random::<f64>() * 2.0 - 1.0))
                    .collect()
            })
            .collect();
        let v_max: Vec<f64> = self.bounds.bounds.iter().map(|&(lo, hi)| hi - lo).collect();

        let initial_pop = evaluate_batch(problem, positions.clone());
        let mut evaluations = initial_pop.len();

        // Personal bests start at initial positions.
        let mut pbest_decisions: Vec<Vec<f64>> = positions.clone();
        let mut pbest_evals: Vec<crate::core::evaluation::Evaluation> =
            initial_pop.iter().map(|c| c.evaluation.clone()).collect();

        // External archive seeded with the non-dominated subset.
        let mut archive = ParetoArchive::new(objectives.clone());
        for c in initial_pop {
            archive.insert(c);
        }
        archive.truncate(self.config.archive_size);

        for _ in 0..self.config.generations {
            // --- Phase 1: serial position/velocity updates (uses RNG) ---
            for i in 0..n {
                let leader = archive
                    .members()
                    .choose(&mut rng)
                    .map(|c| c.decision.clone())
                    .unwrap_or_else(|| positions[i].clone());
                #[allow(clippy::needless_range_loop)] // body indexes velocities/positions/bounds.
                for j in 0..dim {
                    let r1: f64 = rng.random();
                    let r2: f64 = rng.random();
                    let cognitive_term =
                        self.config.cognitive * r1 * (pbest_decisions[i][j] - positions[i][j]);
                    let social_term = self.config.social * r2 * (leader[j] - positions[i][j]);
                    let mut v = self.config.inertia * velocities[i][j]
                        + cognitive_term
                        + social_term;
                    if v > v_max[j] {
                        v = v_max[j];
                    } else if v < -v_max[j] {
                        v = -v_max[j];
                    }
                    velocities[i][j] = v;
                    let (lo, hi) = self.bounds.bounds[j];
                    positions[i][j] = (positions[i][j] + v).clamp(lo, hi);
                }
            }

            // --- Phase 2: parallel-friendly batch evaluation ---
            let evaluated = evaluate_batch(problem, positions.clone());
            evaluations += evaluated.len();

            // --- Phase 3: serial pbest + archive updates ---
            for (i, cand) in evaluated.iter().enumerate() {
                let dominance =
                    pareto_compare(&cand.evaluation, &pbest_evals[i], &objectives);
                let replace = match dominance {
                    Dominance::Dominates => true,
                    Dominance::DominatedBy => false,
                    Dominance::Equal | Dominance::NonDominated => rng.random_bool(0.5),
                };
                if replace {
                    pbest_decisions[i] = cand.decision.clone();
                    pbest_evals[i] = cand.evaluation.clone();
                }
            }
            for c in evaluated {
                archive.insert(c);
            }
            archive.truncate(self.config.archive_size);
        }

        let members = archive.into_vec();
        let front = pareto_front(&members, &objectives);
        let best = best_candidate(&members, &objectives);
        OptimizationResult::new(
            Population::new(members),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(seed: u64) -> Mopso {
        Mopso::new(
            MopsoConfig {
                swarm_size: 30,
                generations: 30,
                archive_size: 30,
                inertia: 0.7,
                cognitive: 1.5,
                social: 1.5,
                seed,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
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
        let oa: Vec<Vec<f64>> =
            ra.pareto_front.iter().map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> =
            rb.pareto_front.iter().map(|c| c.evaluation.objectives.clone()).collect();
        assert_eq!(oa, ob);
    }

    #[test]
    #[should_panic(expected = "multi-objective")]
    fn single_objective_panics() {
        let mut opt = make_optimizer(0);
        let _ = opt.run(&Sphere1D);
    }
}
