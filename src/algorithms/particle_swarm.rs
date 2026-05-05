//! `ParticleSwarm` — Eberhart & Kennedy 1995 PSO for `Vec<f64>` decisions.

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::operators::real::RealBounds;
use crate::pareto::front::best_candidate;
use crate::traits::Optimizer;

/// Configuration for [`ParticleSwarm`].
#[derive(Debug, Clone)]
pub struct ParticleSwarmConfig {
    /// Number of particles in the swarm.
    pub swarm_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Inertia weight `w`. Typical: 0.4–0.9.
    pub inertia: f64,
    /// Cognitive coefficient `c_1`. Typical: 1.5–2.0.
    pub cognitive: f64,
    /// Social coefficient `c_2`. Typical: 1.5–2.0.
    pub social: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for ParticleSwarmConfig {
    fn default() -> Self {
        Self {
            swarm_size: 40,
            generations: 200,
            inertia: 0.7,
            cognitive: 1.5,
            social: 1.5,
            seed: 42,
        }
    }
}

/// Single-objective real-valued PSO.
///
/// Particles update with the standard inertia-weight rule:
///
/// ```text
/// v[i,t+1] = w·v[i,t] + c1·r1·(pbest[i] - x[i,t]) + c2·r2·(gbest - x[i,t])
/// x[i,t+1] = clamp(x[i,t] + v[i,t+1], bounds)
/// ```
///
/// Velocities are clamped to `±(hi - lo)` per dimension to prevent
/// "swarm explosion." Pair with `RealBounds` for both the search bounds
/// and the initial particle positions.
#[derive(Debug, Clone)]
pub struct ParticleSwarm {
    /// Algorithm configuration.
    pub config: ParticleSwarmConfig,
    /// Per-variable bounds — used both to seed the swarm and to clamp positions.
    pub bounds: RealBounds,
}

impl ParticleSwarm {
    /// Construct a `ParticleSwarm`.
    pub fn new(config: ParticleSwarmConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for ParticleSwarm
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.swarm_size >= 1,
            "ParticleSwarm swarm_size must be >= 1",
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "ParticleSwarm requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let dim = self.bounds.bounds.len();
        let n = self.config.swarm_size;
        let mut rng = rng_from_seed(self.config.seed);

        // Initialize positions via the bounds initializer.
        let mut positions: Vec<Vec<f64>> = {
            use crate::traits::Initializer as _;
            self.bounds.initialize(n, &mut rng)
        };
        // Initial velocities: small random perturbations within ±0.1·range.
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

        // Initial evaluation.
        let initial_pop = evaluate_batch(problem, positions.clone());
        let mut evaluations = initial_pop.len();

        // Personal bests start at initial positions.
        let mut pbest_decisions: Vec<Vec<f64>> = positions.clone();
        let mut pbest_evals: Vec<f64> = initial_pop
            .iter()
            .map(|c| c.evaluation.objectives[0])
            .collect();

        // Global best.
        let mut gbest_idx = best_index(&pbest_evals, direction);
        let mut gbest_decision = pbest_decisions[gbest_idx].clone();
        let mut gbest_eval = pbest_evals[gbest_idx];

        for _ in 0..self.config.generations {
            // --- Phase 1: serial position/velocity updates (uses RNG) ---
            for i in 0..n {
                #[allow(clippy::needless_range_loop)] // body indexes velocities/positions/bounds.
                for j in 0..dim {
                    let r1: f64 = rng.random();
                    let r2: f64 = rng.random();
                    let cognitive_term =
                        self.config.cognitive * r1 * (pbest_decisions[i][j] - positions[i][j]);
                    let social_term =
                        self.config.social * r2 * (gbest_decision[j] - positions[i][j]);
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

            // --- Phase 3: serial pbest / gbest updates ---
            for (i, cand) in evaluated.iter().enumerate() {
                let f = cand.evaluation.objectives[0];
                let improves = match direction {
                    Direction::Minimize => f < pbest_evals[i],
                    Direction::Maximize => f > pbest_evals[i],
                };
                if improves {
                    pbest_decisions[i] = positions[i].clone();
                    pbest_evals[i] = f;
                    gbest_idx = i;
                    let beats_global = match direction {
                        Direction::Minimize => f < gbest_eval,
                        Direction::Maximize => f > gbest_eval,
                    };
                    if beats_global {
                        gbest_decision = pbest_decisions[i].clone();
                        gbest_eval = f;
                    }
                }
            }
        }
        let _ = gbest_idx;

        // Final population is the current particle positions, evaluated.
        let final_pop = evaluate_batch(problem, positions);
        evaluations += final_pop.len();
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

fn best_index(values: &[f64], direction: Direction) -> usize {
    let mut idx = 0;
    for i in 1..values.len() {
        let better = match direction {
            Direction::Minimize => values[i] < values[idx],
            Direction::Maximize => values[i] > values[idx],
        };
        if better {
            idx = i;
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(seed: u64) -> ParticleSwarm {
        ParticleSwarm::new(
            ParticleSwarmConfig {
                swarm_size: 30,
                generations: 100,
                inertia: 0.7,
                cognitive: 1.5,
                social: 1.5,
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
