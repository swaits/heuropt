//! `AntColonyTsp` — Dorigo-style Ant System for permutation problems on a
//! complete graph (TSP-style).

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::traits::Optimizer;

/// Configuration for [`AntColonyTsp`].
#[derive(Debug, Clone)]
pub struct AntColonyTspConfig {
    /// Number of ants per generation.
    pub ants: usize,
    /// Number of generations.
    pub generations: usize,
    /// Pheromone weight `α`.
    pub alpha: f64,
    /// Heuristic weight `β`.
    pub beta: f64,
    /// Pheromone evaporation rate `ρ` ∈ [0, 1].
    pub evaporation: f64,
    /// Pheromone deposit constant `Q`. Reinforcement on edge (i, j) is
    /// `Q / tour_length` for every ant whose tour uses (i, j).
    pub deposit: f64,
    /// Initial pheromone level on every edge.
    pub initial_pheromone: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for AntColonyTspConfig {
    fn default() -> Self {
        Self {
            ants: 30,
            generations: 100,
            alpha: 1.0,
            beta: 2.0,
            evaporation: 0.5,
            deposit: 1.0,
            initial_pheromone: 1.0,
            seed: 42,
        }
    }
}

/// Ant Colony Optimization for permutation-style problems on a complete graph.
///
/// `Vec<usize>` decisions only (the permutation `[0, 1, …, n_cities - 1]`).
/// Single-objective only — typically minimizing total tour length, but the
/// algorithm is direction-aware for completeness.
///
/// Each ant builds a tour by repeatedly choosing the next node with
/// probability `∝ τ_ij^α · η_ij^β` over the unvisited cities, where
/// `η_ij = 1 / distance_ij` is the heuristic desirability.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// struct Tsp { distances: Vec<Vec<f64>> }
/// impl Problem for Tsp {
///     type Decision = Vec<usize>;
///     fn objectives(&self) -> ObjectiveSpace {
///         ObjectiveSpace::new(vec![Objective::minimize("length")])
///     }
///     fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
///         let mut len = 0.0;
///         for w in tour.windows(2) { len += self.distances[w[0]][w[1]]; }
///         len += self.distances[*tour.last().unwrap()][tour[0]];
///         Evaluation::new(vec![len])
///     }
/// }
///
/// // 5 cities laid out in a small square + center. The optimal tour
/// // is the perimeter; the diagonal is suboptimal.
/// let cities = [(0.0_f64, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (1.5, 1.5)];
/// let n = cities.len();
/// let mut d = vec![vec![0.0; n]; n];
/// for i in 0..n {
///     for j in 0..n {
///         let dx = cities[i].0 - cities[j].0;
///         let dy = cities[i].1 - cities[j].1;
///         d[i][j] = (dx * dx + dy * dy).sqrt();
///     }
/// }
/// let problem = Tsp { distances: d.clone() };
///
/// let mut opt = AntColonyTsp::new(AntColonyTspConfig {
///     ants: 10,
///     generations: 50,
///     alpha: 1.0,
///     beta: 5.0,
///     evaporation: 0.5,
///     deposit: 1.0,
///     initial_pheromone: 0.1,
///     seed: 42,
/// }, d);
/// let r = opt.run(&problem);
/// assert!(r.best.is_some());
/// ```
pub struct AntColonyTsp {
    /// Algorithm configuration.
    pub config: AntColonyTspConfig,
    /// Symmetric distance matrix; size `n_cities × n_cities`. Diagonal must
    /// be zero.
    pub distances: Vec<Vec<f64>>,
}

impl AntColonyTsp {
    /// Construct an `AntColonyTsp`. Validates that `distances` is square
    /// and has a zero diagonal.
    pub fn new(config: AntColonyTspConfig, distances: Vec<Vec<f64>>) -> Self {
        let n = distances.len();
        assert!(
            n >= 2,
            "AntColonyTsp distances matrix must have >= 2 cities"
        );
        for (i, row) in distances.iter().enumerate() {
            assert_eq!(row.len(), n, "AntColonyTsp distances matrix must be square");
            assert_eq!(
                row[i], 0.0,
                "AntColonyTsp distance from city to itself must be 0"
            );
        }
        Self { config, distances }
    }
}

impl<P> Optimizer<P> for AntColonyTsp
where
    P: Problem<Decision = Vec<usize>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(self.config.ants >= 1, "AntColonyTsp ants must be >= 1");
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "AntColonyTsp requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let n = self.distances.len();
        let mut rng = rng_from_seed(self.config.seed);

        // Heuristic desirability: 1 / distance (with a small floor to avoid
        // division by zero for very-close cities).
        let eta: Vec<Vec<f64>> = self
            .distances
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
                    .collect()
            })
            .collect();

        // Pheromone matrix.
        let mut pheromone: Vec<Vec<f64>> = vec![vec![self.config.initial_pheromone; n]; n];

        let mut best_decision: Option<Vec<usize>> = None;
        let mut best_eval: Option<crate::core::evaluation::Evaluation> = None;
        let mut evaluations = 0usize;

        for _ in 0..self.config.generations {
            let mut tours: Vec<Vec<usize>> = Vec::with_capacity(self.config.ants);
            let mut tour_evals: Vec<crate::core::evaluation::Evaluation> =
                Vec::with_capacity(self.config.ants);

            for _ in 0..self.config.ants {
                let start = rng.random_range(0..n);
                let tour = build_tour(
                    n,
                    start,
                    &pheromone,
                    &eta,
                    self.config.alpha,
                    self.config.beta,
                    &mut rng,
                );
                let eval = problem.evaluate(&tour);
                evaluations += 1;
                tours.push(tour);
                tour_evals.push(eval);
            }

            // Update best.
            for (tour, eval) in tours.iter().zip(tour_evals.iter()) {
                let beats = match &best_eval {
                    None => true,
                    Some(b) => better_than_so(eval, b, direction),
                };
                if beats {
                    best_decision = Some(tour.clone());
                    best_eval = Some(eval.clone());
                }
            }

            // Pheromone evaporation.
            for row in pheromone.iter_mut() {
                for v in row.iter_mut() {
                    *v *= 1.0 - self.config.evaporation;
                }
            }

            // Pheromone deposit on each ant's tour.
            for (tour, eval) in tours.iter().zip(tour_evals.iter()) {
                let length = eval
                    .objectives
                    .first()
                    .copied()
                    .unwrap_or(f64::INFINITY)
                    .max(1e-12);
                let deposit = self.config.deposit / length;
                for w in tour.windows(2) {
                    let (i, j) = (w[0], w[1]);
                    pheromone[i][j] += deposit;
                    pheromone[j][i] += deposit;
                }
                // Close the loop.
                let (i, j) = (*tour.last().unwrap(), tour[0]);
                pheromone[i][j] += deposit;
                pheromone[j][i] += deposit;
            }
        }

        let best = Candidate::new(best_decision.unwrap(), best_eval.unwrap());
        let population = Population::new(vec![best.clone()]);
        let front = vec![best.clone()];
        OptimizationResult::new(
            population,
            front,
            Some(best),
            evaluations,
            self.config.generations,
        )
    }
}

#[cfg(feature = "async")]
impl AntColonyTsp {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations per generation.
    pub async fn run_async<P>(
        &mut self,
        problem: &P,
        concurrency: usize,
    ) -> OptimizationResult<Vec<usize>>
    where
        P: crate::core::async_problem::AsyncProblem<Decision = Vec<usize>>,
    {
        use crate::algorithms::parallel_eval_async::evaluate_batch_async;

        assert!(self.config.ants >= 1, "AntColonyTsp ants must be >= 1");
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "AntColonyTsp requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let n = self.distances.len();
        let mut rng = rng_from_seed(self.config.seed);

        let eta: Vec<Vec<f64>> = self
            .distances
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
                    .collect()
            })
            .collect();

        let mut pheromone: Vec<Vec<f64>> = vec![vec![self.config.initial_pheromone; n]; n];

        let mut best_decision: Option<Vec<usize>> = None;
        let mut best_eval: Option<crate::core::evaluation::Evaluation> = None;
        let mut evaluations = 0usize;

        for _ in 0..self.config.generations {
            let mut tours: Vec<Vec<usize>> = Vec::with_capacity(self.config.ants);
            for _ in 0..self.config.ants {
                let start = rng.random_range(0..n);
                let tour = build_tour(
                    n,
                    start,
                    &pheromone,
                    &eta,
                    self.config.alpha,
                    self.config.beta,
                    &mut rng,
                );
                tours.push(tour);
            }

            let cands = evaluate_batch_async(problem, tours.clone(), concurrency).await;
            evaluations += cands.len();
            let tour_evals: Vec<crate::core::evaluation::Evaluation> =
                cands.into_iter().map(|c| c.evaluation).collect();

            for (tour, eval) in tours.iter().zip(tour_evals.iter()) {
                let beats = match &best_eval {
                    None => true,
                    Some(b) => better_than_so(eval, b, direction),
                };
                if beats {
                    best_decision = Some(tour.clone());
                    best_eval = Some(eval.clone());
                }
            }

            for row in pheromone.iter_mut() {
                for v in row.iter_mut() {
                    *v *= 1.0 - self.config.evaporation;
                }
            }

            for (tour, eval) in tours.iter().zip(tour_evals.iter()) {
                let length = eval
                    .objectives
                    .first()
                    .copied()
                    .unwrap_or(f64::INFINITY)
                    .max(1e-12);
                let deposit = self.config.deposit / length;
                for w in tour.windows(2) {
                    let (i, j) = (w[0], w[1]);
                    pheromone[i][j] += deposit;
                    pheromone[j][i] += deposit;
                }
                let (i, j) = (*tour.last().unwrap(), tour[0]);
                pheromone[i][j] += deposit;
                pheromone[j][i] += deposit;
            }
        }

        let best = Candidate::new(best_decision.unwrap(), best_eval.unwrap());
        let population = Population::new(vec![best.clone()]);
        let front = vec![best.clone()];
        OptimizationResult::new(
            population,
            front,
            Some(best),
            evaluations,
            self.config.generations,
        )
    }
}

fn build_tour(
    n: usize,
    start: usize,
    pheromone: &[Vec<f64>],
    eta: &[Vec<f64>],
    alpha: f64,
    beta: f64,
    rng: &mut crate::core::rng::Rng,
) -> Vec<usize> {
    let mut tour = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    tour.push(start);
    visited[start] = true;

    for _ in 1..n {
        let current = *tour.last().unwrap();
        // Build a probability vector over the unvisited candidates.
        let probs: Vec<(usize, f64)> = (0..n)
            .filter(|&j| !visited[j])
            .map(|j| {
                let p = pheromone[current][j].max(0.0).powf(alpha) * eta[current][j].powf(beta);
                (j, p)
            })
            .collect();
        let total: f64 = probs.iter().map(|(_, p)| *p).sum();
        let next = if total > 0.0 {
            let r: f64 = rng.random::<f64>() * total;
            let mut acc = 0.0;
            let mut chosen = probs.last().unwrap().0;
            for (j, p) in &probs {
                acc += *p;
                if r <= acc {
                    chosen = *j;
                    break;
                }
            }
            chosen
        } else {
            // Degenerate case: pheromone × heuristic is 0 for every
            // unvisited city. Fall back to uniform random.
            let &(j, _) = probs.choose_uniform(rng);
            j
        };
        let _ = probs;
        tour.push(next);
        visited[next] = true;
    }
    tour
}

trait ChooseUniform<T> {
    fn choose_uniform(&self, rng: &mut crate::core::rng::Rng) -> &T;
}
impl<T> ChooseUniform<T> for [T] {
    fn choose_uniform(&self, rng: &mut crate::core::rng::Rng) -> &T {
        &self[rng.random_range(0..self.len())]
    }
}

fn better_than_so(
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

impl crate::traits::AlgorithmInfo for AntColonyTsp {
    fn name(&self) -> &'static str {
        "AntColonyTsp"
    }
    fn seed(&self) -> Option<u64> {
        Some(self.config.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Objective, ObjectiveSpace};

    /// A 5-city ring problem: cities placed at `(cos(2πi/5), sin(2πi/5))`.
    /// Optimal tour length: 2·5·sin(π/5) ≈ 5.878 (a regular pentagon).
    struct RingTsp {
        distances: Vec<Vec<f64>>,
    }
    impl RingTsp {
        fn new(n: usize) -> Self {
            use std::f64::consts::PI;
            let pts: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    let a = 2.0 * PI * (i as f64) / (n as f64);
                    (a.cos(), a.sin())
                })
                .collect();
            let distances = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let (xi, yi) = pts[i];
                            let (xj, yj) = pts[j];
                            ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt()
                        })
                        .collect()
                })
                .collect();
            Self { distances }
        }
    }
    impl Problem for RingTsp {
        type Decision = Vec<usize>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("tour_length")])
        }

        fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
            let n = tour.len();
            let mut total = 0.0;
            for w in tour.windows(2) {
                total += self.distances[w[0]][w[1]];
            }
            total += self.distances[tour[n - 1]][tour[0]];
            Evaluation::new(vec![total])
        }
    }

    /// Trivial single-objective problem to test the multi-objective panic.
    struct DummyMo;
    impl Problem for DummyMo {
        type Decision = Vec<usize>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("a"), Objective::minimize("b")])
        }

        fn evaluate(&self, _tour: &Vec<usize>) -> Evaluation {
            Evaluation::new(vec![0.0, 0.0])
        }
    }

    #[test]
    fn finds_near_optimum_on_5_city_ring() {
        let problem = RingTsp::new(5);
        let mut opt = AntColonyTsp::new(
            AntColonyTspConfig {
                ants: 10,
                generations: 30,
                alpha: 1.0,
                beta: 3.0,
                evaporation: 0.5,
                deposit: 1.0,
                initial_pheromone: 1.0,
                seed: 1,
            },
            problem.distances.clone(),
        );
        let r = opt.run(&problem);
        let best = r.best.unwrap();
        // Optimal pentagon perimeter ≈ 5.878. ACO should hit close.
        assert!(
            best.evaluation.objectives[0] < 5.95,
            "got tour length = {}",
            best.evaluation.objectives[0],
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let problem = RingTsp::new(5);
        let cfg = AntColonyTspConfig {
            ants: 8,
            generations: 10,
            alpha: 1.0,
            beta: 2.0,
            evaporation: 0.5,
            deposit: 1.0,
            initial_pheromone: 1.0,
            seed: 99,
        };
        let mut a = AntColonyTsp::new(cfg.clone(), problem.distances.clone());
        let mut b = AntColonyTsp::new(cfg, problem.distances.clone());
        let ra = a.run(&problem);
        let rb = b.run(&problem);
        assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    #[should_panic(expected = "exactly one objective")]
    fn multi_objective_panics() {
        let mut opt = AntColonyTsp::new(
            AntColonyTspConfig::default(),
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        );
        let _ = opt.run(&DummyMo);
    }
}
