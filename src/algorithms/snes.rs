//! `SeparableNes` — Wierstra et al. 2008/2014 Natural Evolution Strategy
//! with diagonal covariance (sNES).

use rand_distr::{Distribution, Normal};

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::operators::real::RealBounds;
use crate::traits::Optimizer;

/// Configuration for [`SeparableNes`].
#[derive(Debug, Clone)]
pub struct SeparableNesConfig {
    /// Population size `λ` per generation. NES recommends `4 + ⌊3·ln(n)⌋`.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Initial step size `σ_0`.
    pub initial_sigma: f64,
    /// Mean learning rate `η_μ`. NES default is 1.0.
    pub mean_learning_rate: f64,
    /// Sigma learning rate `η_σ`. NES default is `(3 + ln(n)) / (5·sqrt(n))`,
    /// computed at runtime if you set this to `None`.
    pub sigma_learning_rate: Option<f64>,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for SeparableNesConfig {
    fn default() -> Self {
        Self {
            population_size: 16,
            generations: 200,
            initial_sigma: 0.5,
            mean_learning_rate: 1.0,
            sigma_learning_rate: None,
            seed: 42,
        }
    }
}

/// Separable Natural Evolution Strategy (sNES).
///
/// `Vec<f64>` decisions only. Single-objective only. Maintains a sampling
/// distribution `N(μ, diag(σ²))` and updates `μ`, `σ` each generation by
/// following the natural gradient of expected fitness, with rank-shaped
/// fitness utilities for invariance to monotone transforms of the
/// objective.
#[derive(Debug, Clone)]
pub struct SeparableNes {
    /// Algorithm configuration.
    pub config: SeparableNesConfig,
    /// Per-variable bounds — used to seed `μ` (midpoint) and clamp every
    /// sampled offspring.
    pub bounds: RealBounds,
}

impl SeparableNes {
    /// Construct a `SeparableNes`.
    pub fn new(config: SeparableNesConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for SeparableNes
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size >= 2,
            "SeparableNes population_size must be >= 2",
        );
        assert!(self.config.initial_sigma > 0.0, "SeparableNes initial_sigma must be > 0");
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "SeparableNes requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let n = self.bounds.bounds.len();
        let lambda = self.config.population_size;
        let mut rng = rng_from_seed(self.config.seed);

        // Initial state.
        let mut mean: Vec<f64> = self
            .bounds
            .bounds
            .iter()
            .map(|&(lo, hi)| 0.5 * (lo + hi))
            .collect();
        let mut sigma = vec![self.config.initial_sigma; n];

        // Default sigma learning rate (Wierstra et al. 2014, Eq. 11).
        let eta_sigma = self.config.sigma_learning_rate.unwrap_or_else(|| {
            (3.0 + (n as f64).ln()) / (5.0 * (n as f64).sqrt())
        });
        let eta_mean = self.config.mean_learning_rate;

        // Rank utilities — the standard NES weighting:
        //   u_i = max(0, ln(λ/2 + 1) - ln(i)) / Σ - 1/λ
        // (positive total mass, zero sum after the shift).
        let utilities = nes_utilities(lambda);

        let mut best_seen: Option<Candidate<Vec<f64>>> = None;
        let mut total_evaluations = 0usize;

        for _ in 0..self.config.generations {
            // Sample λ offspring.
            let mut z_samples: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut x_samples: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut evals: Vec<Evaluation> = Vec::with_capacity(lambda);
            for _ in 0..lambda {
                let z: Vec<f64> = (0..n)
                    .map(|_| Normal::new(0.0, 1.0).unwrap().sample(&mut rng))
                    .collect();
                let x: Vec<f64> = (0..n)
                    .map(|j| {
                        let v = mean[j] + sigma[j] * z[j];
                        let (lo, hi) = self.bounds.bounds[j];
                        v.clamp(lo, hi)
                    })
                    .collect();
                let e = problem.evaluate(&x);
                total_evaluations += 1;
                let beats_best = match &best_seen {
                    None => true,
                    Some(b) => better(&e, &b.evaluation, direction),
                };
                if beats_best {
                    best_seen = Some(Candidate::new(x.clone(), e.clone()));
                }
                z_samples.push(z);
                x_samples.push(x);
                evals.push(e);
            }

            // Sort offspring best → worst (so utility[0] goes to the best).
            let mut order: Vec<usize> = (0..lambda).collect();
            order.sort_by(|&a, &b| compare(&evals[a], &evals[b], direction));

            // Update mean: μ ← μ + η_μ · σ · Σ u_i · z_i
            let mut grad_mean = vec![0.0_f64; n];
            for k in 0..lambda {
                let u = utilities[k];
                let z = &z_samples[order[k]];
                for j in 0..n {
                    grad_mean[j] += u * z[j];
                }
            }
            for j in 0..n {
                mean[j] += eta_mean * sigma[j] * grad_mean[j];
                let (lo, hi) = self.bounds.bounds[j];
                mean[j] = mean[j].clamp(lo, hi);
            }

            // Update sigma: σ_j ← σ_j · exp((η_σ/2) · Σ u_i · (z_i,j² - 1))
            for j in 0..n {
                let mut grad_sigma_j = 0.0;
                for k in 0..lambda {
                    let u = utilities[k];
                    let z = &z_samples[order[k]];
                    grad_sigma_j += u * (z[j] * z[j] - 1.0);
                }
                sigma[j] *= (0.5 * eta_sigma * grad_sigma_j).exp();
                if !sigma[j].is_finite() || sigma[j] < 1e-30 {
                    sigma[j] = 1e-30;
                }
            }
        }

        let best = best_seen.expect("at least one generation evaluated");
        let population = Population::new(vec![best.clone()]);
        let front = vec![best.clone()];
        OptimizationResult::new(
            population,
            front,
            Some(best),
            total_evaluations,
            self.config.generations,
        )
    }
}

fn nes_utilities(lambda: usize) -> Vec<f64> {
    let half = lambda as f64 / 2.0 + 1.0;
    let raw: Vec<f64> = (0..lambda)
        .map(|i| {
            let v = half.ln() - ((i + 1) as f64).ln();
            v.max(0.0)
        })
        .collect();
    let sum: f64 = raw.iter().sum::<f64>().max(1e-12);
    let inv_lambda = 1.0 / lambda as f64;
    raw.iter().map(|u| u / sum - inv_lambda).collect()
}

fn compare(a: &Evaluation, b: &Evaluation, direction: Direction) -> std::cmp::Ordering {
    match (a.is_feasible(), b.is_feasible()) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a
            .constraint_violation
            .partial_cmp(&b.constraint_violation)
            .unwrap_or(std::cmp::Ordering::Equal),
        (true, true) => match direction {
            Direction::Minimize => a.objectives[0]
                .partial_cmp(&b.objectives[0])
                .unwrap_or(std::cmp::Ordering::Equal),
            Direction::Maximize => b.objectives[0]
                .partial_cmp(&a.objectives[0])
                .unwrap_or(std::cmp::Ordering::Equal),
        },
    }
}

fn better(a: &Evaluation, b: &Evaluation, direction: Direction) -> bool {
    compare(a, b, direction) == std::cmp::Ordering::Less
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(seed: u64) -> SeparableNes {
        SeparableNes::new(
            SeparableNesConfig {
                population_size: 16,
                generations: 200,
                initial_sigma: 0.5,
                mean_learning_rate: 1.0,
                sigma_learning_rate: None,
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
            best.evaluation.objectives[0] < 1e-6,
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
    fn utilities_sum_to_zero() {
        let u = nes_utilities(8);
        let s: f64 = u.iter().sum();
        assert!(s.abs() < 1e-12, "utilities sum to {s}, not 0");
    }

    #[test]
    #[should_panic(expected = "exactly one objective")]
    fn multi_objective_panics() {
        let mut opt = make_optimizer(0);
        let _ = opt.run(&SchafferN1);
    }
}
