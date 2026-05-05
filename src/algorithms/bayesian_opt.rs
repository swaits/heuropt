//! `BayesianOpt` — Gaussian-process-based Bayesian Optimization.
//!
//! Sample-efficient sequential optimizer for expensive black-box
//! single-objective real-valued problems. Builds a GP surrogate of the
//! objective and selects the next evaluation point by maximizing the
//! Expected Improvement acquisition.

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::internal::cholesky::{cholesky, solve};
use crate::operators::real::RealBounds;
use crate::traits::Optimizer;

/// Configuration for [`BayesianOpt`].
#[derive(Debug, Clone)]
pub struct BayesianOptConfig {
    /// Number of uniform-random initial samples before the BO loop starts.
    /// Hansen-style rule of thumb: 5×dim, but small budgets often work.
    pub initial_samples: usize,
    /// Number of BO iterations after the initial design.
    pub iterations: usize,
    /// Per-axis RBF length scales (one per dimension). Smaller = more
    /// "wiggly" surrogate. Reasonable default: 0.2 × bound range per axis.
    pub length_scales: Option<Vec<f64>>,
    /// GP signal variance (the "amplitude" of the surrogate).
    pub signal_variance: f64,
    /// GP noise variance (small jitter to keep the kernel matrix SPD even
    /// at duplicate or near-duplicate points).
    pub noise_variance: f64,
    /// Number of random samples used to maximize the acquisition function
    /// each step. The best-EI sample is chosen as the next evaluation.
    pub acquisition_samples: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for BayesianOptConfig {
    fn default() -> Self {
        Self {
            initial_samples: 10,
            iterations: 40,
            length_scales: None,
            signal_variance: 1.0,
            noise_variance: 1e-6,
            acquisition_samples: 1_000,
            seed: 42,
        }
    }
}

/// Gaussian-process Bayesian Optimization with Expected Improvement.
///
/// `Vec<f64>` decisions only. Single-objective only. Targets expensive
/// evaluation budgets (50–500). The GP kernel is anisotropic RBF; the
/// acquisition function is EI; both are optimized by best-of-N random
/// sampling each step (simple, predictable cost).
#[derive(Debug, Clone)]
pub struct BayesianOpt {
    /// Algorithm configuration.
    pub config: BayesianOptConfig,
    /// Per-variable bounds.
    pub bounds: RealBounds,
}

impl BayesianOpt {
    /// Construct a `BayesianOpt`.
    pub fn new(config: BayesianOptConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for BayesianOpt
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.initial_samples >= 2,
            "BayesianOpt initial_samples must be >= 2",
        );
        assert!(self.config.signal_variance > 0.0, "BayesianOpt signal_variance must be > 0");
        assert!(self.config.noise_variance > 0.0, "BayesianOpt noise_variance must be > 0");
        assert!(
            self.config.acquisition_samples >= 1,
            "BayesianOpt acquisition_samples must be >= 1",
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "BayesianOpt requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let dim = self.bounds.bounds.len();
        if let Some(ls) = &self.config.length_scales {
            assert_eq!(ls.len(), dim, "BayesianOpt length_scales.len() must equal dim");
        }
        let length_scales: Vec<f64> = self
            .config
            .length_scales
            .clone()
            .unwrap_or_else(|| {
                self.bounds
                    .bounds
                    .iter()
                    .map(|&(lo, hi)| 0.2 * (hi - lo).max(1e-9))
                    .collect()
            });
        let mut rng = rng_from_seed(self.config.seed);

        // ---------------- Initial random design ----------------
        let mut decisions: Vec<Vec<f64>> = Vec::with_capacity(
            self.config.initial_samples + self.config.iterations,
        );
        let mut targets: Vec<f64> = Vec::with_capacity(decisions.capacity());
        let mut evaluations = Vec::with_capacity(decisions.capacity());
        for _ in 0..self.config.initial_samples {
            let x = sample_uniform_in_bounds(&self.bounds, &mut rng);
            let e = problem.evaluate(&x);
            // GP works on minimization-oriented "want low" targets.
            let t = oriented_target(&e, direction);
            decisions.push(x);
            targets.push(t);
            evaluations.push(e);
        }

        // ---------------- Sequential BO loop ----------------
        for _ in 0..self.config.iterations {
            // Build the GP posterior around current observations.
            let posterior = match GpPosterior::fit(
                &decisions,
                &targets,
                &length_scales,
                self.config.signal_variance,
                self.config.noise_variance,
            ) {
                Ok(p) => p,
                Err(_) => {
                    // SPD failure (typically numerical): fall back to a
                    // single uniform-random sample this step.
                    let x = sample_uniform_in_bounds(&self.bounds, &mut rng);
                    let e = problem.evaluate(&x);
                    targets.push(oriented_target(&e, direction));
                    decisions.push(x);
                    evaluations.push(e);
                    continue;
                }
            };

            let best_target =
                targets.iter().cloned().fold(f64::INFINITY, f64::min);

            // Maximize EI by best-of-N random sampling.
            let mut best_x = sample_uniform_in_bounds(&self.bounds, &mut rng);
            let mut best_ei = -f64::INFINITY;
            for _ in 0..self.config.acquisition_samples {
                let cand = sample_uniform_in_bounds(&self.bounds, &mut rng);
                let (mu, sigma) = posterior.predict(&cand);
                let ei = expected_improvement(mu, sigma, best_target);
                if ei > best_ei {
                    best_ei = ei;
                    best_x = cand;
                }
            }

            let e = problem.evaluate(&best_x);
            targets.push(oriented_target(&e, direction));
            decisions.push(best_x);
            evaluations.push(e);
        }

        // Build the final population/best.
        let final_pop: Vec<Candidate<Vec<f64>>> = decisions
            .into_iter()
            .zip(evaluations)
            .map(|(d, e)| Candidate::new(d, e))
            .collect();
        let mut best_idx = 0;
        for i in 1..final_pop.len() {
            if better(&final_pop[i].evaluation, &final_pop[best_idx].evaluation, direction) {
                best_idx = i;
            }
        }
        let total_evaluations = final_pop.len();
        let best = final_pop[best_idx].clone();
        let front = vec![best.clone()];
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            Some(best),
            total_evaluations,
            self.config.iterations + self.config.initial_samples,
        )
    }
}

/// Convert an Evaluation into a "smaller is better" target. For Maximize
/// problems we negate; infeasibles get a large penalty proportional to
/// the violation magnitude.
fn oriented_target(e: &Evaluation, direction: Direction) -> f64 {
    let base = match direction {
        Direction::Minimize => e.objectives[0],
        Direction::Maximize => -e.objectives[0],
    };
    if e.is_feasible() {
        base
    } else {
        // Penalize so the GP learns to avoid this region.
        base + 1e6 * e.constraint_violation
    }
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

fn sample_uniform_in_bounds(bounds: &RealBounds, rng: &mut Rng) -> Vec<f64> {
    bounds
        .bounds
        .iter()
        .map(|&(lo, hi)| if lo == hi { lo } else { lo + (hi - lo) * rng.random::<f64>() })
        .collect()
}

/// Anisotropic RBF kernel: `k(x, y) = σ² · exp(-0.5 · Σ ((x_i - y_i)/ℓ_i)²)`.
fn rbf_kernel(x: &[f64], y: &[f64], length_scales: &[f64], signal_variance: f64) -> f64 {
    let mut sum = 0.0;
    for ((a, b), l) in x.iter().zip(y.iter()).zip(length_scales.iter()) {
        let d = (a - b) / l.max(1e-12);
        sum += d * d;
    }
    signal_variance * (-0.5 * sum).exp()
}

struct GpPosterior {
    decisions: Vec<Vec<f64>>,
    length_scales: Vec<f64>,
    signal_variance: f64,
    /// `α = K^{-1} · y_target`, precomputed for the mean prediction.
    alpha: Vec<f64>,
    /// Cholesky factor of `K + σ_n² · I`, kept for variance prediction.
    chol_l: Vec<Vec<f64>>,
}

impl GpPosterior {
    fn fit(
        decisions: &[Vec<f64>],
        targets: &[f64],
        length_scales: &[f64],
        signal_variance: f64,
        noise_variance: f64,
    ) -> Result<Self, &'static str> {
        let n = decisions.len();
        let mut k = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let v = rbf_kernel(&decisions[i], &decisions[j], length_scales, signal_variance);
                k[i][j] = v;
                k[j][i] = v;
            }
            k[i][i] += noise_variance;
        }
        let chol_l = cholesky(&k)?;
        let alpha = solve(&chol_l, targets);
        Ok(Self {
            decisions: decisions.to_vec(),
            length_scales: length_scales.to_vec(),
            signal_variance,
            alpha,
            chol_l,
        })
    }

    fn predict(&self, x: &[f64]) -> (f64, f64) {
        let n = self.decisions.len();
        let mut k_star = vec![0.0_f64; n];
        for (i, k_star_i) in k_star.iter_mut().enumerate() {
            *k_star_i = rbf_kernel(x, &self.decisions[i], &self.length_scales, self.signal_variance);
        }
        let _ = n;
        let mu: f64 = k_star.iter().zip(self.alpha.iter()).map(|(a, b)| a * b).sum();
        // Var = k(x,x) - k_star^T · K^{-1} · k_star
        // Compute K^{-1}·k_star = solve_upper_transpose(L, solve_lower(L, k_star))
        let v_temp = crate::internal::cholesky::solve_lower(&self.chol_l, &k_star);
        let v: f64 = v_temp.iter().map(|x| x * x).sum();
        let var = (self.signal_variance - v).max(0.0);
        (mu, var.sqrt())
    }
}

/// Expected Improvement (minimization-oriented) at a point with predicted
/// mean `mu` and standard deviation `sigma`, given the current best
/// observed target `f_best`. Returns 0 if `sigma` is effectively zero.
fn expected_improvement(mu: f64, sigma: f64, f_best: f64) -> f64 {
    if sigma < 1e-12 {
        return 0.0;
    }
    let improvement = f_best - mu;
    let z = improvement / sigma;
    improvement * normal_cdf(z) + sigma * normal_pdf(z)
}

fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(z: f64) -> f64 {
    // Approximate Φ(z) via erf. Abramowitz & Stegun 7.1.26 series-free
    // rational approximation good to ~1.5e-7.
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Numerical Recipes-style erf, accurate to ~1e-7.
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(seed: u64) -> BayesianOpt {
        BayesianOpt::new(
            BayesianOptConfig {
                initial_samples: 5,
                iterations: 25,
                length_scales: None,
                signal_variance: 1.0,
                noise_variance: 1e-6,
                acquisition_samples: 500,
                seed,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
        )
    }

    #[test]
    fn finds_minimum_of_sphere_quickly() {
        // BO's whole point is sample efficiency: 30 evals ought to be
        // enough for a 1-D sphere. (Pop-based methods needed thousands.)
        let mut opt = make_optimizer(1);
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-3,
            "BO should converge fast on 1-D sphere; got f = {}",
            best.evaluation.objectives[0],
        );
        assert!(r.evaluations <= 30 + 1);
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

    #[test]
    #[should_panic(expected = "length_scales.len() must equal dim")]
    fn length_scales_dim_mismatch_panics() {
        let mut opt = BayesianOpt::new(
            BayesianOptConfig {
                initial_samples: 5,
                iterations: 5,
                length_scales: Some(vec![1.0, 1.0]),
                signal_variance: 1.0,
                noise_variance: 1e-6,
                acquisition_samples: 100,
                seed: 0,
            },
            RealBounds::new(vec![(-1.0, 1.0)]),
        );
        let _ = opt.run(&Sphere1D);
    }
}
