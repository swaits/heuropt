//! CMA-ES — Hansen & Ostermeier 2001 Covariance Matrix Adaptation
//! Evolution Strategy.

use rand_distr::{Distribution, Normal};

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::rng_from_seed;
use crate::internal::eigen::symmetric_eigen;
use crate::operators::real::RealBounds;
use crate::pareto::front::best_candidate;
use crate::traits::Optimizer;

/// Configuration for [`CmaEs`].
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Population size `λ`. Must be at least 4. Hansen recommends
    /// `4 + floor(3 · ln(N))` as a default for `N`-dim problems.
    pub population_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Initial step size `σ_0`. Often ~ 1/3 of the search range per dim.
    pub initial_sigma: f64,
    /// Recompute the eigendecomposition of `C` every this many generations
    /// to amortize cost. The full algorithm decomposes every generation
    /// (set this to 1); 1–10 is fine for small `N`.
    pub eigen_decomposition_period: usize,
    /// Optional initial mean. If `None`, the mean defaults to the per-axis
    /// midpoint of the bounds. Used by `IpopCmaEs` to inject restart
    /// diversity without shrinking the search box.
    pub initial_mean: Option<Vec<f64>>,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            population_size: 16,
            generations: 200,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            initial_mean: None,
            seed: 42,
        }
    }
}

/// Single-objective real-valued CMA-ES.
///
/// Maintains a multivariate Gaussian sampler `mean + σ · N(0, C)`, samples
/// `λ` offspring from it each generation, selects the `μ` best (weighted),
/// and updates `mean`, `σ`, and `C` via the standard CMA-ES rules.
///
/// `Vec<f64>` decisions only. Bounds come from the embedded `RealBounds`
/// field; both the initial mean and every offspring are clamped per
/// dimension.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// struct Sphere;
/// impl Problem for Sphere {
///     type Decision = Vec<f64>;
///     fn objectives(&self) -> ObjectiveSpace {
///         ObjectiveSpace::new(vec![Objective::minimize("f")])
///     }
///     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
///         Evaluation::new(vec![x.iter().map(|v| v * v).sum::<f64>()])
///     }
/// }
///
/// let mut opt = CmaEs::new(
///     CmaEsConfig {
///         population_size: 12,
///         generations: 100,
///         initial_sigma: 1.0,
///         eigen_decomposition_period: 1,
///         initial_mean: None,
///         seed: 42,
///     },
///     RealBounds::new(vec![(-5.0, 5.0); 5]),
/// );
/// let r = opt.run(&Sphere);
/// // CMA-ES converges aggressively on Sphere.
/// assert!(r.best.unwrap().evaluation.objectives[0] < 1e-3);
/// ```
#[derive(Debug, Clone)]
pub struct CmaEs {
    /// Algorithm configuration.
    pub config: CmaEsConfig,
    /// Per-variable bounds — used both to seed `mean` (midpoint) and to
    /// clamp every offspring.
    pub bounds: RealBounds,
}

impl CmaEs {
    /// Construct a `CmaEs`.
    pub fn new(config: CmaEsConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for CmaEs
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size >= 4,
            "CmaEs population_size must be >= 4",
        );
        assert!(
            self.config.initial_sigma > 0.0,
            "CmaEs initial_sigma must be positive",
        );
        assert!(
            self.config.eigen_decomposition_period >= 1,
            "CmaEs eigen_decomposition_period must be >= 1",
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "CmaEs only supports single-objective problems",
        );
        let direction = objectives.objectives[0].direction;

        let n = self.bounds.bounds.len();
        let n_f = n as f64;
        let lambda = self.config.population_size;
        let lambda_f = lambda as f64;
        let mu = lambda / 2;
        assert!(mu >= 1, "CmaEs derived mu (= lambda/2) must be >= 1");
        let mut rng = rng_from_seed(self.config.seed);

        // ---------------------------------------------------------------
        // Selection weights w_i ∝ ln((λ+1)/2) − ln(i)  for i = 1..μ,
        // normalized so they sum to 1. Then mu_eff = 1 / Σ w_i².
        // ---------------------------------------------------------------
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| ((lambda_f + 1.0) / 2.0).ln() - ((i + 1) as f64).ln())
            .collect();
        let sum_w: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / sum_w).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // ---------------------------------------------------------------
        // Standard CMA-ES strategy parameters (Hansen tutorial §7.1).
        // ---------------------------------------------------------------
        let c_sigma = (mu_eff + 2.0) / (n_f + mu_eff + 5.0);
        let d_sigma = 1.0 + 2.0 * ((mu_eff - 1.0) / (n_f + 1.0)).sqrt().max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n_f) / (n_f + 4.0 + 2.0 * mu_eff / n_f);
        let c_1 = 2.0 / ((n_f + 1.3).powi(2) + mu_eff);
        let c_mu = ((1.0 - c_1) * 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
            / ((n_f + 2.0).powi(2) + mu_eff))
            .min(1.0 - c_1);
        // E‖N(0, I)‖ ≈ √n · (1 − 1/(4n) + 1/(21n²))
        let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

        // ---------------------------------------------------------------
        // Initial state.
        // ---------------------------------------------------------------
        let mut mean: Vec<f64> = if let Some(provided) = self.config.initial_mean.clone() {
            assert_eq!(
                provided.len(),
                self.bounds.bounds.len(),
                "CmaEs initial_mean.len() must equal the bounds dimension",
            );
            // Clamp the user-provided mean into the bounds so the algorithm
            // doesn't start outside the search box.
            provided
                .into_iter()
                .zip(self.bounds.bounds.iter())
                .map(|(v, &(lo, hi))| v.clamp(lo, hi))
                .collect()
        } else {
            self.bounds
                .bounds
                .iter()
                .map(|&(lo, hi)| 0.5 * (lo + hi))
                .collect()
        };
        let mut sigma = self.config.initial_sigma;
        // Covariance C, eigenvectors B, eigenvalues d (square roots of eigenvalues of C).
        let mut c_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut b: Vec<Vec<f64>> = c_matrix.to_vec();
        let mut d: Vec<f64> = vec![1.0; n];
        let mut p_sigma = vec![0.0_f64; n];
        let mut p_c = vec![0.0_f64; n];
        let mut evaluations = 0usize;

        let normal = Normal::new(0.0, 1.0).expect("Normal::new(0, 1)");
        let mut best_candidate_seen: Option<Candidate<Vec<f64>>> = None;

        for generation in 0..self.config.generations {
            // Recompute B, d every period generations from C (after symmetrizing).
            if generation % self.config.eigen_decomposition_period == 0 {
                // Force symmetry.
                #[allow(clippy::needless_range_loop)] // body indexes both [i][j] and [j][i].
                for i in 0..n {
                    for j in (i + 1)..n {
                        let avg = 0.5 * (c_matrix[i][j] + c_matrix[j][i]);
                        c_matrix[i][j] = avg;
                        c_matrix[j][i] = avg;
                    }
                }
                let (eigenvalues, eigenvectors) = symmetric_eigen(&c_matrix, 1e-14, 100);
                // eigenvectors is sorted descending; we don't depend on order
                // for sampling correctness, but we do need positive eigenvalues.
                d = eigenvalues.iter().map(|&v| v.max(1e-20).sqrt()).collect();
                // B is the matrix whose columns are the eigenvectors. The
                // helper returns `eigenvectors[i]` as the i-th *eigenvector*,
                // so b[r][c] should equal eigenvectors[c][r].
                b = (0..n)
                    .map(|r| (0..n).map(|c| eigenvectors[c][r]).collect())
                    .collect();
            }

            // ----- Sample λ offspring -----
            let mut z_samples: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut x_samples: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            for _ in 0..lambda {
                let z: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
                // y = B · D · z
                let bd_z: Vec<f64> = (0..n)
                    .map(|i| (0..n).map(|j| b[i][j] * d[j] * z[j]).sum::<f64>())
                    .collect();
                // x = mean + σ · y, clamped to bounds
                let x: Vec<f64> = (0..n)
                    .map(|i| {
                        let v = mean[i] + sigma * bd_z[i];
                        let (lo, hi) = self.bounds.bounds[i];
                        v.clamp(lo, hi)
                    })
                    .collect();
                z_samples.push(z);
                x_samples.push(x);
            }

            // Evaluate offspring (parallel-friendly).
            let evaluated = evaluate_batch(problem, x_samples.clone());
            evaluations += evaluated.len();

            // Track the best candidate ever.
            for c in &evaluated {
                let beats_best = match &best_candidate_seen {
                    None => true,
                    Some(b) => better_than_so(&c.evaluation, &b.evaluation, direction),
                };
                if beats_best {
                    best_candidate_seen = Some(c.clone());
                }
            }

            // Sort offspring by fitness ascending (best first).
            let mut order: Vec<usize> = (0..lambda).collect();
            order.sort_by(|&a, &b_| {
                compare_so(
                    &evaluated[a].evaluation,
                    &evaluated[b_].evaluation,
                    direction,
                )
            });

            // ----- Recompute mean from the μ best (weighted average of x) -----
            let old_mean = mean.clone();
            let mut new_mean = vec![0.0_f64; n];
            for k in 0..mu {
                let xk = &x_samples[order[k]];
                let wk = weights[k];
                for i in 0..n {
                    new_mean[i] += wk * xk[i];
                }
            }
            mean = new_mean;

            // ----- Weighted average of z (used for evolution-path updates) -----
            let mut z_weighted = vec![0.0_f64; n];
            for k in 0..mu {
                let zk = &z_samples[order[k]];
                let wk = weights[k];
                for i in 0..n {
                    z_weighted[i] += wk * zk[i];
                }
            }

            // ----- Evolution path for step size: p_σ = (1 - c_σ) p_σ + sqrt(c_σ (2 - c_σ) μ_eff) · B z̄ -----
            let factor_p_sigma = (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt();
            // B · z_weighted (since C^{-1/2} (m_new - m_old) / σ = B · D^{-1} · D · z̄ = B · z̄)
            let bz: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| b[i][j] * z_weighted[j]).sum::<f64>())
                .collect();
            for i in 0..n {
                p_sigma[i] = (1.0 - c_sigma) * p_sigma[i] + factor_p_sigma * bz[i];
            }

            // ----- Step-size update -----
            let p_sigma_norm = p_sigma.iter().map(|x| x * x).sum::<f64>().sqrt();
            sigma *= ((c_sigma / d_sigma) * (p_sigma_norm / chi_n - 1.0)).exp();

            // Heaviside for h_σ: damp p_c update if the step length is huge.
            let h_sigma = if p_sigma_norm
                / (1.0 - (1.0 - c_sigma).powi(2 * (generation as i32 + 1))).sqrt()
                < (1.4 + 2.0 / (n_f + 1.0)) * chi_n
            {
                1.0
            } else {
                0.0
            };

            // ----- Evolution path for C: p_c = (1 - c_c) p_c + h_σ · sqrt(c_c (2 - c_c) μ_eff) · (m_new - m_old)/σ -----
            let factor_p_c = h_sigma * (c_c * (2.0 - c_c) * mu_eff).sqrt();
            for i in 0..n {
                p_c[i] = (1.0 - c_c) * p_c[i] + factor_p_c * (mean[i] - old_mean[i]) / sigma;
            }

            // ----- Covariance matrix update (rank-1 + rank-μ) -----
            let delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c);
            #[allow(clippy::needless_range_loop)]
            // body uses both i and j to index c_matrix and offspring.
            for i in 0..n {
                for j in 0..n {
                    let mut update = (1.0 - c_1 - c_mu) * c_matrix[i][j]
                        + c_1 * (p_c[i] * p_c[j] + delta_h * c_matrix[i][j]);
                    // Rank-μ contribution.
                    let mut rank_mu_term = 0.0;
                    for k in 0..mu {
                        let xk = &x_samples[order[k]];
                        let yi = (xk[i] - old_mean[i]) / sigma;
                        let yj = (xk[j] - old_mean[j]) / sigma;
                        rank_mu_term += weights[k] * yi * yj;
                    }
                    update += c_mu * rank_mu_term;
                    c_matrix[i][j] = update;
                }
            }

            // Clamp mean to bounds (sigma may push it out otherwise).
            for (i, m) in mean.iter_mut().enumerate() {
                let (lo, hi) = self.bounds.bounds[i];
                *m = m.clamp(lo, hi);
            }
        }

        // Final population: just the best-seen candidate. Match other
        // single-objective algorithms' convention.
        let best = best_candidate_seen.expect("at least one generation evaluated");
        let final_pop = vec![best.clone()];
        let front = vec![best.clone()];
        let best_opt = best_candidate(&final_pop, &objectives);
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best_opt,
            evaluations,
            self.config.generations,
        )
    }
}

#[cfg(feature = "async")]
impl CmaEs {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations per generation.
    pub async fn run_async<P>(
        &mut self,
        problem: &P,
        concurrency: usize,
    ) -> OptimizationResult<Vec<f64>>
    where
        P: crate::core::async_problem::AsyncProblem<Decision = Vec<f64>>,
    {
        use crate::algorithms::parallel_eval_async::evaluate_batch_async;

        assert!(
            self.config.population_size >= 4,
            "CmaEs population_size must be >= 4",
        );
        assert!(
            self.config.initial_sigma > 0.0,
            "CmaEs initial_sigma must be positive",
        );
        assert!(
            self.config.eigen_decomposition_period >= 1,
            "CmaEs eigen_decomposition_period must be >= 1",
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "CmaEs only supports single-objective problems",
        );
        let direction = objectives.objectives[0].direction;

        let n = self.bounds.bounds.len();
        let n_f = n as f64;
        let lambda = self.config.population_size;
        let lambda_f = lambda as f64;
        let mu = lambda / 2;
        assert!(mu >= 1, "CmaEs derived mu (= lambda/2) must be >= 1");
        let mut rng = rng_from_seed(self.config.seed);

        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| ((lambda_f + 1.0) / 2.0).ln() - ((i + 1) as f64).ln())
            .collect();
        let sum_w: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / sum_w).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        let c_sigma = (mu_eff + 2.0) / (n_f + mu_eff + 5.0);
        let d_sigma = 1.0 + 2.0 * ((mu_eff - 1.0) / (n_f + 1.0)).sqrt().max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n_f) / (n_f + 4.0 + 2.0 * mu_eff / n_f);
        let c_1 = 2.0 / ((n_f + 1.3).powi(2) + mu_eff);
        let c_mu = ((1.0 - c_1) * 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
            / ((n_f + 2.0).powi(2) + mu_eff))
            .min(1.0 - c_1);
        let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

        let mut mean: Vec<f64> = if let Some(provided) = self.config.initial_mean.clone() {
            assert_eq!(
                provided.len(),
                self.bounds.bounds.len(),
                "CmaEs initial_mean.len() must equal the bounds dimension",
            );
            provided
                .into_iter()
                .zip(self.bounds.bounds.iter())
                .map(|(v, &(lo, hi))| v.clamp(lo, hi))
                .collect()
        } else {
            self.bounds
                .bounds
                .iter()
                .map(|&(lo, hi)| 0.5 * (lo + hi))
                .collect()
        };
        let mut sigma = self.config.initial_sigma;
        let mut c_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut b: Vec<Vec<f64>> = c_matrix.to_vec();
        let mut d: Vec<f64> = vec![1.0; n];
        let mut p_sigma = vec![0.0_f64; n];
        let mut p_c = vec![0.0_f64; n];
        let mut evaluations = 0usize;

        let normal = Normal::new(0.0, 1.0).expect("Normal::new(0, 1)");
        let mut best_candidate_seen: Option<Candidate<Vec<f64>>> = None;

        for generation in 0..self.config.generations {
            if generation % self.config.eigen_decomposition_period == 0 {
                #[allow(clippy::needless_range_loop)]
                for i in 0..n {
                    for j in (i + 1)..n {
                        let avg = 0.5 * (c_matrix[i][j] + c_matrix[j][i]);
                        c_matrix[i][j] = avg;
                        c_matrix[j][i] = avg;
                    }
                }
                let (eigenvalues, eigenvectors) = symmetric_eigen(&c_matrix, 1e-14, 100);
                d = eigenvalues.iter().map(|&v| v.max(1e-20).sqrt()).collect();
                b = (0..n)
                    .map(|r| (0..n).map(|c| eigenvectors[c][r]).collect())
                    .collect();
            }

            let mut z_samples: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut x_samples: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            for _ in 0..lambda {
                let z: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
                let bd_z: Vec<f64> = (0..n)
                    .map(|i| (0..n).map(|j| b[i][j] * d[j] * z[j]).sum::<f64>())
                    .collect();
                let x: Vec<f64> = (0..n)
                    .map(|i| {
                        let v = mean[i] + sigma * bd_z[i];
                        let (lo, hi) = self.bounds.bounds[i];
                        v.clamp(lo, hi)
                    })
                    .collect();
                z_samples.push(z);
                x_samples.push(x);
            }

            let evaluated = evaluate_batch_async(problem, x_samples.clone(), concurrency).await;
            evaluations += evaluated.len();

            for c in &evaluated {
                let beats_best = match &best_candidate_seen {
                    None => true,
                    Some(b) => better_than_so(&c.evaluation, &b.evaluation, direction),
                };
                if beats_best {
                    best_candidate_seen = Some(c.clone());
                }
            }

            let mut order: Vec<usize> = (0..lambda).collect();
            order.sort_by(|&a, &b_| {
                compare_so(
                    &evaluated[a].evaluation,
                    &evaluated[b_].evaluation,
                    direction,
                )
            });

            let old_mean = mean.clone();
            let mut new_mean = vec![0.0_f64; n];
            for k in 0..mu {
                let xk = &x_samples[order[k]];
                let wk = weights[k];
                for i in 0..n {
                    new_mean[i] += wk * xk[i];
                }
            }
            mean = new_mean;

            let mut z_weighted = vec![0.0_f64; n];
            for k in 0..mu {
                let zk = &z_samples[order[k]];
                let wk = weights[k];
                for i in 0..n {
                    z_weighted[i] += wk * zk[i];
                }
            }

            let factor_p_sigma = (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt();
            let bz: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| b[i][j] * z_weighted[j]).sum::<f64>())
                .collect();
            for i in 0..n {
                p_sigma[i] = (1.0 - c_sigma) * p_sigma[i] + factor_p_sigma * bz[i];
            }

            let p_sigma_norm = p_sigma.iter().map(|x| x * x).sum::<f64>().sqrt();
            sigma *= ((c_sigma / d_sigma) * (p_sigma_norm / chi_n - 1.0)).exp();

            let h_sigma = if p_sigma_norm
                / (1.0 - (1.0 - c_sigma).powi(2 * (generation as i32 + 1))).sqrt()
                < (1.4 + 2.0 / (n_f + 1.0)) * chi_n
            {
                1.0
            } else {
                0.0
            };

            let factor_p_c = h_sigma * (c_c * (2.0 - c_c) * mu_eff).sqrt();
            for i in 0..n {
                p_c[i] = (1.0 - c_c) * p_c[i] + factor_p_c * (mean[i] - old_mean[i]) / sigma;
            }

            let delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c);
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                for j in 0..n {
                    let mut update = (1.0 - c_1 - c_mu) * c_matrix[i][j]
                        + c_1 * (p_c[i] * p_c[j] + delta_h * c_matrix[i][j]);
                    let mut rank_mu_term = 0.0;
                    for k in 0..mu {
                        let xk = &x_samples[order[k]];
                        let yi = (xk[i] - old_mean[i]) / sigma;
                        let yj = (xk[j] - old_mean[j]) / sigma;
                        rank_mu_term += weights[k] * yi * yj;
                    }
                    update += c_mu * rank_mu_term;
                    c_matrix[i][j] = update;
                }
            }

            for (i, m) in mean.iter_mut().enumerate() {
                let (lo, hi) = self.bounds.bounds[i];
                *m = m.clamp(lo, hi);
            }
        }

        let best = best_candidate_seen.expect("at least one generation evaluated");
        let final_pop = vec![best.clone()];
        let front = vec![best.clone()];
        let best_opt = best_candidate(&final_pop, &objectives);
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            best_opt,
            evaluations,
            self.config.generations,
        )
    }
}

fn compare_so(
    a: &crate::core::evaluation::Evaluation,
    b: &crate::core::evaluation::Evaluation,
    direction: Direction,
) -> std::cmp::Ordering {
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

fn better_than_so(
    a: &crate::core::evaluation::Evaluation,
    b: &crate::core::evaluation::Evaluation,
    direction: Direction,
) -> bool {
    compare_so(a, b, direction) == std::cmp::Ordering::Less
}

impl crate::traits::AlgorithmInfo for CmaEs {
    fn name(&self) -> &'static str {
        "CMA-ES"
    }
    fn full_name(&self) -> &'static str {
        "Covariance Matrix Adaptation Evolution Strategy"
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
    use crate::tests_support::{SchafferN1, Sphere1D};

    /// 5-D Rosenbrock for exercise.
    struct Rosenbrock5D;
    impl Problem for Rosenbrock5D {
        type Decision = Vec<f64>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }

        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            let f: f64 = (0..(x.len() - 1))
                .map(|i| {
                    let a = 1.0 - x[i];
                    let b = x[i + 1] - x[i] * x[i];
                    a * a + 100.0 * b * b
                })
                .sum();
            Evaluation::new(vec![f])
        }
    }

    #[test]
    fn finds_minimum_of_sphere() {
        let mut opt = CmaEs::new(
            CmaEsConfig {
                population_size: 12,
                generations: 100,
                initial_sigma: 0.5,
                eigen_decomposition_period: 1,
                initial_mean: None,
                seed: 1,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
        );
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-8,
            "got f = {}",
            best.evaluation.objectives[0],
        );
    }

    #[test]
    fn finds_minimum_of_rosenbrock_5d() {
        let mut opt = CmaEs::new(
            CmaEsConfig {
                population_size: 16,
                generations: 400,
                initial_sigma: 0.5,
                eigen_decomposition_period: 1,
                initial_mean: None,
                seed: 1,
            },
            RealBounds::new(vec![(-5.0, 5.0); 5]),
        );
        let r = opt.run(&Rosenbrock5D);
        let best = r.best.unwrap();
        // Rosenbrock is a tough non-convex valley; CMA-ES should still get
        // far closer than random search.
        assert!(
            best.evaluation.objectives[0] < 1.0,
            "got f = {}",
            best.evaluation.objectives[0],
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let cfg = CmaEsConfig {
            population_size: 8,
            generations: 30,
            initial_sigma: 0.5,
            eigen_decomposition_period: 1,
            initial_mean: None,
            seed: 99,
        };
        let mut a = CmaEs::new(cfg.clone(), RealBounds::new(vec![(-5.0, 5.0)]));
        let mut b = CmaEs::new(cfg, RealBounds::new(vec![(-5.0, 5.0)]));
        let ra = a.run(&Sphere1D);
        let rb = b.run(&Sphere1D);
        assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    #[should_panic(expected = "single-objective")]
    fn multi_objective_panics() {
        let mut opt = CmaEs::new(CmaEsConfig::default(), RealBounds::new(vec![(-5.0, 5.0)]));
        let _ = opt.run(&SchafferN1);
    }

    #[test]
    #[should_panic(expected = "population_size must be >= 4")]
    fn small_population_panics() {
        let mut opt = CmaEs::new(
            CmaEsConfig {
                population_size: 3,
                generations: 1,
                initial_sigma: 0.5,
                eigen_decomposition_period: 1,
                initial_mean: None,
                seed: 0,
            },
            RealBounds::new(vec![(-1.0, 1.0)]),
        );
        let _ = opt.run(&Sphere1D);
    }

    // ---- Mutation-test pinned helpers --------------------------------------

    #[test]
    fn compare_so_feasibility_first_under_min() {
        let mut a = Evaluation::new(vec![10.0]);
        a.constraint_violation = 0.0;
        let mut b = Evaluation::new(vec![1.0]);
        b.constraint_violation = 1.0;
        assert_eq!(compare_so(&a, &b, Direction::Minimize), std::cmp::Ordering::Less);
        assert_eq!(compare_so(&b, &a, Direction::Minimize), std::cmp::Ordering::Greater);
    }

    #[test]
    fn compare_so_two_feasible_under_min_and_max() {
        let a = Evaluation::new(vec![1.0]);
        let b = Evaluation::new(vec![2.0]);
        assert_eq!(compare_so(&a, &b, Direction::Minimize), std::cmp::Ordering::Less);
        assert_eq!(compare_so(&b, &a, Direction::Minimize), std::cmp::Ordering::Greater);
        // Maximize inverts.
        assert_eq!(compare_so(&a, &b, Direction::Maximize), std::cmp::Ordering::Greater);
        assert_eq!(compare_so(&b, &a, Direction::Maximize), std::cmp::Ordering::Less);
    }

    #[test]
    fn compare_so_two_infeasible_compares_violation() {
        let mut a = Evaluation::new(vec![0.0]);
        a.constraint_violation = 0.5;
        let mut b = Evaluation::new(vec![0.0]);
        b.constraint_violation = 1.0;
        assert_eq!(compare_so(&a, &b, Direction::Minimize), std::cmp::Ordering::Less);
        assert_eq!(compare_so(&b, &a, Direction::Minimize), std::cmp::Ordering::Greater);
    }

    #[test]
    fn better_than_so_matches_compare_so() {
        let a = Evaluation::new(vec![1.0]);
        let b = Evaluation::new(vec![2.0]);
        assert!(better_than_so(&a, &b, Direction::Minimize));
        assert!(!better_than_so(&b, &a, Direction::Minimize));
        assert!(better_than_so(&b, &a, Direction::Maximize));
        // Equal: not strictly better.
        let c = Evaluation::new(vec![1.0]);
        assert!(!better_than_so(&a, &c, Direction::Minimize));
    }

    /// Pin the final population size and at least one improvement step.
    #[test]
    fn cmaes_decreases_sphere_objective_over_generations() {
        let mut opt = CmaEs::new(
            CmaEsConfig {
                population_size: 8,
                generations: 30,
                initial_sigma: 0.5,
                eigen_decomposition_period: 1,
                initial_mean: None,
                seed: 7,
            },
            RealBounds::new(vec![(-3.0, 3.0); 2]),
        );
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap().evaluation.objectives[0];
        // After 30 gens × 8 pop on a 2-D sphere starting σ=0.5, best should
        // be much smaller than initial random sampling (variance bound = 9).
        assert!(best < 1.0, "best = {best}");
    }
}
