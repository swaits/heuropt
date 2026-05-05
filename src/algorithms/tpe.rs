//! `Tpe` — Bergstra et al. 2011 Tree-structured Parzen Estimator.

use rand::Rng as _;
use rand_distr::{Distribution, Normal};

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::operators::real::RealBounds;
use crate::traits::Optimizer;

/// Configuration for [`Tpe`].
#[derive(Debug, Clone)]
pub struct TpeConfig {
    /// Number of uniform-random initial samples before the TPE loop starts.
    pub initial_samples: usize,
    /// Number of TPE iterations after the initial design.
    pub iterations: usize,
    /// Top-γ fraction of observations classified as "good." Bergstra
    /// et al. recommend γ = 0.25.
    pub good_fraction: f64,
    /// Number of candidate samples drawn from the 'good' KDE per step.
    /// The one with the largest `l(x) / g(x)` ratio is chosen.
    pub candidate_samples: usize,
    /// Bandwidth multiplier on the per-axis KDE (Scott's rule × this
    /// factor). 1.0 is the standard rule; 0.5–2.0 is the practical range.
    pub bandwidth_factor: f64,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            initial_samples: 10,
            iterations: 90,
            good_fraction: 0.25,
            candidate_samples: 24,
            bandwidth_factor: 1.0,
            seed: 42,
        }
    }
}

/// Tree-structured Parzen Estimator.
///
/// Sample-efficient sequential optimizer for `Vec<f64>` decisions. Unlike
/// `BayesianOpt`, no GP — TPE models `p(x | y < y*)` and `p(x | y >= y*)`
/// as per-axis Gaussian KDEs and picks the next candidate by maximizing
/// the ratio of the two densities.
#[derive(Debug, Clone)]
pub struct Tpe {
    /// Algorithm configuration.
    pub config: TpeConfig,
    /// Per-variable bounds.
    pub bounds: RealBounds,
}

impl Tpe {
    /// Construct a `Tpe`.
    pub fn new(config: TpeConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for Tpe
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.initial_samples >= 2,
            "Tpe initial_samples must be >= 2"
        );
        assert!(
            self.config.good_fraction > 0.0 && self.config.good_fraction < 1.0,
            "Tpe good_fraction must be in (0, 1)",
        );
        assert!(
            self.config.candidate_samples >= 1,
            "Tpe candidate_samples must be >= 1",
        );
        assert!(
            self.config.bandwidth_factor > 0.0,
            "Tpe bandwidth_factor must be > 0"
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "Tpe requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let dim = self.bounds.bounds.len();
        let mut rng = rng_from_seed(self.config.seed);

        let mut decisions: Vec<Vec<f64>> = Vec::new();
        let mut targets: Vec<f64> = Vec::new();
        let mut evals: Vec<Evaluation> = Vec::new();
        for _ in 0..self.config.initial_samples {
            let x = sample_uniform_in_bounds(&self.bounds, &mut rng);
            let e = problem.evaluate(&x);
            targets.push(oriented_target(&e, direction));
            decisions.push(x);
            evals.push(e);
        }

        for _ in 0..self.config.iterations {
            // Split into good vs bad observations.
            let (good_idx, bad_idx) = split_good_bad(&targets, self.config.good_fraction);

            // Sample candidates from the good KDE.
            let mut best_x: Option<Vec<f64>> = None;
            let mut best_ratio = f64::NEG_INFINITY;
            for _ in 0..self.config.candidate_samples {
                let cand = sample_from_kde(
                    &decisions,
                    &good_idx,
                    &self.bounds,
                    self.config.bandwidth_factor,
                    &mut rng,
                );
                let l = log_kde_density(
                    &cand,
                    &decisions,
                    &good_idx,
                    &self.bounds,
                    self.config.bandwidth_factor,
                );
                let g = log_kde_density(
                    &cand,
                    &decisions,
                    &bad_idx,
                    &self.bounds,
                    self.config.bandwidth_factor,
                );
                let ratio = l - g;
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_x = Some(cand);
                }
            }
            let x = best_x.expect("at least one candidate sampled");
            let _ = dim;
            let e = problem.evaluate(&x);
            targets.push(oriented_target(&e, direction));
            decisions.push(x);
            evals.push(e);
        }

        // Identify the best observation.
        let mut best_idx = 0;
        for i in 1..evals.len() {
            if better(&evals[i], &evals[best_idx], direction) {
                best_idx = i;
            }
        }
        let total_evals = evals.len();
        let final_pop: Vec<Candidate<Vec<f64>>> = decisions
            .into_iter()
            .zip(evals)
            .map(|(d, e)| Candidate::new(d, e))
            .collect();
        let best = final_pop[best_idx].clone();
        let front = vec![best.clone()];
        OptimizationResult::new(
            Population::new(final_pop),
            front,
            Some(best),
            total_evals,
            self.config.iterations + self.config.initial_samples,
        )
    }
}

fn oriented_target(e: &Evaluation, direction: Direction) -> f64 {
    let base = match direction {
        Direction::Minimize => e.objectives[0],
        Direction::Maximize => -e.objectives[0],
    };
    if e.is_feasible() {
        base
    } else {
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
        .map(|&(lo, hi)| {
            if lo == hi {
                lo
            } else {
                lo + (hi - lo) * rng.random::<f64>()
            }
        })
        .collect()
}

/// Split observation indices into a "good" set (top `good_fraction` by
/// minimization target) and a "bad" set. Both sets are guaranteed
/// non-empty when there are at least 2 observations.
fn split_good_bad(targets: &[f64], good_fraction: f64) -> (Vec<usize>, Vec<usize>) {
    let n = targets.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        targets[a]
            .partial_cmp(&targets[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_good = ((n as f64) * good_fraction).round() as usize;
    let n_good = n_good.clamp(1, n.saturating_sub(1));
    let good = order[..n_good].to_vec();
    let bad = order[n_good..].to_vec();
    (good, bad)
}

/// Sample one decision from a per-axis Gaussian mixture KDE on the
/// indices `support`. Each support point contributes a Gaussian per axis
/// with bandwidth chosen by Scott's rule (`σ̂ · n^(-1/5)`) scaled by
/// `bandwidth_factor`. The mixture weight is uniform over the support.
fn sample_from_kde(
    decisions: &[Vec<f64>],
    support: &[usize],
    bounds: &RealBounds,
    bandwidth_factor: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    if support.is_empty() {
        return sample_uniform_in_bounds(bounds, rng);
    }
    let dim = bounds.bounds.len();
    let bandwidths = scott_bandwidths(decisions, support, bandwidth_factor);

    let pick = support[rng.random_range(0..support.len())];
    let center = &decisions[pick];
    let mut x = vec![0.0_f64; dim];
    for j in 0..dim {
        let normal = Normal::new(center[j], bandwidths[j].max(1e-12)).unwrap();
        let v = normal.sample(rng);
        let (lo, hi) = bounds.bounds[j];
        x[j] = v.clamp(lo, hi);
    }
    x
}

/// Per-axis log-density at `x` of the KDE built on `support`.
fn log_kde_density(
    x: &[f64],
    decisions: &[Vec<f64>],
    support: &[usize],
    bounds: &RealBounds,
    bandwidth_factor: f64,
) -> f64 {
    if support.is_empty() {
        return f64::NEG_INFINITY;
    }
    let dim = bounds.bounds.len();
    let bandwidths = scott_bandwidths(decisions, support, bandwidth_factor);

    // Sum of per-axis log-densities, with the kernel a product of 1-D
    // Gaussians. Using log-sum-exp for numerical stability would be more
    // accurate, but the per-axis-product form is what TPE uses in
    // practice and is fine for our optimization goal of *ranking*
    // candidates by ratio.
    let mut total = 0.0;
    for j in 0..dim {
        let h = bandwidths[j].max(1e-12);
        let mut s = 0.0;
        for &i in support {
            let z = (x[j] - decisions[i][j]) / h;
            s += (-0.5 * z * z).exp() / (h * (2.0 * std::f64::consts::PI).sqrt());
        }
        let mean_density = s / support.len() as f64;
        total += mean_density.max(1e-300).ln();
    }
    total
}

/// Per-axis bandwidths via Scott's rule: `σ̂ · n^(-1/5)`, with `σ̂` the
/// per-axis standard deviation of the support.
fn scott_bandwidths(decisions: &[Vec<f64>], support: &[usize], factor: f64) -> Vec<f64> {
    let dim = decisions[0].len();
    let mut means = vec![0.0_f64; dim];
    for &i in support {
        for j in 0..dim {
            means[j] += decisions[i][j];
        }
    }
    let n = support.len() as f64;
    for m in means.iter_mut() {
        *m /= n;
    }
    let mut vars = vec![0.0_f64; dim];
    for &i in support {
        for j in 0..dim {
            let d = decisions[i][j] - means[j];
            vars[j] += d * d;
        }
    }
    let denom = (support.len().saturating_sub(1).max(1)) as f64;
    for v in vars.iter_mut() {
        *v /= denom;
    }
    let scott_n = (support.len() as f64).powf(-0.2);
    vars.into_iter()
        .map(|v| factor * v.sqrt().max(1e-6) * scott_n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{SchafferN1, Sphere1D};

    fn make_optimizer(seed: u64) -> Tpe {
        Tpe::new(
            TpeConfig {
                initial_samples: 10,
                iterations: 50,
                good_fraction: 0.25,
                candidate_samples: 24,
                bandwidth_factor: 1.0,
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
        // TPE without bandwidth tuning, 60 evals on 1-D sphere: clearly
        // beats random search (which averages ≈ f = 8) but not as
        // aggressive as well-tuned BO.
        assert!(
            best.evaluation.objectives[0] < 0.1,
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
    fn split_handles_small_n() {
        let (good, bad) = split_good_bad(&[3.0, 1.0, 2.0, 4.0, 5.0], 0.25);
        // 0.25 × 5 = 1.25 → round to 1, so 1 good + 4 bad.
        assert_eq!(good.len(), 1);
        assert_eq!(bad.len(), 4);
        assert_eq!(good[0], 1); // index of value 1.0 (the minimum)
    }

    #[test]
    #[should_panic(expected = "exactly one objective")]
    fn multi_objective_panics() {
        let mut opt = make_optimizer(0);
        let _ = opt.run(&SchafferN1);
    }
}
