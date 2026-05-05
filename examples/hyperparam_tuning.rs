//! Tune a synthetic ML model's hyperparameters with Bayesian Optimization
//! and (separately) Tree-structured Parzen Estimator.
//!
//! The "model" here is a deterministic function over `(learning_rate,
//! weight_decay, depth)` that mimics the shape of a real validation-loss
//! surface — a noisy minimum near sensible hyperparameters with sharp
//! penalties as you stray. It's compute-cheap so the example runs in
//! seconds, but the *workflow* is exactly what you'd use on a real
//! 30-second-per-eval model.
//!
//! Demonstrates:
//! - Sample-efficient optimization: 60 evaluations total, not 60,000.
//! - Comparing BO vs TPE on the same problem with the same budget.
//! - Decoding decision vectors with mixed scales (log-uniform learning
//!   rate, integer-valued depth) using transforms inside `evaluate`.
//!
//! Run with: `cargo run --release --example hyperparam_tuning`

use heuropt::prelude::*;

/// A pretend deep-learning model whose validation loss is a
/// reproducible analytic function of three hyperparameters.
struct ModelTuning;

impl Problem for ModelTuning {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("val_loss")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        // The decision vector is in [0, 1] per dim; we decode each axis
        // into the "real" hyperparameter space.
        let lr = log_uniform(x[0], 1e-5, 1e-1); // learning rate
        let wd = log_uniform(x[1], 1e-6, 1e-2); // weight decay
        let depth = scale_to_int(x[2], 2, 12); // num layers

        // Synthetic validation loss surface:
        //   * minimum at lr ≈ 1e-3, wd ≈ 1e-4, depth = 6
        //   * log-quadratic in lr / wd (typical hyperparameter shape)
        //   * mild penalty for depth far from 6
        //   * tiny deterministic "noise" so flat regions don't all tie
        let lr_term = (lr.log10() - (-3.0)).powi(2);
        let wd_term = (wd.log10() - (-4.0)).powi(2);
        let depth_term = 0.05 * ((depth as f64 - 6.0).abs());
        let noise = 0.02 * ((10.0 * x[0] + 17.0 * x[1] + 23.0 * x[2]).sin());

        let val_loss = 0.05 + 0.3 * lr_term + 0.2 * wd_term + depth_term + noise;
        Evaluation::new(vec![val_loss])
    }
}

fn log_uniform(unit: f64, lo: f64, hi: f64) -> f64 {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (log_lo + unit * (log_hi - log_lo)).exp()
}

fn scale_to_int(unit: f64, lo: i32, hi: i32) -> i32 {
    let span = (hi - lo + 1) as f64;
    let i = (unit * span).floor() as i32;
    (lo + i).min(hi)
}

fn run_bo(seed: u64) -> OptimizationResult<Vec<f64>> {
    let mut opt = BayesianOpt::new(
        BayesianOptConfig {
            initial_samples: 10,
            iterations: 50, // 60 total evals
            length_scales: None,
            signal_variance: 1.0,
            noise_variance: 1e-6,
            acquisition_samples: 200,
            seed,
        },
        RealBounds::new(vec![(0.0, 1.0); 3]),
    );
    opt.run(&ModelTuning)
}

fn run_tpe(seed: u64) -> OptimizationResult<Vec<f64>> {
    let mut opt = Tpe::new(
        TpeConfig {
            initial_samples: 10,
            iterations: 50, // 60 total evals
            good_fraction: 0.25,
            candidate_samples: 64,
            bandwidth_factor: 1.0,
            seed,
        },
        RealBounds::new(vec![(0.0, 1.0); 3]),
    );
    opt.run(&ModelTuning)
}

fn report(name: &str, r: &OptimizationResult<Vec<f64>>) {
    let best = r.best.as_ref().expect("at least one feasible candidate");
    let lr = log_uniform(best.decision[0], 1e-5, 1e-1);
    let wd = log_uniform(best.decision[1], 1e-6, 1e-2);
    let depth = scale_to_int(best.decision[2], 2, 12);
    println!(
        "{:<8} val_loss = {:>7.4}  | lr = {:>10.2e}  wd = {:>10.2e}  depth = {}  | evals = {}",
        name, best.evaluation.objectives[0], lr, wd, depth, r.evaluations,
    );
}

fn main() {
    println!("Tuning ModelTuning (synthetic 3-D loss surface)");
    println!("Optimum: lr ≈ 1e-3, wd ≈ 1e-4, depth = 6, val_loss ≈ 0.03");
    println!();
    println!(
        "{:<8} {:<26}  {:<24}  {:<24}",
        "alg", "best", "(decoded hyperparams)", "(eval budget)"
    );
    for seed in 0..5 {
        println!();
        println!("seed {}:", seed);
        let bo = run_bo(seed);
        let tpe = run_tpe(seed);
        report("BO", &bo);
        report("TPE", &tpe);
    }
}
