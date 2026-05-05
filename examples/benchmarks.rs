//! Two canonical optimization benchmarks: ZDT1 and Rastrigin.
//!
//! - **ZDT1** (Zitzler, Deb, Thiele, 2000) is the standard 2-objective
//!   continuous benchmark. The analytical Pareto front is
//!   `f2 = 1 - sqrt(f1)` for `f1 ∈ [0, 1]`, which we use to score the
//!   solution quality of a short NSGA-II run.
//! - **Rastrigin** is the textbook multimodal single-objective trap:
//!   `f(x) = 10·n + Σ (x_i² − 10·cos(2π·x_i))`, global minimum `f = 0`
//!   at the origin. We solve it with DE.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example benchmarks
//! ```
//!
//! Both problems are defined by simple public-domain math and have no
//! licensing concerns.

use std::f64::consts::PI;

use heuropt::prelude::*;

/// ZDT1: 30-dimensional, two minimization objectives.
struct Zdt1 {
    dim: usize,
}

impl Problem for Zdt1 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        debug_assert_eq!(x.len(), self.dim);
        // GaussianMutation does not enforce bounds (spec §11.2); clamp here so
        // the example stays well-defined on a bounded benchmark. This is the
        // spec-recommended pattern for handling bounds in v1.
        let x0 = x[0].clamp(0.0, 1.0);
        let tail_sum: f64 = x[1..].iter().map(|v| v.clamp(0.0, 1.0)).sum();
        let g = 1.0 + 9.0 * tail_sum / (self.dim as f64 - 1.0);
        let f1 = x0;
        let f2 = g * (1.0 - (f1 / g).sqrt());
        Evaluation::new(vec![f1, f2])
    }
}

/// Rastrigin: n-dimensional, single minimization objective.
struct Rastrigin {
    dim: usize,
}

impl Problem for Rastrigin {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let n = self.dim as f64;
        let value = 10.0 * n
            + x.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>();
        Evaluation::new(vec![value])
    }
}

/// Mean L2 distance from each front point to the nearest point on the
/// analytical ZDT1 Pareto front (`f2 = 1 - sqrt(f1)` for `f1 ∈ [0, 1]`).
///
/// Since `(f1, 1 - sqrt(f1))` is monotone in `f1`, the minimum-distance
/// projection on a fine sample of the curve is good enough for a
/// human-readable quality signal.
fn mean_distance_to_zdt1_front(front: &[Candidate<Vec<f64>>]) -> f64 {
    let samples: Vec<(f64, f64)> = (0..=1000)
        .map(|i| {
            let f1 = i as f64 / 1000.0;
            (f1, 1.0 - f1.sqrt())
        })
        .collect();
    let mut total = 0.0;
    for c in front {
        let f1 = c.evaluation.objectives[0];
        let f2 = c.evaluation.objectives[1];
        let mut best = f64::INFINITY;
        for &(rf1, rf2) in &samples {
            let d = ((rf1 - f1).powi(2) + (rf2 - f2).powi(2)).sqrt();
            if d < best {
                best = d;
            }
        }
        total += best;
    }
    total / front.len().max(1) as f64
}

fn run_zdt1() {
    let dim = 30;
    let problem = Zdt1 { dim };
    let initializer = RealBounds::new(vec![(0.0, 1.0); dim]);
    let variation = GaussianMutation { sigma: 0.05 };
    let config = Nsga2Config { population_size: 100, generations: 400, seed: 42 };
    let mut optimizer = Nsga2::new(config, initializer, variation);

    let result = optimizer.run(&problem);
    let dist = mean_distance_to_zdt1_front(&result.pareto_front);

    println!("== ZDT1 (NSGA-II, dim={dim}) ==");
    println!("  population:    {}", result.population.len());
    println!("  pareto front:  {}", result.pareto_front.len());
    println!("  evaluations:   {}", result.evaluations);
    println!("  mean L2 to true front: {dist:.5}");
    println!(
        "  example front point: f1={:.4}, f2={:.4}",
        result.pareto_front[0].evaluation.objectives[0],
        result.pareto_front[0].evaluation.objectives[1],
    );
}

fn run_rastrigin() {
    let dim = 5;
    let problem = Rastrigin { dim };
    let bounds = RealBounds::new(vec![(-5.12, 5.12); dim]);
    let config = DifferentialEvolutionConfig {
        population_size: 100,
        generations: 1000,
        differential_weight: 0.5,
        crossover_probability: 0.9,
        seed: 1,
    };
    let mut optimizer = DifferentialEvolution::new(config, bounds);

    let result = optimizer.run(&problem);
    let best = result.best.expect("single-objective always has a best");

    println!("== Rastrigin (DE, dim={dim}) ==");
    println!("  evaluations:   {}", result.evaluations);
    println!("  best f:        {:.6e}", best.evaluation.objectives[0]);
    println!("  best x:        {:?}", best.decision);
}

fn main() {
    run_zdt1();
    println!();
    run_rastrigin();
}
