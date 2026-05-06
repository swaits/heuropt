//! Async evaluation example: optimize hyperparameters where each
//! evaluation is an awaitable (simulated HTTP) call.
//!
//! Demonstrates:
//! - Implementing [`AsyncProblem`].
//! - Driving the optimizer through `tokio` with bounded concurrency.
//! - Comparing wall-clock time at concurrency = 1 vs 8.
//!
//! Run with: `cargo run --release --features async --example async_eval`

use std::time::Instant;

use heuropt::core::async_problem::AsyncProblem;
use heuropt::prelude::*;

struct RemoteService;

impl AsyncProblem for RemoteService {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("loss")])
    }

    async fn evaluate_async(&self, x: &Vec<f64>) -> Evaluation {
        // Simulate a 20 ms remote-service round-trip per evaluation.
        // The compute itself is ~free; the latency is the bottleneck.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let loss: f64 = x.iter().map(|v| v * v).sum();
        Evaluation::new(vec![loss])
    }
}

#[tokio::main]
async fn main() {
    let bounds = vec![(-1.0_f64, 1.0_f64); 4];
    let problem = RemoteService;

    println!("RandomSearch with 200 evaluations (20 ms each)");
    println!();

    for &concurrency in &[1_usize, 4, 16] {
        let mut opt = RandomSearch::new(
            RandomSearchConfig {
                iterations: 100,
                batch_size: 2,
                seed: 42,
            },
            RealBounds::new(bounds.clone()),
        );
        let started = Instant::now();
        let result = opt.run_async(&problem, concurrency).await;
        let elapsed = started.elapsed();
        println!(
            "concurrency = {:>2}  elapsed = {:>5} ms  best loss = {:>8.5}  evaluations = {}",
            concurrency,
            elapsed.as_millis(),
            result.best.unwrap().evaluation.objectives[0],
            result.evaluations,
        );
    }

    println!();
    println!("DifferentialEvolution at concurrency=8");
    let started = Instant::now();
    let mut de = DifferentialEvolution::new(
        DifferentialEvolutionConfig {
            population_size: 8,
            generations: 10,
            differential_weight: 0.5,
            crossover_probability: 0.9,
            seed: 42,
        },
        RealBounds::new(bounds.clone()),
    );
    let result = de.run_async(&problem, 8).await;
    let elapsed = started.elapsed();
    println!(
        "elapsed = {:>5} ms  best loss = {:>8.5}  evaluations = {}",
        elapsed.as_millis(),
        result.best.unwrap().evaluation.objectives[0],
        result.evaluations,
    );
}
