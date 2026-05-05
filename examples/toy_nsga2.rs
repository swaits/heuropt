//! Solve the Schaffer N.1 two-objective problem with NSGA-II.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example toy_nsga2
//! ```

use heuropt::prelude::*;

struct SchafferN1;

impl Problem for SchafferN1 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let v = x[0];
        Evaluation::new(vec![v * v, (v - 2.0).powi(2)])
    }
}

fn main() {
    let initializer = RealBounds::new(vec![(-5.0, 5.0)]);
    let variation = GaussianMutation { sigma: 0.2 };
    let config = Nsga2Config { population_size: 60, generations: 80, seed: 42 };
    let mut optimizer = Nsga2::new(config, initializer, variation);

    let result = optimizer.run(&SchafferN1);

    println!("Final population: {}", result.population.len());
    println!("Pareto front size: {}", result.pareto_front.len());
    println!("Total evaluations: {}", result.evaluations);
    println!("First few front points (f1, f2):");
    for c in result.pareto_front.iter().take(8) {
        let o = &c.evaluation.objectives;
        println!("  ({:.4}, {:.4})", o[0], o[1]);
    }
}
