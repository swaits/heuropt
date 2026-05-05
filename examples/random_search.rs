//! Run a 2D sphere problem under `RandomSearch`.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example random_search
//! ```

use heuropt::prelude::*;

struct Sphere2D;

impl Problem for Sphere2D {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        Evaluation::new(vec![x.iter().map(|v| v * v).sum()])
    }
}

fn main() {
    let initializer = RealBounds::new(vec![(-5.0, 5.0), (-5.0, 5.0)]);
    let config = RandomSearchConfig { iterations: 500, batch_size: 1, seed: 7 };
    let mut optimizer = RandomSearch::new(config, initializer);

    let result = optimizer.run(&Sphere2D);

    let best = result.best.expect("single-objective always has a best");
    println!("Total evaluations: {}", result.evaluations);
    println!("Best decision: {:?}", best.decision);
    println!("Best f: {:.6}", best.evaluation.objectives[0]);
}
