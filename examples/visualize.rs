//! Visualize an NSGA-II run on Schaffer N.1 with the `heuropt-plot`
//! companion crate — produces a `pareto_front.svg` of the final
//! Pareto front.
//!
//! Run with: `cargo run --release --example visualize`

use heuropt::prelude::*;
use heuropt_plot::pareto_front_svg;

struct Schaffer;

impl Problem for Schaffer {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }
    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        Evaluation::new(vec![x[0] * x[0], (x[0] - 2.0).powi(2)])
    }
}

fn main() {
    let problem = Schaffer;
    let bounds = vec![(-5.0_f64, 5.0_f64)];
    let space = problem.objectives();

    let mut opt = Nsga2::new(
        Nsga2Config {
            population_size: 50,
            generations: 100,
            seed: 42,
        },
        RealBounds::new(bounds.clone()),
        CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        },
    );

    let result = opt.run(&problem);

    let svg = pareto_front_svg(
        &result.pareto_front,
        &space,
        700,
        450,
        "NSGA-II on Schaffer N.1 — final Pareto front",
    );
    std::fs::write("pareto_front.svg", svg).expect("write pareto_front.svg");

    println!("Final front size: {}", result.pareto_front.len());
    println!("Wrote pareto_front.svg");
}
