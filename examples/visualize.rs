//! Visualize an NSGA-II run on Schaffer N.1 — produces two SVGs:
//! `pareto_front.svg` (scatter plot of the final front) and
//! `convergence.svg` (best-so-far hypervolume per generation).
//!
//! Uses the `heuropt-plot` companion crate plus the v0.6 observer
//! API (`Periodic`) to record per-generation hypervolume into a Vec
//! during the run.
//!
//! Run with: `cargo run --release --example visualize`

use std::cell::RefCell;
use std::ops::ControlFlow;

use heuropt::metrics::hypervolume_2d;
use heuropt::prelude::*;
use heuropt_plot::{convergence_svg, pareto_front_svg};

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
    let ref_point = [10.0, 10.0];

    // Per-generation hypervolume trace, recorded by the observer.
    let history: RefCell<Vec<f64>> = RefCell::new(Vec::new());

    let mut recorder = |snap: &Snapshot<'_, Vec<f64>>| -> ControlFlow<()> {
        let hv = match snap.pareto_front {
            Some(front) => hypervolume_2d(front, snap.objectives, ref_point),
            None => 0.0,
        };
        history.borrow_mut().push(hv);
        ControlFlow::Continue(())
    };

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

    let result = opt.run_with(&problem, &mut recorder);

    let front_svg = pareto_front_svg(
        &result.pareto_front,
        &space,
        700,
        450,
        "NSGA-II on Schaffer N.1 — final Pareto front",
    );
    std::fs::write("pareto_front.svg", front_svg).expect("write pareto_front.svg");

    let trace = history.borrow();
    let conv_svg = convergence_svg(
        &trace,
        700,
        450,
        "NSGA-II on Schaffer N.1 — hypervolume per generation",
        "hypervolume",
        false, // higher is better
    );
    std::fs::write("convergence.svg", conv_svg).expect("write convergence.svg");

    println!("Final front size:        {}", result.pareto_front.len());
    println!(
        "Final hypervolume:       {:.4}",
        trace.last().copied().unwrap_or(0.0)
    );
    println!("Wrote pareto_front.svg and convergence.svg");
}
