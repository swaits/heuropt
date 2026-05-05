//! Implement a custom optimizer by implementing `Optimizer<P>` directly.
//!
//! Demonstrates spec §2.3: a junior engineer can add a new algorithm by
//! implementing a single trait, with no framework gymnastics required.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example custom_optimizer
//! ```

use heuropt::prelude::*;

/// A trivial single-objective hill-climber: sample one point, then repeatedly
/// jitter it with `GaussianMutation` and keep the better feasible result.
struct HillClimber {
    iterations: usize,
    sigma: f64,
    initializer: RealBounds,
    seed: u64,
}

impl<P> Optimizer<P> for HillClimber
where
    P: Problem<Decision = Vec<f64>>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        let objectives = problem.objectives();
        assert!(objectives.is_single_objective(), "HillClimber needs one objective");
        let mut rng = rng_from_seed(self.seed);
        let mut variation = GaussianMutation { sigma: self.sigma };

        let mut current = self.initializer.initialize(1, &mut rng).remove(0);
        let mut current_eval = problem.evaluate(&current);
        let mut evaluations = 1;

        for _ in 0..self.iterations {
            let children = variation.vary(&[current.clone()], &mut rng);
            let candidate = children.into_iter().next().unwrap();
            let candidate_eval = problem.evaluate(&candidate);
            evaluations += 1;
            // Accept on direction-correct improvement (Sphere is minimize).
            let accept = candidate_eval.objectives[0] < current_eval.objectives[0];
            if accept {
                current = candidate;
                current_eval = candidate_eval;
            }
        }

        let best = Candidate::new(current.clone(), current_eval.clone());
        let population = Population::new(vec![best.clone()]);
        let pareto_front = vec![best.clone()];
        OptimizationResult::new(
            population,
            pareto_front,
            Some(best),
            evaluations,
            self.iterations,
        )
    }
}

struct Sphere1D;

impl Problem for Sphere1D {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        Evaluation::new(vec![x[0] * x[0]])
    }
}

fn main() {
    let mut climber = HillClimber {
        iterations: 500,
        sigma: 0.5,
        initializer: RealBounds::new(vec![(-5.0, 5.0)]),
        seed: 11,
    };

    let result = climber.run(&Sphere1D);
    let best = result.best.expect("single-objective always has a best");
    println!("Hill-climber best f = {:.6}", best.evaluation.objectives[0]);
    println!("Total evaluations: {}", result.evaluations);
}
