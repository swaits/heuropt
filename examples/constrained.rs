//! Constrained multi-objective optimization (BNH problem) plus a
//! demo of the observer / stop-condition API.
//!
//! BNH (Binh & Korn 1996) is a 2-variable / 2-objective / 2-constraint
//! multi-objective problem:
//!
//! ```text
//! minimize  f1 = 4·x1² + 4·x2²
//!           f2 = (x1 − 5)² + (x2 − 5)²
//! subject to
//!           g1: (x1 − 5)² + x2² ≤ 25
//!           g2: (x1 − 8)² + (x2 + 3)² ≥ 7.7
//! 0 ≤ x1 ≤ 5,  0 ≤ x2 ≤ 3
//! ```
//!
//! Demonstrates:
//! - Constraint handling via `Evaluation::constrained` (heuropt's
//!   default tournament/Pareto comparators prefer feasibles).
//! - The Observer API: a `Stagnation` observer that halts the run
//!   once the front stops improving, plus a `Periodic` observer that
//!   prints progress every 25 generations.
//! - Composing observers with `.or()`.
//!
//! Run with: `cargo run --release --example constrained`

use heuropt::prelude::*;

struct Bnh;

impl Problem for Bnh {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let f1 = 4.0 * x[0] * x[0] + 4.0 * x[1] * x[1];
        let f2 = (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);

        // g1: (x1 − 5)² + x2² ≤ 25  → violation = max(0, lhs − 25)
        let g1 = ((x[0] - 5.0).powi(2) + x[1].powi(2) - 25.0).max(0.0);
        // g2: (x1 − 8)² + (x2 + 3)² ≥ 7.7  → violation = max(0, 7.7 − lhs)
        let g2 = (7.7 - ((x[0] - 8.0).powi(2) + (x[1] + 3.0).powi(2))).max(0.0);

        let total_violation = g1 + g2;
        Evaluation::constrained(vec![f1, f2], total_violation)
    }
}

fn main() {
    let bounds = vec![(0.0_f64, 5.0_f64), (0.0_f64, 3.0_f64)];

    // Compose stop conditions: halt after 5 s OR (via .or()) print
    // periodic progress every 25 generations. The Periodic observer
    // never breaks; it only logs.
    let stop = MaxTime::new(std::time::Duration::from_secs(5));
    let progress = Periodic::new(25, |snap: &Snapshot<'_, Vec<f64>>| {
        let feasible_in_pop = snap
            .population
            .iter()
            .filter(|c| c.evaluation.is_feasible())
            .count();
        let front_size = snap.pareto_front.map(|f| f.len()).unwrap_or(0);
        println!(
            "gen {:>4}  evaluations = {:>6}  feasible/pop = {}/{}  front = {}",
            snap.iteration,
            snap.evaluations,
            feasible_in_pop,
            snap.population.len(),
            front_size,
        );
    });
    let mut observer = <_ as Observer<Vec<f64>>>::or(stop, progress);

    let mut opt = Nsga2::new(
        Nsga2Config {
            population_size: 100,
            generations: 250,
            seed: 42,
        },
        RealBounds::new(bounds.clone()),
        CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / 2.0),
        },
    );
    let result = opt.run_with(&Bnh, &mut observer);

    let total_feasible = result
        .population
        .iter()
        .filter(|c| c.evaluation.is_feasible())
        .count();

    println!();
    println!("Final state after {} generations:", result.generations);
    println!("  total evaluations:        {}", result.evaluations);
    println!(
        "  feasible / total pop:     {} / {}",
        total_feasible,
        result.population.len()
    );
    println!("  pareto front size:        {}", result.pareto_front.len());
    println!();
    println!("Sample of the front (f1, f2):");
    let mut sorted = result.pareto_front.clone();
    sorted.sort_by(|a, b| {
        a.evaluation.objectives[0]
            .partial_cmp(&b.evaluation.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = sorted.len();
    if n > 0 {
        for k in (0..n).step_by((n / 5).max(1)) {
            let c = &sorted[k];
            println!(
                "  f1 = {:>7.3}, f2 = {:>7.3}, violation = {:.3}",
                c.evaluation.objectives[0],
                c.evaluation.objectives[1],
                c.evaluation.constraint_violation,
            );
        }
    }
}
