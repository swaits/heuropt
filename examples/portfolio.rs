//! Multi-objective portfolio optimization with a budget constraint.
//!
//! Real-world flavor: pick a portfolio over five synthetic assets that
//! trades off **return** (maximize) against **risk** (minimize). Weights
//! must be non-negative and sum to 1.0 (the standard probability-simplex
//! budget constraint).
//!
//! Demonstrates:
//! - Multi-objective formulation with a maximize axis (return) and a
//!   minimize axis (variance-based risk).
//! - The `ProjectToSimplex` repair operator wired into a `Repair`-aware
//!   variation pipeline so every offspring respects the budget.
//! - NSGA-II producing a Pareto front of trade-offs.
//! - Picking one answer off the front via a-posteriori weighting (see
//!   `docs/book/src/cookbook/pick-one.md`).
//!
//! Run with: `cargo run --release --example portfolio`

use heuropt::prelude::*;

/// Five-asset toy market. Means and a covariance matrix you'd estimate
/// from real returns; here they're synthetic but realistic-shape.
struct Portfolio {
    /// Expected per-period returns (one per asset).
    expected_returns: [f64; 5],
    /// Symmetric 5×5 covariance matrix.
    covariance: [[f64; 5]; 5],
}

impl Problem for Portfolio {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::maximize("return"),
            Objective::minimize("risk"),
        ])
    }

    fn evaluate(&self, weights: &Vec<f64>) -> Evaluation {
        // Expected return: w · μ
        let r: f64 = weights
            .iter()
            .zip(self.expected_returns.iter())
            .map(|(w, m)| w * m)
            .sum();

        // Risk (portfolio variance): w · Σ · w
        let mut risk = 0.0;
        for i in 0..5 {
            for j in 0..5 {
                risk += weights[i] * self.covariance[i][j] * weights[j];
            }
        }

        Evaluation::new(vec![r, risk])
    }
}

/// Variation pipeline that respects the simplex constraint: SBX +
/// PolyMut produce real-valued children, then `ProjectToSimplex` projects
/// them back onto `{ w : w ≥ 0, Σw = 1 }`.
struct SimplexVariation {
    crossover: SimulatedBinaryCrossover,
    mutation: PolynomialMutation,
    repair: ProjectToSimplex,
}

impl Variation<Vec<f64>> for SimplexVariation {
    fn vary(&mut self, parents: &[Vec<f64>], rng: &mut Rng) -> Vec<Vec<f64>> {
        let crossed = self.crossover.vary(parents, rng);
        let mut out = Vec::with_capacity(crossed.len());
        for child in crossed {
            let mut mutated = self
                .mutation
                .vary(std::slice::from_ref(&child), rng)
                .pop()
                .expect("PolynomialMutation returned no child");
            self.repair.repair(&mut mutated);
            out.push(mutated);
        }
        out
    }
}

/// `Initializer` that uniformly samples points on the simplex via the
/// standard "log-and-normalize" trick. Every initial member is feasible
/// by construction.
struct SimplexInit {
    dim: usize,
}

impl Initializer<Vec<f64>> for SimplexInit {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        use rand::Rng as _;
        let mut out = Vec::with_capacity(size);
        for _ in 0..size {
            // Sample exponentials, normalize → uniform on simplex.
            let mut e: Vec<f64> = (0..self.dim)
                .map(|_| -(1.0_f64 - rng.random::<f64>()).ln())
                .collect();
            let s: f64 = e.iter().sum();
            for v in e.iter_mut() {
                *v /= s;
            }
            out.push(e);
        }
        out
    }
}

fn main() {
    let problem = Portfolio {
        // Synthetic but plausible: 8% / 12% / 5% / 15% / 3% expected
        // returns. The two "stocks" (B, D) have higher expected return
        // and higher variance than the bonds / cash equivalents.
        expected_returns: [0.08, 0.12, 0.05, 0.15, 0.03],
        covariance: [
            [0.04, 0.02, 0.01, 0.03, 0.005],
            [0.02, 0.10, 0.01, 0.05, 0.005],
            [0.01, 0.01, 0.02, 0.01, 0.005],
            [0.03, 0.05, 0.01, 0.16, 0.005],
            [0.005, 0.005, 0.005, 0.005, 0.001],
        ],
    };

    let bounds = vec![(0.0_f64, 1.0_f64); 5];

    let variation = SimplexVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 1.0),
        mutation: PolynomialMutation::new(bounds.clone(), 20.0, 1.0 / 5.0),
        repair: ProjectToSimplex::new(1.0),
    };

    let mut opt = Nsga2::new(
        Nsga2Config {
            population_size: 100,
            generations: 200,
            seed: 42,
        },
        SimplexInit { dim: 5 },
        variation,
    );

    let result = opt.run(&problem);

    println!("Pareto front size: {}", result.pareto_front.len());
    println!("Total evaluations: {}", result.evaluations);

    // Pick one: a-posteriori weighted decision favoring return slightly.
    // Lower score = preferred. We compare in oriented space (maximize
    // axis already flipped to negative by `as_minimization`).
    let space = problem.objectives();
    let weights = [1.0, 1.5]; // weight risk a bit more than -return
    let chosen = result
        .pareto_front
        .iter()
        .min_by(|a, b| {
            let ax: f64 = space
                .as_minimization(&a.evaluation.objectives)
                .iter()
                .zip(&weights)
                .map(|(v, w)| v * w)
                .sum();
            let bx: f64 = space
                .as_minimization(&b.evaluation.objectives)
                .iter()
                .zip(&weights)
                .map(|(v, w)| v * w)
                .sum();
            ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("non-empty front");

    println!();
    println!(
        "Picked portfolio: weights = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        chosen.decision[0],
        chosen.decision[1],
        chosen.decision[2],
        chosen.decision[3],
        chosen.decision[4],
    );
    println!(
        "  expected return: {:>6.4}",
        chosen.evaluation.objectives[0]
    );
    println!(
        "  risk (variance): {:>6.4}",
        chosen.evaluation.objectives[1]
    );

    // Print 5 representative points across the front.
    println!();
    println!("Sample of the front (return, risk):");
    let mut sorted = result.pareto_front.clone();
    sorted.sort_by(|a, b| {
        a.evaluation.objectives[0]
            .partial_cmp(&b.evaluation.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = sorted.len();
    for k in (0..n).step_by((n / 5).max(1)) {
        let c = &sorted[k];
        println!(
            "  return = {:.4}, risk = {:.4}",
            c.evaluation.objectives[0], c.evaluation.objectives[1],
        );
    }
}
