//! Single-machine job-shop scheduling: minimize total weighted
//! completion time given per-job processing times and due-date weights.
//!
//! The decision is a permutation `Vec<usize>` — the order in which
//! jobs are processed. We use `SimulatedAnnealing` paired with
//! `SwapMutation` (the standard generic-permutation pair).
//!
//! Demonstrates:
//! - Permutation decisions (`Vec<usize>`).
//! - Simulated annealing with a custom `Initializer` that produces a
//!   randomly shuffled identity permutation.
//! - `SwapMutation` preserving the permutation invariant for free.
//!
//! Run with: `cargo run --release --example scheduling`

use heuropt::prelude::*;

/// Single-machine weighted-completion-time problem (1 || Σwᵢ Cᵢ).
struct Scheduling {
    /// Processing time for each job.
    process_times: Vec<f64>,
    /// Importance weight for each job. Higher weight = more
    /// punishing if the job finishes late.
    weights: Vec<f64>,
}

impl Problem for Scheduling {
    type Decision = Vec<usize>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("total_wct")])
    }

    fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
        // Compute each job's completion time as the running sum of
        // processing times in the chosen order.
        let mut clock = 0.0_f64;
        let mut total_wct = 0.0_f64;
        for &job in schedule {
            clock += self.process_times[job];
            total_wct += self.weights[job] * clock;
        }
        Evaluation::new(vec![total_wct])
    }
}

/// Initializer that produces a single randomly-shuffled permutation
/// `[0, 1, …, n-1]`. Simulated annealing only needs one initial decision.
struct ShuffledPerm {
    n: usize,
}

impl Initializer<Vec<usize>> for ShuffledPerm {
    fn initialize(&mut self, _size: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
        use rand::seq::SliceRandom;
        let mut perm: Vec<usize> = (0..self.n).collect();
        perm.shuffle(rng);
        vec![perm]
    }
}

fn main() {
    // 12 jobs. The optimal policy is the Smith's-rule order: sort by
    // p_i / w_i ascending (shortest weighted processing time first).
    // We can compute that directly to compare against the search result.
    let jobs = [
        (3.0_f64, 2.0_f64),
        (5.0, 1.0),
        (2.0, 4.0),
        (8.0, 3.0),
        (4.0, 5.0),
        (1.0, 2.0),
        (7.0, 6.0),
        (6.0, 1.0),
        (3.0, 3.0),
        (5.0, 4.0),
        (2.0, 2.0),
        (4.0, 1.0),
    ];
    let process_times: Vec<f64> = jobs.iter().map(|j| j.0).collect();
    let weights: Vec<f64> = jobs.iter().map(|j| j.1).collect();
    let n = jobs.len();

    let problem = Scheduling {
        process_times: process_times.clone(),
        weights: weights.clone(),
    };

    // Smith's rule oracle: sort jobs by p / w ascending.
    let mut smith_order: Vec<usize> = (0..n).collect();
    smith_order.sort_by(|&a, &b| {
        let ra = process_times[a] / weights[a];
        let rb = process_times[b] / weights[b];
        ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let smith_score = problem.evaluate(&smith_order).objectives[0];

    // Search via simulated annealing with swap mutation.
    let mut opt = SimulatedAnnealing::new(
        SimulatedAnnealingConfig {
            iterations: 5_000,
            initial_temperature: 50.0,
            final_temperature: 1e-3,
            seed: 42,
        },
        ShuffledPerm { n },
        SwapMutation,
    );
    let result = opt.run(&problem);

    let best = result.best.unwrap();
    println!("Single-machine weighted completion time, {} jobs", n);
    println!();
    println!(
        "Smith's-rule oracle:           {:>8.2}  order = {:?}",
        smith_score, smith_order
    );
    println!(
        "Simulated annealing best:      {:>8.2}  order = {:?}",
        best.evaluation.objectives[0], best.decision,
    );
    println!(
        "Random initial schedule:       {:>8.2}  order = {:?}",
        problem.evaluate(&(0..n).collect()).objectives[0],
        (0..n).collect::<Vec<usize>>(),
    );
    println!();
    println!(
        "SA reached optimum (Smith): {}",
        (best.evaluation.objectives[0] - smith_score).abs() < 1e-9
    );
}
