//! Bi-objective 0/1 knapsack — Zitzler & Thiele's textbook multi-objective
//! combinatorial benchmark, solved with NSGA-II.
//!
//! - **Benchmark family**: Zitzler & Thiele (1999) bi-objective knapsack.
//!   Each item has two profit values and a single weight; a single capacity
//!   constraint. We use a 30-item instance with values drawn from the same
//!   U(10, 100) distribution scheme as the published instances, embedded as
//!   `const` tables so the example stays self-contained.
//! - **Algorithm**: [`Nsga2`].
//! - **Decision**: `Vec<bool>` of length 30 (take / leave each item).
//! - **Variation**: a local one-point crossover (binary GAs' workhorse) piped
//!   into [`BitFlipMutation`] via [`CompositeVariation`]. **A future PR could
//!   lift `OnePointCrossover` / `UniformCrossover` into the library proper**
//!   so users don't need to roll their own.
//! - **Initializer**: a tiny local `RandomBinary` (one-liner; would be a
//!   reasonable library addition too).
//! - **Constraint handling**: weight overruns are penalized in both
//!   objectives by `-large * overrun`. With the penalty dominating profit
//!   range, the Pareto front is composed entirely of feasible solutions
//!   (standard heuristic-MO practice).
//!
//! Sources:
//! - Zitzler & Thiele (1999), "Multiobjective evolutionary algorithms: A
//!   comparative case study and the Strength Pareto approach."
//! - Deb (2001), "Multi-Objective Optimization Using Evolutionary Algorithms"
//!   for the standard penalty-based MO constraint handling.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example mo_knapsack
//! ```

use heuropt::metrics::hypervolume_2d;
use heuropt::prelude::*;
use rand::Rng as _;

const N_ITEMS: usize = 30;

/// Profit vector A (one of two objectives), U(10, 100) style.
const PROFITS_A: [f64; N_ITEMS] = [
    61.0, 17.0, 92.0, 49.0, 73.0, 28.0, 84.0, 36.0, 55.0, 78.0, 23.0, 91.0, 12.0, 67.0, 45.0, 58.0,
    33.0, 71.0, 14.0, 26.0, 87.0, 42.0, 19.0, 65.0, 30.0, 51.0, 79.0, 22.0, 47.0, 88.0,
];

/// Profit vector B (the other objective). Intentionally anti-correlated with
/// A on many items so the Pareto front spans a wide trade-off.
const PROFITS_B: [f64; N_ITEMS] = [
    24.0, 81.0, 16.0, 67.0, 29.0, 73.0, 41.0, 60.0, 52.0, 19.0, 77.0, 34.0, 95.0, 22.0, 71.0, 88.0,
    56.0, 27.0, 64.0, 90.0, 18.0, 43.0, 79.0, 31.0, 85.0, 25.0, 38.0, 92.0, 70.0, 13.0,
];

/// Item weights.
const WEIGHTS: [f64; N_ITEMS] = [
    35.0, 58.0, 22.0, 71.0, 14.0, 86.0, 31.0, 53.0, 78.0, 19.0, 44.0, 16.0, 67.0, 88.0, 25.0, 51.0,
    33.0, 74.0, 12.0, 47.0, 63.0, 28.0, 91.0, 36.0, 55.0, 17.0, 82.0, 41.0, 24.0, 68.0,
];

/// Capacity = roughly half the total weight (standard Zitzler-Thiele convention).
fn capacity() -> f64 {
    0.5 * WEIGHTS.iter().sum::<f64>()
}

struct BiKnapsack {
    cap: f64,
}

impl Problem for BiKnapsack {
    type Decision = Vec<bool>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::maximize("profit_A"),
            Objective::maximize("profit_B"),
        ])
    }

    fn evaluate(&self, take: &Vec<bool>) -> Evaluation {
        let (pa, pb, w) =
            take.iter()
                .enumerate()
                .fold((0.0_f64, 0.0_f64, 0.0_f64), |(pa, pb, w), (i, &t)| {
                    if t {
                        (pa + PROFITS_A[i], pb + PROFITS_B[i], w + WEIGHTS[i])
                    } else {
                        (pa, pb, w)
                    }
                });
        // Penalty: large coefficient on weight overrun, applied to both objectives.
        let overrun = (w - self.cap).max(0.0);
        let penalty = 1000.0 * overrun;
        Evaluation::new(vec![pa - penalty, pb - penalty])
    }

    fn decision_schema(&self) -> Vec<DecisionVariable> {
        (0..N_ITEMS)
            .map(|i| DecisionVariable::new(format!("item_take_{i}")))
            .collect()
    }
}

/// Random binary initializer — each bit is 50/50 independently.
#[derive(Debug, Clone, Copy)]
struct RandomBinary {
    n: usize,
}

impl Initializer<Vec<bool>> for RandomBinary {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<Vec<bool>> {
        (0..size)
            .map(|_| (0..self.n).map(|_| rng.random_bool(0.5)).collect())
            .collect()
    }
}

/// One-point crossover for binary chromosomes.
#[derive(Debug, Clone, Copy, Default)]
struct OnePointCrossoverBool;

impl Variation<Vec<bool>> for OnePointCrossoverBool {
    fn vary(&mut self, parents: &[Vec<bool>], rng: &mut Rng) -> Vec<Vec<bool>> {
        assert!(
            parents.len() >= 2,
            "OnePointCrossoverBool requires 2 parents"
        );
        let p1 = &parents[0];
        let p2 = &parents[1];
        assert_eq!(p1.len(), p2.len(), "parent lengths differ");
        let n = p1.len();
        if n < 2 {
            return vec![p1.clone(), p2.clone()];
        }
        let cut = rng.random_range(1..n);
        let mut c1 = Vec::with_capacity(n);
        let mut c2 = Vec::with_capacity(n);
        c1.extend_from_slice(&p1[..cut]);
        c1.extend_from_slice(&p2[cut..]);
        c2.extend_from_slice(&p2[..cut]);
        c2.extend_from_slice(&p1[cut..]);
        vec![c1, c2]
    }
}

fn main() {
    let cap = capacity();
    let problem = BiKnapsack { cap };

    let mut optimizer = Nsga2::new(
        Nsga2Config {
            population_size: 120,
            generations: 400,
            seed: 19,
        },
        RandomBinary { n: N_ITEMS },
        CompositeVariation {
            crossover: OnePointCrossoverBool,
            mutation: BitFlipMutation {
                probability: 1.0 / N_ITEMS as f64,
            },
        },
    );
    let result = optimizer.run(&problem);

    println!("Bi-objective 0/1 knapsack — Zitzler–Thiele style, 30 items");
    println!(
        "Capacity = {:.0} (≈ half of total weight {:.0})",
        cap,
        WEIGHTS.iter().sum::<f64>()
    );
    println!();
    println!("Total evaluations: {}", result.evaluations);
    println!("Pareto-front size: {}", result.pareto_front.len());
    println!();

    // Sort by profit_A descending for display, dedupe by integer-rounded objective values.
    let mut front: Vec<&Candidate<Vec<bool>>> = result.pareto_front.iter().collect();
    front.sort_by(|a, b| {
        b.evaluation.objectives[0]
            .partial_cmp(&a.evaluation.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen: Vec<(i64, i64)> = Vec::new();
    println!("    profit_A     profit_B   weight");
    for c in &front {
        let o = &c.evaluation.objectives;
        let key = (o[0] as i64, o[1] as i64);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let w: f64 = c
            .decision
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t)
            .map(|(i, _)| WEIGHTS[i])
            .sum();
        println!("    {:>8.0}     {:>8.0}   {:>6.0}", o[0], o[1], w);
    }
    println!("    ({} unique objective-space points)", seen.len());

    // Hypervolume against a reference point of (0, 0): since these are
    // maximization objectives, we transform to minimization by negation in
    // the metric — hypervolume_2d uses ObjectiveSpace::as_minimization() so
    // it Just Works.
    let ref_point = [0.0, 0.0];
    let owned: Vec<Candidate<Vec<bool>>> = result.pareto_front.to_vec();
    let hv = hypervolume_2d(&owned, &problem.objectives(), ref_point);
    println!();
    println!(
        "Hypervolume vs. reference (profit_A=0, profit_B=0): {:.0}",
        hv
    );
}
