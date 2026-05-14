//! 3-objective Job-Shop Scheduling on Lawrence's LA01 instance, solved with
//! NSGA-III (the many-objective successor to NSGA-II).
//!
//! - **Benchmark**: Lawrence LA01 (1984), 10 jobs × 5 machines, 50 operations
//!   total. Each operation has a fixed machine and processing time;
//!   operations within a job run in order. Data taken from the OR-Library /
//!   JSPLIB la01 instance file.
//! - **Three objectives** (this example):
//!   - f₁ = makespan
//!   - f₂ = total flow time Σⱼ Cⱼ
//!   - f₃ = total tardiness Σⱼ max(0, Cⱼ − dⱼ), with synthetic due dates
//!     dⱼ = 1.3 × (sum of processing times of job j)
//! - **Algorithm**: [`Nsga3`] — designed for ≥ 3 objectives (NSGA-II's
//!   crowding distance degrades in higher dim).
//! - **Encoding**: operation-based string of length 50.
//! - **Variation**: a local POX (multiset-preserving) crossover piped through
//!   a small randomly-chosen mutation that alternates between
//!   [`InsertionMutation`] and [`ScrambleMutation`]. Strict-permutation
//!   crossovers cannot be used on multiset encodings.
//! - **Initializer**: [`ShuffledMultisetPermutation`].
//!
//! Sources:
//! - Lawrence (1984), thesis benchmark instances.
//! - OR-Library / JSPLIB LA01 instance file.
//! - Deb & Jain (2014), "An evolutionary many-objective optimization
//!   algorithm using reference-point based non-dominated sorting approach,
//!   Part I" — NSGA-III.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example mo_jss_la01
//! ```

use heuropt::prelude::*;
use rand::Rng as _;

const N_JOBS: usize = 10;
const N_MACHINES: usize = 5;

/// LA01 routing — machine id for the k-th operation of job j.
const LA01_MACHINE: [[usize; N_MACHINES]; N_JOBS] = [
    [1, 0, 4, 3, 2],
    [0, 3, 4, 2, 1],
    [3, 4, 1, 2, 0],
    [1, 0, 4, 2, 3],
    [0, 3, 2, 1, 4],
    [1, 2, 4, 0, 3],
    [3, 4, 1, 2, 0],
    [2, 0, 1, 3, 4],
    [3, 1, 4, 0, 2],
    [4, 3, 1, 2, 0],
];

/// LA01 processing times — duration of the k-th operation of job j.
const LA01_TIME: [[f64; N_MACHINES]; N_JOBS] = [
    [21.0, 53.0, 95.0, 55.0, 34.0],
    [21.0, 52.0, 16.0, 26.0, 71.0],
    [39.0, 98.0, 42.0, 31.0, 12.0],
    [77.0, 55.0, 79.0, 66.0, 77.0],
    [83.0, 34.0, 64.0, 19.0, 37.0],
    [54.0, 43.0, 79.0, 92.0, 62.0],
    [69.0, 77.0, 87.0, 87.0, 93.0],
    [38.0, 60.0, 41.0, 24.0, 66.0],
    [17.0, 49.0, 25.0, 44.0, 98.0],
    [77.0, 79.0, 43.0, 75.0, 96.0],
];

/// Synthetic due dates: 1.3 × total processing time of each job.
fn due_dates() -> [f64; N_JOBS] {
    let mut d = [0.0_f64; N_JOBS];
    for (j, row) in LA01_TIME.iter().enumerate() {
        d[j] = 1.3 * row.iter().sum::<f64>();
    }
    d
}

struct La01ThreeObjective {
    due: [f64; N_JOBS],
}

impl Problem for La01ThreeObjective {
    type Decision = Vec<usize>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("makespan"),
            Objective::minimize("total_flow_time"),
            Objective::minimize("total_tardiness"),
        ])
    }

    fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
        let mut job_next = [0_usize; N_JOBS];
        let mut job_clock = [0.0_f64; N_JOBS];
        let mut machine_clock = [0.0_f64; N_MACHINES];
        for &job in schedule {
            let k = job_next[job];
            let m = LA01_MACHINE[job][k];
            let t = LA01_TIME[job][k];
            let start = job_clock[job].max(machine_clock[m]);
            let end = start + t;
            job_clock[job] = end;
            machine_clock[m] = end;
            job_next[job] = k + 1;
        }
        let makespan = machine_clock.iter().cloned().fold(0.0_f64, f64::max);
        let flow_time: f64 = job_clock.iter().sum();
        let tardiness: f64 = job_clock
            .iter()
            .zip(self.due.iter())
            .map(|(&c, &d)| (c - d).max(0.0))
            .sum();
        Evaluation::new(vec![makespan, flow_time, tardiness])
    }

    fn decision_schema(&self) -> Vec<DecisionVariable> {
        (0..N_JOBS * N_MACHINES)
            .map(|k| DecisionVariable::new(format!("op_slot_{k}")))
            .collect()
    }
}

/// POX — multiset-preserving crossover for operation-string encodings.
/// (Identical in spirit to the one in `jss_ft06_bi.rs`; copied locally so
/// each example stays self-contained.)
#[derive(Debug, Clone, Copy, Default)]
struct PrecedenceOrderCrossover;

impl Variation<Vec<usize>> for PrecedenceOrderCrossover {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(parents.len() >= 2, "POX requires 2 parents");
        let p1 = &parents[0];
        let p2 = &parents[1];
        let mut in_j1 = [false; N_JOBS];
        loop {
            for slot in &mut in_j1 {
                *slot = rng.random_bool(0.5);
            }
            let n_in_j1 = in_j1.iter().filter(|&&b| b).count();
            if n_in_j1 > 0 && n_in_j1 < N_JOBS {
                break;
            }
        }
        vec![pox_child(p1, p2, &in_j1), pox_child(p2, p1, &in_j1)]
    }
}

fn pox_child(donor: &[usize], filler: &[usize], in_donor_set: &[bool]) -> Vec<usize> {
    let n = donor.len();
    let mut child = vec![usize::MAX; n];
    for k in 0..n {
        if in_donor_set[donor[k]] {
            child[k] = donor[k];
        }
    }
    let mut fill_idx = 0;
    for &v in filler {
        if !in_donor_set[v] {
            while fill_idx < n && child[fill_idx] != usize::MAX {
                fill_idx += 1;
            }
            child[fill_idx] = v;
            fill_idx += 1;
        }
    }
    child
}

/// Per-call random choice between Insertion and Scramble. Both preserve the
/// multiset; flipping a coin gives the schedule access to two complementary
/// neighborhood moves.
#[derive(Debug, Clone, Copy, Default)]
struct InsertionOrScramble;

impl Variation<Vec<usize>> for InsertionOrScramble {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        if rng.random_bool(0.5) {
            InsertionMutation.vary(parents, rng)
        } else {
            ScrambleMutation.vary(parents, rng)
        }
    }
}

fn main() {
    let problem = La01ThreeObjective { due: due_dates() };

    let mut optimizer = Nsga3::new(
        Nsga3Config {
            population_size: 120,
            generations: 600,
            reference_divisions: 12,
            seed: 9,
        },
        ShuffledMultisetPermutation::new(vec![N_MACHINES; N_JOBS]),
        CompositeVariation {
            crossover: PrecedenceOrderCrossover,
            mutation: InsertionOrScramble,
        },
    );
    let result = optimizer.run(&problem);

    println!("LA01 — 3-objective JSS via NSGA-III");
    println!("Source: Lawrence (1984), OR-Library la01 instance");
    println!();
    println!("Objectives: f1 = makespan,  f2 = total flow time,  f3 = total tardiness");
    println!("Due dates:  dⱼ = 1.3 × Σ(processing times of job j)");
    println!();
    println!("Total evaluations: {}", result.evaluations);
    println!("Pareto-front size: {}", result.pareto_front.len());
    println!();

    // Sort by makespan and print up to 12 well-spaced rows.
    let mut front: Vec<&Candidate<Vec<usize>>> = result.pareto_front.iter().collect();
    front.sort_by(|a, b| {
        a.evaluation.objectives[0]
            .partial_cmp(&b.evaluation.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let stride = (front.len() / 12).max(1);
    println!("    f1 makespan   f2 flow time   f3 tardiness");
    let mut printed = 0_usize;
    for (i, c) in front.iter().enumerate() {
        if i % stride == 0 || i + 1 == front.len() {
            let o = &c.evaluation.objectives;
            println!("    {:>11.0}   {:>12.0}   {:>11.0}", o[0], o[1], o[2]);
            printed += 1;
            if printed >= 12 {
                break;
            }
        }
    }
    println!();

    if let (Some(corner_ms), Some(corner_ft), Some(corner_td)) = (
        front.first(),
        front.iter().min_by(|a, b| {
            a.evaluation.objectives[1]
                .partial_cmp(&b.evaluation.objectives[1])
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        front.iter().min_by(|a, b| {
            a.evaluation.objectives[2]
                .partial_cmp(&b.evaluation.objectives[2])
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
    ) {
        println!(
            "Makespan corner:  f1={:.0}, f2={:.0}, f3={:.0}",
            corner_ms.evaluation.objectives[0],
            corner_ms.evaluation.objectives[1],
            corner_ms.evaluation.objectives[2],
        );
        println!(
            "Flow-time corner: f1={:.0}, f2={:.0}, f3={:.0}",
            corner_ft.evaluation.objectives[0],
            corner_ft.evaluation.objectives[1],
            corner_ft.evaluation.objectives[2],
        );
        println!(
            "Tardiness corner: f1={:.0}, f2={:.0}, f3={:.0}",
            corner_td.evaluation.objectives[0],
            corner_td.evaluation.objectives[1],
            corner_td.evaluation.objectives[2],
        );
    }
}
