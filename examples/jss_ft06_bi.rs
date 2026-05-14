//! Solve a bi-objective extension of the Fisher–Thompson FT06 job-shop
//! scheduling benchmark using NSGA-II.
//!
//! - **Benchmark**: FT06 (Fisher & Thompson, 1963), 6 jobs × 6 machines, 36
//!   operations total. Each operation has a fixed machine and processing
//!   time; operations within a job must run in the given order.
//! - **Canonical (single-objective) optimum**: makespan **55**.
//! - **Bi-objective extension** (this example):
//!     - f₁ = makespan (Cₘₐₓ)
//!     - f₂ = total flow time Σⱼ Cⱼ
//!
//!   Both are standard JSS objectives in the multi-objective literature.
//! - **Algorithm**: [`Nsga2`].
//! - **Encoding**: operation-based string of length 36, each job id appears
//!   6 times. The k-th occurrence of job `j` represents the k-th operation
//!   of job `j`.
//! - **Variation**: a local `PrecedenceOrderCrossover` (POX) piped into
//!   [`InversionMutation`] via [`CompositeVariation`]. The strict-permutation
//!   crossovers shipped in the library (OX, PMX, CX, ERX) would break the
//!   operation-string multiset, so this example defines a small JSS-aware
//!   crossover inline. POX is the standard crossover for operation-based JSS
//!   GAs (Lee & Yamakawa, 1996; Bierwirth et al., 1996).
//! - **Initializer**: [`ShuffledMultisetPermutation`].
//!
//! Sources:
//! - Fisher, H., Thompson, G. L. (1963). *Probabilistic learning combinations
//!   of local job-shop scheduling rules.*
//! - OR-Library / JSPLIB FT06 instance file.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example jss_ft06_bi
//! ```

use heuropt::prelude::*;
use rand::Rng as _;

/// Precedence-preserving Order-based Crossover for operation-string JSS
/// encodings. Partitions job ids into two sets J1 / J2; the child takes
/// positions occupied by J1 from parent A and fills the remaining positions
/// with J2's operations in parent B's order. Two children are produced by
/// reversing the parent roles.
///
/// Preserves the JSS multiset invariant (each job id appears `N_MACHINES`
/// times) because every operation in the multiset is covered exactly once:
/// J1 ops by parent A, J2 ops by parent B.
#[derive(Debug, Clone, Copy, Default)]
struct PrecedenceOrderCrossover;

impl Variation<Vec<usize>> for PrecedenceOrderCrossover {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(parents.len() >= 2, "POX requires 2 parents");
        let p1 = &parents[0];
        let p2 = &parents[1];
        let mut in_j1 = [false; N_JOBS];
        // Ensure both partitions are non-empty to avoid degenerate (child == one parent).
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

/// FT06 routing — machine id for the k-th operation of job j.
const FT06_MACHINE: [[usize; 6]; 6] = [
    [2, 0, 1, 3, 5, 4],
    [1, 2, 4, 5, 0, 3],
    [2, 3, 5, 0, 1, 4],
    [1, 0, 2, 3, 4, 5],
    [2, 1, 4, 5, 0, 3],
    [1, 3, 5, 0, 4, 2],
];

/// FT06 processing times — duration of the k-th operation of job j on the
/// machine given by `FT06_MACHINE[j][k]`.
const FT06_TIME: [[f64; 6]; 6] = [
    [1.0, 3.0, 6.0, 7.0, 3.0, 6.0],
    [8.0, 5.0, 10.0, 10.0, 10.0, 4.0],
    [5.0, 4.0, 8.0, 9.0, 1.0, 7.0],
    [5.0, 5.0, 5.0, 3.0, 8.0, 9.0],
    [9.0, 3.0, 5.0, 4.0, 3.0, 1.0],
    [3.0, 3.0, 9.0, 10.0, 4.0, 1.0],
];

const N_JOBS: usize = 6;
const N_MACHINES: usize = 6;
const KNOWN_MAKESPAN_OPTIMUM: f64 = 55.0;

struct Ft06BiObjective;

impl Problem for Ft06BiObjective {
    type Decision = Vec<usize>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("makespan"),
            Objective::minimize("total_flow_time"),
        ])
    }

    fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
        let mut job_next = [0_usize; N_JOBS];
        let mut job_clock = [0.0_f64; N_JOBS];
        let mut machine_clock = [0.0_f64; N_MACHINES];

        for &job in schedule {
            let k = job_next[job];
            let m = FT06_MACHINE[job][k];
            let t = FT06_TIME[job][k];
            let start = job_clock[job].max(machine_clock[m]);
            let end = start + t;
            job_clock[job] = end;
            machine_clock[m] = end;
            job_next[job] = k + 1;
        }
        let makespan = machine_clock.iter().cloned().fold(0.0_f64, f64::max);
        let flow_time: f64 = job_clock.iter().sum();
        Evaluation::new(vec![makespan, flow_time])
    }

    fn decision_schema(&self) -> Vec<DecisionVariable> {
        (0..N_JOBS * N_MACHINES)
            .map(|k| DecisionVariable::new(format!("op_slot_{k}")))
            .collect()
    }
}

fn main() {
    let problem = Ft06BiObjective;

    let mut optimizer = Nsga2::new(
        Nsga2Config {
            population_size: 200,
            generations: 1500,
            seed: 7,
        },
        ShuffledMultisetPermutation::new(vec![N_MACHINES; N_JOBS]),
        CompositeVariation {
            crossover: PrecedenceOrderCrossover,
            mutation: SwapMutation,
        },
    );
    let result = optimizer.run(&problem);

    println!("FT06 — bi-objective JSS via NSGA-II");
    println!("Source: Fisher & Thompson (1963); known single-objective optimum makespan = 55");
    println!();
    println!("Total evaluations: {}", result.evaluations);
    println!("Pareto-front size: {}", result.pareto_front.len());
    println!();

    // Sort front by makespan ascending and print a sample of points.
    let mut front: Vec<&Candidate<Vec<usize>>> = result.pareto_front.iter().collect();
    front.sort_by(|a, b| {
        a.evaluation.objectives[0]
            .partial_cmp(&b.evaluation.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate by objective values so the output isn't a wall of identical rows.
    let mut seen: Vec<(i64, i64)> = Vec::new();
    println!("    makespan   total flow time");
    for c in &front {
        let o = &c.evaluation.objectives;
        let key = (o[0] as i64, o[1] as i64);
        if !seen.contains(&key) {
            seen.push(key);
            println!("    {:>8.0}   {:>15.0}", o[0], o[1]);
        }
    }
    println!("    ({} unique objective-space points)", seen.len());
    println!();

    // Compare the makespan-corner against the known optimum.
    if let Some(makespan_corner) = front.first() {
        let best_makespan = makespan_corner.evaluation.objectives[0];
        let gap_abs = best_makespan - KNOWN_MAKESPAN_OPTIMUM;
        let gap_pct = 100.0 * gap_abs / KNOWN_MAKESPAN_OPTIMUM;
        println!(
            "Makespan corner: {:.0}   vs. known optimum 55   (gap {:+.0}, {:+.2}%)",
            best_makespan, gap_abs, gap_pct
        );
    }
}
