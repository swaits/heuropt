#![no_main]
//! Fuzz `non_dominated_sort` for partition correctness.
//!
//! Invariants checked:
//!   * Every population index appears in exactly one front.
//!   * Earlier fronts dominate later fronts (no backwards domination).
//!   * No panics on any vector of finite or non-finite objective values.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::pareto::dominance::{Dominance, pareto_compare};
use heuropt::pareto::sort::non_dominated_sort;

#[derive(Arbitrary, Debug)]
struct Input {
    objectives: Vec<(f64, f64)>,
}

fuzz_target!(|input: Input| {
    if input.objectives.is_empty() || input.objectives.len() > 32 {
        return;
    }
    let space = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
    let pop: Vec<Candidate<()>> = input
        .objectives
        .iter()
        .map(|&(a, b)| Candidate::new((), Evaluation::new(vec![a, b])))
        .collect();

    let fronts = non_dominated_sort(&pop, &space);

    // Partition: every index appears exactly once.
    let mut seen = vec![false; pop.len()];
    for front in &fronts {
        for &idx in front {
            assert!(!seen[idx], "index {idx} in multiple fronts");
            seen[idx] = true;
        }
    }
    for (i, &was) in seen.iter().enumerate() {
        assert!(was, "index {i} missing from all fronts");
    }

    // Earlier fronts cannot be dominated by later fronts.
    for (k, fk) in fronts.iter().enumerate() {
        for fl in fronts.iter().skip(k + 1) {
            for &i in fk {
                for &j in fl {
                    let r = pareto_compare(&pop[i].evaluation, &pop[j].evaluation, &space);
                    assert!(
                        !matches!(r, Dominance::DominatedBy),
                        "front-{k}/{i} dominated by later front",
                    );
                }
            }
        }
    }
});
