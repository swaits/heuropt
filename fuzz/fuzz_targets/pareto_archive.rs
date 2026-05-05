#![no_main]
//! Fuzz `ParetoArchive` for the non-domination invariant under arbitrary
//! insertion/truncation sequences.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::pareto::archive::ParetoArchive;
use heuropt::pareto::dominance::{Dominance, pareto_compare};

#[derive(Arbitrary, Debug)]
enum Op {
    Insert(f64, f64),
    Truncate(u8),
}

#[derive(Arbitrary, Debug)]
struct Input {
    ops: Vec<Op>,
}

fuzz_target!(|input: Input| {
    if input.ops.len() > 64 {
        return;
    }
    let space = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
    let mut archive: ParetoArchive<()> = ParetoArchive::new(space.clone());

    for op in input.ops {
        match op {
            Op::Insert(a, b) => {
                let cand = Candidate::new((), Evaluation::new(vec![a, b]));
                archive.insert(cand);
            }
            Op::Truncate(n) => archive.truncate(n as usize),
        }
    }

    // Members must be pairwise non-dominated.
    let m = archive.members();
    for i in 0..m.len() {
        for j in 0..m.len() {
            if i == j {
                continue;
            }
            let r = pareto_compare(&m[i].evaluation, &m[j].evaluation, &space);
            assert!(
                !matches!(r, Dominance::DominatedBy),
                "archive member {i} dominated by {j}: {:?} vs {:?}",
                m[i].evaluation.objectives,
                m[j].evaluation.objectives,
            );
        }
    }
});
