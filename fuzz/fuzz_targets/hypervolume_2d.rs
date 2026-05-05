#![no_main]
//! Fuzz `hypervolume_2d` for non-negativity and reference-point handling.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::metrics::hypervolume::hypervolume_2d;

#[derive(Arbitrary, Debug)]
struct Input {
    points: Vec<(f64, f64)>,
    ref_point: (f64, f64),
}

fuzz_target!(|input: Input| {
    if input.points.len() > 64 {
        return;
    }
    // Non-finite floats are permitted by Evaluation, but HV is undefined
    // there — restrict to finite for this property.
    if !input.ref_point.0.is_finite() || !input.ref_point.1.is_finite() {
        return;
    }
    if input
        .points
        .iter()
        .any(|&(a, b)| !a.is_finite() || !b.is_finite())
    {
        return;
    }

    let space = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
    let pop: Vec<Candidate<()>> = input
        .points
        .iter()
        .map(|&(a, b)| Candidate::new((), Evaluation::new(vec![a, b])))
        .collect();
    let hv = hypervolume_2d(&pop, &space, [input.ref_point.0, input.ref_point.1]);
    // HV can be +∞ when the dominated rectangle area overflows f64 (e.g. a
    // ref point at f64::MAX with deeply negative front coords). The
    // contracted invariants are non-negativity and non-NaN.
    assert!(hv >= 0.0, "HV negative: {hv}");
    assert!(!hv.is_nan(), "HV is NaN");
});
