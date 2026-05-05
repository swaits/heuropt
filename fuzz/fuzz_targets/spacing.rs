#![no_main]
//! Fuzz the `spacing` metric for non-negativity.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::metrics::spacing::spacing;

#[derive(Arbitrary, Debug)]
struct Input {
    points: Vec<(f64, f64)>,
}

fuzz_target!(|input: Input| {
    if input.points.len() > 64 {
        return;
    }
    // Bound magnitudes so distance computations don't overflow to
    // inf-inf=NaN — `spacing` is documented to operate on values produced
    // by `as_minimization` of problem evaluations, not arbitrary f64s.
    if input
        .points
        .iter()
        .any(|&(a, b)| !a.is_finite() || !b.is_finite() || a.abs() > 1e150 || b.abs() > 1e150)
    {
        return;
    }
    let space = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
    let pop: Vec<Candidate<()>> = input
        .points
        .iter()
        .map(|&(a, b)| Candidate::new((), Evaluation::new(vec![a, b])))
        .collect();
    let s = spacing(&pop, &space);
    // Spacing can overflow to +∞ when point coordinates straddle ±f64::MAX.
    // Contract is non-negative + non-NaN.
    assert!(s >= 0.0, "spacing negative: {s}");
    assert!(!s.is_nan(), "spacing is NaN");
});
