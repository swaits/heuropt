#![no_main]
//! Fuzz `crowding_distance` for shape and non-negativity.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::pareto::crowding::crowding_distance;

#[derive(Arbitrary, Debug)]
struct Input {
    points: Vec<(f64, f64)>,
}

fuzz_target!(|input: Input| {
    if input.points.len() > 64 {
        return;
    }
    // Bound magnitudes — crowding's `(max - min)` and per-axis gaps can
    // both overflow to +∞ when points span ±f64::MAX, yielding inf/inf=NaN.
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

    let front: Vec<usize> = (0..pop.len()).collect();
    let d = crowding_distance(&pop, &front, &space);
    assert_eq!(d.len(), front.len(), "crowding distance length mismatch");
    for (i, &v) in d.iter().enumerate() {
        assert!(v >= 0.0 || v.is_infinite(), "negative crowding[{i}] = {v}");
        assert!(!v.is_nan(), "NaN crowding[{i}]");
    }
    // If size <= 2, every entry is +∞.
    if pop.len() <= 2 {
        for (i, &v) in d.iter().enumerate() {
            assert!(v.is_infinite(), "size<=2 crowding[{i}] not inf: {v}");
        }
    }
});
