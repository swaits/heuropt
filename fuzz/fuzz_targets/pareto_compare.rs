#![no_main]
//! Fuzz `pareto_compare` for anti-symmetry and reflexivity.
//!
//! Invariants checked:
//!   * `compare(a, b)` and `compare(b, a)` form an anti-symmetric pair
//!     (`Dominates ↔ DominatedBy`, `Equal ↔ Equal`, `NonDominated ↔ NonDominated`).
//!   * `compare(a, a) == Equal`.
//!   * No panics on any combination of finite/non-finite floats.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::pareto::dominance::{Dominance, pareto_compare};

#[derive(Arbitrary, Debug)]
struct Input {
    a_objs: Vec<f64>,
    b_objs: Vec<f64>,
    a_violation: f64,
    b_violation: f64,
    minimize_mask: u8,
}

fuzz_target!(|input: Input| {
    if input.a_objs.is_empty() || input.a_objs.len() != input.b_objs.len() {
        return;
    }
    if input.a_objs.len() > 8 {
        return;
    }
    let m = input.a_objs.len();
    let space = ObjectiveSpace::new(
        (0..m)
            .map(|i| {
                if (input.minimize_mask >> i) & 1 == 0 {
                    Objective::minimize(format!("f{i}"))
                } else {
                    Objective::maximize(format!("f{i}"))
                }
            })
            .collect(),
    );

    let a = Evaluation::constrained(input.a_objs.clone(), input.a_violation);
    let b = Evaluation::constrained(input.b_objs.clone(), input.b_violation);

    let ab = pareto_compare(&a, &b, &space);
    let ba = pareto_compare(&b, &a, &space);
    let aa = pareto_compare(&a, &a, &space);

    // Anti-symmetry pairs.
    let antisymmetric = matches!(
        (ab, ba),
        (Dominance::Dominates, Dominance::DominatedBy)
            | (Dominance::DominatedBy, Dominance::Dominates)
            | (Dominance::Equal, Dominance::Equal)
            | (Dominance::NonDominated, Dominance::NonDominated),
    );
    assert!(antisymmetric, "asymmetric: ab={ab:?}, ba={ba:?}");

    // Reflexivity (when objectives are finite — NaNs make equality
    // ill-defined, so skip the check there).
    if input.a_objs.iter().all(|v| v.is_finite()) && input.a_violation.is_finite() {
        assert_eq!(aa, Dominance::Equal);
    }
});
