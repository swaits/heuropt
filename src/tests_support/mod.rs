//! Shared, deliberately tiny test problems used by algorithm unit tests.
//!
//! Not part of the public API.

use crate::core::evaluation::Evaluation;
use crate::core::objective::{Objective, ObjectiveSpace};
use crate::core::problem::Problem;

/// 1-D minimization sphere `f(x) = x^2`. Single objective, always feasible.
pub struct Sphere1D;

impl Problem for Sphere1D {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, decision: &Vec<f64>) -> Evaluation {
        Evaluation::new(vec![decision[0] * decision[0]])
    }
}

/// Schaffer N.1 — the textbook two-objective minimization warm-up:
///
/// `f1(x) = x^2`, `f2(x) = (x - 2)^2`. Always feasible.
pub struct SchafferN1;

impl Problem for SchafferN1 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let v = x[0];
        Evaluation::new(vec![v * v, (v - 2.0).powi(2)])
    }
}
