//! Objective values and total constraint violation for a single decision.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The result of evaluating a decision: objective values plus total constraint violation.
///
/// A non-positive `constraint_violation` means the candidate is feasible.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    /// Objective values in the order declared by the problem.
    pub objectives: Vec<f64>,
    /// Total constraint violation. `<= 0.0` is feasible; positive is infeasible.
    pub constraint_violation: f64,
}

impl Evaluation {
    /// Build a feasible evaluation from objective values.
    pub fn new(objectives: Vec<f64>) -> Self {
        Self {
            objectives,
            constraint_violation: 0.0,
        }
    }

    /// Build an evaluation with a known total constraint violation.
    pub fn constrained(objectives: Vec<f64>, constraint_violation: f64) -> Self {
        Self {
            objectives,
            constraint_violation,
        }
    }

    /// Returns `true` when `constraint_violation <= 0.0`.
    pub fn is_feasible(&self) -> bool {
        self.constraint_violation <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_feasible() {
        let e = Evaluation::new(vec![1.0, 2.0]);
        assert_eq!(e.constraint_violation, 0.0);
        assert!(e.is_feasible());
    }

    #[test]
    fn constrained_sets_violation() {
        let e = Evaluation::constrained(vec![0.0], 0.5);
        assert!(!e.is_feasible());
        assert_eq!(e.constraint_violation, 0.5);
    }

    #[test]
    fn zero_or_negative_violation_is_feasible() {
        assert!(Evaluation::constrained(vec![0.0], 0.0).is_feasible());
        assert!(Evaluation::constrained(vec![0.0], -1.0).is_feasible());
    }
}
