//! A decision paired with its evaluation.

use crate::core::evaluation::Evaluation;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A decision together with its evaluated objective values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate<D> {
    /// The decision (input to the problem).
    pub decision: D,
    /// The evaluated objective values and constraint violation.
    pub evaluation: Evaluation,
}

impl<D> Candidate<D> {
    /// Pair a decision with its evaluation.
    pub fn new(decision: D, evaluation: Evaluation) -> Self {
        Self {
            decision,
            evaluation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pairs_decision_and_evaluation() {
        let c = Candidate::new(vec![1.0, 2.0], Evaluation::new(vec![5.0]));
        assert_eq!(c.decision, vec![1.0, 2.0]);
        assert_eq!(c.evaluation.objectives, vec![5.0]);
    }
}
