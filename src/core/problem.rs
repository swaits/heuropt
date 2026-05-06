//! The user-implemented `Problem` trait.

use crate::core::decision_variable::DecisionVariable;
use crate::core::evaluation::Evaluation;
use crate::core::objective::ObjectiveSpace;

/// An optimization problem.
///
/// Implement this trait to describe what the optimizer is allowed to vary
/// (`Decision`), how many objectives it has (`objectives`), and how to score a
/// decision (`evaluate`).
///
/// Example decision types: `Vec<f64>`, `Vec<bool>`, `Vec<i64>`, custom domain
/// structs, or permutations represented as `Vec<usize>`.
pub trait Problem {
    /// The thing the optimizer changes. Must be `Clone` because heuristic
    /// algorithms routinely clone decisions.
    type Decision: Clone;

    /// Return the objectives for this problem.
    ///
    /// Returned by value for ergonomics — problems do not need to store an
    /// `ObjectiveSpace` field.
    fn objectives(&self) -> ObjectiveSpace;

    /// Evaluate a decision. Must not mutate `self`.
    fn evaluate(&self, decision: &Self::Decision) -> Evaluation;

    /// Optional schema describing each decision variable — names,
    /// labels, units, and bounds. Used by the explorer JSON export
    /// to label decision-variable axes with the user's preferred
    /// names and units. Default: empty (the exporter generates
    /// fallback names like `x[0]`, `x[1]`).
    ///
    /// Override this on your `Problem` impl to provide pretty
    /// metadata. The returned vector should have one entry per
    /// element of the decision; if its length doesn't match, the
    /// exporter fills the remainder with `x[i]` defaults.
    fn decision_schema(&self) -> Vec<DecisionVariable> {
        Vec::new()
    }
}
