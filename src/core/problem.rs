//! The user-implemented `Problem` trait.

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
}
