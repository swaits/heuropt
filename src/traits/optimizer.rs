//! The single trait users implement to add a new optimizer.

use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;

/// An optimizer that runs to completion in a single call.
///
/// Implementations own their main loop, manage their own state, and return an
/// [`OptimizationResult`]. v1 deliberately does not expose a step-by-step API
/// or an associated error type — invalid configuration may panic with a clear
/// message.
pub trait Optimizer<P>
where
    P: Problem,
{
    /// Run the optimizer to completion against `problem`.
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision>;
}
