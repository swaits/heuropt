//! Async-evaluable problems for IO-bound workloads.
//!
//! Most heuropt algorithms operate synchronously: their `Problem::evaluate`
//! returns immediately. For workloads where evaluation is *IO-bound* — calling
//! an HTTP service, querying a remote model, spawning a subprocess —
//! awaiting an async fn is much more efficient than blocking a worker
//! thread.
//!
//! [`AsyncProblem`] mirrors [`Problem`](crate::core::Problem) but its
//! `evaluate_async` returns a future. Every algorithm in heuropt exposes
//! a `run_async` method that drives evaluations through a user-chosen
//! async runtime (typically tokio). Hyperband uses
//! [`AsyncPartialProblem`] instead, which mirrors
//! [`PartialProblem`](crate::core::partial_problem::PartialProblem) for
//! multi-fidelity workloads.
//!
//! Available only with the `async` feature.

use std::future::Future;

use crate::core::evaluation::Evaluation;
use crate::core::objective::ObjectiveSpace;

/// A problem whose evaluation is async — useful when `evaluate` does
/// IO (HTTP, RPC, subprocess) rather than pure CPU work.
///
/// Mirrors [`Problem`](crate::core::Problem) one-for-one except that
/// `evaluate_async` returns a future. The returned future must be
/// `Send` so the algorithm can run many evaluations concurrently
/// across a runtime's worker pool.
///
/// Implementors who already have a synchronous `Problem` can adapt
/// to `AsyncProblem` with a one-line wrapper:
///
/// ```ignore
/// impl AsyncProblem for MyProblem {
///     type Decision = <Self as Problem>::Decision;
///     fn objectives(&self) -> ObjectiveSpace { Problem::objectives(self) }
///     async fn evaluate_async(&self, x: &Self::Decision) -> Evaluation {
///         Problem::evaluate(self, x)
///     }
/// }
/// ```
pub trait AsyncProblem: Sync {
    /// The thing the optimizer changes. Same constraints as
    /// [`Problem::Decision`](crate::core::Problem::Decision).
    type Decision: Clone + Send + Sync;

    /// Return the objectives for this problem.
    fn objectives(&self) -> ObjectiveSpace;

    /// Evaluate `decision` asynchronously. The returned future is
    /// driven by whichever runtime the algorithm's `run_async` is
    /// invoked from.
    fn evaluate_async(&self, decision: &Self::Decision) -> impl Future<Output = Evaluation> + Send;
}

/// Async equivalent of [`PartialProblem`](crate::core::partial_problem::PartialProblem)
/// for multi-fidelity workloads — used by Hyperband's `run_async`.
///
/// Like [`AsyncProblem`], `evaluate_at_budget_async` returns a future
/// so callers can fan out budgeted evaluations across an async runtime.
pub trait AsyncPartialProblem: Sync {
    /// The thing the optimizer changes. Same constraints as
    /// [`PartialProblem::Decision`](crate::core::partial_problem::PartialProblem::Decision).
    type Decision: Clone + Send + Sync;

    /// Return the objectives for this problem.
    fn objectives(&self) -> ObjectiveSpace;

    /// Evaluate `decision` at the given fidelity `budget` asynchronously.
    ///
    /// Same monotonicity contract as
    /// [`PartialProblem::evaluate_at_budget`](crate::core::partial_problem::PartialProblem::evaluate_at_budget):
    /// higher budget should give a more accurate estimate.
    fn evaluate_at_budget_async(
        &self,
        decision: &Self::Decision,
        budget: f64,
    ) -> impl Future<Output = Evaluation> + Send;
}
