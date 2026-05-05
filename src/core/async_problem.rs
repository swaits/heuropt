//! Async-evaluable problems for IO-bound workloads.
//!
//! Most heuropt algorithms operate synchronously: their `Problem::evaluate`
//! returns immediately. For workloads where evaluation is *IO-bound* — calling
//! an HTTP service, querying a remote model, spawning a subprocess —
//! awaiting an async fn is much more efficient than blocking a worker
//! thread.
//!
//! [`AsyncProblem`] mirrors [`Problem`](crate::core::Problem) but its
//! `evaluate_async` returns a future. Algorithms that support async
//! evaluation (NSGA-II, DE, RandomSearch as of v0.7.0; others land
//! incrementally) expose a `run_async` method that drives evaluations
//! through a user-chosen async runtime (typically tokio).
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
