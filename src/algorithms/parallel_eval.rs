//! Population-wide evaluation helper used by the population-based algorithms.
//!
//! Two cfg-gated implementations:
//!
//! - With the `parallel` feature: rayon's `into_par_iter` evaluates decisions
//!   on the global thread pool. The result preserves input order so any
//!   downstream sort/dominance/crowding decisions remain reproducible.
//! - Without the feature: plain serial `into_iter`.
//!
//! Both implementations require `P: Problem + Sync` and `P::Decision: Send`,
//! so each algorithm's `Optimizer<P>` impl carries the same bounds regardless
//! of feature state. This keeps the public `Problem` trait itself unchanged.

use crate::core::candidate::Candidate;
use crate::core::problem::Problem;

/// Evaluate every decision in `decisions` against `problem` and return the
/// resulting candidates in the same order.
#[cfg(feature = "parallel")]
pub(crate) fn evaluate_batch<P>(
    problem: &P,
    decisions: Vec<P::Decision>,
) -> Vec<Candidate<P::Decision>>
where
    P: Problem + Sync,
    P::Decision: Send,
{
    use rayon::prelude::*;
    decisions
        .into_par_iter()
        .map(|d| {
            let e = problem.evaluate(&d);
            Candidate::new(d, e)
        })
        .collect()
}

/// Serial fallback used when the `parallel` feature is disabled.
#[cfg(not(feature = "parallel"))]
pub(crate) fn evaluate_batch<P>(
    problem: &P,
    decisions: Vec<P::Decision>,
) -> Vec<Candidate<P::Decision>>
where
    P: Problem + Sync,
    P::Decision: Send,
{
    decisions
        .into_iter()
        .map(|d| {
            let e = problem.evaluate(&d);
            Candidate::new(d, e)
        })
        .collect()
}
