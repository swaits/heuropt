//! Async population evaluator.
//!
//! Available only with the `async` feature. Used by the `run_async`
//! method on algorithms that support async problems.

use futures::stream::{FuturesOrdered, StreamExt};

use crate::core::async_problem::AsyncProblem;
use crate::core::candidate::Candidate;

/// Evaluate every decision concurrently against `problem`, preserving
/// input order in the returned vector. Concurrency is bounded by
/// `concurrency` (≥ 1) — too high a value wastes memory and may
/// overload downstream services; too low forfeits parallelism.
///
/// Returns a future that the caller drives via their preferred
/// runtime (typically tokio).
pub async fn evaluate_batch_async<P>(
    problem: &P,
    decisions: Vec<P::Decision>,
    concurrency: usize,
) -> Vec<Candidate<P::Decision>>
where
    P: AsyncProblem,
{
    assert!(
        concurrency >= 1,
        "evaluate_batch_async concurrency must be >= 1"
    );
    let mut out: Vec<Candidate<P::Decision>> = Vec::with_capacity(decisions.len());

    // Process in concurrency-bounded chunks to keep peak memory low
    // and avoid blasting downstream services. Each chunk uses
    // FuturesOrdered to preserve per-chunk order, and chunks are
    // emitted in their natural order.
    let mut iter = decisions.into_iter();
    loop {
        let mut futs = FuturesOrdered::new();
        for _ in 0..concurrency {
            match iter.next() {
                Some(d) => {
                    futs.push_back(async move {
                        let e = problem.evaluate_async(&d).await;
                        Candidate::new(d, e)
                    });
                }
                None => break,
            }
        }
        if futs.is_empty() {
            break;
        }
        while let Some(c) = futs.next().await {
            out.push(c);
        }
    }
    out
}
