//! Per-generation observation, callbacks, and stop conditions.
//!
//! Algorithms accept an [`Observer`] via [`Optimizer::run_with`] and call
//! it once per generation (where "generation" makes sense for that
//! algorithm — see each algorithm's docs). Returning
//! [`std::ops::ControlFlow::Break`] from an observer halts the optimizer
//! and the partial [`OptimizationResult`] is returned to the caller.
//!
//! Observers can be composed with [`builtin::AnyOf`] / [`builtin::AllOf`].
//!
//! [`OptimizationResult`]: crate::core::result::OptimizationResult
//! [`Optimizer::run_with`]: crate::traits::Optimizer::run_with

pub mod builtin;
mod snapshot;

pub use snapshot::Snapshot;

use std::ops::ControlFlow;

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;

/// A callback invoked by an [`Optimizer`](crate::traits::Optimizer)
/// after every generation. Return [`ControlFlow::Break`] to halt
/// the optimizer; [`ControlFlow::Continue`] to keep going.
///
/// Implement directly for stateful observers that need to track
/// history (e.g. stagnation detection, convergence trace logging).
/// For simple stop conditions, use the helpers in
/// [`builtin`](crate::observer::builtin).
pub trait Observer<D> {
    /// Inspect the latest snapshot. Return [`ControlFlow::Break`] to
    /// halt the run; [`ControlFlow::Continue`] to keep going.
    fn observe(&mut self, snapshot: &Snapshot<'_, D>) -> ControlFlow<()>;

    /// Compose with another observer that fires when *either* of them
    /// signals a break.
    fn or<O: Observer<D>>(self, other: O) -> builtin::AnyOf<Self, O>
    where
        Self: Sized,
    {
        builtin::AnyOf { a: self, b: other }
    }

    /// Compose with another observer that fires when *both* of them
    /// signal a break in the same call.
    fn and<O: Observer<D>>(self, other: O) -> builtin::AllOf<Self, O>
    where
        Self: Sized,
    {
        builtin::AllOf { a: self, b: other }
    }
}

/// `()` is the no-op observer. Used as the default when callers don't
/// want any callbacks (it's what `run` uses internally).
impl<D> Observer<D> for () {
    #[inline]
    fn observe(&mut self, _: &Snapshot<'_, D>) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// Closures of the right shape implement Observer too — short-form
/// for one-liner callbacks.
impl<D, F> Observer<D> for F
where
    F: FnMut(&Snapshot<'_, D>) -> ControlFlow<()>,
{
    #[inline]
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        self(snap)
    }
}

/// Build a snapshot for the "final notification" path of the default
/// `run_with` impl on [`Optimizer`](crate::traits::Optimizer).
///
/// Algorithm impls that override `run_with` to call the observer per
/// generation should construct their own snapshots inline rather than
/// using this helper, because they have richer per-generation state.
pub fn finalize_snapshot<'a, D>(
    iteration: usize,
    evaluations: usize,
    elapsed: std::time::Duration,
    population: &'a [Candidate<D>],
    pareto_front: Option<&'a [Candidate<D>]>,
    best: Option<&'a Candidate<D>>,
    objectives: &'a ObjectiveSpace,
) -> Snapshot<'a, D> {
    Snapshot {
        iteration,
        evaluations,
        elapsed,
        population,
        pareto_front,
        best,
        objectives,
    }
}
