//! The single trait users implement to add a new optimizer.

use std::time::{Duration, Instant};

use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::observer::{Observer, Snapshot};

/// An optimizer that runs to completion in a single call.
///
/// Implementations own their main loop, manage their own state, and return an
/// [`OptimizationResult`]. Invalid configuration panics with a clear
/// message rather than returning a `Result`.
pub trait Optimizer<P>
where
    P: Problem,
{
    /// Run the optimizer to completion against `problem`.
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision>;

    /// Run with an [`Observer`] called after each generation.
    ///
    /// The observer can halt the run by returning
    /// [`std::ops::ControlFlow::Break`]; the partial result is still
    /// returned. Built-in observers in
    /// [`heuropt::observer::builtin`](crate::observer::builtin) cover
    /// the common stop conditions (`MaxTime`, `TargetFitness`,
    /// `Stagnation`, …).
    ///
    /// **Default impl:** falls back to `run` plus a single final
    /// notification. Algorithms that override this method get true
    /// per-generation observation; algorithms that don't get a single
    /// notification at the end. The trait-level docstring on each
    /// algorithm calls out which behavior it supports.
    fn run_with<O>(&mut self, problem: &P, observer: &mut O) -> OptimizationResult<P::Decision>
    where
        O: Observer<P::Decision>,
    {
        let started = Instant::now();
        let result = self.run(problem);
        let elapsed = started.elapsed();
        notify_final(&result, elapsed, problem, observer);
        result
    }
}

/// Helper used by the default `run_with` impl: build a single final-
/// state snapshot and hand it to the observer once. Algorithms that
/// override `run_with` for per-generation reporting don't go through
/// this path — they construct their own per-iteration snapshots.
fn notify_final<P, O>(
    result: &OptimizationResult<P::Decision>,
    elapsed: Duration,
    problem: &P,
    observer: &mut O,
) where
    P: Problem,
    O: Observer<P::Decision>,
{
    let objectives = problem.objectives();
    let snap = Snapshot {
        iteration: result.generations,
        evaluations: result.evaluations,
        elapsed,
        population: result.population.as_slice(),
        pareto_front: Some(result.pareto_front.as_slice()),
        best: result.best.as_ref(),
        objectives: &objectives,
    };
    let _ = observer.observe(&snap);
}
