//! Per-generation observation payload passed to [`Observer`](super::Observer).

use std::time::Duration;

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;

/// A view of an optimizer's state at one generation boundary.
///
/// Borrowed (`&'a ...`) rather than owned so the algorithm doesn't
/// have to clone the whole population on every call. Observers that
/// need to retain values across calls should clone what they need
/// out of the snapshot.
#[derive(Debug)]
pub struct Snapshot<'a, D> {
    /// Zero-indexed generation count. The first call is `iteration = 0`
    /// for "after the initial population was built and evaluated";
    /// subsequent calls are after generation 1, 2, …
    pub iteration: usize,

    /// Total `Problem::evaluate` calls so far, including the initial
    /// population.
    pub evaluations: usize,

    /// Wall-clock time since `run_with` started.
    pub elapsed: Duration,

    /// The current population (whatever the algorithm considers the
    /// "live" set this generation). For steady-state algorithms this
    /// is the post-replacement population.
    pub population: &'a [Candidate<D>],

    /// The current Pareto front, if the algorithm tracks one. `None`
    /// for single-objective algorithms.
    pub pareto_front: Option<&'a [Candidate<D>]>,

    /// The current best candidate. `Some` for single-objective
    /// algorithms; `None` for multi-objective unless the algorithm
    /// tracks a notion of best (some don't).
    pub best: Option<&'a Candidate<D>>,

    /// The objective space, useful for observers that need to convert
    /// raw objective values to minimization-oriented form.
    pub objectives: &'a ObjectiveSpace,
}
