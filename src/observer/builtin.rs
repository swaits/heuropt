//! Built-in observers covering the common stop conditions.

use std::ops::ControlFlow;
use std::time::Duration;

use super::{Observer, Snapshot};
use crate::core::objective::Direction;

/// Halt after a fixed wall-clock duration since `run_with` started.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
/// use std::time::Duration;
///
/// let stop = MaxTime::new(Duration::from_millis(50));
/// // pass `&mut stop` to `Optimizer::run_with`.
/// # let _ = stop;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MaxTime {
    pub limit: Duration,
}

impl MaxTime {
    pub fn new(limit: Duration) -> Self {
        Self { limit }
    }
}

impl<D> Observer<D> for MaxTime {
    #[inline]
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        if snap.elapsed >= self.limit {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Halt after a target number of generations.
///
/// Most algorithms already take a `generations` count in their config,
/// so this is mostly useful for capping algorithms whose configured
/// loop is open-ended (or for testing).
#[derive(Debug, Clone, Copy)]
pub struct MaxIterations {
    pub limit: usize,
}

impl MaxIterations {
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<D> Observer<D> for MaxIterations {
    #[inline]
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        if snap.iteration >= self.limit {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Halt as soon as the best single-objective fitness reaches `target`.
///
/// Direction-aware: for `Minimize` axes the target is reached when
/// `best ≤ target`; for `Maximize`, when `best ≥ target`.
///
/// Multi-objective snapshots (where `Snapshot::best` is `None` or the
/// problem has more than one objective) are silently ignored — this
/// observer never breaks them.
#[derive(Debug, Clone, Copy)]
pub struct TargetFitness {
    pub target: f64,
}

impl TargetFitness {
    pub fn new(target: f64) -> Self {
        Self { target }
    }
}

impl<D> Observer<D> for TargetFitness {
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        if !snap.objectives.is_single_objective() {
            return ControlFlow::Continue(());
        }
        let direction = snap.objectives.objectives[0].direction;
        if let Some(best) = snap.best
            && let Some(&v) = best.evaluation.objectives.first()
        {
            let hit = match direction {
                Direction::Minimize => v <= self.target,
                Direction::Maximize => v >= self.target,
            };
            if hit {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

/// Halt when the best single-objective fitness has not improved by
/// more than `tolerance` over the last `window` generations.
///
/// Multi-objective snapshots are silently ignored.
#[derive(Debug, Clone)]
pub struct Stagnation {
    pub window: usize,
    pub tolerance: f64,
    history: std::collections::VecDeque<f64>,
}

impl Stagnation {
    pub fn new(window: usize, tolerance: f64) -> Self {
        assert!(window > 0, "Stagnation window must be > 0");
        assert!(
            tolerance >= 0.0,
            "Stagnation tolerance must be non-negative"
        );
        Self {
            window,
            tolerance,
            history: std::collections::VecDeque::with_capacity(window + 1),
        }
    }
}

impl<D> Observer<D> for Stagnation {
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        if !snap.objectives.is_single_objective() {
            return ControlFlow::Continue(());
        }
        let direction = snap.objectives.objectives[0].direction;
        let v = match snap
            .best
            .and_then(|c| c.evaluation.objectives.first().copied())
        {
            Some(v) => v,
            None => return ControlFlow::Continue(()),
        };
        // Push to history; cap at window+1 so we always have 1 + window samples.
        self.history.push_back(v);
        while self.history.len() > self.window + 1 {
            self.history.pop_front();
        }
        if self.history.len() <= self.window {
            return ControlFlow::Continue(());
        }
        let oldest = self.history.front().copied().unwrap();
        let newest = self.history.back().copied().unwrap();
        let improvement = match direction {
            Direction::Minimize => oldest - newest,
            Direction::Maximize => newest - oldest,
        };
        if improvement <= self.tolerance {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Compose two observers — break if **either** signals a break.
#[derive(Debug, Clone, Copy)]
pub struct AnyOf<A, B> {
    pub a: A,
    pub b: B,
}

impl<D, A, B> Observer<D> for AnyOf<A, B>
where
    A: Observer<D>,
    B: Observer<D>,
{
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        // Always poll both so stateful observers (Stagnation) update
        // their history, then OR the results.
        let ra = self.a.observe(snap);
        let rb = self.b.observe(snap);
        if ra.is_break() || rb.is_break() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Compose two observers — break only if **both** signal a break in
/// the same call.
#[derive(Debug, Clone, Copy)]
pub struct AllOf<A, B> {
    pub a: A,
    pub b: B,
}

impl<D, A, B> Observer<D> for AllOf<A, B>
where
    A: Observer<D>,
    B: Observer<D>,
{
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        let ra = self.a.observe(snap);
        let rb = self.b.observe(snap);
        if ra.is_break() && rb.is_break() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Call a user closure every `every` generations (default 1 = every
/// generation). Useful for periodic logging without bloating callback
/// frequency.
pub struct Periodic<F> {
    pub every: usize,
    counter: usize,
    pub callback: F,
}

impl<F> Periodic<F> {
    pub fn new(every: usize, callback: F) -> Self {
        assert!(every >= 1, "Periodic every must be >= 1");
        Self {
            every,
            counter: 0,
            callback,
        }
    }
}

impl<D, F> Observer<D> for Periodic<F>
where
    F: FnMut(&Snapshot<'_, D>),
{
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        self.counter += 1;
        if self.counter >= self.every {
            self.counter = 0;
            (self.callback)(snap);
        }
        ControlFlow::Continue(())
    }
}

/// Tracing-backed observer — emits a structured `debug!` event per
/// generation with iteration / evaluations / elapsed / best fitness.
///
/// Available only with the `tracing` feature.
#[cfg(feature = "tracing")]
#[derive(Debug, Default, Clone, Copy)]
pub struct TracingObserver;

#[cfg(feature = "tracing")]
impl<D> Observer<D> for TracingObserver {
    fn observe(&mut self, snap: &Snapshot<'_, D>) -> ControlFlow<()> {
        let best = snap
            .best
            .and_then(|c| c.evaluation.objectives.first().copied());
        tracing::debug!(
            iteration = snap.iteration,
            evaluations = snap.evaluations,
            elapsed_ms = snap.elapsed.as_millis() as u64,
            best = ?best,
            front_size = snap.pareto_front.map(|f| f.len()),
            "heuropt generation",
        );
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::candidate::Candidate;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Objective, ObjectiveSpace};

    fn snap_with_best<'a>(
        iteration: usize,
        elapsed_ms: u64,
        best: Option<&'a Candidate<()>>,
        objectives: &'a ObjectiveSpace,
        empty_pop: &'a [Candidate<()>],
    ) -> Snapshot<'a, ()> {
        Snapshot {
            iteration,
            evaluations: 0,
            elapsed: Duration::from_millis(elapsed_ms),
            population: empty_pop,
            pareto_front: None,
            best,
            objectives,
        }
    }

    #[test]
    fn max_time_breaks_after_limit() {
        let space = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop: Vec<Candidate<()>> = vec![];
        let mut o = MaxTime::new(Duration::from_millis(100));
        let s = snap_with_best(0, 50, None, &space, &pop);
        assert!(o.observe(&s).is_continue());
        let s = snap_with_best(1, 100, None, &space, &pop);
        assert!(o.observe(&s).is_break());
    }

    #[test]
    fn target_fitness_minimize() {
        let space = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop: Vec<Candidate<()>> = vec![];
        let cand = Candidate::new((), Evaluation::new(vec![0.005]));
        let mut o = TargetFitness::new(0.01);
        let s = snap_with_best(0, 0, Some(&cand), &space, &pop);
        assert!(o.observe(&s).is_break());
    }

    #[test]
    fn target_fitness_maximize() {
        let space = ObjectiveSpace::new(vec![Objective::maximize("f")]);
        let pop: Vec<Candidate<()>> = vec![];
        let cand_below = Candidate::new((), Evaluation::new(vec![0.5]));
        let cand_above = Candidate::new((), Evaluation::new(vec![1.5]));
        let mut o = TargetFitness::new(1.0);
        let s = snap_with_best(0, 0, Some(&cand_below), &space, &pop);
        assert!(o.observe(&s).is_continue());
        let s = snap_with_best(1, 0, Some(&cand_above), &space, &pop);
        assert!(o.observe(&s).is_break());
    }

    #[test]
    fn stagnation_breaks_on_no_improvement() {
        let space = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop: Vec<Candidate<()>> = vec![];
        let mut o = Stagnation::new(3, 1e-6);

        // Five generations of "no improvement" — same value every time.
        for i in 0..3 {
            let cand = Candidate::new((), Evaluation::new(vec![1.0]));
            let s = snap_with_best(i, 0, Some(&cand), &space, &pop);
            // First `window` calls just fill history; should not break.
            assert!(o.observe(&s).is_continue());
        }
        let cand = Candidate::new((), Evaluation::new(vec![1.0]));
        let s = snap_with_best(3, 0, Some(&cand), &space, &pop);
        assert!(o.observe(&s).is_break());
    }

    #[test]
    fn stagnation_does_not_break_on_improvement() {
        let space = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop: Vec<Candidate<()>> = vec![];
        let mut o = Stagnation::new(2, 1e-6);
        let values = [1.0, 0.9, 0.8, 0.7];
        for (i, &v) in values.iter().enumerate() {
            let cand = Candidate::new((), Evaluation::new(vec![v]));
            let s = snap_with_best(i, 0, Some(&cand), &space, &pop);
            assert!(o.observe(&s).is_continue(), "iter {i}");
        }
    }

    #[test]
    fn anyof_breaks_when_either_breaks() {
        let space = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop: Vec<Candidate<()>> = vec![];
        let cand = Candidate::new((), Evaluation::new(vec![5.0]));
        let mut o =
            <MaxIterations as Observer<()>>::or(MaxIterations::new(3), TargetFitness::new(1.0));
        for i in 0..3 {
            let s = snap_with_best(i, 0, Some(&cand), &space, &pop);
            assert!(o.observe(&s).is_continue(), "iter {i}");
        }
        // iteration = 3 hits MaxIterations limit → break
        let s = snap_with_best(3, 0, Some(&cand), &space, &pop);
        assert!(o.observe(&s).is_break());
    }

    #[test]
    fn periodic_calls_callback_every_n() {
        let space = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop: Vec<Candidate<()>> = vec![];
        let mut count = 0_usize;
        {
            let mut o = Periodic::new(3, |_: &Snapshot<'_, ()>| count += 1);
            for i in 0..10 {
                let s = snap_with_best(i, 0, None, &space, &pop);
                let _ = o.observe(&s);
            }
        }
        assert_eq!(count, 3); // every 3rd of 10 = generations 2, 5, 8
    }

    #[test]
    fn closure_implements_observer() {
        let space = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop: Vec<Candidate<()>> = vec![];
        let mut count = 0_usize;
        let mut closure = |_: &Snapshot<'_, ()>| -> ControlFlow<()> {
            count += 1;
            if count >= 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let s = snap_with_best(0, 0, None, &space, &pop);
        assert!(<_ as Observer<()>>::observe(&mut closure, &s).is_continue());
        assert!(<_ as Observer<()>>::observe(&mut closure, &s).is_break());
    }
}
