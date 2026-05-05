//! Trait for restoring decisions to feasibility.

/// Transforms an infeasible (or possibly-infeasible) decision into a
/// feasible one, in place.
///
/// `Repair` is the projection-style alternative to penalty-style
/// constraint handling (which uses `Evaluation::constraint_violation`).
/// Wrap a `Variation` operator's output through a `Repair` to guarantee
/// feasibility; or call `repair()` inside your `Problem::evaluate` if
/// the constraint structure is best handled at evaluation time.
pub trait Repair<D> {
    /// Mutate `decision` in place to satisfy the repair's constraints.
    fn repair(&mut self, decision: &mut D);
}
