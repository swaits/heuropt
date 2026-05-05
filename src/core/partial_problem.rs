//! Trait for multi-fidelity optimization problems.

use crate::core::evaluation::Evaluation;
use crate::core::objective::ObjectiveSpace;

/// A problem whose evaluation cost can be controlled by a fidelity
/// "budget" parameter — for example, an ML training run that gets
/// trained for `budget` epochs, a CFD simulation that runs for `budget`
/// timesteps, or a Monte Carlo evaluation that draws `budget` samples.
///
/// Multi-fidelity optimizers (Hyperband, Successive Halving, BOHB) use
/// this trait to evaluate cheap low-budget previews of many
/// configurations, then "promote" the survivors to higher budgets.
///
/// `PartialProblem` is intentionally NOT a sub-trait of [`Problem`]
/// because the evaluation contract is different: `Problem::evaluate`
/// is single-shot, while `evaluate_at_budget` is parameterized by
/// fidelity. Implementors who already have a `Problem` and want their
/// `evaluate_at_budget` to ignore the budget can write a one-line
/// wrapper that just calls `Problem::evaluate`.
///
/// [`Problem`]: crate::core::Problem
pub trait PartialProblem {
    /// The thing the optimizer changes. Same constraints as
    /// [`Problem::Decision`](crate::core::Problem::Decision).
    type Decision: Clone;

    /// Return the objectives for this problem.
    fn objectives(&self) -> ObjectiveSpace;

    /// Evaluate `decision` at the given fidelity `budget`.
    ///
    /// Higher `budget` should give a more accurate (and more expensive)
    /// estimate of the same underlying objective. Hyperband requires
    /// monotonicity: a higher-budget evaluation should not be worse
    /// than a lower-budget evaluation by chance — though some noise is
    /// fine and expected.
    fn evaluate_at_budget(&self, decision: &Self::Decision, budget: f64) -> Evaluation;
}
