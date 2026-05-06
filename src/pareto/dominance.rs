//! Pareto dominance enum and pairwise dominance comparison.

use crate::core::evaluation::Evaluation;
use crate::core::objective::ObjectiveSpace;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The Pareto relationship between two evaluations.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dominance {
    /// `a` dominates `b`.
    Dominates,
    /// `a` is dominated by `b`.
    DominatedBy,
    /// Neither dominates the other.
    NonDominated,
    /// All objective values are equal after orientation conversion.
    Equal,
}

/// Pareto-compare two evaluations under the given objective space.
///
/// Behavior (spec §9.2):
///
/// 1. Feasible candidates dominate infeasible candidates.
/// 2. If both are infeasible, the candidate with smaller
///    `constraint_violation` dominates.
/// 3. Otherwise compare objective values after converting both to
///    minimization orientation via [`ObjectiveSpace::as_minimization`].
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let s = ObjectiveSpace::new(vec![
///     Objective::minimize("f1"),
///     Objective::minimize("f2"),
/// ]);
/// let a = Evaluation::new(vec![1.0, 1.0]);
/// let b = Evaluation::new(vec![2.0, 2.0]);
/// assert_eq!(pareto_compare(&a, &b, &s), Dominance::Dominates);
/// assert_eq!(pareto_compare(&b, &a, &s), Dominance::DominatedBy);
/// ```
pub fn pareto_compare(a: &Evaluation, b: &Evaluation, objectives: &ObjectiveSpace) -> Dominance {
    let a_feasible = a.is_feasible();
    let b_feasible = b.is_feasible();
    match (a_feasible, b_feasible) {
        (true, false) => return Dominance::Dominates,
        (false, true) => return Dominance::DominatedBy,
        (false, false) => {
            return if a.constraint_violation < b.constraint_violation {
                Dominance::Dominates
            } else if a.constraint_violation > b.constraint_violation {
                Dominance::DominatedBy
            } else {
                Dominance::Equal
            };
        }
        (true, true) => {}
    }

    let am = objectives.as_minimization(&a.objectives);
    let bm = objectives.as_minimization(&b.objectives);

    let mut a_better_anywhere = false;
    let mut b_better_anywhere = false;
    for (av, bv) in am.iter().zip(bm.iter()) {
        if av < bv {
            a_better_anywhere = true;
        } else if av > bv {
            b_better_anywhere = true;
        }
    }

    match (a_better_anywhere, b_better_anywhere) {
        (true, false) => Dominance::Dominates,
        (false, true) => Dominance::DominatedBy,
        (false, false) => Dominance::Equal,
        (true, true) => Dominance::NonDominated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::objective::Objective;

    fn space_min2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    #[test]
    fn a_dominates_b() {
        let s = space_min2();
        let a = Evaluation::new(vec![1.0, 1.0]);
        let b = Evaluation::new(vec![2.0, 2.0]);
        assert_eq!(pareto_compare(&a, &b, &s), Dominance::Dominates);
    }

    #[test]
    fn a_dominated_by_b() {
        let s = space_min2();
        let a = Evaluation::new(vec![3.0, 4.0]);
        let b = Evaluation::new(vec![1.0, 2.0]);
        assert_eq!(pareto_compare(&a, &b, &s), Dominance::DominatedBy);
    }

    #[test]
    fn non_dominated_pair() {
        let s = space_min2();
        let a = Evaluation::new(vec![1.0, 4.0]);
        let b = Evaluation::new(vec![3.0, 2.0]);
        assert_eq!(pareto_compare(&a, &b, &s), Dominance::NonDominated);
    }

    #[test]
    fn equal_pair() {
        let s = space_min2();
        let a = Evaluation::new(vec![2.0, 2.0]);
        let b = Evaluation::new(vec![2.0, 2.0]);
        assert_eq!(pareto_compare(&a, &b, &s), Dominance::Equal);
    }

    #[test]
    fn feasible_beats_infeasible() {
        let s = space_min2();
        let a = Evaluation::new(vec![10.0, 10.0]);
        let b = Evaluation::constrained(vec![1.0, 1.0], 5.0);
        assert_eq!(pareto_compare(&a, &b, &s), Dominance::Dominates);
        assert_eq!(pareto_compare(&b, &a, &s), Dominance::DominatedBy);
    }

    #[test]
    fn smaller_violation_wins_among_infeasible() {
        let s = space_min2();
        let a = Evaluation::constrained(vec![100.0, 100.0], 1.0);
        let b = Evaluation::constrained(vec![0.0, 0.0], 5.0);
        assert_eq!(pareto_compare(&a, &b, &s), Dominance::Dominates);
    }

    #[test]
    fn maximize_axis_handled_correctly() {
        let s = ObjectiveSpace::new(vec![
            Objective::minimize("cost"),
            Objective::maximize("accuracy"),
        ]);
        // a: cost=1 (better), accuracy=0.9 (better) → dominates b
        let a = Evaluation::new(vec![1.0, 0.9]);
        let b = Evaluation::new(vec![2.0, 0.8]);
        assert_eq!(pareto_compare(&a, &b, &s), Dominance::Dominates);
    }
}
