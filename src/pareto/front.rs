//! Compute the non-dominated front of a population, and the single-objective best.

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::pareto::dominance::{Dominance, pareto_compare};

/// Return all candidates that are not dominated by any other candidate.
///
/// O(N²·M) in v1 (spec §9.3). Input order is preserved among returned
/// candidates.
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
/// let pop = [
///     Candidate::new(1u32, Evaluation::new(vec![1.0, 4.0])), // non-dominated
///     Candidate::new(2u32, Evaluation::new(vec![3.0, 2.0])), // non-dominated
///     Candidate::new(3u32, Evaluation::new(vec![5.0, 5.0])), // dominated
/// ];
/// let front = pareto_front(&pop, &s);
/// let kept: Vec<u32> = front.iter().map(|c| c.decision).collect();
/// assert_eq!(kept, vec![1, 2]);
/// ```
pub fn pareto_front<D: Clone>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
) -> Vec<Candidate<D>> {
    let mut out = Vec::new();
    'outer: for (i, a) in population.iter().enumerate() {
        for (j, b) in population.iter().enumerate() {
            if i == j {
                continue;
            }
            if matches!(
                pareto_compare(&a.evaluation, &b.evaluation, objectives),
                Dominance::DominatedBy
            ) {
                continue 'outer;
            }
        }
        out.push(a.clone());
    }
    out
}

/// Return the best candidate for a single-objective problem.
///
/// Returns `None` if there is not exactly one objective, if the population is
/// empty, or if every candidate is infeasible (spec §9.4).
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
/// let pop = [
///     Candidate::new(1u32, Evaluation::new(vec![3.0])),
///     Candidate::new(2u32, Evaluation::new(vec![1.0])),
///     Candidate::new(3u32, Evaluation::new(vec![2.0])),
/// ];
/// let best = best_candidate(&pop, &s).unwrap();
/// assert_eq!(best.decision, 2);
/// ```
pub fn best_candidate<D: Clone>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
) -> Option<Candidate<D>> {
    if !objectives.is_single_objective() {
        return None;
    }

    let mut best: Option<&Candidate<D>> = None;
    let mut best_min: f64 = f64::INFINITY;
    for c in population {
        if !c.evaluation.is_feasible() {
            continue;
        }
        let m = objectives.as_minimization(&c.evaluation.objectives);
        let v = m.first().copied().unwrap_or(f64::INFINITY);
        if best.is_none() || v < best_min {
            best = Some(c);
            best_min = v;
        }
    }
    best.cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::Objective;

    fn cand(decision: u32, obj: Vec<f64>) -> Candidate<u32> {
        Candidate::new(decision, Evaluation::new(obj))
    }

    fn space_min2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    #[test]
    fn empty_population_returns_empty_front() {
        let s = space_min2();
        let front = pareto_front::<u32>(&[], &s);
        assert!(front.is_empty());
    }

    #[test]
    fn single_candidate_is_its_own_front() {
        let s = space_min2();
        let pop = [cand(1, vec![1.0, 2.0])];
        let front = pareto_front(&pop, &s);
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].decision, 1);
    }

    #[test]
    fn dominated_points_removed_non_dominated_kept() {
        let s = space_min2();
        let pop = [
            cand(1, vec![1.0, 4.0]), // non-dominated
            cand(2, vec![3.0, 2.0]), // non-dominated
            cand(3, vec![5.0, 5.0]), // dominated by 1 and 2
            cand(4, vec![2.0, 3.0]), // non-dominated
        ];
        let front = pareto_front(&pop, &s);
        let kept: Vec<u32> = front.iter().map(|c| c.decision).collect();
        assert_eq!(kept, vec![1, 2, 4]);
    }

    #[test]
    fn best_candidate_none_when_multi_objective() {
        let s = space_min2();
        let pop = [cand(1, vec![1.0, 1.0])];
        assert!(best_candidate(&pop, &s).is_none());
    }

    #[test]
    fn best_candidate_returns_min_for_minimize() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [cand(1, vec![3.0]), cand(2, vec![1.0]), cand(3, vec![2.0])];
        let best = best_candidate(&pop, &s).unwrap();
        assert_eq!(best.decision, 2);
    }

    #[test]
    fn best_candidate_returns_max_for_maximize() {
        let s = ObjectiveSpace::new(vec![Objective::maximize("score")]);
        let pop = [cand(1, vec![3.0]), cand(2, vec![5.0]), cand(3, vec![2.0])];
        let best = best_candidate(&pop, &s).unwrap();
        assert_eq!(best.decision, 2);
    }

    #[test]
    fn best_candidate_skips_infeasible() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [
            Candidate::new(1u32, Evaluation::constrained(vec![0.0], 5.0)), // infeasible
            Candidate::new(2u32, Evaluation::new(vec![10.0])),
            Candidate::new(3u32, Evaluation::new(vec![3.0])),
        ];
        let best = best_candidate(&pop, &s).unwrap();
        assert_eq!(best.decision, 3);
    }

    #[test]
    fn best_candidate_none_when_all_infeasible() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [
            Candidate::new(1u32, Evaluation::constrained(vec![0.0], 1.0)),
            Candidate::new(2u32, Evaluation::constrained(vec![0.0], 2.0)),
        ];
        assert!(best_candidate(&pop, &s).is_none());
    }

    /// `best_candidate` keeps the *first* minimum on a tie — pins the strict
    /// `v < best_min` (a `<=` mutant would keep the last tied candidate).
    #[test]
    fn best_candidate_keeps_first_on_tie() {
        use crate::core::objective::Objective;
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [
            Candidate::new(1u32, Evaluation::new(vec![1.0])),
            Candidate::new(2u32, Evaluation::new(vec![1.0])),
        ];
        let best = best_candidate(&pop, &s).unwrap();
        assert_eq!(best.decision, 1, "should keep the first of two tied minima");
    }
}
