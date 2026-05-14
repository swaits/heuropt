//! A concrete Pareto archive that maintains an approximate non-dominated set.

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;

/// A growable, dominance-pruned archive of candidates.
///
/// Built around a single concrete struct rather than a trait (spec §13). The
/// archive insert/extend operations maintain the non-domination property among
/// members; `truncate` enforces a maximum size by simple tail-truncation in
/// v1.
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
/// let mut a: ParetoArchive<u32> = ParetoArchive::new(s);
/// a.insert(Candidate::new(1, Evaluation::new(vec![1.0, 4.0])));
/// a.insert(Candidate::new(2, Evaluation::new(vec![3.0, 2.0])));
/// // Dominated by both — should be discarded:
/// a.insert(Candidate::new(3, Evaluation::new(vec![5.0, 5.0])));
/// assert_eq!(a.members().len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ParetoArchive<D> {
    /// The current approximate non-dominated set.
    pub members: Vec<Candidate<D>>,
    /// The objective space used for dominance comparisons.
    pub objectives: ObjectiveSpace,
}

impl<D: Clone> ParetoArchive<D> {
    /// Build an empty archive against the given objective space.
    pub fn new(objectives: ObjectiveSpace) -> Self {
        Self {
            members: Vec::new(),
            objectives,
        }
    }

    /// Insert a candidate, preserving the non-domination property.
    ///
    /// - If any existing member dominates the new candidate, discard it.
    /// - Otherwise, drop existing members that the new candidate dominates,
    ///   then keep the new candidate.
    pub fn insert(&mut self, candidate: Candidate<D>) {
        // The naïve formulation calls `pareto_compare` twice per member
        // (once each pass), and `pareto_compare` re-allocates two
        // Vec<f64>s via `as_minimization` per call → 4N allocations per
        // insert. Cache the candidate's oriented + feasibility once, and
        // each member's oriented once, then inline the dominance checks.
        let n = self.members.len();
        let m_dim = self.objectives.len();
        let cand_oriented = self
            .objectives
            .as_minimization(&candidate.evaluation.objectives);
        let cand_feasible = candidate.evaluation.is_feasible();
        let cand_violation = candidate.evaluation.constraint_violation;
        let member_oriented: Vec<Vec<f64>> = self
            .members
            .iter()
            .map(|c| self.objectives.as_minimization(&c.evaluation.objectives))
            .collect();

        // First pass: bail if any existing member dominates-or-equals
        // the candidate.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let m_eval = &self.members[i].evaluation;
            if member_dominates_or_equals(
                &member_oriented[i],
                m_eval.is_feasible(),
                m_eval.constraint_violation,
                &cand_oriented,
                cand_feasible,
                cand_violation,
                m_dim,
            ) {
                return;
            }
        }

        // Second pass: drop existing members the candidate dominates.
        let mut keep_mask = Vec::with_capacity(n);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let m_eval = &self.members[i].evaluation;
            let cand_dominates_member = candidate_dominates_member(
                &cand_oriented,
                cand_feasible,
                cand_violation,
                &member_oriented[i],
                m_eval.is_feasible(),
                m_eval.constraint_violation,
                m_dim,
            );
            keep_mask.push(!cand_dominates_member);
        }
        let mut idx = 0;
        self.members.retain(|_| {
            let keep = keep_mask[idx];
            idx += 1;
            keep
        });
        self.members.push(candidate);
    }

    /// Insert each candidate from `candidates`.
    pub fn extend<I>(&mut self, candidates: I)
    where
        I: IntoIterator<Item = Candidate<D>>,
    {
        for c in candidates {
            self.insert(c);
        }
    }

    /// Truncate the archive to at most `max_size` members.
    ///
    /// In v1 this is simple tail-truncation; future versions may use crowding
    /// distance to preferentially keep diverse members.
    pub fn truncate(&mut self, max_size: usize) {
        if self.members.len() > max_size {
            self.members.truncate(max_size);
        }
    }

    /// View the current members.
    pub fn members(&self) -> &[Candidate<D>] {
        &self.members
    }

    /// Consume the archive, returning the members.
    pub fn into_vec(self) -> Vec<Candidate<D>> {
        self.members
    }
}

/// Inline `pareto_compare(member, candidate, objectives) ∈ {Dominates, Equal}`
/// against the cached oriented + feasibility/violation values, returning the
/// boolean directly.
#[inline]
fn member_dominates_or_equals(
    m_oriented: &[f64],
    m_feasible: bool,
    m_violation: f64,
    c_oriented: &[f64],
    c_feasible: bool,
    c_violation: f64,
    m_dim: usize,
) -> bool {
    match (m_feasible, c_feasible) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => m_violation <= c_violation,
        (true, true) => {
            let mut c_better = false;
            for k in 0..m_dim {
                if c_oriented[k] < m_oriented[k] {
                    c_better = true;
                    break;
                }
            }
            !c_better
        }
    }
}

/// Inline `pareto_compare(candidate, member, objectives) == Dominates`.
#[inline]
fn candidate_dominates_member(
    c_oriented: &[f64],
    c_feasible: bool,
    c_violation: f64,
    m_oriented: &[f64],
    m_feasible: bool,
    m_violation: f64,
    m_dim: usize,
) -> bool {
    match (c_feasible, m_feasible) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => c_violation < m_violation,
        (true, true) => {
            let mut c_better_anywhere = false;
            let mut m_better_anywhere = false;
            for k in 0..m_dim {
                let cv = c_oriented[k];
                let mv = m_oriented[k];
                if cv < mv {
                    c_better_anywhere = true;
                } else if cv > mv {
                    m_better_anywhere = true;
                }
            }
            c_better_anywhere && !m_better_anywhere
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::Objective;

    fn space_min2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn cand(decision: u32, obj: Vec<f64>) -> Candidate<u32> {
        Candidate::new(decision, Evaluation::new(obj))
    }

    #[test]
    fn dominated_insertion_is_rejected() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        a.insert(cand(1, vec![1.0, 1.0]));
        a.insert(cand(2, vec![2.0, 2.0])); // dominated, discarded
        assert_eq!(a.members().len(), 1);
        assert_eq!(a.members()[0].decision, 1);
    }

    #[test]
    fn dominating_insertion_evicts_existing() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        a.insert(cand(1, vec![3.0, 3.0]));
        a.insert(cand(2, vec![5.0, 0.0])); // non-dominated with 1, kept
        a.insert(cand(3, vec![1.0, 1.0])); // dominates 1, non-dom with 2 (worse f2)
        let dec: Vec<u32> = a.members().iter().map(|c| c.decision).collect();
        assert!(dec.contains(&3));
        assert!(dec.contains(&2));
        assert!(!dec.contains(&1));
    }

    #[test]
    fn equal_candidate_is_treated_as_dominated() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        a.insert(cand(1, vec![1.0, 1.0]));
        a.insert(cand(2, vec![1.0, 1.0])); // equal, treated as already covered
        assert_eq!(a.members().len(), 1);
    }

    #[test]
    fn truncate_simple_tail() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        // Mutually non-dominated front of 4 points along a trade-off curve.
        a.insert(cand(1, vec![0.0, 4.0]));
        a.insert(cand(2, vec![1.0, 3.0]));
        a.insert(cand(3, vec![2.0, 2.0]));
        a.insert(cand(4, vec![3.0, 1.0]));
        assert_eq!(a.members().len(), 4);
        a.truncate(2);
        assert_eq!(a.members().len(), 2);
    }

    #[test]
    fn extend_accepts_iterator() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        a.extend(vec![cand(1, vec![1.0, 4.0]), cand(2, vec![3.0, 2.0])]);
        assert_eq!(a.members().len(), 2);
    }

    /// `truncate` keeps the archive untouched when it is already at or
    /// below `max_size`, and trims it when over. Pins the `>` boundary.
    #[test]
    fn truncate_boundary_behavior() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        // Three mutually non-dominated members.
        a.insert(cand(1, vec![1.0, 3.0]));
        a.insert(cand(2, vec![2.0, 2.0]));
        a.insert(cand(3, vec![3.0, 1.0]));
        assert_eq!(a.members().len(), 3);
        // max_size == len → no-op (kills `>` → `>=`).
        a.truncate(3);
        assert_eq!(a.members().len(), 3);
        // max_size > len → no-op.
        a.truncate(10);
        assert_eq!(a.members().len(), 3);
        // max_size < len → trims.
        a.truncate(2);
        assert_eq!(a.members().len(), 2);
    }

    /// A trade-off candidate (better on one axis, worse on the other) is
    /// neither dominated nor dominating — it must be *added* alongside the
    /// existing member. Pins the per-axis `<` / `>` scan in both
    /// `member_dominates_or_equals` and `candidate_dominates_member`.
    #[test]
    fn trade_off_candidate_is_kept_alongside() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        a.insert(cand(1, vec![1.0, 5.0]));
        a.insert(cand(2, vec![5.0, 1.0])); // trade-off — must be kept
        assert_eq!(a.members().len(), 2);
    }

    /// An equal-objectives candidate is rejected (a member dominates-or-
    /// equals it). Pins the Equal branch — distinguishes `<=` from `<` in
    /// `candidate_dominates_member` and the `<=` in
    /// `member_dominates_or_equals`'s infeasible branch.
    #[test]
    fn equal_candidate_is_rejected() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        a.insert(cand(1, vec![2.0, 2.0]));
        a.insert(cand(2, vec![2.0, 2.0])); // identical objectives → rejected
        assert_eq!(a.members().len(), 1);
        assert_eq!(a.members()[0].decision, 1);
    }

    /// Two infeasible candidates: the one with smaller constraint violation
    /// wins. Pins the `<` / `<=` in the infeasible branches.
    #[test]
    fn infeasible_candidate_with_smaller_violation_evicts_larger() {
        let mut a = ParetoArchive::<u32>::new(space_min2());
        a.insert(Candidate::new(
            1u32,
            Evaluation::constrained(vec![0.0, 0.0], 1.0),
        ));
        // Smaller violation → dominates the existing infeasible member.
        a.insert(Candidate::new(
            2u32,
            Evaluation::constrained(vec![9.0, 9.0], 0.5),
        ));
        assert_eq!(a.members().len(), 1);
        assert_eq!(a.members()[0].decision, 2);
    }
}
