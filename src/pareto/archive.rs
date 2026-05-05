//! A concrete Pareto archive that maintains an approximate non-dominated set.

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::pareto::dominance::{Dominance, pareto_compare};

/// A growable, dominance-pruned archive of candidates.
///
/// Built around a single concrete struct rather than a trait (spec §13). The
/// archive insert/extend operations maintain the non-domination property among
/// members; `truncate` enforces a maximum size by simple tail-truncation in
/// v1.
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
        Self { members: Vec::new(), objectives }
    }

    /// Insert a candidate, preserving the non-domination property.
    ///
    /// - If any existing member dominates the new candidate, discard it.
    /// - Otherwise, drop existing members that the new candidate dominates,
    ///   then keep the new candidate.
    pub fn insert(&mut self, candidate: Candidate<D>) {
        for m in &self.members {
            if matches!(
                pareto_compare(&candidate.evaluation, &m.evaluation, &self.objectives),
                Dominance::DominatedBy | Dominance::Equal
            ) {
                return;
            }
        }
        self.members.retain(|m| {
            !matches!(
                pareto_compare(&candidate.evaluation, &m.evaluation, &self.objectives),
                Dominance::Dominates
            )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::Objective;

    fn space_min2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
        ])
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
}
