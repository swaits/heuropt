//! Fast non-dominated sorting (Deb et al., NSGA-II).

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::pareto::dominance::{Dominance, pareto_compare};

/// Partition the population into Pareto fronts by dominance rank.
///
/// `fronts[0]` is the non-dominated set, `fronts[1]` is what becomes
/// non-dominated after removing `fronts[0]`, and so on. Each entry is an index
/// into the input population. Equal-objective candidates land on the same
/// front. O(N²·M) is acceptable for v1 (spec §9.5).
pub fn non_dominated_sort<D>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
) -> Vec<Vec<usize>> {
    let n = population.len();
    if n == 0 {
        return Vec::new();
    }

    let mut dominates: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dominated_by_count: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front: Vec<usize> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            match pareto_compare(
                &population[i].evaluation,
                &population[j].evaluation,
                objectives,
            ) {
                Dominance::Dominates => dominates[i].push(j),
                Dominance::DominatedBy => dominated_by_count[i] += 1,
                _ => {}
            }
        }
        if dominated_by_count[i] == 0 {
            first_front.push(i);
        }
    }

    fronts.push(first_front);
    let mut k = 0;
    while k < fronts.len() && !fronts[k].is_empty() {
        let mut next: Vec<usize> = Vec::new();
        // Borrow-friendly: collect dominated indices for the current front first.
        let to_visit: Vec<usize> = fronts[k].clone();
        for i in to_visit {
            for &j in &dominates[i] {
                dominated_by_count[j] -= 1;
                if dominated_by_count[j] == 0 {
                    next.push(j);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        fronts.push(next);
        k += 1;
    }

    fronts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::Objective;

    fn cand(obj: Vec<f64>) -> Candidate<()> {
        Candidate::new((), Evaluation::new(obj))
    }

    fn space_min2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
        ])
    }

    #[test]
    fn empty_population_no_fronts() {
        let s = space_min2();
        let fronts = non_dominated_sort::<()>(&[], &s);
        assert!(fronts.is_empty());
    }

    #[test]
    fn known_population_yields_expected_fronts() {
        let s = space_min2();
        // Indices 0..4 deliberately mix layers:
        //   0: (1, 5)  ← front 0
        //   1: (2, 3)  ← front 0
        //   2: (4, 1)  ← front 0
        //   3: (3, 4)  ← front 1 (dominated by 1)
        //   4: (5, 6)  ← front 2 (dominated by 1, 2, 3)
        let pop = [
            cand(vec![1.0, 5.0]),
            cand(vec![2.0, 3.0]),
            cand(vec![4.0, 1.0]),
            cand(vec![3.0, 4.0]),
            cand(vec![5.0, 6.0]),
        ];
        let fronts = non_dominated_sort(&pop, &s);
        assert_eq!(fronts.len(), 3);
        let mut f0 = fronts[0].clone();
        let mut f1 = fronts[1].clone();
        let mut f2 = fronts[2].clone();
        f0.sort();
        f1.sort();
        f2.sort();
        assert_eq!(f0, vec![0, 1, 2]);
        assert_eq!(f1, vec![3]);
        assert_eq!(f2, vec![4]);
    }
}
