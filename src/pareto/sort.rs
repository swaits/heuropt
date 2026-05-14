//! Fast non-dominated sorting (Deb et al., NSGA-II).

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;

/// Partition the population into Pareto fronts by dominance rank.
///
/// `fronts[0]` is the non-dominated set, `fronts[1]` is what becomes
/// non-dominated after removing `fronts[0]`, and so on. Each entry is an index
/// into the input population. Equal-objective candidates land on the same
/// front. O(N²·M) is acceptable for v1 (spec §9.5).
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
///     Candidate::new((), Evaluation::new(vec![1.0, 5.0])), // front 0
///     Candidate::new((), Evaluation::new(vec![2.0, 3.0])), // front 0
///     Candidate::new((), Evaluation::new(vec![4.0, 1.0])), // front 0
///     Candidate::new((), Evaluation::new(vec![3.0, 4.0])), // front 1
///     Candidate::new((), Evaluation::new(vec![5.0, 6.0])), // front 2
/// ];
/// let fronts = non_dominated_sort(&pop, &s);
/// assert_eq!(fronts.len(), 3);
/// ```
pub fn non_dominated_sort<D>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
) -> Vec<Vec<usize>> {
    let n = population.len();
    if n == 0 {
        return Vec::new();
    }

    // Precompute the per-individual feasibility, violation, and
    // minimization-oriented objective vectors. The naïve formulation
    // calls `pareto_compare` (and therefore `as_minimization`) twice for
    // every pair, allocating two fresh Vec<f64>s per call; doing it once
    // up front cuts that to one allocation per individual.
    let feasible: Vec<bool> = population
        .iter()
        .map(|c| c.evaluation.is_feasible())
        .collect();
    let violation: Vec<f64> = population
        .iter()
        .map(|c| c.evaluation.constraint_violation)
        .collect();
    let oriented: Vec<Vec<f64>> = population
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();
    let m = objectives.len();

    let mut dominates: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dominated_by_count: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front: Vec<usize> = Vec::new();

    // Compare each unordered pair {i, j} exactly once. The dominance
    // relation is antisymmetric — the outcome of `compare(i, j)` fully
    // determines `compare(j, i)` — so iterating `j > i` and applying the
    // result in both directions does identical work in half the iterations.
    for i in 0..n {
        let ai_feasible = feasible[i];
        let ai_violation = violation[i];
        let ai = &oriented[i];
        for j in (i + 1)..n {
            let bi_feasible = feasible[j];
            let bi_violation = violation[j];
            // Inline the body of `pareto_compare`. We only care about
            // `Dominates` vs `DominatedBy`; `Equal` and `NonDominated`
            // are no-ops here.
            let dominates_outcome = match (ai_feasible, bi_feasible) {
                (true, false) => Some(true),  // i dominates j
                (false, true) => Some(false), // j dominates i
                (false, false) => {
                    if ai_violation < bi_violation {
                        Some(true)
                    } else if ai_violation > bi_violation {
                        Some(false)
                    } else {
                        None
                    }
                }
                (true, true) => {
                    let bj = &oriented[j];
                    let mut a_better_anywhere = false;
                    let mut b_better_anywhere = false;
                    for k in 0..m {
                        let av = ai[k];
                        let bv = bj[k];
                        if av < bv {
                            a_better_anywhere = true;
                        } else if av > bv {
                            b_better_anywhere = true;
                        }
                    }
                    match (a_better_anywhere, b_better_anywhere) {
                        (true, false) => Some(true),
                        (false, true) => Some(false),
                        _ => None,
                    }
                }
            };
            match dominates_outcome {
                Some(true) => {
                    // i dominates j
                    dominates[i].push(j);
                    dominated_by_count[j] += 1;
                }
                Some(false) => {
                    // j dominates i
                    dominates[j].push(i);
                    dominated_by_count[i] += 1;
                }
                None => {}
            }
        }
    }

    for (i, &count) in dominated_by_count.iter().enumerate() {
        if count == 0 {
            first_front.push(i);
        }
    }

    fronts.push(first_front);
    let mut k = 0;
    let mut assigned = vec![false; n];
    for &i in &fronts[0] {
        assigned[i] = true;
    }
    while k < fronts.len() && !fronts[k].is_empty() {
        let mut next: Vec<usize> = Vec::new();
        // Borrow-friendly: collect dominated indices for the current front first.
        let to_visit: Vec<usize> = fronts[k].clone();
        for i in to_visit {
            for &j in &dominates[i] {
                dominated_by_count[j] -= 1;
                if dominated_by_count[j] == 0 {
                    next.push(j);
                    assigned[j] = true;
                }
            }
        }
        if next.is_empty() {
            break;
        }
        fronts.push(next);
        k += 1;
    }

    // Any indices still unassigned correspond to dominance-graph cycles
    // (which can arise when objectives or constraint violations contain
    // NaN — `pareto_compare` becomes intransitive). Place them all in a
    // final residual front so the partition invariant holds.
    let residual: Vec<usize> = (0..n).filter(|&i| !assigned[i]).collect();
    if !residual.is_empty() {
        fronts.push(residual);
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
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    #[test]
    fn empty_population_no_fronts() {
        let s = space_min2();
        let fronts = non_dominated_sort::<()>(&[], &s);
        assert!(fronts.is_empty());
    }

    /// Regression: discovered by the `non_dominated_sort` fuzzer. NaN
    /// objectives make `pareto_compare` intransitive, which can leave a
    /// cycle in the dominance graph where no node has zero in-degree.
    /// Previously the algorithm dropped those indices silently; now they
    /// land in a final residual front so the partition invariant holds.
    #[test]
    fn nan_objective_cycle_indices_partitioned_into_residual_front() {
        let s = space_min2();
        // Three points whose pairwise comparisons form a 3-cycle under NaN
        // intransitivity (the original fuzz-found case had 5 points; this
        // 3-point case is the minimal reproduction).
        let pop = [
            cand(vec![f64::NAN, 1.0]),
            cand(vec![1.0, f64::NAN]),
            cand(vec![f64::NAN, f64::NAN]),
        ];
        let fronts = non_dominated_sort(&pop, &s);
        let mut all_indices: Vec<usize> = fronts.iter().flatten().copied().collect();
        all_indices.sort();
        assert_eq!(all_indices, vec![0, 1, 2]);
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

    /// Three mutually non-dominated points all land in front 0; a fourth
    /// point dominated by all three lands in front 1. Pins the `<` / `>`
    /// comparisons in the inline dominance check.
    #[test]
    fn three_nondominated_then_one_dominated() {
        let s = space_min2();
        let pop = [
            cand(vec![1.0, 3.0]),
            cand(vec![2.0, 2.0]),
            cand(vec![3.0, 1.0]),
            cand(vec![5.0, 5.0]), // dominated by all three
        ];
        let fronts = non_dominated_sort(&pop, &s);
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 3);
        assert_eq!(fronts[1], vec![3]);
    }

    /// A strict chain a ▷ b ▷ c produces three singleton fronts. Pins the
    /// front-peeling `while` loop and the `&&` guard at line 127.
    #[test]
    fn strict_chain_produces_three_singleton_fronts() {
        let s = space_min2();
        let pop = [
            cand(vec![1.0, 1.0]), // dominates everything
            cand(vec![2.0, 2.0]),
            cand(vec![3.0, 3.0]),
        ];
        let fronts = non_dominated_sort(&pop, &s);
        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0], vec![0]);
        assert_eq!(fronts[1], vec![1]);
        assert_eq!(fronts[2], vec![2]);
    }
}
