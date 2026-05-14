//! Crowding distance for diversity preservation in NSGA-II-style algorithms.

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;

/// Compute crowding distance for the given front (a slice of indices into the
/// population).
///
/// Returns a `Vec<f64>` aligned with `front` (so `result[i]` is the crowding
/// distance of `population[front[i]]`). Boundary points receive
/// `f64::INFINITY`. If the front has 0 entries an empty vector is returned;
/// 1 or 2 entries return all `f64::INFINITY`. All comparisons happen on
/// minimization-oriented objective values (spec §9.6).
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
/// // Three points along a Pareto-like trade-off; the interior point gets
/// // a finite crowding distance, the boundaries get +∞.
/// let pop = [
///     Candidate::new((), Evaluation::new(vec![0.0, 4.0])),
///     Candidate::new((), Evaluation::new(vec![2.0, 2.0])),
///     Candidate::new((), Evaluation::new(vec![4.0, 0.0])),
/// ];
/// let d = crowding_distance(&pop, &[0, 1, 2], &s);
/// assert!(d[0].is_infinite());
/// assert!(d[1].is_finite() && d[1] > 0.0);
/// assert!(d[2].is_infinite());
/// ```
pub fn crowding_distance<D>(
    population: &[Candidate<D>],
    front: &[usize],
    objectives: &ObjectiveSpace,
) -> Vec<f64> {
    let n = front.len();
    if n == 0 {
        return Vec::new();
    }
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let m = objectives.len();
    let mut distance = vec![0.0_f64; n];

    // Cache minimization-oriented objective values for each front member.
    let oriented: Vec<Vec<f64>> = front
        .iter()
        .map(|&idx| objectives.as_minimization(&population[idx].evaluation.objectives))
        .collect();

    #[allow(clippy::needless_range_loop)] // `k` indexes into nested vectors below.
    for k in 0..m {
        // Sort indices into `front` by objective k.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            oriented[a][k]
                .partial_cmp(&oriented[b][k])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        distance[order[0]] = f64::INFINITY;
        distance[order[n - 1]] = f64::INFINITY;

        let f_min = oriented[order[0]][k];
        let f_max = oriented[order[n - 1]][k];
        let span = f_max - f_min;
        if span == 0.0 {
            continue;
        }

        for i in 1..n - 1 {
            if distance[order[i]] == f64::INFINITY {
                continue;
            }
            let prev = oriented[order[i - 1]][k];
            let next = oriented[order[i + 1]][k];
            distance[order[i]] += (next - prev) / span;
        }
    }

    distance
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
    fn empty_front_returns_empty_vec() {
        let s = space_min2();
        let pop: [Candidate<()>; 0] = [];
        let d = crowding_distance(&pop, &[], &s);
        assert!(d.is_empty());
    }

    #[test]
    fn single_point_is_infinity() {
        let s = space_min2();
        let pop = [cand(vec![1.0, 2.0])];
        let d = crowding_distance(&pop, &[0], &s);
        assert_eq!(d, vec![f64::INFINITY]);
    }

    #[test]
    fn two_points_both_infinity() {
        let s = space_min2();
        let pop = [cand(vec![1.0, 2.0]), cand(vec![2.0, 1.0])];
        let d = crowding_distance(&pop, &[0, 1], &s);
        assert_eq!(d, vec![f64::INFINITY, f64::INFINITY]);
    }

    #[test]
    fn boundary_infinity_interior_finite() {
        let s = space_min2();
        // Three non-dominated points along a Pareto-like trade-off:
        //   0: (0, 4)
        //   1: (2, 2)  ← interior on both axes
        //   2: (4, 0)
        let pop = [
            cand(vec![0.0, 4.0]),
            cand(vec![2.0, 2.0]),
            cand(vec![4.0, 0.0]),
        ];
        let d = crowding_distance(&pop, &[0, 1, 2], &s);
        assert!(d[0].is_infinite());
        assert!(d[2].is_infinite());
        assert!(d[1].is_finite());
        assert!(d[1] > 0.0);
    }

    #[test]
    fn equal_objective_axis_does_not_panic() {
        // All points share the same f2 value; the f2 axis contributes zero,
        // and the f1 axis still produces sensible boundary infinities.
        let s = space_min2();
        let pop = [
            cand(vec![0.0, 1.0]),
            cand(vec![1.0, 1.0]),
            cand(vec![2.0, 1.0]),
        ];
        let d = crowding_distance(&pop, &[0, 1, 2], &s);
        assert!(d[0].is_infinite());
        assert!(d[2].is_infinite());
        assert!(d[1].is_finite());
    }

    /// Crowding distance pins the exact interior contribution: for a 3-point
    /// 2-objective front, the middle point's distance is the sum over both
    /// objectives of (next - prev) / span. With evenly-spaced points the
    /// value is exactly 2.0 (1.0 per objective).
    #[test]
    fn interior_point_distance_is_pinned() {
        let s = space_min2();
        // Front along the line f1 + f2 = 4: (0,4), (2,2), (4,0).
        let pop = [cand(vec![0.0, 4.0]), cand(vec![2.0, 2.0]), cand(vec![4.0, 0.0])];
        let d = crowding_distance(&pop, &[0, 1, 2], &s);
        // Boundary points are infinite; the middle point gets
        // (4-0)/4 + (4-0)/4 = 2.0 (objective 0 span 4, objective 1 span 4).
        assert!(d[0].is_infinite());
        assert!(d[2].is_infinite());
        assert!((d[1] - 2.0).abs() < 1e-12, "interior distance = {}", d[1]);
    }

    /// An asymmetric front pins the per-objective `(next - prev) / span`
    /// arithmetic: catches the `-` ↔ `+`/`/` and `/` ↔ `*` mutants.
    #[test]
    fn asymmetric_interior_distance_is_pinned() {
        let s = space_min2();
        // (0,10), (1,2), (10,0): objective-0 span = 10, objective-1 span = 10.
        let pop = [cand(vec![0.0, 10.0]), cand(vec![1.0, 2.0]), cand(vec![10.0, 0.0])];
        let d = crowding_distance(&pop, &[0, 1, 2], &s);
        // middle point: obj0 (10-0)/10 = 1.0; obj1 (10-0)/10 = 1.0 → 2.0.
        assert!((d[1] - 2.0).abs() < 1e-12, "got {}", d[1]);
    }
}
