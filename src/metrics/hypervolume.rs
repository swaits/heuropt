//! Exact 2D and N-D hypervolume against a fixed reference point.

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::ObjectiveSpace;

/// Compute the dominated hypervolume of a 2D front against `reference_point`.
///
/// Both `reference_point` coordinates are interpreted in the same
/// minimization-oriented frame as `objectives.as_minimization`. The reference
/// point should be worse than every point you intend to count; points that do
/// not strictly dominate the reference along both axes are silently skipped
/// (spec §14.2).
///
/// # Panics
/// If `objectives` does not have exactly two objectives.
pub fn hypervolume_2d<D>(
    front: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    reference_point: [f64; 2],
) -> f64 {
    assert_eq!(
        objectives.len(),
        2,
        "hypervolume_2d requires exactly 2 objectives",
    );
    if front.is_empty() {
        return 0.0;
    }

    let mut points: Vec<[f64; 2]> = front
        .iter()
        .filter_map(|c| {
            let m = objectives.as_minimization(&c.evaluation.objectives);
            let p = [m[0], m[1]];
            if p[0] < reference_point[0] && p[1] < reference_point[1] {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    if points.is_empty() {
        return 0.0;
    }
    points.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut area = 0.0;
    let mut last_y = reference_point[1];
    for p in &points {
        if p[1] >= last_y {
            // Dominated by an already-counted point on the second axis: skip.
            continue;
        }
        let width = reference_point[0] - p[0];
        let height = last_y - p[1];
        area += width * height;
        last_y = p[1];
    }
    area
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
    fn known_three_point_front_area() {
        // Reference (4, 4); front at (1,3), (2,2), (3,1).
        // Dominated region area = 4*4 - sum of "outside" rectangles
        //   stripes: x∈[1,2] y∈[3,4]→1, x∈[2,3] y∈[2,4]→2, x∈[3,4] y∈[1,4]→3 → total dominated = 1+2+3 = 6.
        let s = space_min2();
        let front = [
            cand(vec![1.0, 3.0]),
            cand(vec![2.0, 2.0]),
            cand(vec![3.0, 1.0]),
        ];
        let hv = hypervolume_2d(&front, &s, [4.0, 4.0]);
        assert!((hv - 6.0).abs() < 1e-12, "expected 6.0, got {hv}");
    }

    #[test]
    fn empty_front_is_zero() {
        let s = space_min2();
        let front: [Candidate<()>; 0] = [];
        assert_eq!(hypervolume_2d(&front, &s, [10.0, 10.0]), 0.0);
    }

    #[test]
    fn point_not_dominating_reference_skipped() {
        let s = space_min2();
        // Reference at (1, 1); the front point (2, 0.5) does not dominate the
        // reference along axis 0 → contributes nothing.
        let front = [cand(vec![2.0, 0.5])];
        assert_eq!(hypervolume_2d(&front, &s, [1.0, 1.0]), 0.0);
    }

    #[test]
    fn maximize_axis_handled_via_orientation() {
        // Maximize axis flips sign; reference must be in the same oriented
        // frame. With maximize on axis 1, raw value 0.9 becomes -0.9 and the
        // reference 0.0 must be passed as 0.0 (worse than -0.9).
        let s = ObjectiveSpace::new(vec![
            Objective::minimize("cost"),
            Objective::maximize("score"),
        ]);
        let front = [cand(vec![1.0, 0.9])];
        let hv = hypervolume_2d(&front, &s, [2.0, 0.0]);
        // width = 2.0 - 1.0 = 1.0; height = 0.0 - (-0.9) = 0.9 → 0.9
        assert!((hv - 0.9).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "exactly 2 objectives")]
    fn panics_on_non_2d() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("only")]);
        let front = [cand(vec![1.0])];
        let _ = hypervolume_2d(&front, &s, [10.0, 10.0]);
    }
}

/// Compute the dominated hypervolume in arbitrary dimensions using the
/// **Hypervolume-by-Slicing-Objectives (HSO)** algorithm of While et al. 2006.
///
/// `objectives.len()` must equal `reference_point.len()`. Like
/// [`hypervolume_2d`], the reference point is interpreted in the same
/// minimization-oriented frame as `ObjectiveSpace::as_minimization`, and
/// points that don't strictly dominate the reference are silently skipped.
///
/// For 2-D problems prefer [`hypervolume_2d`] (it has the same exact result
/// but a tighter sweep loop). This function calls [`hypervolume_2d`]
/// internally as the recursion base case.
///
/// Worst-case complexity is O((N · M)!) which sounds awful but in practice
/// HSO is competitive with WFG up through ~5 objectives at population sizes
/// of 100–200 — i.e. exactly the regime heuropt targets.
///
/// # Panics
/// If `objectives.len() != reference_point.len()`, or if either is zero.
pub fn hypervolume_nd<D>(
    front: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    reference_point: &[f64],
) -> f64 {
    assert_eq!(
        objectives.len(),
        reference_point.len(),
        "hypervolume_nd: ObjectiveSpace and reference_point must agree on dimension",
    );
    assert!(
        !reference_point.is_empty(),
        "hypervolume_nd: dimension must be >= 1"
    );

    if front.is_empty() {
        return 0.0;
    }

    // Project each point into minimization-oriented space, then keep only
    // points that strictly dominate the reference along every axis.
    let oriented: Vec<Vec<f64>> = front
        .iter()
        .filter_map(|c| {
            let m = objectives.as_minimization(&c.evaluation.objectives);
            if m.iter().zip(reference_point.iter()).all(|(p, r)| p < r) {
                Some(m)
            } else {
                None
            }
        })
        .collect();
    if oriented.is_empty() {
        return 0.0;
    }

    hso_recursive(&oriented, reference_point)
}

fn hso_recursive(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    let m = reference.len();
    if m == 1 {
        // 1-D HV: distance from the best (minimum) point to the reference.
        let best = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        return (reference[0] - best).max(0.0);
    }
    if m == 2 {
        // 2-D HV via the same sweep used by hypervolume_2d. Inlined here
        // because we already have the points in oriented form.
        let mut sorted: Vec<&Vec<f64>> = points.iter().collect();
        sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
        let mut area = 0.0;
        let mut last_y = reference[1];
        for p in sorted {
            if p[1] >= last_y {
                continue;
            }
            let width = reference[0] - p[0];
            let height = last_y - p[1];
            area += width * height;
            last_y = p[1];
        }
        return area;
    }

    // M ≥ 3: sweep along the last axis from the reference downward,
    // peeling off bands. At each band:
    //   - the active set is "all points whose last-axis value ≤ band_top";
    //   - its (M-1)-dim HV (on the first M-1 axes against the
    //     corresponding sub-reference), multiplied by band thickness, is
    //     the band's HV contribution.
    //
    // We sort points ascending by the last axis once, then iterate from
    // the largest last-axis value downward. The active set at iteration
    // `k` is exactly the prefix `sorted[..=k]` — no allocations or
    // linear-scan removals needed.
    let last = m - 1;
    let mut sorted: Vec<Vec<f64>> = points.to_vec();
    sorted.sort_by(|a, b| {
        a[last]
            .partial_cmp(&b[last])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Pre-project all points onto the first M-1 axes once. The active
    // set at each band is the prefix `projected_all[..=k]`; we slice
    // that prefix instead of rebuilding it per band.
    let projected_all: Vec<Vec<f64>> = sorted.iter().map(|q| q[..last].to_vec()).collect();
    let sub_reference: &[f64] = &reference[..last];
    let mut total = 0.0;
    let mut prev = reference[last];
    for k in (0..sorted.len()).rev() {
        let p = &sorted[k];
        let depth = prev - p[last];
        if depth > 0.0 {
            let active = &projected_all[..=k];
            // The 2-D base case sweeps in sorted-x order and skips any
            // point with `y >= last_y`, which is exactly the dominance
            // filter — so for M=3 (sub_reference len 2) we can hand
            // `active` straight to `hso_recursive` without paying for
            // an O(K²) `non_dominated_projection` first. For M≥4 we
            // still need the explicit filter to keep the recursion's
            // upper levels honest.
            let inner = if sub_reference.len() == 2 {
                hso_recursive(active, sub_reference)
            } else {
                let nd = non_dominated_projection(active);
                hso_recursive(&nd, sub_reference)
            };
            total += depth * inner;
        }
        prev = p[last];
    }

    total
}

/// Drop dominated members of a projected point set.
fn non_dominated_projection(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = if let Some(first) = points.first() {
        first.len()
    } else {
        return Vec::new();
    };
    let mut out: Vec<Vec<f64>> = Vec::new();
    'outer: for p in points {
        // Skip if dominated by any kept point.
        for q in &out {
            if dominates(q, p, m) {
                continue 'outer;
            }
        }
        // Drop already-kept points that this one dominates.
        out.retain(|q| !dominates(p, q, m));
        out.push(p.clone());
    }
    out
}

fn dominates(a: &[f64], b: &[f64], m: usize) -> bool {
    let mut strictly_better = false;
    for i in 0..m {
        if a[i] > b[i] {
            return false;
        }
        if a[i] < b[i] {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Convenience wrapper that takes raw `Evaluation`s. Useful inside SMS-EMOA
/// where we want to compute "front HV minus point's contribution."
pub(crate) fn hypervolume_nd_from_evaluations(
    evaluations: &[&Evaluation],
    objectives: &ObjectiveSpace,
    reference_point: &[f64],
) -> f64 {
    if evaluations.is_empty() {
        return 0.0;
    }
    let oriented: Vec<Vec<f64>> = evaluations
        .iter()
        .filter_map(|e| {
            let m = objectives.as_minimization(&e.objectives);
            if m.iter().zip(reference_point.iter()).all(|(p, r)| p < r) {
                Some(m)
            } else {
                None
            }
        })
        .collect();
    if oriented.is_empty() {
        return 0.0;
    }
    hso_recursive(&oriented, reference_point)
}

#[cfg(test)]
mod nd_tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::Objective;

    fn cand_n(obj: Vec<f64>) -> Candidate<()> {
        Candidate::new((), Evaluation::new(obj))
    }

    #[test]
    fn nd_matches_2d_on_known_case() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
        let front = [
            cand_n(vec![1.0, 3.0]),
            cand_n(vec![2.0, 2.0]),
            cand_n(vec![3.0, 1.0]),
        ];
        let hv2 = hypervolume_2d(&front, &s, [4.0, 4.0]);
        let hvn = hypervolume_nd(&front, &s, &[4.0, 4.0]);
        assert!((hv2 - hvn).abs() < 1e-12, "{hv2} vs {hvn}");
        assert!((hvn - 6.0).abs() < 1e-12);
    }

    #[test]
    fn nd_three_d_single_point_at_origin() {
        let s = ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
            Objective::minimize("f3"),
        ]);
        let front = [cand_n(vec![0.0, 0.0, 0.0])];
        // Reference at (1, 1, 1): one point fully dominates the cube
        // → HV = 1·1·1 = 1.
        let hv = hypervolume_nd(&front, &s, &[1.0, 1.0, 1.0]);
        assert!((hv - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nd_three_d_two_points_no_overlap() {
        let s = ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
            Objective::minimize("f3"),
        ]);
        // Reference (2, 2, 2). Two non-dominated points, projecting cleanly:
        //   p1 = (0, 1, 1)  → contributes a 2 × 1 × 1 = 2 box
        //   p2 = (1, 0, 1)  → contributes 1 × 2 × 1 = 2 minus the overlap with p1
        //                     overlap (where x<=1 AND y<=1 AND z<=1) is 1·1·1 = 1
        //   p3 = (1, 1, 0)  → ... and so on
        // Manual computation is annoying; instead verify monotonicity:
        // adding more non-dominated points must strictly increase HV.
        let front_one = [cand_n(vec![0.0, 1.0, 1.0])];
        let front_two = [cand_n(vec![0.0, 1.0, 1.0]), cand_n(vec![1.0, 0.0, 1.0])];
        let front_three = [
            cand_n(vec![0.0, 1.0, 1.0]),
            cand_n(vec![1.0, 0.0, 1.0]),
            cand_n(vec![1.0, 1.0, 0.0]),
        ];
        let hv1 = hypervolume_nd(&front_one, &s, &[2.0, 2.0, 2.0]);
        let hv2 = hypervolume_nd(&front_two, &s, &[2.0, 2.0, 2.0]);
        let hv3 = hypervolume_nd(&front_three, &s, &[2.0, 2.0, 2.0]);
        assert!(hv1 < hv2, "{hv1} should be < {hv2}");
        assert!(hv2 < hv3, "{hv2} should be < {hv3}");
        // Sanity bound: each point is a (2,2,2)-box minus an L-shape;
        // total can't exceed the box volume of 8.
        assert!(hv3 < 8.0);
    }

    #[test]
    fn nd_empty_is_zero() {
        let s = ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
            Objective::minimize("f3"),
        ]);
        let front: [Candidate<()>; 0] = [];
        assert_eq!(hypervolume_nd(&front, &s, &[1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn nd_skips_points_not_dominating_reference() {
        let s = ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
            Objective::minimize("f3"),
        ]);
        // (3, 0, 0) is not dominated by reference (1, 1, 1) on axis 0.
        let front = [cand_n(vec![3.0, 0.0, 0.0])];
        assert_eq!(hypervolume_nd(&front, &s, &[1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    #[should_panic(expected = "must agree on dimension")]
    fn nd_panics_on_dim_mismatch() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
        let front = [cand_n(vec![1.0, 1.0])];
        let _ = hypervolume_nd(&front, &s, &[1.0, 1.0, 1.0]);
    }

    /// Sanity test: dominated points shouldn't increase HV.
    #[test]
    fn nd_dominated_points_dont_increase_hv() {
        let s = ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
            Objective::minimize("f3"),
        ]);
        let base = vec![cand_n(vec![0.0, 1.0, 1.0]), cand_n(vec![1.0, 0.0, 1.0])];
        // Add a dominated point — HV should be unchanged.
        let mut with_dominated = base.clone();
        with_dominated.push(cand_n(vec![1.5, 1.5, 1.5]));
        let hv_base = hypervolume_nd(&base, &s, &[2.0, 2.0, 2.0]);
        let hv_with = hypervolume_nd(&with_dominated, &s, &[2.0, 2.0, 2.0]);
        assert!((hv_base - hv_with).abs() < 1e-12, "{hv_base} vs {hv_with}");
    }
}
