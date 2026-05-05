//! Exact 2D hypervolume against a fixed reference point.

use crate::core::candidate::Candidate;
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
        ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
        ])
    }

    #[test]
    fn known_three_point_front_area() {
        // Reference (4, 4); front at (1,3), (2,2), (3,1).
        // Dominated region area = 4*4 - sum of "outside" rectangles
        //   stripes: x∈[1,2] y∈[3,4]→1, x∈[2,3] y∈[2,4]→2, x∈[3,4] y∈[1,4]→3 → total dominated = 1+2+3 = 6.
        let s = space_min2();
        let front = [cand(vec![1.0, 3.0]), cand(vec![2.0, 2.0]), cand(vec![3.0, 1.0])];
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
