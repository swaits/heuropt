//! Structured reference-point generators on the unit simplex.

/// Das–Dennis reference points: all compositions of `divisions` into
/// `num_objectives` non-negative integer parts, divided by `divisions`.
///
/// Returns `binomial(divisions + num_objectives - 1, num_objectives - 1)`
/// points, each a `Vec<f64>` of length `num_objectives` summing to `1.0`.
///
/// This is the canonical NSGA-III / MOEA/D weight-vector generator.
///
/// # Panics
/// If `num_objectives == 0`.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// // 3 objectives, 4 divisions → binomial(6, 2) = 15 points.
/// let pts = das_dennis(3, 4);
/// assert_eq!(pts.len(), 15);
/// for w in &pts {
///     assert_eq!(w.len(), 3);
///     let sum: f64 = w.iter().sum();
///     assert!((sum - 1.0).abs() < 1e-12);
/// }
/// ```
pub fn das_dennis(num_objectives: usize, divisions: usize) -> Vec<Vec<f64>> {
    assert!(
        num_objectives > 0,
        "das_dennis requires num_objectives >= 1"
    );
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(num_objectives);
    recurse(num_objectives, divisions, divisions, &mut current, &mut out);
    out
}

fn recurse(
    remaining_axes: usize,
    remaining_units: usize,
    total: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<f64>>,
) {
    if remaining_axes == 1 {
        current.push(remaining_units);
        let scale = (total as f64).max(1.0);
        out.push(current.iter().map(|&v| v as f64 / scale).collect());
        current.pop();
        return;
    }
    for take in 0..=remaining_units {
        current.push(take);
        recurse(
            remaining_axes - 1,
            remaining_units - take,
            total,
            current,
            out,
        );
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn two_objective_four_divisions() {
        let pts = das_dennis(2, 4);
        // Expected: (0,4),(1,3),(2,2),(3,1),(4,0) → /4 → 5 points.
        assert_eq!(pts.len(), 5);
        for p in &pts {
            assert_eq!(p.len(), 2);
            assert!(approx_eq(p[0] + p[1], 1.0));
        }
        let first = &pts[0];
        let last = &pts[pts.len() - 1];
        assert!(approx_eq(first[0], 0.0) && approx_eq(first[1], 1.0));
        assert!(approx_eq(last[0], 1.0) && approx_eq(last[1], 0.0));
    }

    #[test]
    fn three_objective_twelve_divisions_has_91_points() {
        // C(12+3-1, 3-1) = C(14, 2) = 91 — the canonical NSGA-III 3-obj set.
        let pts = das_dennis(3, 12);
        assert_eq!(pts.len(), 91);
        for p in &pts {
            assert_eq!(p.len(), 3);
            assert!(approx_eq(p.iter().sum::<f64>(), 1.0));
        }
    }

    #[test]
    fn five_objective_six_divisions_has_210_points() {
        // C(6+5-1, 5-1) = C(10, 4) = 210.
        let pts = das_dennis(5, 6);
        assert_eq!(pts.len(), 210);
        for p in &pts {
            assert!(approx_eq(p.iter().sum::<f64>(), 1.0));
        }
    }

    #[test]
    fn zero_divisions_yields_one_zero_point() {
        // With 0 divisions every axis must take 0 → a single all-zero point.
        let pts = das_dennis(3, 0);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "num_objectives >= 1")]
    fn zero_objectives_panics() {
        let _ = das_dennis(0, 4);
    }
}
