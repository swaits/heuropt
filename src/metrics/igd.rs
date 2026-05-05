//! Inverted Generational Distance (IGD) and IGD+ performance indicators.
//!
//! Both quantify how well an approximation set covers a reference set
//! (typically the true Pareto front). Smaller values are better.

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::ObjectiveSpace;

/// Inverted Generational Distance.
///
/// For each point in the `reference` set, compute the Euclidean distance
/// to its nearest neighbor in the `approximation` set (in minimization-
/// oriented objective space), then average:
///
/// ```text
/// IGD(A) = (1 / |R|) · Σ_{r ∈ R} min_{a ∈ A} ‖a − r‖₂
/// ```
///
/// Lower is better. IGD captures both convergence (close to the front)
/// and spread (the approximation must cover the reference).
///
/// # Panics
///
/// If `reference` is empty.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
/// use heuropt::metrics::igd::igd;
///
/// let space = ObjectiveSpace::new(vec![
///     Objective::minimize("f1"),
///     Objective::minimize("f2"),
/// ]);
/// // Approximation: a sparse 2-point front.
/// let approx = [
///     Candidate::new((), Evaluation::new(vec![0.0, 1.0])),
///     Candidate::new((), Evaluation::new(vec![1.0, 0.0])),
/// ];
/// // Reference: a dense 3-point sample of the true front.
/// let reference = [
///     Evaluation::new(vec![0.0, 1.0]),
///     Evaluation::new(vec![0.5, 0.5]),
///     Evaluation::new(vec![1.0, 0.0]),
/// ];
/// let v = igd(&approx, &reference, &space);
/// // The middle reference point is unfortunately distance √(0.5²+0.5²) = 0.707
/// // from each approximation point; the boundary points are 0 away.
/// // IGD = (0 + 0.707 + 0) / 3 ≈ 0.236.
/// assert!((v - 0.2357).abs() < 1e-3);
/// ```
pub fn igd<D>(
    approximation: &[Candidate<D>],
    reference: &[Evaluation],
    objectives: &ObjectiveSpace,
) -> f64 {
    assert!(
        !reference.is_empty(),
        "igd: reference set must not be empty"
    );
    let approx_oriented: Vec<Vec<f64>> = approximation
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();
    if approx_oriented.is_empty() {
        return f64::INFINITY;
    }
    let mut total = 0.0_f64;
    for r in reference {
        let r_oriented = objectives.as_minimization(&r.objectives);
        let mut min_d = f64::INFINITY;
        for a in &approx_oriented {
            let d: f64 = a
                .iter()
                .zip(r_oriented.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt();
            if d < min_d {
                min_d = d;
            }
        }
        total += min_d;
    }
    total / reference.len() as f64
}

/// IGD+ — a dominance-respecting variant of IGD.
///
/// For each reference point `r`, the distance to an approximation
/// point `a` is computed only on objectives where `a` is *worse than*
/// `r` — i.e. on the "violation" component of the gap. This makes
/// IGD+ a Pareto-compliant indicator: adding a dominated point to the
/// approximation never improves the score.
///
/// ```text
/// IGD+(A) = (1 / |R|) · Σ_{r ∈ R} min_{a ∈ A} ‖max(a − r, 0)‖₂
/// ```
///
/// Lower is better.
///
/// # Panics
///
/// If `reference` is empty.
pub fn igd_plus<D>(
    approximation: &[Candidate<D>],
    reference: &[Evaluation],
    objectives: &ObjectiveSpace,
) -> f64 {
    assert!(
        !reference.is_empty(),
        "igd_plus: reference set must not be empty"
    );
    let approx_oriented: Vec<Vec<f64>> = approximation
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();
    if approx_oriented.is_empty() {
        return f64::INFINITY;
    }
    let mut total = 0.0_f64;
    for r in reference {
        let r_oriented = objectives.as_minimization(&r.objectives);
        let mut min_d = f64::INFINITY;
        for a in &approx_oriented {
            let d: f64 = a
                .iter()
                .zip(r_oriented.iter())
                .map(|(x, y)| (x - y).max(0.0).powi(2))
                .sum::<f64>()
                .sqrt();
            if d < min_d {
                min_d = d;
            }
        }
        total += min_d;
    }
    total / reference.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::objective::Objective;

    fn space_min2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn cand(obj: Vec<f64>) -> Candidate<()> {
        Candidate::new((), Evaluation::new(obj))
    }

    #[test]
    fn igd_perfect_match_is_zero() {
        let s = space_min2();
        let approx = [cand(vec![0.0, 1.0]), cand(vec![1.0, 0.0])];
        let reference = [
            Evaluation::new(vec![0.0, 1.0]),
            Evaluation::new(vec![1.0, 0.0]),
        ];
        let v = igd(&approx, &reference, &s);
        assert!(v < 1e-12);
    }

    #[test]
    fn igd_known_value() {
        let s = space_min2();
        let approx = [cand(vec![0.0, 0.0])];
        let reference = [Evaluation::new(vec![1.0, 1.0])];
        let v = igd(&approx, &reference, &s);
        assert!((v - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn igd_plus_dominated_point_does_not_improve() {
        let s = space_min2();
        let reference = [
            Evaluation::new(vec![0.0, 1.0]),
            Evaluation::new(vec![1.0, 0.0]),
        ];
        let base = vec![cand(vec![0.5, 0.5])];
        let with_dominated = vec![cand(vec![0.5, 0.5]), cand(vec![1.0, 1.0])];
        let v_base = igd_plus(&base, &reference, &s);
        let v_with = igd_plus(&with_dominated, &reference, &s);
        // Adding a dominated point should not improve the score.
        assert!(v_with >= v_base - 1e-12);
    }

    #[test]
    fn igd_empty_approximation_is_infinity() {
        let s = space_min2();
        let approx: [Candidate<()>; 0] = [];
        let reference = [Evaluation::new(vec![0.0, 1.0])];
        assert!(igd(&approx, &reference, &s).is_infinite());
    }

    #[test]
    #[should_panic(expected = "reference set must not be empty")]
    fn igd_empty_reference_panics() {
        let s = space_min2();
        let approx = [cand(vec![0.0, 1.0])];
        let _ = igd::<()>(&approx, &[], &s);
    }
}
