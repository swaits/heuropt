//! Schott's spacing metric for Pareto fronts.

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;

/// Schott's spacing metric.
///
/// For each point on the front, compute the Manhattan distance to its nearest
/// neighbor (in minimization-oriented objective space). The metric is the
/// (population) standard deviation of those per-point distances. A perfectly
/// uniform front has spacing 0.
///
/// Returns `0.0` for empty or single-point fronts (spec §14.1).
pub fn spacing<D>(front: &[Candidate<D>], objectives: &ObjectiveSpace) -> f64 {
    let n = front.len();
    if n < 2 {
        return 0.0;
    }
    let oriented: Vec<Vec<f64>> = front
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();

    let mut nearest = vec![f64::INFINITY; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = oriented[i]
                .iter()
                .zip(oriented[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if d < nearest[i] {
                nearest[i] = d;
            }
        }
    }

    let mean = nearest.iter().sum::<f64>() / n as f64;
    let variance = nearest.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt()
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
    fn empty_front_is_zero() {
        let s = space_min2();
        let pts: [Candidate<()>; 0] = [];
        assert_eq!(spacing(&pts, &s), 0.0);
    }

    #[test]
    fn single_point_is_zero() {
        let s = space_min2();
        assert_eq!(spacing(&[cand(vec![1.0, 1.0])], &s), 0.0);
    }

    #[test]
    fn uniform_front_has_zero_spacing() {
        let s = space_min2();
        // Points evenly spaced along a line: each interior point's nearest
        // neighbor is at the same distance as its boundary neighbors',
        // and the boundary points share that distance too.
        let pts = [
            cand(vec![0.0, 4.0]),
            cand(vec![1.0, 3.0]),
            cand(vec![2.0, 2.0]),
            cand(vec![3.0, 1.0]),
            cand(vec![4.0, 0.0]),
        ];
        let s_val = spacing(&pts, &s);
        assert!(s_val.abs() < 1e-12);
    }

    #[test]
    fn non_uniform_front_has_positive_spacing() {
        let s = space_min2();
        // Clustered + isolated points → uneven nearest-neighbor distances.
        let pts = [
            cand(vec![0.0, 0.0]),
            cand(vec![0.1, 0.1]),
            cand(vec![5.0, 5.0]),
        ];
        let s_val = spacing(&pts, &s);
        assert!(s_val > 0.0);
    }
}
