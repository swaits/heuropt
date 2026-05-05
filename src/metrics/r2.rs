//! R2 indicator — a unary quality measure for Pareto fronts.
//!
//! For each weight vector `λ` in a user-supplied set, find the
//! best (smallest) weighted Tchebycheff value across the front;
//! average over all weight vectors. Lower is better.

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;

/// R2 indicator using the weighted Tchebycheff utility.
///
/// ```text
/// R2(A) = (1 / |Λ|) · Σ_{λ ∈ Λ} min_{a ∈ A} max_i { λ_i · |a_i − z*_i| }
/// ```
///
/// where `z*` is the ideal point (per-axis minimum across the
/// approximation, in minimization-oriented coordinates) and `Λ` is
/// a set of unit-simplex weight vectors. Lower is better.
///
/// Use [`das_dennis`](crate::pareto::das_dennis) to generate the
/// canonical structured weight set.
///
/// # Panics
///
/// If the approximation is empty, or any weight vector has wrong
/// length / negative entries / zero sum.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
/// use heuropt::metrics::r2::r2;
///
/// let space = ObjectiveSpace::new(vec![
///     Objective::minimize("f1"),
///     Objective::minimize("f2"),
/// ]);
/// let approx = [
///     Candidate::new((), Evaluation::new(vec![0.0, 1.0])),
///     Candidate::new((), Evaluation::new(vec![1.0, 0.0])),
/// ];
/// // Two weight vectors: (1, 0) and (0, 1) — extreme directions.
/// let weights = [vec![1.0, 0.0], vec![0.0, 1.0]];
/// let v = r2(&approx, &weights, &space);
/// // For each direction, the best front member matches that axis exactly.
/// // R2 = 0 since the ideal point is achieved on each direction.
/// assert!(v < 1e-12);
/// ```
pub fn r2<D>(
    approximation: &[Candidate<D>],
    weights: &[Vec<f64>],
    objectives: &ObjectiveSpace,
) -> f64 {
    assert!(
        !approximation.is_empty(),
        "r2: approximation must not be empty"
    );
    assert!(!weights.is_empty(), "r2: weight set must not be empty");
    let m = objectives.len();
    for (i, w) in weights.iter().enumerate() {
        assert_eq!(
            w.len(),
            m,
            "r2: weight {i} has wrong length ({} vs {m})",
            w.len()
        );
        assert!(
            w.iter().all(|&v| v >= 0.0),
            "r2: weight {i} has a negative entry"
        );
        assert!(w.iter().sum::<f64>() > 0.0, "r2: weight {i} has zero sum");
    }

    // Convert all approximation members to minimization orientation once.
    let oriented: Vec<Vec<f64>> = approximation
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();

    // Ideal point z* (per-axis minimum).
    let mut z_star = vec![f64::INFINITY; m];
    for o in &oriented {
        for k in 0..m {
            if o[k] < z_star[k] {
                z_star[k] = o[k];
            }
        }
    }

    let mut total = 0.0_f64;
    for w in weights {
        let mut best = f64::INFINITY;
        for o in &oriented {
            // Weighted Tchebycheff: max_i { w_i · |o_i − z*_i| }
            let mut t = 0.0_f64;
            for k in 0..m {
                let dk = (o[k] - z_star[k]).abs() * w[k];
                if dk > t {
                    t = dk;
                }
            }
            if t < best {
                best = t;
            }
        }
        total += best;
    }
    total / weights.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::Objective;
    use crate::pareto::das_dennis;

    fn space_min2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn cand(obj: Vec<f64>) -> Candidate<()> {
        Candidate::new((), Evaluation::new(obj))
    }

    #[test]
    fn r2_extremes_are_perfect_at_endpoints() {
        let s = space_min2();
        let front = [cand(vec![0.0, 1.0]), cand(vec![1.0, 0.0])];
        let weights = [vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(r2(&front, &weights, &s) < 1e-12);
    }

    #[test]
    fn r2_dense_dasdennis_finite_for_uniform_front() {
        let s = space_min2();
        let weights = das_dennis(2, 5);
        let front: Vec<Candidate<()>> = (0..=10)
            .map(|i| {
                let t = i as f64 / 10.0;
                cand(vec![t, 1.0 - t])
            })
            .collect();
        let v = r2(&front, &weights, &s);
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    #[should_panic(expected = "approximation must not be empty")]
    fn r2_empty_approximation_panics() {
        let s = space_min2();
        let weights = vec![vec![1.0, 0.0]];
        let _: f64 = r2::<()>(&[], &weights, &s);
    }

    #[test]
    #[should_panic(expected = "weight set must not be empty")]
    fn r2_empty_weights_panics() {
        let s = space_min2();
        let front = [cand(vec![0.0, 1.0])];
        let _ = r2(&front, &[], &s);
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn r2_wrong_dim_weight_panics() {
        let s = space_min2();
        let front = [cand(vec![0.0, 1.0])];
        let weights = vec![vec![1.0, 0.0, 0.0]];
        let _ = r2(&front, &weights, &s);
    }
}
