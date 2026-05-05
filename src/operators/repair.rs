//! Repair operators: in-place projections that restore decisions to
//! feasibility.

use crate::traits::Repair;

/// Clamp every variable of a `Vec<f64>` to per-axis inclusive bounds.
///
/// The simplest possible repair — pair with `GaussianMutation` (which
/// doesn't enforce bounds in v1) to produce a bounds-respecting variant
/// without writing a custom Variation impl.
#[derive(Debug, Clone)]
pub struct ClampToBounds {
    /// Per-variable inclusive bounds.
    pub bounds: Vec<(f64, f64)>,
}

impl ClampToBounds {
    /// Construct a `ClampToBounds`.
    ///
    /// # Panics
    /// If any `(lo, hi)` has `lo > hi`.
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            assert!(
                lo <= hi,
                "ClampToBounds bound at index {i} has lo > hi: ({lo}, {hi})",
            );
        }
        Self { bounds }
    }
}

impl Repair<Vec<f64>> for ClampToBounds {
    fn repair(&mut self, decision: &mut Vec<f64>) {
        for (j, x) in decision.iter_mut().enumerate() {
            if let Some(&(lo, hi)) = self.bounds.get(j) {
                *x = x.clamp(lo, hi);
            }
        }
    }
}

/// Project a `Vec<f64>` onto the simplex `{ x : x ≥ 0, Σ x = total }`.
///
/// Implements the standard O(n log n) projection algorithm of Wang & Carreira-
/// Perpiñán 2013. Useful for portfolio-style problems where the
/// decision must sum to a budget, and for normalizing reference
/// directions onto the unit simplex.
#[derive(Debug, Clone)]
pub struct ProjectToSimplex {
    /// Target sum (the simplex's "size"). Standard probability simplex
    /// uses `total = 1.0`.
    pub total: f64,
}

impl ProjectToSimplex {
    /// Construct a `ProjectToSimplex`.
    ///
    /// # Panics
    /// If `total <= 0.0`.
    pub fn new(total: f64) -> Self {
        assert!(total > 0.0, "ProjectToSimplex total must be > 0");
        Self { total }
    }
}

impl Repair<Vec<f64>> for ProjectToSimplex {
    fn repair(&mut self, decision: &mut Vec<f64>) {
        let n = decision.len();
        if n == 0 {
            return;
        }
        // Sort copy descending.
        let mut sorted: Vec<f64> = decision.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Find ρ = max{ j : sorted[j-1] - (Σ_{i<=j} sorted[i] - total) / j > 0 }.
        let mut cumsum = 0.0;
        let mut rho = 0;
        let mut tau_at_rho = 0.0;
        for (j, &val) in sorted.iter().enumerate() {
            cumsum += val;
            let tau = (cumsum - self.total) / (j as f64 + 1.0);
            if val - tau > 0.0 {
                rho = j + 1;
                tau_at_rho = tau;
            }
        }
        let _ = rho;
        // Apply: x_i ← max(x_i - τ, 0).
        for x in decision.iter_mut() {
            *x = (*x - tau_at_rho).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn clamp_to_bounds_clips() {
        let mut r = ClampToBounds::new(vec![(-1.0, 1.0); 3]);
        let mut x = vec![-2.5, 0.5, 5.0];
        r.repair(&mut x);
        assert_eq!(x, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn clamp_passthrough_when_already_in_bounds() {
        let mut r = ClampToBounds::new(vec![(-1.0, 1.0); 3]);
        let mut x = vec![-0.3, 0.0, 0.7];
        let original = x.clone();
        r.repair(&mut x);
        assert_eq!(x, original);
    }

    #[test]
    #[should_panic(expected = "lo > hi")]
    fn clamp_invalid_bounds_panics() {
        let _ = ClampToBounds::new(vec![(1.0, -1.0)]);
    }

    #[test]
    fn project_to_unit_simplex_sums_to_total() {
        let mut r = ProjectToSimplex::new(1.0);
        let mut x = vec![0.5, 0.3, 0.2, -0.5];
        r.repair(&mut x);
        let s: f64 = x.iter().sum();
        assert!(approx_eq(s, 1.0, 1e-12));
        for &v in &x {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn project_already_on_simplex_unchanged() {
        let mut r = ProjectToSimplex::new(1.0);
        let mut x = vec![0.5, 0.3, 0.2];
        r.repair(&mut x);
        let s: f64 = x.iter().sum();
        assert!(approx_eq(s, 1.0, 1e-12));
        // Within tolerance, the values should be roughly preserved (no
        // clipping needed).
        assert!(approx_eq(x[0], 0.5, 1e-12));
        assert!(approx_eq(x[1], 0.3, 1e-12));
        assert!(approx_eq(x[2], 0.2, 1e-12));
    }

    #[test]
    fn project_arbitrary_total() {
        let mut r = ProjectToSimplex::new(10.0);
        let mut x = vec![100.0, 50.0, -20.0, 30.0];
        r.repair(&mut x);
        let s: f64 = x.iter().sum();
        assert!(approx_eq(s, 10.0, 1e-9));
        for &v in &x {
            assert!(v >= 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "total must be > 0")]
    fn project_non_positive_total_panics() {
        let _ = ProjectToSimplex::new(0.0);
    }
}
