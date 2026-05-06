//! Repair operators: in-place projections that restore decisions to
//! feasibility.

use crate::traits::Repair;

/// Clamp every variable of a `Vec<f64>` to per-axis inclusive bounds.
///
/// The simplest possible repair — pair with `GaussianMutation` (which
/// doesn't enforce bounds in v1) to produce a bounds-respecting variant
/// without writing a custom Variation impl.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut r = ClampToBounds::new(vec![(-1.0, 1.0); 3]);
/// let mut x = vec![-2.0, 0.5, 5.0];
/// r.repair(&mut x);
/// assert_eq!(x, vec![-1.0, 0.5, 1.0]);
/// ```
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
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut r = ProjectToSimplex::new(1.0);
/// let mut x = vec![0.6, 0.5, -0.1, 0.3];
/// r.repair(&mut x);
/// let sum: f64 = x.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-12);
/// assert!(x.iter().all(|&v| v >= 0.0));
/// ```
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
        // If any |x_i| dwarfs `total` so badly that `x_i - total == x_i` in
        // f64, the standard Duchi/Held-Wolfe projection loses all precision
        // in τ and silently returns the all-zero vector. In that pathological
        // regime the projection is effectively concentrated on argmax(x), so
        // assign all mass there directly.
        let max_abs = decision
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        if max_abs > self.total * 1e15 {
            let mut argmax = 0;
            for (i, &v) in decision.iter().enumerate().skip(1) {
                if v > decision[argmax] {
                    argmax = i;
                }
            }
            for (i, x) in decision.iter_mut().enumerate() {
                *x = if i == argmax { self.total } else { 0.0 };
            }
            return;
        }

        // Sort copy descending.
        let mut sorted: Vec<f64> = decision.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Find ρ = max{ j : sorted[j-1] - (Σ_{i<=j} sorted[i] - total) / j > 0 }.
        // Mathematically the j=0 case always satisfies the condition (since
        // total > 0), so we initialize from it before the loop — that guards
        // against floating-point precision loss when |sorted[0]| ≫ total,
        // where the subtraction `sorted[0] - tau` could otherwise round to
        // zero and leave τ unset (yielding the all-zero output bug).
        let mut cumsum = 0.0;
        let mut tau_at_rho = sorted[0] - self.total;
        for (j, &val) in sorted.iter().enumerate() {
            cumsum += val;
            let tau = (cumsum - self.total) / (j as f64 + 1.0);
            if val - tau > 0.0 {
                tau_at_rho = tau;
            }
        }
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

    /// Regression: discovered by the `clamp_to_bounds` fuzzer. When
    /// `|max(x)|` dwarfs `total` so badly that the subtraction `x - τ`
    /// rounds away `total`, the standard algorithm previously returned
    /// the all-zero vector. The degenerate-magnitude fallback now
    /// concentrates all mass on argmax(x).
    #[test]
    fn project_extreme_magnitudes_concentrates_on_argmax() {
        let mut r = ProjectToSimplex::new(1.0);
        let mut x = vec![1e20, 5e19, -1e20];
        r.repair(&mut x);
        let s: f64 = x.iter().sum();
        assert!(approx_eq(s, 1.0, 1e-12));
        // Argmax is index 0; all mass should be there.
        assert!(approx_eq(x[0], 1.0, 1e-12));
        assert_eq!(x[1], 0.0);
        assert_eq!(x[2], 0.0);
    }

    /// Regression: when the input is "all zeros", τ is small (0 - total),
    /// the projection should distribute total evenly. This tests the
    /// loop's handling of equal entries.
    #[test]
    fn project_all_zeros_distributes_evenly() {
        let mut r = ProjectToSimplex::new(1.0);
        let mut x = vec![0.0, 0.0, 0.0, 0.0];
        r.repair(&mut x);
        let s: f64 = x.iter().sum();
        assert!(approx_eq(s, 1.0, 1e-12));
        for &v in &x {
            assert!(approx_eq(v, 0.25, 1e-12));
        }
    }
}
