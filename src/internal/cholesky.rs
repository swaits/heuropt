//! Cholesky factorization (`A = L · L^T`) plus triangular solves for
//! symmetric positive-definite matrices.
//!
//! Used internally by Bayesian Optimization for the GP posterior. Hand-
//! rolled to avoid pulling in a linear-algebra dependency.

/// Factorize a symmetric positive-definite matrix `a` as `L · L^T`,
/// returning `L` (lower triangular). Returns `Err` if `a` is not SPD,
/// which the caller typically responds to by adding jitter to the
/// diagonal and retrying.
pub(crate) fn cholesky(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, &'static str> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    debug_assert!(a.iter().all(|row| row.len() == n));
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err("matrix is not positive-definite");
                }
                l[i][j] = sum.sqrt();
            } else {
                if l[j][j].abs() < 1e-300 {
                    return Err("zero on diagonal during Cholesky");
                }
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Solve `L · y = b` (forward substitution) for lower-triangular `L`.
pub(crate) fn solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    y
}

/// Solve `L^T · x = y` (backward substitution) for lower-triangular `L`
/// (so `L^T` is upper-triangular).
pub(crate) fn solve_upper_transpose(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// Solve `A · x = b` given the Cholesky factor `L` of `A`. One forward
/// substitution + one back substitution.
pub(crate) fn solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let y = solve_lower(l, b);
    solve_upper_transpose(l, &y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn two_by_two_spd_factors() {
        // A = [[4, 2], [2, 5]] → L = [[2, 0], [1, 2]]
        let a = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let l = cholesky(&a).unwrap();
        assert!(approx_eq(l[0][0], 2.0, 1e-12));
        assert!(approx_eq(l[1][0], 1.0, 1e-12));
        assert!(approx_eq(l[1][1], 2.0, 1e-12));
        // L · L^T should reconstruct A.
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0;
                for k in 0..2 {
                    s += l[i][k] * l[j][k];
                }
                assert!(approx_eq(s, a[i][j], 1e-12));
            }
        }
    }

    #[test]
    fn three_by_three_solve_round_trip() {
        // SPD 3x3 with a known answer.
        let a = vec![
            vec![25.0, 15.0, -5.0],
            vec![15.0, 18.0, 0.0],
            vec![-5.0, 0.0, 11.0],
        ];
        let l = cholesky(&a).unwrap();
        // Choose a vector and check A · x = b round trip.
        let x_truth = vec![1.0, -2.0, 0.5];
        let b: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[i][j] * x_truth[j]).sum())
            .collect();
        let x = solve(&l, &b);
        for k in 0..3 {
            assert!(approx_eq(x[k], x_truth[k], 1e-9));
        }
    }

    #[test]
    fn non_psd_returns_err() {
        // [[1, 2], [2, 1]] has eigenvalues 3 and -1 → not PD.
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(cholesky(&a).is_err());
    }

    #[test]
    fn empty_matrix() {
        let a: Vec<Vec<f64>> = Vec::new();
        let l = cholesky(&a).unwrap();
        assert_eq!(l.len(), 0);
    }
}
