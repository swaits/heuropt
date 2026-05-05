//! Symmetric-matrix eigendecomposition via cyclic Jacobi rotations.
//!
//! Used internally by CMA-ES to maintain the covariance matrix's
//! eigendecomposition each generation. Hand-rolled to avoid pulling in a
//! linear-algebra dependency for one algorithm.

/// Symmetric eigendecomposition of an `n × n` matrix.
///
/// `matrix` must be square and symmetric (caller's responsibility — this is
/// `pub(crate)`). Returns `(eigenvalues, eigenvectors)` where:
///
/// - `eigenvalues[i]` is the i-th eigenvalue, in **descending** order.
/// - `eigenvectors[i]` is the corresponding unit eigenvector (row).
///
/// Iterates cyclic Jacobi rotations until the largest off-diagonal magnitude
/// is below `tol` or `max_sweeps` sweeps have completed. For typical CMA-ES
/// usage (small N, well-conditioned C) convergence is fast.
pub(crate) fn symmetric_eigen(
    matrix: &[Vec<f64>],
    tol: f64,
    max_sweeps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    debug_assert!(matrix.iter().all(|row| row.len() == n), "matrix must be square");

    // Working copy of the matrix; converges to a diagonal of eigenvalues.
    let mut a: Vec<Vec<f64>> = matrix.iter().map(|row| row.clone()).collect();
    // Eigenvector accumulator, starts as identity.
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for _ in 0..max_sweeps {
        let mut max_off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let abs_off = a[i][j].abs();
                if abs_off > max_off {
                    max_off = abs_off;
                }
            }
        }
        if max_off < tol {
            break;
        }

        // Cyclic sweep: rotate every (i, j) pair once.
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() < tol {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                // Rotation angle (Givens) chosen to zero out a[p][q].
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                // Update diagonal entries.
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                // Update other off-diagonal entries in rows/cols p and q.
                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r][p];
                        let arq = a[r][q];
                        a[r][p] = arp - s * (arq + tau * arp);
                        a[r][q] = arq + s * (arp - tau * arq);
                        a[p][r] = a[r][p];
                        a[q][r] = a[r][q];
                    }
                }

                // Update accumulated eigenvectors.
                for r in 0..n {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = vrp - s * (vrq + tau * vrp);
                    v[r][q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }

    // Extract eigenvalues from the diagonal of `a` and pair them with their
    // eigenvectors (columns of `v`).
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|i| {
            let val = a[i][i];
            let vec: Vec<f64> = (0..n).map(|r| v[r][i]).collect();
            (val, vec)
        })
        .collect();
    // Sort by eigenvalue descending.
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = pairs.iter().map(|(v, _)| *v).collect();
    let eigenvectors: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();
    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn diagonal_matrix_keeps_eigenvalues_on_diagonal() {
        let m = vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 5.0],
        ];
        let (vals, vecs) = symmetric_eigen(&m, 1e-12, 50);
        // Sorted descending: 5, 3, 1.
        assert!(approx_eq(vals[0], 5.0, 1e-10));
        assert!(approx_eq(vals[1], 3.0, 1e-10));
        assert!(approx_eq(vals[2], 1.0, 1e-10));
        for v in &vecs {
            assert!(approx_eq(norm(v), 1.0, 1e-10));
        }
    }

    #[test]
    fn two_by_two_known_case() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1, eigenvectors (1,1)/√2 and (1,-1)/√2.
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (vals, vecs) = symmetric_eigen(&m, 1e-12, 50);
        assert!(approx_eq(vals[0], 3.0, 1e-10));
        assert!(approx_eq(vals[1], 1.0, 1e-10));
        // Each eigenvector has unit norm.
        for v in &vecs {
            assert!(approx_eq(norm(v), 1.0, 1e-10));
        }
        // (1,1)/√2 ≈ (0.7071, 0.7071): components have the same sign.
        assert!((vecs[0][0] - vecs[0][1]).abs() < 1e-10);
        // (1,-1)/√2: components have opposite signs.
        assert!((vecs[1][0] + vecs[1][1]).abs() < 1e-10);
    }

    #[test]
    fn reconstruct_via_a_v_equals_lambda_v() {
        // Reconstruct A · v_i ≈ λ_i · v_i for a small symmetric matrix.
        let m = vec![
            vec![4.0, 1.0, -2.0],
            vec![1.0, 2.0, 0.5],
            vec![-2.0, 0.5, 3.0],
        ];
        let (vals, vecs) = symmetric_eigen(&m, 1e-12, 100);
        for (lambda, v) in vals.iter().zip(vecs.iter()) {
            // A · v
            let av: Vec<f64> = (0..3)
                .map(|i| (0..3).map(|j| m[i][j] * v[j]).sum::<f64>())
                .collect();
            // λ · v
            let lv: Vec<f64> = v.iter().map(|x| lambda * x).collect();
            for (x, y) in av.iter().zip(lv.iter()) {
                assert!(approx_eq(*x, *y, 1e-9), "Av != λv: {x} vs {y}");
            }
        }
    }

    #[test]
    fn eigenvectors_are_orthogonal() {
        let m = vec![
            vec![4.0, 1.0, -2.0],
            vec![1.0, 2.0, 0.5],
            vec![-2.0, 0.5, 3.0],
        ];
        let (_, vecs) = symmetric_eigen(&m, 1e-12, 100);
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(approx_eq(dot(&vecs[i], &vecs[j]), 0.0, 1e-9));
            }
        }
    }
}
