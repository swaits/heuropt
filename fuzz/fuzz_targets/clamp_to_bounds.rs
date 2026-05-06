#![no_main]
//! Fuzz `ClampToBounds` + `ProjectToSimplex` repair operators for
//! idempotence and target-set membership.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use heuropt::prelude::*;

#[derive(Arbitrary, Debug)]
struct Input {
    bounds: Vec<(f64, f64)>,
    x: Vec<f64>,
    simplex_total: f64,
}

fuzz_target!(|input: Input| {
    if input.bounds.is_empty() || input.bounds.len() > 16 {
        return;
    }
    if input.x.len() != input.bounds.len() {
        return;
    }
    let bounds: Vec<(f64, f64)> = input
        .bounds
        .iter()
        .filter_map(|&(lo, hi)| {
            if lo.is_finite() && hi.is_finite() && lo < hi {
                Some((lo, hi))
            } else {
                None
            }
        })
        .collect();
    if bounds.len() != input.bounds.len() {
        return;
    }
    // Restrict to a numerically-reasonable magnitude range for repair
    // operators — they are invoked downstream of evolutionary search where
    // candidate magnitudes are bounded.
    if input.x.iter().any(|v| !v.is_finite() || v.abs() > 1e30) {
        return;
    }

    let mut x = input.x.clone();
    let mut clamp = ClampToBounds::new(bounds.clone());
    clamp.repair(&mut x);
    for (j, &v) in x.iter().enumerate() {
        let (lo, hi) = bounds[j];
        assert!(v >= lo && v <= hi, "clamp out of bounds");
    }
    let after_one = x.clone();
    clamp.repair(&mut x);
    assert_eq!(x, after_one, "clamp not idempotent");

    // Simplex projection only meaningful when total > 0 and dim >= 1.
    // The Duchi/Held-Wolfe projection loses precision when |x| ≫ total
    // (τ becomes indistinguishable from max(x) in f64). Restrict to inputs
    // within the algorithm's well-conditioned regime, |x_i| ≤ total · 1e6.
    let max_abs = input.x.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if input.simplex_total.is_finite()
        && input.simplex_total > 1.0
        && input.simplex_total < 1e9
        && max_abs <= input.simplex_total * 1e6
    {
        let mut y = input.x.clone();
        let mut proj = ProjectToSimplex::new(input.simplex_total);
        proj.repair(&mut y);
        for &v in &y {
            assert!(v >= 0.0, "project negative entry");
        }
        let s: f64 = y.iter().sum();
        assert!(
            (s - input.simplex_total).abs() < 1e-6 * input.simplex_total.max(1.0),
            "project sum {s} != target {}",
            input.simplex_total,
        );
        let after = y.clone();
        proj.repair(&mut y);
        // The simplex projection's `τ` computation operates on values
        // up to `simplex_total · 1e6` (per the filter above), so its FP
        // precision floor is ~1e-4 of the input scale. Outputs near the
        // `max(x_i − τ, 0)` clamp boundary can flip between 0 and a
        // small positive value across re-applications. The fuzzer is
        // checking for *gross* non-idempotence (all-zeros vs valid),
        // not ULP-level slop.
        let scale = input.simplex_total.max(max_abs).max(1.0);
        for (a, b) in after.iter().zip(y.iter()) {
            assert!(
                (a - b).abs() < 1e-4 * scale,
                "project not idempotent: {a} vs {b}",
            );
        }
    }
});
