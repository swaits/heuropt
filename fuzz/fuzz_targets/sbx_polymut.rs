#![no_main]
//! Fuzz SBX + PolynomialMutation: in-bounds parents must produce in-bounds
//! children for any seed and any (η, per-variable-probability) pair.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use heuropt::core::rng::rng_from_seed;
use heuropt::prelude::*;

#[derive(Arbitrary, Debug)]
struct Input {
    bounds: Vec<(f64, f64)>,
    eta_sbx: f64,
    eta_pm: f64,
    pvp_sbx: f64,
    pvp_pm: f64,
    a_frac: Vec<f64>,
    b_frac: Vec<f64>,
    seed: u64,
}

fuzz_target!(|input: Input| {
    let n = input.bounds.len();
    if n == 0 || n > 8 {
        return;
    }
    if !(input.eta_sbx.is_finite() && input.eta_pm.is_finite()) {
        return;
    }
    if !(input.eta_sbx >= 1.0 && input.eta_sbx <= 100.0) {
        return;
    }
    if !(input.eta_pm >= 1.0 && input.eta_pm <= 100.0) {
        return;
    }
    let pvp_sbx = match input.pvp_sbx {
        v if v.is_finite() && (0.0..=1.0).contains(&v) => v,
        _ => return,
    };
    let pvp_pm = match input.pvp_pm {
        v if v.is_finite() && (0.0..=1.0).contains(&v) => v,
        _ => return,
    };
    // Sanitize bounds: lo < hi, finite.
    let bounds: Vec<(f64, f64)> = input
        .bounds
        .iter()
        .filter_map(|&(lo, hi)| {
            if lo.is_finite() && hi.is_finite() && hi - lo > 1e-9 {
                Some((lo, hi))
            } else {
                None
            }
        })
        .collect();
    if bounds.len() != n {
        return;
    }
    if input.a_frac.len() < n || input.b_frac.len() < n {
        return;
    }

    let p1: Vec<f64> = bounds
        .iter()
        .zip(&input.a_frac)
        .map(|(&(lo, hi), &f)| {
            let frac = if f.is_finite() { f.fract().abs() } else { 0.5 };
            lo + frac * (hi - lo)
        })
        .collect();
    let p2: Vec<f64> = bounds
        .iter()
        .zip(&input.b_frac)
        .map(|(&(lo, hi), &f)| {
            let frac = if f.is_finite() { f.fract().abs() } else { 0.5 };
            lo + frac * (hi - lo)
        })
        .collect();

    let mut rng = rng_from_seed(input.seed);
    let mut sbx = SimulatedBinaryCrossover::new(bounds.clone(), input.eta_sbx, pvp_sbx);
    let kids = sbx.vary(&[p1, p2], &mut rng);
    assert_eq!(kids.len(), 2);
    for c in &kids {
        for (j, &v) in c.iter().enumerate() {
            let (lo, hi) = bounds[j];
            assert!(v >= lo && v <= hi, "SBX child[{j}] = {v} out of [{lo}, {hi}]");
        }
    }

    let mut pm = PolynomialMutation::new(bounds.clone(), input.eta_pm, pvp_pm);
    let mutated = pm.vary(std::slice::from_ref(&kids[0]), &mut rng);
    assert_eq!(mutated.len(), 1);
    for (j, &v) in mutated[0].iter().enumerate() {
        let (lo, hi) = bounds[j];
        assert!(v >= lo && v <= hi, "PM child[{j}] = {v} out of [{lo}, {hi}]");
    }
});
