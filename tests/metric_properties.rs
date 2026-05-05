//! Per-metric property tests for the Pareto-quality metrics.

use proptest::prelude::*;

use heuropt::core::candidate::Candidate;
use heuropt::core::evaluation::Evaluation;
use heuropt::core::objective::{Objective, ObjectiveSpace};
use heuropt::metrics::hypervolume::{hypervolume_2d, hypervolume_nd};
use heuropt::metrics::spacing::spacing;

fn space_2d() -> ObjectiveSpace {
    ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
}

fn space_3d() -> ObjectiveSpace {
    ObjectiveSpace::new(vec![
        Objective::minimize("f1"),
        Objective::minimize("f2"),
        Objective::minimize("f3"),
    ])
}

fn cand_2d(a: f64, b: f64) -> Candidate<()> {
    Candidate::new((), Evaluation::new(vec![a, b]))
}
fn cand_3d(a: f64, b: f64, c: f64) -> Candidate<()> {
    Candidate::new((), Evaluation::new(vec![a, b, c]))
}

proptest! {
    /// hypervolume_2d is non-negative.
    #[test]
    fn hv2_non_negative(
        front in prop::collection::vec((0.0_f64..10.0, 0.0_f64..10.0), 0..15),
    ) {
        let s = space_2d();
        let pop: Vec<Candidate<()>> = front.iter().map(|&(a, b)| cand_2d(a, b)).collect();
        let hv = hypervolume_2d(&pop, &s, [11.0, 11.0]);
        prop_assert!(hv >= 0.0);
        prop_assert!(hv.is_finite());
    }

    /// hypervolume_2d is bounded above by the (reference - 0)² = 121 box.
    #[test]
    fn hv2_bounded_by_box(
        front in prop::collection::vec((0.0_f64..10.0, 0.0_f64..10.0), 1..15),
    ) {
        let s = space_2d();
        let pop: Vec<Candidate<()>> = front.iter().map(|&(a, b)| cand_2d(a, b)).collect();
        let hv = hypervolume_2d(&pop, &s, [11.0, 11.0]);
        prop_assert!(hv <= 121.0_f64 + 1e-9);
    }

    /// Adding a dominated point doesn't change hypervolume_2d.
    #[test]
    fn hv2_dominated_invariant(
        a in 0.0_f64..5.0,
        b in 0.0_f64..5.0,
        d_offset in 0.001_f64..3.0,
    ) {
        let s = space_2d();
        let base = vec![cand_2d(a, b)];
        let mut with_dominated = base.clone();
        // (a + offset, b + offset) is strictly worse than (a, b) on both
        // axes, so it's dominated.
        with_dominated.push(cand_2d(a + d_offset, b + d_offset));
        let hv1 = hypervolume_2d(&base, &s, [11.0, 11.0]);
        let hv2 = hypervolume_2d(&with_dominated, &s, [11.0, 11.0]);
        prop_assert!((hv1 - hv2).abs() < 1e-9);
    }

    /// hypervolume_nd agrees with hypervolume_2d on 2-D inputs.
    #[test]
    fn hv_nd_matches_2d(
        front in prop::collection::vec((0.0_f64..10.0, 0.0_f64..10.0), 1..10),
    ) {
        let s = space_2d();
        let pop: Vec<Candidate<()>> = front.iter().map(|&(a, b)| cand_2d(a, b)).collect();
        let hv2 = hypervolume_2d(&pop, &s, [11.0, 11.0]);
        let hvn = hypervolume_nd(&pop, &s, &[11.0, 11.0]);
        prop_assert!((hv2 - hvn).abs() < 1e-9, "{hv2} vs {hvn}");
    }

    /// hypervolume_nd in 3-D is non-negative and bounded.
    #[test]
    fn hv3_non_negative_bounded(
        pts in prop::collection::vec(
            (0.0_f64..2.0, 0.0_f64..2.0, 0.0_f64..2.0),
            0..10,
        ),
    ) {
        let s = space_3d();
        let pop: Vec<Candidate<()>> = pts.iter().map(|&(a, b, c)| cand_3d(a, b, c)).collect();
        let hv = hypervolume_nd(&pop, &s, &[3.0, 3.0, 3.0]);
        prop_assert!(hv >= 0.0);
        prop_assert!(hv.is_finite());
        // Reference box has volume 27.
        prop_assert!(hv <= 27.0 + 1e-9);
    }

    /// spacing is non-negative and zero on a single point.
    #[test]
    fn spacing_non_negative(
        front in prop::collection::vec((0.0_f64..10.0, 0.0_f64..10.0), 0..15),
    ) {
        let s = space_2d();
        let pop: Vec<Candidate<()>> = front.iter().map(|&(a, b)| cand_2d(a, b)).collect();
        let sp = spacing(&pop, &s);
        prop_assert!(sp >= 0.0);
        prop_assert!(sp.is_finite());
    }

    /// spacing on a single-point front is exactly 0.
    #[test]
    fn spacing_single_point_is_zero(a in 0.0_f64..10.0, b in 0.0_f64..10.0) {
        let s = space_2d();
        let pop = vec![cand_2d(a, b)];
        let sp = spacing(&pop, &s);
        prop_assert_eq!(sp, 0.0);
    }
}
