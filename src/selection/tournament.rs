//! Single-objective tournament selection.

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::objective::{Direction, ObjectiveSpace};
use crate::core::rng::Rng;

/// Tournament selection for single-objective problems.
///
/// Each tournament samples `tournament_size` candidates uniformly with
/// replacement; the best one's decision is cloned into the output. Tiebreak
/// rules (spec §10.2):
///
/// 1. Feasible candidates beat infeasible candidates.
/// 2. Among infeasibles, smaller `constraint_violation` wins.
/// 3. Among feasibles, the direction-correct best objective wins.
///
/// # Panics
/// If `objectives` does not contain exactly one objective, or if `population`
/// is empty when `count > 0`, or if `tournament_size == 0`.
pub fn tournament_select_single_objective<D: Clone>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    tournament_size: usize,
    count: usize,
    rng: &mut Rng,
) -> Vec<D> {
    assert!(
        objectives.is_single_objective(),
        "tournament_select_single_objective requires exactly one objective",
    );
    assert!(
        tournament_size > 0,
        "tournament_size must be greater than 0",
    );
    if count == 0 {
        return Vec::new();
    }
    assert!(
        !population.is_empty(),
        "tournament_select_single_objective called on empty population with count > 0",
    );

    let direction = objectives.objectives[0].direction;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best_idx = rng.random_range(0..population.len());
        for _ in 1..tournament_size {
            let challenger = rng.random_range(0..population.len());
            if challenger_wins(&population[challenger], &population[best_idx], direction) {
                best_idx = challenger;
            }
        }
        out.push(population[best_idx].decision.clone());
    }
    out
}

fn challenger_wins<D>(c: &Candidate<D>, b: &Candidate<D>, dir: Direction) -> bool {
    match (c.evaluation.is_feasible(), b.evaluation.is_feasible()) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => c.evaluation.constraint_violation < b.evaluation.constraint_violation,
        (true, true) => {
            let cv = c
                .evaluation
                .objectives
                .first()
                .copied()
                .unwrap_or(f64::INFINITY);
            let bv = b
                .evaluation
                .objectives
                .first()
                .copied()
                .unwrap_or(f64::INFINITY);
            match dir {
                Direction::Minimize => cv < bv,
                Direction::Maximize => cv > bv,
            }
        }
    }
}

/// Stochastic-ranking selection (Runarsson & Yao 2000) for single-objective
/// constrained problems.
///
/// Performs a probabilistic bubble-sort pass on the population — each
/// pairwise comparison uses the *objective* value with probability `pf`,
/// otherwise it uses the standard feasibility-then-violation-then-objective
/// rule. The classic value is `pf = 0.45`; values close to `0.5` weight
/// objective improvement against constraint satisfaction.
///
/// Returns `count` decisions cloned from the top of the ranked
/// population. Useful when constraint satisfaction is hard and strict
/// feasibility-first selection traps the search outside the feasible
/// region.
///
/// # Panics
/// If `objectives` does not contain exactly one objective, if `pf` is
/// outside `[0.0, 1.0]`, or if the population is empty when `count > 0`.
pub fn stochastic_ranking_select<D: Clone>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    pf: f64,
    count: usize,
    rng: &mut Rng,
) -> Vec<D> {
    assert!(
        objectives.is_single_objective(),
        "stochastic_ranking_select requires exactly one objective",
    );
    assert!(
        (0.0..=1.0).contains(&pf),
        "stochastic_ranking_select pf must be in [0.0, 1.0]",
    );
    if count == 0 {
        return Vec::new();
    }
    assert!(
        !population.is_empty(),
        "stochastic_ranking_select called on empty population with count > 0",
    );

    let direction = objectives.objectives[0].direction;
    let n = population.len();
    let mut order: Vec<usize> = (0..n).collect();

    // Bubble-sort with at most n full sweeps (Runarsson & Yao §3).
    for _ in 0..n {
        let mut swapped = false;
        for i in 0..n - 1 {
            let a = &population[order[i]].evaluation;
            let b = &population[order[i + 1]].evaluation;
            let use_objective = rng.random::<f64>() < pf;
            let a_first = if use_objective || (a.is_feasible() && b.is_feasible()) {
                better_by_objective(a, b, direction)
            } else {
                better_by_feasibility(a, b, direction)
            };
            if !a_first {
                order.swap(i, i + 1);
                swapped = true;
            }
        }
        if !swapped {
            break;
        }
    }

    let mut out = Vec::with_capacity(count);
    for k in 0..count {
        out.push(population[order[k % n]].decision.clone());
    }
    out
}

fn better_by_objective(
    a: &crate::core::evaluation::Evaluation,
    b: &crate::core::evaluation::Evaluation,
    direction: Direction,
) -> bool {
    let av = a.objectives.first().copied().unwrap_or(f64::INFINITY);
    let bv = b.objectives.first().copied().unwrap_or(f64::INFINITY);
    match direction {
        Direction::Minimize => av < bv,
        Direction::Maximize => av > bv,
    }
}

fn better_by_feasibility(
    a: &crate::core::evaluation::Evaluation,
    b: &crate::core::evaluation::Evaluation,
    direction: Direction,
) -> bool {
    match (a.is_feasible(), b.is_feasible()) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => a.constraint_violation < b.constraint_violation,
        (true, true) => better_by_objective(a, b, direction),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::Objective;
    use crate::core::rng::rng_from_seed;

    fn cand_min(d: u32, v: f64) -> Candidate<u32> {
        Candidate::new(d, Evaluation::new(vec![v]))
    }

    #[test]
    fn large_tournament_picks_best_minimize() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [cand_min(1, 10.0), cand_min(2, 1.0), cand_min(3, 5.0)];
        let mut rng = rng_from_seed(1);
        // Tournament size equal to population almost always returns the best.
        let picks = tournament_select_single_objective(&pop, &s, 100, 10, &mut rng);
        assert!(picks.iter().all(|&d| d == 2));
    }

    #[test]
    fn large_tournament_picks_best_maximize() {
        let s = ObjectiveSpace::new(vec![Objective::maximize("score")]);
        let pop = [cand_min(1, 10.0), cand_min(2, 1.0), cand_min(3, 5.0)];
        let mut rng = rng_from_seed(2);
        let picks = tournament_select_single_objective(&pop, &s, 100, 10, &mut rng);
        assert!(picks.iter().all(|&d| d == 1));
    }

    #[test]
    fn feasible_beats_infeasible() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [
            Candidate::new(1u32, Evaluation::constrained(vec![0.0], 5.0)),
            Candidate::new(2u32, Evaluation::new(vec![100.0])),
        ];
        let mut rng = rng_from_seed(3);
        let picks = tournament_select_single_objective(&pop, &s, 50, 20, &mut rng);
        // Feasible candidate (decision 2) wins regardless of objective value.
        assert!(picks.iter().all(|&d| d == 2));
    }

    #[test]
    #[should_panic(expected = "exactly one objective")]
    fn multi_objective_panics() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
        let pop = [cand_min(1, 1.0)];
        let mut rng = rng_from_seed(0);
        let _ = tournament_select_single_objective(&pop, &s, 2, 1, &mut rng);
    }

    #[test]
    fn stochastic_ranking_returns_count_decisions() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [cand_min(1, 5.0), cand_min(2, 1.0), cand_min(3, 9.0)];
        let mut rng = rng_from_seed(7);
        let picks = stochastic_ranking_select(&pop, &s, 0.45, 4, &mut rng);
        assert_eq!(picks.len(), 4);
        for p in &picks {
            assert!([1, 2, 3].contains(p));
        }
    }

    #[test]
    fn stochastic_ranking_pf_zero_is_feasibility_first() {
        // With pf = 0, the algorithm reduces to strict feasibility-first
        // ordering, so the best feasible candidate should top the rank.
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [
            Candidate::new(1u32, Evaluation::constrained(vec![0.0], 5.0)), // infeasible
            Candidate::new(2u32, Evaluation::new(vec![10.0])),             // feasible, big f
            Candidate::new(3u32, Evaluation::new(vec![3.0])),              // feasible, small f
        ];
        let mut rng = rng_from_seed(0);
        let picks = stochastic_ranking_select(&pop, &s, 0.0, 3, &mut rng);
        assert_eq!(picks[0], 3); // best feasible first
    }

    #[test]
    #[should_panic(expected = "pf must be in [0.0, 1.0]")]
    fn stochastic_ranking_pf_out_of_range_panics() {
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [cand_min(1, 1.0)];
        let mut rng = rng_from_seed(0);
        let _ = stochastic_ranking_select(&pop, &s, 1.5, 1, &mut rng);
    }

    // ---- Mutation-test pinned helpers --------------------------------------

    fn constrained(d: u32, obj: f64, cv: f64) -> Candidate<u32> {
        Candidate::new(d, Evaluation::constrained(vec![obj], cv))
    }

    #[test]
    fn challenger_wins_feasibility_first() {
        // Feasible challenger beats infeasible best, regardless of objective.
        let feasible = cand_min(1, 100.0);
        let infeasible = constrained(2, 0.0, 1.0);
        assert!(challenger_wins(&feasible, &infeasible, Direction::Minimize));
        assert!(!challenger_wins(
            &infeasible,
            &feasible,
            Direction::Minimize
        ));
    }

    #[test]
    fn challenger_wins_two_infeasible_compares_violation() {
        let less_violating = constrained(1, 0.0, 0.5);
        let more_violating = constrained(2, 0.0, 1.0);
        assert!(challenger_wins(
            &less_violating,
            &more_violating,
            Direction::Minimize
        ));
        assert!(!challenger_wins(
            &more_violating,
            &less_violating,
            Direction::Minimize
        ));
    }

    #[test]
    fn challenger_wins_two_feasible_under_min_and_max() {
        let lower = cand_min(1, 1.0);
        let higher = cand_min(2, 2.0);
        assert!(challenger_wins(&lower, &higher, Direction::Minimize));
        assert!(!challenger_wins(&higher, &lower, Direction::Minimize));
        assert!(challenger_wins(&higher, &lower, Direction::Maximize));
        assert!(!challenger_wins(&lower, &higher, Direction::Maximize));
    }

    #[test]
    fn challenger_wins_equal_objectives_does_not_win() {
        // Strict comparison: equal objectives → challenger does NOT win.
        let a = cand_min(1, 1.0);
        let b = cand_min(2, 1.0);
        assert!(!challenger_wins(&a, &b, Direction::Minimize));
        assert!(!challenger_wins(&a, &b, Direction::Maximize));
    }

    #[test]
    fn better_by_objective_min_and_max() {
        let a = Evaluation::new(vec![1.0]);
        let b = Evaluation::new(vec![2.0]);
        assert!(better_by_objective(&a, &b, Direction::Minimize));
        assert!(!better_by_objective(&b, &a, Direction::Minimize));
        assert!(better_by_objective(&b, &a, Direction::Maximize));
        assert!(!better_by_objective(&a, &b, Direction::Maximize));
        // Equal → not strictly better.
        let c = Evaluation::new(vec![1.0]);
        assert!(!better_by_objective(&a, &c, Direction::Minimize));
    }

    #[test]
    fn better_by_feasibility_all_four_branches() {
        let feasible_a = Evaluation::new(vec![10.0]);
        let infeasible_b = Evaluation::constrained(vec![0.0], 1.0);
        // feasible vs infeasible
        assert!(better_by_feasibility(
            &feasible_a,
            &infeasible_b,
            Direction::Minimize
        ));
        assert!(!better_by_feasibility(
            &infeasible_b,
            &feasible_a,
            Direction::Minimize
        ));
        // two infeasible: smaller violation wins
        let low_cv = Evaluation::constrained(vec![0.0], 0.3);
        let high_cv = Evaluation::constrained(vec![0.0], 0.9);
        assert!(better_by_feasibility(
            &low_cv,
            &high_cv,
            Direction::Minimize
        ));
        assert!(!better_by_feasibility(
            &high_cv,
            &low_cv,
            Direction::Minimize
        ));
        // two feasible: delegates to better_by_objective
        let feasible_lower = Evaluation::new(vec![1.0]);
        let feasible_higher = Evaluation::new(vec![2.0]);
        assert!(better_by_feasibility(
            &feasible_lower,
            &feasible_higher,
            Direction::Minimize
        ));
    }

    #[test]
    fn stochastic_ranking_select_pf_zero_is_pure_feasibility_order() {
        // pf = 0 → always compare by feasibility. The feasible candidate
        // must rank first regardless of objective value.
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [
            constrained(1, 0.0, 2.0), // infeasible, great objective
            cand_min(2, 100.0),       // feasible, terrible objective
        ];
        let mut rng = rng_from_seed(7);
        let picks = stochastic_ranking_select(&pop, &s, 0.0, 1, &mut rng);
        // With pf=0, feasibility dominates → candidate 2 ranked first.
        assert_eq!(picks, vec![2]);
    }

    #[test]
    fn stochastic_ranking_select_count_wraps_modulo_population() {
        // count > population size wraps around via `order[k % n]`.
        let s = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        let pop = [cand_min(1, 1.0), cand_min(2, 2.0)];
        let mut rng = rng_from_seed(0);
        let picks = stochastic_ranking_select(&pop, &s, 0.0, 5, &mut rng);
        assert_eq!(picks.len(), 5);
        // Best (candidate 1) is at index 0; index 2 wraps to it again.
        assert_eq!(picks[0], 1);
        assert_eq!(picks[2], 1);
    }
}
