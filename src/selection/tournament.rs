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
            let cv = c.evaluation.objectives.first().copied().unwrap_or(f64::INFINITY);
            let bv = b.evaluation.objectives.first().copied().unwrap_or(f64::INFINITY);
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
        let s = ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
        ]);
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
            Candidate::new(2u32, Evaluation::new(vec![10.0])),              // feasible, big f
            Candidate::new(3u32, Evaluation::new(vec![3.0])),               // feasible, small f
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
}
