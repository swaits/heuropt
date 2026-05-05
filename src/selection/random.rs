//! Uniform random selection (with replacement).

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::rng::Rng;

/// Select `count` decisions uniformly at random with replacement.
///
/// Returns cloned decisions. Panics if `population` is empty and `count > 0`
/// (spec §10.1).
pub fn select_random<D: Clone>(
    population: &[Candidate<D>],
    count: usize,
    rng: &mut Rng,
) -> Vec<D> {
    if count == 0 {
        return Vec::new();
    }
    assert!(
        !population.is_empty(),
        "select_random called on empty population with count > 0",
    );
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let idx = rng.random_range(0..population.len());
        out.push(population[idx].decision.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::rng::rng_from_seed;

    fn cand(d: u32) -> Candidate<u32> {
        Candidate::new(d, Evaluation::new(vec![d as f64]))
    }

    #[test]
    fn returns_count_decisions_from_population() {
        let pop = [cand(1), cand(2), cand(3)];
        let mut rng = rng_from_seed(42);
        let picks = select_random(&pop, 5, &mut rng);
        assert_eq!(picks.len(), 5);
        for p in &picks {
            assert!([1, 2, 3].contains(p));
        }
    }

    #[test]
    fn count_zero_returns_empty() {
        let pop = [cand(1)];
        let mut rng = rng_from_seed(0);
        assert!(select_random(&pop, 0, &mut rng).is_empty());
    }

    #[test]
    #[should_panic(expected = "empty population")]
    fn empty_population_panics() {
        let pop: Vec<Candidate<u32>> = Vec::new();
        let mut rng = rng_from_seed(0);
        let _ = select_random(&pop, 1, &mut rng);
    }
}
