//! Standard return type for optimizers.

use crate::core::candidate::Candidate;
use crate::core::population::Population;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The output of an optimization run.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct OptimizationResult<D> {
    /// The final population (or all sampled candidates, depending on algorithm).
    pub population: Population<D>,
    /// The non-dominated subset of the final population or archive.
    pub pareto_front: Vec<Candidate<D>>,
    /// The single-objective best, or `None` for multi-objective problems.
    pub best: Option<Candidate<D>>,
    /// Total number of `Problem::evaluate` calls.
    pub evaluations: usize,
    /// Total number of major optimizer iterations.
    pub generations: usize,
}

impl<D> OptimizationResult<D> {
    /// Construct an `OptimizationResult` from its parts.
    pub fn new(
        population: Population<D>,
        pareto_front: Vec<Candidate<D>>,
        best: Option<Candidate<D>>,
        evaluations: usize,
        generations: usize,
    ) -> Self {
        Self {
            population,
            pareto_front,
            best,
            evaluations,
            generations,
        }
    }

    /// The final population.
    pub fn population(&self) -> &Population<D> {
        &self.population
    }

    /// The non-dominated subset.
    pub fn pareto_front(&self) -> &[Candidate<D>] {
        &self.pareto_front
    }

    /// The single-objective best, when meaningful.
    pub fn best(&self) -> Option<&Candidate<D>> {
        self.best.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;

    #[test]
    fn accessors_return_expected_data() {
        let cand = Candidate::new(1.0_f64, Evaluation::new(vec![1.0]));
        let pop = Population::new(vec![cand.clone()]);
        let r = OptimizationResult::new(pop, vec![cand.clone()], Some(cand.clone()), 5, 2);
        assert_eq!(r.population().len(), 1);
        assert_eq!(r.pareto_front().len(), 1);
        assert!(r.best().is_some());
        assert_eq!(r.evaluations, 5);
        assert_eq!(r.generations, 2);
    }
}
