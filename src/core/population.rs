//! A friendly wrapper around `Vec<Candidate<D>>`.

use crate::core::candidate::Candidate;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A collection of evaluated candidates.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Population<D> {
    /// The candidates.
    pub candidates: Vec<Candidate<D>>,
}

impl<D> Population<D> {
    /// Wrap a vector of candidates as a `Population`.
    pub fn new(candidates: Vec<Candidate<D>>) -> Self {
        Self { candidates }
    }

    /// Number of candidates.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Returns `true` if there are no candidates.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Iterate over the candidates by reference.
    pub fn iter(&self) -> impl Iterator<Item = &Candidate<D>> {
        self.candidates.iter()
    }

    /// Unwrap into the inner `Vec<Candidate<D>>`.
    pub fn into_vec(self) -> Vec<Candidate<D>> {
        self.candidates
    }
}

impl<D> From<Vec<Candidate<D>>> for Population<D> {
    fn from(candidates: Vec<Candidate<D>>) -> Self {
        Self::new(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;

    fn cand(x: f64) -> Candidate<f64> {
        Candidate::new(x, Evaluation::new(vec![x]))
    }

    #[test]
    fn new_len_iter_into_vec() {
        let pop = Population::new(vec![cand(1.0), cand(2.0)]);
        assert_eq!(pop.len(), 2);
        assert!(!pop.is_empty());
        assert_eq!(pop.iter().count(), 2);
        assert_eq!(pop.into_vec().len(), 2);
    }

    #[test]
    fn from_vec_works() {
        let pop: Population<f64> = vec![cand(1.0)].into();
        assert_eq!(pop.len(), 1);
    }

    #[test]
    fn empty_population() {
        let pop: Population<f64> = Population::new(Vec::new());
        assert!(pop.is_empty());
    }
}
