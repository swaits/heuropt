//! Compose two `Variation` operators into a pipeline (typically crossover → mutation).

use crate::core::rng::Rng;
use crate::traits::Variation;

/// A two-stage variation pipeline.
///
/// On each call to `vary`:
///
/// 1. The `crossover` operator is run on the input `parents`, producing one
///    or more children.
/// 2. For every child, the `mutation` operator is run with that child as its
///    sole parent, and the resulting children are concatenated into the
///    output.
///
/// Use this to build the canonical NSGA-II operator stack — SBX followed by
/// polynomial mutation — out of the existing primitives:
///
/// ```rust
/// use heuropt::prelude::*;
///
/// let bounds = vec![(0.0, 1.0); 30];
/// let variation = CompositeVariation {
///     crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
///     mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / 30.0),
/// };
/// let _ = variation;
/// ```
#[derive(Debug, Clone)]
pub struct CompositeVariation<C, M> {
    /// First-stage operator; typically a crossover that consumes ≥ 2 parents.
    pub crossover: C,
    /// Second-stage operator; typically a mutation that consumes 1 parent.
    pub mutation: M,
}

impl<D, C, M> Variation<D> for CompositeVariation<C, M>
where
    D: Clone,
    C: Variation<D>,
    M: Variation<D>,
{
    fn vary(&mut self, parents: &[D], rng: &mut Rng) -> Vec<D> {
        let crossed = self.crossover.vary(parents, rng);
        let mut out = Vec::with_capacity(crossed.len());
        for child in crossed {
            let mutated = self.mutation.vary(std::slice::from_ref(&child), rng);
            out.extend(mutated);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rng::rng_from_seed;
    use crate::operators::real::{
        BoundedGaussianMutation, PolynomialMutation, SimulatedBinaryCrossover,
    };

    #[test]
    fn pipes_sbx_into_polynomial_mutation() {
        let bounds = vec![(-1.0, 1.0); 4];
        let mut variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 1.0),
            mutation: PolynomialMutation::new(bounds, 20.0, 0.25),
        };
        let mut rng = rng_from_seed(123);
        let p1 = vec![0.1, -0.2, 0.3, -0.4];
        let p2 = vec![-0.3, 0.4, -0.1, 0.2];
        let children = variation.vary(&[p1, p2], &mut rng);
        // SBX produces 2 children; polynomial mutation produces 1 child each.
        assert_eq!(children.len(), 2);
        for c in &children {
            assert_eq!(c.len(), 4);
            for &x in c {
                assert!((-1.0..=1.0).contains(&x));
            }
        }
    }

    #[test]
    fn output_count_equals_inner_crossover_count_when_mutation_is_1to1() {
        // BoundedGaussianMutation always returns 1 child.
        let bounds = vec![(0.0, 1.0); 3];
        let mut variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 10.0, 0.5),
            mutation: BoundedGaussianMutation::new(0.05, bounds),
        };
        let mut rng = rng_from_seed(0);
        let parents = vec![vec![0.5, 0.5, 0.5], vec![0.25, 0.75, 0.5]];
        let children = variation.vary(&parents, &mut rng);
        assert_eq!(children.len(), 2);
    }
}
