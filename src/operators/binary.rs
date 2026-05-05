//! Operators for binary (`Vec<bool>`) decisions.

use rand::Rng as _;

use crate::core::rng::Rng;
use crate::traits::Variation;

/// Flip each bit of the first parent independently with probability `p`.
///
/// Always returns exactly one child (spec §11.3). Panics if `probability` is
/// outside `[0.0, 1.0]` or if no parents are provided.
#[derive(Debug, Clone)]
pub struct BitFlipMutation {
    /// Per-bit flip probability. Must lie in `[0.0, 1.0]`.
    pub probability: f64,
}

impl Variation<Vec<bool>> for BitFlipMutation {
    fn vary(&mut self, parents: &[Vec<bool>], rng: &mut Rng) -> Vec<Vec<bool>> {
        assert!(
            (0.0..=1.0).contains(&self.probability),
            "BitFlipMutation probability must be in [0.0, 1.0]",
        );
        assert!(
            !parents.is_empty(),
            "BitFlipMutation requires at least one parent",
        );
        let mut child = parents[0].clone();
        for bit in child.iter_mut() {
            if rng.random_bool(self.probability) {
                *bit = !*bit;
            }
        }
        vec![child]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rng::rng_from_seed;

    #[test]
    fn probability_zero_changes_nothing() {
        let mut m = BitFlipMutation { probability: 0.0 };
        let mut rng = rng_from_seed(0);
        let parent = vec![false, true, false, true, true];
        let children = m.vary(&[parent.clone()], &mut rng);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0], parent);
    }

    #[test]
    fn probability_one_flips_everything() {
        let mut m = BitFlipMutation { probability: 1.0 };
        let mut rng = rng_from_seed(0);
        let parent = vec![false, true, false, true, true];
        let children = m.vary(&[parent.clone()], &mut rng);
        let expected: Vec<bool> = parent.iter().map(|b| !b).collect();
        assert_eq!(children[0], expected);
    }

    #[test]
    #[should_panic(expected = "must be in [0.0, 1.0]")]
    fn probability_above_one_panics() {
        let mut m = BitFlipMutation { probability: 1.5 };
        let mut rng = rng_from_seed(0);
        m.vary(&[vec![true]], &mut rng);
    }
}
