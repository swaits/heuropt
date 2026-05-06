//! Operators for permutation (`Vec<usize>`) decisions.

use rand::Rng as _;

use crate::core::rng::Rng;
use crate::traits::Variation;

/// Swap two distinct random indices in the first parent (spec §11.4).
///
/// If the parent has length `< 2` the child is returned unchanged.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(42);
/// let mut m = SwapMutation;
/// let parent: Vec<usize> = (0..6).collect();
/// let children = m.vary(std::slice::from_ref(&parent), &mut rng);
/// assert_eq!(children.len(), 1);
/// // Still a permutation of [0, 1, 2, 3, 4, 5]:
/// let mut sorted = children[0].clone();
/// sorted.sort();
/// assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SwapMutation;

impl Variation<Vec<usize>> for SwapMutation {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(
            !parents.is_empty(),
            "SwapMutation requires at least one parent",
        );
        let mut child = parents[0].clone();
        let n = child.len();
        if n >= 2 {
            let i = rng.random_range(0..n);
            let mut j = rng.random_range(0..n);
            while j == i {
                j = rng.random_range(0..n);
            }
            child.swap(i, j);
        }
        vec![child]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rng::rng_from_seed;

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort();
        v
    }

    #[test]
    fn preserves_multiset_contents() {
        let mut m = SwapMutation;
        let mut rng = rng_from_seed(11);
        let parent = vec![0_usize, 1, 2, 3, 4];
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        assert_eq!(children.len(), 1);
        assert_eq!(sorted(children[0].clone()), sorted(parent));
    }

    #[test]
    fn single_element_unchanged() {
        let mut m = SwapMutation;
        let mut rng = rng_from_seed(0);
        let parent = vec![42_usize];
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        assert_eq!(children[0], parent);
    }

    #[test]
    fn two_elements_always_swapped() {
        let mut m = SwapMutation;
        let mut rng = rng_from_seed(0);
        let parent = vec![1_usize, 2];
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        assert_eq!(children[0], vec![2, 1]);
    }
}
