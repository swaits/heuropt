//! Operators for permutation (`Vec<usize>`) decisions.
//!
//! This module ships initializers, crossovers, and mutations that cover the
//! common permutation-encoded problem families: TSP and other strict-permutation
//! problems (cities labeled `0..n`), and JSS-style multiset / operation-string
//! encodings (each job id repeated `k` times).
//!
//! | Operator                          | Strict perm | Multiset / op-string |
//! |-----------------------------------|:-----------:|:--------------------:|
//! | `SwapMutation`                    | ✓           | ✓                    |
//! | `InversionMutation`               | ✓           | ✓                    |
//! | `InsertionMutation`               | ✓           | ✓                    |
//! | `ScrambleMutation`                | ✓           | ✓                    |
//! | `OrderCrossover` (OX)             | ✓           | ✗                    |
//! | `PartiallyMappedCrossover` (PMX)  | ✓           | ✗                    |
//! | `CycleCrossover` (CX)             | ✓           | ✗                    |
//! | `EdgeRecombinationCrossover`      | ✓ (`0..n`)  | ✗                    |
//!
//! The four crossovers all assume *strict* permutations — every value appears
//! exactly once. Combining them with multiset encodings (e.g., the
//! operation-based JSS encoding produced by [`ShuffledMultisetPermutation`])
//! will break the multiset invariant. For multiset encodings, drive variation
//! with the mutation operators alone.

use rand::Rng as _;
use rand::seq::SliceRandom;

use crate::core::rng::Rng;
use crate::traits::{Initializer, Variation};

// ---------------------------------------------------------------------------
// Initializers
// ---------------------------------------------------------------------------

/// Initializer that produces independent random shuffles of `[0..n)`.
///
/// Use for any strict-permutation problem (TSP, single-machine scheduling,
/// QAP, …).
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(7);
/// let mut init = ShuffledPermutation { n: 5 };
/// let pop = init.initialize(3, &mut rng);
/// assert_eq!(pop.len(), 3);
/// for p in &pop {
///     let mut sorted = p.clone();
///     sorted.sort();
///     assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ShuffledPermutation {
    /// Number of distinct elements; each produced shuffle is a permutation of `[0..n)`.
    pub n: usize,
}

impl Initializer<Vec<usize>> for ShuffledPermutation {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
        (0..size)
            .map(|_| {
                let mut p: Vec<usize> = (0..self.n).collect();
                p.shuffle(rng);
                p
            })
            .collect()
    }
}

/// Initializer that produces independent random shuffles of a multiset.
///
/// The multiset is `[0]*r[0] ++ [1]*r[1] ++ ... ++ [k-1]*r[k-1]`, where
/// `r = repeats_per_id`. The canonical use is the operation-based encoding of
/// job-shop scheduling: for `n_jobs × n_machines` JSS, set
/// `repeats_per_id = vec![n_machines; n_jobs]` and each produced shuffle is a
/// valid operation order in which each job appears exactly `n_machines` times.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// // 3 jobs × 2 machines: each job id 0, 1, 2 appears twice.
/// let mut rng = rng_from_seed(42);
/// let mut init = ShuffledMultisetPermutation::new(vec![2, 2, 2]);
/// let pop = init.initialize(4, &mut rng);
/// for p in &pop {
///     assert_eq!(p.len(), 6);
///     let mut counts = [0_usize; 3];
///     for &v in p { counts[v] += 1; }
///     assert_eq!(counts, [2, 2, 2]);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ShuffledMultisetPermutation {
    /// Number of repetitions of each id; element `i` of the multiset appears
    /// `repeats_per_id[i]` times.
    pub repeats_per_id: Vec<usize>,
}

impl ShuffledMultisetPermutation {
    /// Construct a multiset initializer with the given per-id repetition counts.
    pub fn new(repeats_per_id: Vec<usize>) -> Self {
        Self { repeats_per_id }
    }
}

impl Initializer<Vec<usize>> for ShuffledMultisetPermutation {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
        let template: Vec<usize> = self
            .repeats_per_id
            .iter()
            .enumerate()
            .flat_map(|(id, &r)| std::iter::repeat_n(id, r))
            .collect();
        (0..size)
            .map(|_| {
                let mut v = template.clone();
                v.shuffle(rng);
                v
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

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

/// Reverse a random sub-slice `[i, j]` of the first parent.
///
/// Often called *2-opt-style* mutation in TSP literature. Preserves both strict
/// permutations and multiset encodings (since reversing a slice only permutes
/// the values within it). For TSP this is typically the most effective
/// mutation: it directly searches over edge-swap neighbors.
///
/// If the parent has length `< 2` the child is returned unchanged.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(0);
/// let mut m = InversionMutation;
/// let parent: Vec<usize> = (0..8).collect();
/// let children = m.vary(std::slice::from_ref(&parent), &mut rng);
/// let mut sorted = children[0].clone();
/// sorted.sort();
/// assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct InversionMutation;

impl Variation<Vec<usize>> for InversionMutation {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(
            !parents.is_empty(),
            "InversionMutation requires at least one parent",
        );
        let mut child = parents[0].clone();
        let n = child.len();
        if n >= 2 {
            let i = rng.random_range(0..n);
            let j = rng.random_range(0..n);
            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
            child[lo..=hi].reverse();
        }
        vec![child]
    }
}

/// Remove a random element and re-insert it at a different random position
/// ("shift" mutation).
///
/// Preserves both strict permutations and multisets. Particularly effective
/// for sequencing problems (flow shop, JSS) where moving a single
/// operation/job to a new position is a meaningful neighborhood move.
///
/// If the parent has length `< 2` the child is returned unchanged.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(5);
/// let mut m = InsertionMutation;
/// let parent: Vec<usize> = vec![0, 1, 2, 3, 4];
/// let children = m.vary(std::slice::from_ref(&parent), &mut rng);
/// let mut sorted = children[0].clone();
/// sorted.sort();
/// assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct InsertionMutation;

impl Variation<Vec<usize>> for InsertionMutation {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(
            !parents.is_empty(),
            "InsertionMutation requires at least one parent",
        );
        let mut child = parents[0].clone();
        let n = child.len();
        if n >= 2 {
            let from = rng.random_range(0..n);
            let v = child.remove(from);
            let to = rng.random_range(0..=child.len());
            child.insert(to, v);
        }
        vec![child]
    }
}

/// Randomly permute the contents of a random sub-slice.
///
/// Preserves both strict permutations and multisets. Provides a stronger
/// neighborhood than swap/inversion: a single application can rearrange up to
/// `n` positions at once, useful as a diversification operator.
///
/// If the parent has length `< 2` the child is returned unchanged.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(13);
/// let mut m = ScrambleMutation;
/// let parent: Vec<usize> = (0..10).collect();
/// let children = m.vary(std::slice::from_ref(&parent), &mut rng);
/// let mut sorted = children[0].clone();
/// sorted.sort();
/// assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ScrambleMutation;

impl Variation<Vec<usize>> for ScrambleMutation {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(
            !parents.is_empty(),
            "ScrambleMutation requires at least one parent",
        );
        let mut child = parents[0].clone();
        let n = child.len();
        if n >= 2 {
            let i = rng.random_range(0..n);
            let j = rng.random_range(0..n);
            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
            child[lo..=hi].shuffle(rng);
        }
        vec![child]
    }
}

// ---------------------------------------------------------------------------
// Crossovers (strict permutations only)
// ---------------------------------------------------------------------------

/// Order Crossover (OX) — strict-permutation crossover by Davis (1985).
///
/// Pick a random segment `[lo, hi)` from parent A and copy it into the child
/// at those positions. Fill the remaining positions by walking parent B
/// (starting just after `hi`, wrapping), inserting each unused value in order.
///
/// Returns two children: one with parents in the order `(A, B)`, one in the
/// order `(B, A)`. Both share the same crossover points.
///
/// # Panics
/// - If fewer than two parents are supplied.
/// - If the two parents have different lengths.
///
/// # Note
/// OX assumes *strict* permutations — every value appears exactly once. Using
/// it on multiset / operation-string encodings (e.g., JSS) will produce
/// invalid children.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(1);
/// let mut ox = OrderCrossover;
/// let p1: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
/// let p2: Vec<usize> = vec![7, 6, 5, 4, 3, 2, 1, 0];
/// let children = ox.vary(&[p1.clone(), p2.clone()], &mut rng);
/// assert_eq!(children.len(), 2);
/// for c in &children {
///     let mut sorted = c.clone();
///     sorted.sort();
///     assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct OrderCrossover;

impl Variation<Vec<usize>> for OrderCrossover {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(
            parents.len() >= 2,
            "OrderCrossover requires at least 2 parents",
        );
        let p1 = &parents[0];
        let p2 = &parents[1];
        assert_eq!(
            p1.len(),
            p2.len(),
            "OrderCrossover requires equal-length parents",
        );
        let n = p1.len();
        if n < 2 {
            return vec![p1.clone(), p2.clone()];
        }
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let (lo, hi_inclusive) = if i <= j { (i, j) } else { (j, i) };
        let hi = hi_inclusive + 1; // exclusive end
        vec![ox_child(p1, p2, lo, hi), ox_child(p2, p1, lo, hi)]
    }
}

fn ox_child(donor: &[usize], filler: &[usize], lo: usize, hi: usize) -> Vec<usize> {
    let n = donor.len();
    let mut child = vec![0_usize; n];
    let segment = &donor[lo..hi];
    child[lo..hi].copy_from_slice(segment);
    let mut fill_pos = hi % n;
    let mut filler_pos = hi % n;
    let mut placed = hi - lo;
    while placed < n {
        let v = filler[filler_pos];
        if !segment.contains(&v) {
            child[fill_pos] = v;
            fill_pos = (fill_pos + 1) % n;
            placed += 1;
        }
        filler_pos = (filler_pos + 1) % n;
    }
    child
}

/// Partially Mapped Crossover (PMX) — Goldberg & Lingle (1985).
///
/// Builds a child by starting from a copy of parent B, then sliding parent A's
/// segment `[lo, hi)` into place via swaps. The result has A's segment exactly
/// in the same positions and B's order outside the segment, with internal
/// swaps maintaining the permutation property.
///
/// Returns two children: one built from `(A, B)`, one from `(B, A)`.
///
/// # Panics
/// - If fewer than two parents are supplied.
/// - If the two parents have different lengths.
///
/// # Note
/// PMX is strict-permutation only.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(2);
/// let mut pmx = PartiallyMappedCrossover;
/// let p1: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
/// let p2: Vec<usize> = vec![3, 7, 5, 1, 6, 4, 2, 0];
/// let children = pmx.vary(&[p1, p2], &mut rng);
/// assert_eq!(children.len(), 2);
/// for c in &children {
///     let mut sorted = c.clone();
///     sorted.sort();
///     assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct PartiallyMappedCrossover;

impl Variation<Vec<usize>> for PartiallyMappedCrossover {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(
            parents.len() >= 2,
            "PartiallyMappedCrossover requires at least 2 parents",
        );
        let p1 = &parents[0];
        let p2 = &parents[1];
        assert_eq!(
            p1.len(),
            p2.len(),
            "PartiallyMappedCrossover requires equal-length parents",
        );
        let n = p1.len();
        if n < 2 {
            return vec![p1.clone(), p2.clone()];
        }
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let (lo, hi_inclusive) = if i <= j { (i, j) } else { (j, i) };
        let hi = hi_inclusive + 1;
        vec![pmx_child(p1, p2, lo, hi), pmx_child(p2, p1, lo, hi)]
    }
}

fn pmx_child(donor: &[usize], base: &[usize], lo: usize, hi: usize) -> Vec<usize> {
    // Start from a copy of `base`; for each position k in [lo, hi), swap so
    // that child[k] == donor[k]. Each swap preserves the permutation.
    let mut child = base.to_vec();
    for k in lo..hi {
        let v = donor[k];
        if child[k] == v {
            continue;
        }
        let cur = child
            .iter()
            .position(|&x| x == v)
            .expect("permutation invariant: every value must appear");
        child.swap(k, cur);
    }
    child
}

/// Cycle Crossover (CX) — Oliver, Smith & Holland (1987).
///
/// Partitions positions into cycles using the bijection `A[i] ↔ B[i]`. Cycles
/// alternate which parent supplies their values: cycle 1 from A, cycle 2 from
/// B, cycle 3 from A, …
///
/// Returns two children, the second using the opposite cycle assignment.
///
/// # Panics
/// - If fewer than two parents are supplied.
/// - If the two parents have different lengths.
///
/// # Note
/// CX is strict-permutation only and additionally requires that both parents
/// contain exactly the same set of values (otherwise no cycle closes).
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(3);
/// let mut cx = CycleCrossover;
/// let p1: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
/// let p2: Vec<usize> = vec![7, 3, 1, 4, 2, 5, 6, 0];
/// let children = cx.vary(&[p1, p2], &mut rng);
/// assert_eq!(children.len(), 2);
/// for c in &children {
///     let mut sorted = c.clone();
///     sorted.sort();
///     assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CycleCrossover;

impl Variation<Vec<usize>> for CycleCrossover {
    fn vary(&mut self, parents: &[Vec<usize>], _rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(
            parents.len() >= 2,
            "CycleCrossover requires at least 2 parents",
        );
        let p1 = &parents[0];
        let p2 = &parents[1];
        assert_eq!(
            p1.len(),
            p2.len(),
            "CycleCrossover requires equal-length parents",
        );
        let n = p1.len();
        if n == 0 {
            return vec![Vec::new(), Vec::new()];
        }
        vec![cx_child(p1, p2), cx_child(p2, p1)]
    }
}

fn cx_child(start_parent: &[usize], other_parent: &[usize]) -> Vec<usize> {
    let n = start_parent.len();
    let mut child = vec![0_usize; n];
    let mut visited = vec![false; n];
    let mut cycle_index = 0_usize;
    for seed in 0..n {
        if visited[seed] {
            continue;
        }
        let (from, switch_through) = if cycle_index % 2 == 0 {
            (start_parent, other_parent)
        } else {
            (other_parent, start_parent)
        };
        let mut k = seed;
        loop {
            if visited[k] {
                break;
            }
            visited[k] = true;
            child[k] = from[k];
            let next_val = switch_through[k];
            let next_pos = from
                .iter()
                .position(|&x| x == next_val)
                .expect("CycleCrossover: parents must share the same value multiset");
            k = next_pos;
        }
        cycle_index += 1;
    }
    child
}

/// Edge Recombination Crossover (ERX) — Whitley, Starkweather & Fuquay (1989).
///
/// The standard high-quality TSP crossover. Builds an adjacency table listing
/// each city's neighbors across both parent tours, then walks the table
/// greedily: at each step the next city is the unvisited neighbor of the
/// current city that has the fewest remaining edges (random tie-break).
/// Dead-ends are filled with any unvisited city.
///
/// Two children are produced by starting at parents A's and parents B's first
/// city respectively.
///
/// # Panics
/// - If fewer than two parents are supplied.
/// - If the two parents have different lengths.
///
/// # Note
/// ERX assumes cities are labeled `0..n` (the standard TSP convention) for
/// O(1) adjacency-table indexing. Strict permutations only.
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// let mut rng = rng_from_seed(4);
/// let mut erx = EdgeRecombinationCrossover;
/// let p1: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
/// let p2: Vec<usize> = vec![5, 3, 1, 4, 2, 0];
/// let children = erx.vary(&[p1, p2], &mut rng);
/// assert_eq!(children.len(), 2);
/// for c in &children {
///     let mut sorted = c.clone();
///     sorted.sort();
///     assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeRecombinationCrossover;

impl Variation<Vec<usize>> for EdgeRecombinationCrossover {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(
            parents.len() >= 2,
            "EdgeRecombinationCrossover requires at least 2 parents",
        );
        let p1 = &parents[0];
        let p2 = &parents[1];
        assert_eq!(
            p1.len(),
            p2.len(),
            "EdgeRecombinationCrossover requires equal-length parents",
        );
        let n = p1.len();
        if n == 0 {
            return vec![Vec::new(), Vec::new()];
        }
        vec![erx_child(p1, p2, p1[0], rng), erx_child(p1, p2, p2[0], rng)]
    }
}

fn erx_child(p1: &[usize], p2: &[usize], start: usize, rng: &mut Rng) -> Vec<usize> {
    let n = p1.len();
    // adj[v] = neighbors of city v across both parent tours (no duplicates).
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for tour in [p1, p2] {
        for i in 0..n {
            let cur = tour[i];
            let prev = tour[(i + n - 1) % n];
            let next = tour[(i + 1) % n];
            debug_assert!(
                cur < n && prev < n && next < n,
                "ERX requires cities labeled 0..n",
            );
            for nb in [prev, next] {
                if !adj[cur].contains(&nb) {
                    adj[cur].push(nb);
                }
            }
        }
    }
    let mut visited = vec![false; n];
    let mut child = Vec::with_capacity(n);
    let mut current = start;
    for _ in 0..n {
        child.push(current);
        visited[current] = true;
        // Remove `current` from every adjacency list so it isn't picked again.
        for list in adj.iter_mut() {
            list.retain(|&x| x != current);
        }
        if child.len() == n {
            break;
        }
        let neighbors: Vec<usize> = adj[current]
            .iter()
            .copied()
            .filter(|&c| !visited[c])
            .collect();
        let next = if neighbors.is_empty() {
            // Dead-end: pick any unvisited city. Iterating in index order gives
            // a deterministic fallback; ERX is rarely sensitive to this choice.
            (0..n)
                .find(|&c| !visited[c])
                .expect("at least one unvisited city remains")
        } else {
            let min_deg = neighbors
                .iter()
                .map(|&c| adj[c].len())
                .min()
                .expect("non-empty neighbors");
            let ties: Vec<usize> = neighbors
                .into_iter()
                .filter(|&c| adj[c].len() == min_deg)
                .collect();
            if ties.len() == 1 {
                ties[0]
            } else {
                ties[rng.random_range(0..ties.len())]
            }
        };
        current = next;
    }
    child
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rng::rng_from_seed;

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort();
        v
    }

    fn is_strict_perm(v: &[usize]) -> bool {
        let n = v.len();
        let mut seen = vec![false; n];
        for &x in v {
            if x >= n || seen[x] {
                return false;
            }
            seen[x] = true;
        }
        true
    }

    fn multiset_eq(a: &[usize], b: &[usize]) -> bool {
        sorted(a.to_vec()) == sorted(b.to_vec())
    }

    // -------- SwapMutation (kept) --------

    #[test]
    fn swap_preserves_multiset_contents() {
        let mut m = SwapMutation;
        let mut rng = rng_from_seed(11);
        let parent = vec![0_usize, 1, 2, 3, 4];
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        assert_eq!(children.len(), 1);
        assert_eq!(sorted(children[0].clone()), sorted(parent));
    }

    #[test]
    fn swap_single_element_unchanged() {
        let mut m = SwapMutation;
        let mut rng = rng_from_seed(0);
        let parent = vec![42_usize];
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        assert_eq!(children[0], parent);
    }

    #[test]
    fn swap_two_elements_always_swapped() {
        let mut m = SwapMutation;
        let mut rng = rng_from_seed(0);
        let parent = vec![1_usize, 2];
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        assert_eq!(children[0], vec![2, 1]);
    }

    // -------- ShuffledPermutation --------

    #[test]
    fn shuffled_permutation_returns_requested_count() {
        let mut init = ShuffledPermutation { n: 7 };
        let mut rng = rng_from_seed(1);
        let pop = init.initialize(10, &mut rng);
        assert_eq!(pop.len(), 10);
        for p in &pop {
            assert!(is_strict_perm(p));
            assert_eq!(p.len(), 7);
        }
    }

    #[test]
    fn shuffled_permutation_n0_yields_empty_vecs() {
        let mut init = ShuffledPermutation { n: 0 };
        let mut rng = rng_from_seed(1);
        let pop = init.initialize(3, &mut rng);
        assert_eq!(pop.len(), 3);
        assert!(pop.iter().all(|p| p.is_empty()));
    }

    // -------- ShuffledMultisetPermutation --------

    #[test]
    fn shuffled_multiset_preserves_counts() {
        let repeats = vec![3_usize, 2, 4, 1]; // total 10
        let mut init = ShuffledMultisetPermutation::new(repeats.clone());
        let mut rng = rng_from_seed(2);
        let pop = init.initialize(5, &mut rng);
        for p in &pop {
            assert_eq!(p.len(), 10);
            let mut counts = [0_usize; 4];
            for &v in p {
                counts[v] += 1;
            }
            assert_eq!(counts, [3, 2, 4, 1]);
        }
    }

    // -------- InversionMutation --------

    #[test]
    fn inversion_preserves_strict_permutation() {
        let mut m = InversionMutation;
        let parent: Vec<usize> = (0..9).collect();
        for seed in 0..50 {
            let mut rng = rng_from_seed(seed);
            let children = m.vary(std::slice::from_ref(&parent), &mut rng);
            assert_eq!(children.len(), 1);
            assert!(is_strict_perm(&children[0]));
        }
    }

    #[test]
    fn inversion_preserves_multiset() {
        let mut m = InversionMutation;
        let parent = vec![0_usize, 0, 1, 1, 2, 2];
        for seed in 0..50 {
            let mut rng = rng_from_seed(seed);
            let children = m.vary(std::slice::from_ref(&parent), &mut rng);
            assert!(multiset_eq(&children[0], &parent));
        }
    }

    #[test]
    fn inversion_single_element_unchanged() {
        let mut m = InversionMutation;
        let mut rng = rng_from_seed(0);
        let parent = vec![42_usize];
        let children = m.vary(std::slice::from_ref(&parent), &mut rng);
        assert_eq!(children[0], parent);
    }

    // -------- InsertionMutation --------

    #[test]
    fn insertion_preserves_strict_permutation() {
        let mut m = InsertionMutation;
        let parent: Vec<usize> = (0..7).collect();
        for seed in 0..50 {
            let mut rng = rng_from_seed(seed);
            let children = m.vary(std::slice::from_ref(&parent), &mut rng);
            assert_eq!(children[0].len(), 7);
            assert!(is_strict_perm(&children[0]));
        }
    }

    #[test]
    fn insertion_preserves_multiset() {
        let mut m = InsertionMutation;
        let parent = vec![0_usize, 1, 1, 2, 2, 2];
        for seed in 0..50 {
            let mut rng = rng_from_seed(seed);
            let children = m.vary(std::slice::from_ref(&parent), &mut rng);
            assert!(multiset_eq(&children[0], &parent));
        }
    }

    // -------- ScrambleMutation --------

    #[test]
    fn scramble_preserves_strict_permutation() {
        let mut m = ScrambleMutation;
        let parent: Vec<usize> = (0..8).collect();
        for seed in 0..50 {
            let mut rng = rng_from_seed(seed);
            let children = m.vary(std::slice::from_ref(&parent), &mut rng);
            assert!(is_strict_perm(&children[0]));
        }
    }

    #[test]
    fn scramble_preserves_multiset() {
        let mut m = ScrambleMutation;
        let parent = vec![0_usize, 1, 1, 2, 3, 3, 3];
        for seed in 0..50 {
            let mut rng = rng_from_seed(seed);
            let children = m.vary(std::slice::from_ref(&parent), &mut rng);
            assert!(multiset_eq(&children[0], &parent));
        }
    }

    // -------- OrderCrossover (OX) --------

    #[test]
    fn ox_returns_two_valid_permutations() {
        let mut ox = OrderCrossover;
        let p1: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let p2: Vec<usize> = vec![3, 7, 5, 1, 6, 4, 2, 0];
        for seed in 0..30 {
            let mut rng = rng_from_seed(seed);
            let children = ox.vary(&[p1.clone(), p2.clone()], &mut rng);
            assert_eq!(children.len(), 2);
            for c in &children {
                assert!(is_strict_perm(c), "child not a permutation: {:?}", c);
            }
        }
    }

    #[test]
    fn ox_with_identical_parents_reproduces_parent() {
        let mut ox = OrderCrossover;
        let p: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
        let mut rng = rng_from_seed(0);
        let children = ox.vary(&[p.clone(), p.clone()], &mut rng);
        assert_eq!(children, vec![p.clone(), p]);
    }

    // -------- PartiallyMappedCrossover (PMX) --------

    #[test]
    fn pmx_returns_two_valid_permutations() {
        let mut pmx = PartiallyMappedCrossover;
        let p1: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let p2: Vec<usize> = vec![3, 7, 5, 1, 6, 4, 2, 0];
        for seed in 0..30 {
            let mut rng = rng_from_seed(seed);
            let children = pmx.vary(&[p1.clone(), p2.clone()], &mut rng);
            assert_eq!(children.len(), 2);
            for c in &children {
                assert!(is_strict_perm(c), "child not a permutation: {:?}", c);
            }
        }
    }

    #[test]
    fn pmx_with_identical_parents_reproduces_parent() {
        let mut pmx = PartiallyMappedCrossover;
        let p: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
        let mut rng = rng_from_seed(0);
        let children = pmx.vary(&[p.clone(), p.clone()], &mut rng);
        assert_eq!(children, vec![p.clone(), p]);
    }

    // -------- CycleCrossover (CX) --------

    #[test]
    fn cx_returns_two_valid_permutations() {
        let mut cx = CycleCrossover;
        let p1: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let p2: Vec<usize> = vec![7, 3, 1, 4, 2, 5, 6, 0];
        let mut rng = rng_from_seed(0);
        let children = cx.vary(&[p1, p2], &mut rng);
        assert_eq!(children.len(), 2);
        for c in &children {
            assert!(is_strict_perm(c));
        }
    }

    #[test]
    fn cx_with_identical_parents_reproduces_parent() {
        let mut cx = CycleCrossover;
        let p: Vec<usize> = vec![0, 1, 2, 3, 4];
        let mut rng = rng_from_seed(0);
        let children = cx.vary(&[p.clone(), p.clone()], &mut rng);
        assert_eq!(children, vec![p.clone(), p]);
    }

    // -------- EdgeRecombinationCrossover (ERX) --------

    #[test]
    fn erx_returns_two_valid_permutations() {
        let mut erx = EdgeRecombinationCrossover;
        let p1: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let p2: Vec<usize> = vec![3, 5, 7, 1, 6, 4, 2, 0];
        for seed in 0..30 {
            let mut rng = rng_from_seed(seed);
            let children = erx.vary(&[p1.clone(), p2.clone()], &mut rng);
            assert_eq!(children.len(), 2);
            for c in &children {
                assert!(is_strict_perm(c), "child not a permutation: {:?}", c);
            }
        }
    }

    #[test]
    fn erx_with_identical_parents_walks_the_same_tour() {
        // When both parents are identical, the adjacency table reproduces
        // the parent's edge set; the only reachable tour is the parent
        // (possibly reversed). Either way, the multiset is preserved.
        let mut erx = EdgeRecombinationCrossover;
        let p: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
        let mut rng = rng_from_seed(0);
        let children = erx.vary(&[p.clone(), p.clone()], &mut rng);
        for c in &children {
            assert!(is_strict_perm(c));
            assert_eq!(c.len(), p.len());
        }
    }
}
