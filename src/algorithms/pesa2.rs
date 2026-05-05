//! `PesaII` — Corne, Jerram, Knowles & Oates 2001 Pareto Envelope-based
//! Selection Algorithm II.

use std::collections::BTreeMap;

use rand::Rng as _;

use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::pareto::archive::ParetoArchive;
use crate::pareto::front::{best_candidate, pareto_front};
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`PesaII`].
#[derive(Debug, Clone)]
pub struct PesaIIConfig {
    /// Internal population size (used for variation).
    pub population_size: usize,
    /// External non-dominated archive cap.
    pub archive_size: usize,
    /// Number of generations.
    pub generations: usize,
    /// Number of grid divisions per objective axis.
    pub grid_divisions: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for PesaIIConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            archive_size: 100,
            generations: 250,
            grid_divisions: 16,
            seed: 42,
        }
    }
}

/// Pareto Envelope-based Selection Algorithm II.
///
/// Maintains an internal population (used to drive variation) and an
/// external non-dominated archive. Selection biases toward members in
/// sparsely-populated grid boxes so the front spreads out.
#[derive(Debug, Clone)]
pub struct PesaII<I, V> {
    /// Algorithm configuration.
    pub config: PesaIIConfig,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> PesaII<I, V> {
    /// Construct a `PesaII`.
    pub fn new(config: PesaIIConfig, initializer: I, variation: V) -> Self {
        Self { config, initializer, variation }
    }
}

impl<P, I, V> Optimizer<P> for PesaII<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(self.config.population_size > 0, "PesaII population_size must be > 0");
        assert!(self.config.archive_size > 0, "PesaII archive_size must be > 0");
        assert!(self.config.grid_divisions >= 1, "PesaII grid_divisions must be >= 1");
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        // Initial internal population.
        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut internal: Vec<Candidate<P::Decision>> = initial_decisions
            .into_iter()
            .map(|d| {
                let e = problem.evaluate(&d);
                Candidate::new(d, e)
            })
            .collect();
        let mut evaluations = internal.len();

        // External archive.
        let mut archive = ParetoArchive::new(objectives.clone());
        for c in &internal {
            archive.insert(c.clone());
        }
        truncate_by_grid(&mut archive, self.config.archive_size, self.config.grid_divisions);

        for _ in 0..self.config.generations {
            // Build grid + box counts on the archive.
            let (boxes, counts) = build_grid(&archive, &objectives, self.config.grid_divisions);

            // Generate offspring via region-based selection on the archive.
            let mut offspring: Vec<Candidate<P::Decision>> = Vec::with_capacity(n);
            while offspring.len() < n {
                let p1 = region_tournament(&archive, &boxes, &counts, &mut rng);
                let p2 = region_tournament(&archive, &boxes, &counts, &mut rng);
                let parents = vec![archive.members()[p1].decision.clone(), archive.members()[p2].decision.clone()];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "PesaII variation returned no children");
                for child in children {
                    if offspring.len() >= n {
                        break;
                    }
                    let eval = problem.evaluate(&child);
                    evaluations += 1;
                    offspring.push(Candidate::new(child, eval));
                }
            }

            // Internal pop becomes the offspring; archive gets every
            // non-dominated offspring.
            for c in &offspring {
                archive.insert(c.clone());
            }
            truncate_by_grid(&mut archive, self.config.archive_size, self.config.grid_divisions);
            internal = offspring;
        }

        let _ = internal; // not directly returned
        let members = archive.into_vec();
        let front = pareto_front(&members, &objectives);
        let best = best_candidate(&members, &objectives);
        OptimizationResult::new(
            Population::new(members),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

/// Compute per-member box index (M-tuple of grid coordinates) and the
/// population count of each occupied box.
fn build_grid<D: Clone>(
    archive: &ParetoArchive<D>,
    objectives: &ObjectiveSpace,
    divisions: usize,
) -> (Vec<Vec<usize>>, BTreeMap<Vec<usize>, usize>) {
    let m = objectives.len();
    let members = archive.members();
    if members.is_empty() {
        return (Vec::new(), BTreeMap::new());
    }
    let oriented: Vec<Vec<f64>> = members
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();
    let mut lo = vec![f64::INFINITY; m];
    let mut hi = vec![f64::NEG_INFINITY; m];
    for o in &oriented {
        for k in 0..m {
            if o[k] < lo[k] {
                lo[k] = o[k];
            }
            if o[k] > hi[k] {
                hi[k] = o[k];
            }
        }
    }
    let mut boxes: Vec<Vec<usize>> = Vec::with_capacity(members.len());
    for o in &oriented {
        let mut box_idx = Vec::with_capacity(m);
        for k in 0..m {
            let span = (hi[k] - lo[k]).max(1e-12);
            let frac = ((o[k] - lo[k]) / span).clamp(0.0, 1.0 - 1e-9);
            box_idx.push((frac * divisions as f64) as usize);
        }
        boxes.push(box_idx);
    }
    let mut counts: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    for b in &boxes {
        *counts.entry(b.clone()).or_insert(0) += 1;
    }
    (boxes, counts)
}

/// Pick a member by region-based tournament: take two random members,
/// prefer the one whose grid box is less crowded.
fn region_tournament<D: Clone>(
    archive: &ParetoArchive<D>,
    boxes: &[Vec<usize>],
    counts: &BTreeMap<Vec<usize>, usize>,
    rng: &mut Rng,
) -> usize {
    let n = archive.members().len();
    let a = rng.random_range(0..n);
    let b = rng.random_range(0..n);
    let ca = counts.get(&boxes[a]).copied().unwrap_or(1);
    let cb = counts.get(&boxes[b]).copied().unwrap_or(1);
    if ca < cb {
        a
    } else if cb < ca {
        b
    } else if rng.random_bool(0.5) {
        a
    } else {
        b
    }
}

/// Truncate the archive to `max_size` by repeatedly evicting a uniform-random
/// member of the most-occupied grid box (PESA-II's standard approach).
fn truncate_by_grid<D: Clone>(
    archive: &mut ParetoArchive<D>,
    max_size: usize,
    divisions: usize,
) {
    while archive.members().len() > max_size {
        let objectives = archive.objectives.clone();
        let (boxes, counts) = build_grid(archive, &objectives, divisions);
        // Find the most-crowded box.
        let max_count = counts.values().copied().max().unwrap_or(0);
        if max_count <= 1 {
            // No crowding to break: just truncate.
            archive.truncate(max_size);
            break;
        }
        // Indices in that box.
        let crowded_box = counts
            .iter()
            .find(|&(_, &c)| c == max_count)
            .map(|(b, _)| b.clone())
            .unwrap();
        let candidates: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| **b == crowded_box)
            .map(|(i, _)| i)
            .collect();
        // Use a fixed seed-derived RNG would be ideal, but truncation is
        // called from the main RNG indirectly; use a deterministic pick
        // (the first candidate) to avoid sneaking nondeterminism in.
        let evict = *candidates.first().expect("non-empty crowded box");
        archive.members.swap_remove(evict);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{
        CompositeVariation, PolynomialMutation, RealBounds, SimulatedBinaryCrossover,
    };
    use crate::tests_support::SchafferN1;

    fn make_optimizer(
        seed: u64,
    ) -> PesaII<RealBounds, CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>> {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        PesaII::new(
            PesaIIConfig {
                population_size: 20,
                archive_size: 30,
                generations: 15,
                grid_divisions: 8,
                seed,
            },
            initializer,
            variation,
        )
    }

    #[test]
    fn produces_pareto_front() {
        let mut opt = make_optimizer(1);
        let r = opt.run(&SchafferN1);
        assert!(!r.pareto_front.is_empty());
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut a = make_optimizer(99);
        let mut b = make_optimizer(99);
        let ra = a.run(&SchafferN1);
        let rb = b.run(&SchafferN1);
        let oa: Vec<Vec<f64>> =
            ra.pareto_front.iter().map(|c| c.evaluation.objectives.clone()).collect();
        let ob: Vec<Vec<f64>> =
            rb.pareto_front.iter().map(|c| c.evaluation.objectives.clone()).collect();
        assert_eq!(oa, ob);
    }

    #[test]
    #[should_panic(expected = "archive_size must be > 0")]
    fn zero_archive_size_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = PesaII::new(
            PesaIIConfig {
                population_size: 4,
                archive_size: 0,
                generations: 1,
                grid_divisions: 4,
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }

}
