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
///
/// # Example
///
/// ```
/// use heuropt::prelude::*;
///
/// struct Schaffer;
/// impl Problem for Schaffer {
///     type Decision = Vec<f64>;
///     fn objectives(&self) -> ObjectiveSpace {
///         ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
///     }
///     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
///         Evaluation::new(vec![x[0] * x[0], (x[0] - 2.0).powi(2)])
///     }
/// }
///
/// let bounds = vec![(-5.0_f64, 5.0_f64)];
/// let mut opt = PesaII::new(
///     PesaIIConfig {
///         population_size: 20,
///         archive_size: 30,
///         generations: 20,
///         grid_divisions: 8,
///         seed: 42,
///     },
///     RealBounds::new(bounds.clone()),
///     CompositeVariation {
///         crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
///         mutation:  PolynomialMutation::new(bounds, 20.0, 1.0),
///     },
/// );
/// let r = opt.run(&Schaffer);
/// assert!(!r.pareto_front.is_empty());
/// ```
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
        Self {
            config,
            initializer,
            variation,
        }
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
        assert!(
            self.config.population_size > 0,
            "PesaII population_size must be > 0"
        );
        assert!(
            self.config.archive_size > 0,
            "PesaII archive_size must be > 0"
        );
        assert!(
            self.config.grid_divisions >= 1,
            "PesaII grid_divisions must be >= 1"
        );
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
        truncate_by_grid(
            &mut archive,
            self.config.archive_size,
            self.config.grid_divisions,
        );

        for _ in 0..self.config.generations {
            // Build grid + box counts on the archive.
            let (boxes, counts) = build_grid(&archive, &objectives, self.config.grid_divisions);

            // Generate offspring via region-based selection on the archive.
            let mut offspring: Vec<Candidate<P::Decision>> = Vec::with_capacity(n);
            while offspring.len() < n {
                let p1 = region_tournament(&archive, &boxes, &counts, &mut rng);
                let p2 = region_tournament(&archive, &boxes, &counts, &mut rng);
                let parents = vec![
                    archive.members()[p1].decision.clone(),
                    archive.members()[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(
                    !children.is_empty(),
                    "PesaII variation returned no children"
                );
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
            truncate_by_grid(
                &mut archive,
                self.config.archive_size,
                self.config.grid_divisions,
            );
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

#[cfg(feature = "async")]
impl<I, V> PesaII<I, V> {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations of the initial
    /// population. Per-step evaluations are sequential to preserve the
    /// algorithm's exact RNG sequencing.
    pub async fn run_async<P>(
        &mut self,
        problem: &P,
        concurrency: usize,
    ) -> OptimizationResult<P::Decision>
    where
        P: crate::core::async_problem::AsyncProblem,
        I: Initializer<P::Decision>,
        V: Variation<P::Decision>,
    {
        use crate::algorithms::parallel_eval_async::evaluate_batch_async;

        assert!(
            self.config.population_size > 0,
            "PesaII population_size must be > 0"
        );
        assert!(
            self.config.archive_size > 0,
            "PesaII archive_size must be > 0"
        );
        assert!(
            self.config.grid_divisions >= 1,
            "PesaII grid_divisions must be >= 1"
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        let initial_decisions = self.initializer.initialize(n, &mut rng);
        let mut internal: Vec<Candidate<P::Decision>> =
            evaluate_batch_async(problem, initial_decisions, concurrency).await;
        let mut evaluations = internal.len();

        let mut archive = ParetoArchive::new(objectives.clone());
        for c in &internal {
            archive.insert(c.clone());
        }
        truncate_by_grid(
            &mut archive,
            self.config.archive_size,
            self.config.grid_divisions,
        );

        for _ in 0..self.config.generations {
            let (boxes, counts) = build_grid(&archive, &objectives, self.config.grid_divisions);

            let mut offspring: Vec<Candidate<P::Decision>> = Vec::with_capacity(n);
            while offspring.len() < n {
                let p1 = region_tournament(&archive, &boxes, &counts, &mut rng);
                let p2 = region_tournament(&archive, &boxes, &counts, &mut rng);
                let parents = vec![
                    archive.members()[p1].decision.clone(),
                    archive.members()[p2].decision.clone(),
                ];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(
                    !children.is_empty(),
                    "PesaII variation returned no children"
                );
                for child in children {
                    if offspring.len() >= n {
                        break;
                    }
                    let eval = problem.evaluate_async(&child).await;
                    evaluations += 1;
                    offspring.push(Candidate::new(child, eval));
                }
            }

            for c in &offspring {
                archive.insert(c.clone());
            }
            truncate_by_grid(
                &mut archive,
                self.config.archive_size,
                self.config.grid_divisions,
            );
            internal = offspring;
        }

        let _ = internal;
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
fn truncate_by_grid<D: Clone>(archive: &mut ParetoArchive<D>, max_size: usize, divisions: usize) {
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

impl<I, V> crate::traits::AlgorithmInfo for PesaII<I, V> {
    fn name(&self) -> &'static str {
        "PESA-II"
    }
    fn full_name(&self) -> &'static str {
        "Pareto Envelope-based Selection Algorithm II"
    }
    fn seed(&self) -> Option<u64> {
        Some(self.config.seed)
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
        let oa: Vec<Vec<f64>> = ra
            .pareto_front
            .iter()
            .map(|c| c.evaluation.objectives.clone())
            .collect();
        let ob: Vec<Vec<f64>> = rb
            .pareto_front
            .iter()
            .map(|c| c.evaluation.objectives.clone())
            .collect();
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

    // ---- Mutation-test pinned helpers --------------------------------------

    use crate::core::candidate::Candidate;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Objective, ObjectiveSpace};

    fn space2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    #[test]
    fn build_grid_empty_archive_is_empty() {
        let archive = ParetoArchive::<u32>::new(space2());
        let (boxes, counts) = build_grid(&archive, &space2(), 4);
        assert!(boxes.is_empty());
        assert!(counts.is_empty());
    }

    #[test]
    fn build_grid_assigns_corner_points_to_distinct_boxes() {
        let mut archive = ParetoArchive::<u32>::new(space2());
        // Three non-dominated corner points span the grid extremes.
        archive.insert(Candidate::new(1u32, Evaluation::new(vec![0.0, 4.0])));
        archive.insert(Candidate::new(2u32, Evaluation::new(vec![2.0, 2.0])));
        archive.insert(Candidate::new(3u32, Evaluation::new(vec![4.0, 0.0])));
        let (boxes, counts) = build_grid(&archive, &space2(), 4);
        assert_eq!(boxes.len(), 3);
        // The min and max corners land in different boxes — total count
        // across all boxes equals the member count.
        let total: usize = counts.values().sum();
        assert_eq!(total, 3);
        // The two extreme points are in different boxes (grid spreads them).
        assert_ne!(boxes[0], boxes[2]);
    }

    #[test]
    fn region_tournament_prefers_less_crowded_box() {
        use crate::core::rng::rng_from_seed;
        // Members 0 and 1 share a crowded box (count 2); member 2 is alone.
        let mut archive = ParetoArchive::<u32>::new(space2());
        archive.insert(Candidate::new(1u32, Evaluation::new(vec![0.0, 4.0])));
        archive.insert(Candidate::new(2u32, Evaluation::new(vec![2.0, 2.0])));
        archive.insert(Candidate::new(3u32, Evaluation::new(vec![4.0, 0.0])));
        // Hand-build boxes/counts where index 2 is in a singleton box and
        // indices 0,1 share a crowded box.
        let boxes = vec![vec![0usize, 0], vec![0usize, 0], vec![3usize, 3]];
        let mut counts = std::collections::BTreeMap::new();
        counts.insert(vec![0usize, 0], 2usize);
        counts.insert(vec![3usize, 3], 1usize);
        // Across many seeds, the less-crowded index (2) must win whenever
        // the two random draws differ between the crowded/uncrowded boxes.
        let mut picked_uncrowded = 0;
        for seed in 0..300 {
            let mut rng = rng_from_seed(seed);
            if region_tournament(&archive, &boxes, &counts, &mut rng) == 2 {
                picked_uncrowded += 1;
            }
        }
        // Index 2 wins whenever it's drawn against 0 or 1, plus half its
        // self-draws — clear majority.
        assert!(picked_uncrowded > 150, "uncrowded picked {picked_uncrowded}/300");
    }
}
