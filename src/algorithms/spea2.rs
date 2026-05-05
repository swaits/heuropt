//! SPEA2 — Strength Pareto Evolutionary Algorithm 2 (Zitzler, Laumanns, Thiele 2001).

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::pareto::dominance::{Dominance, pareto_compare};
use crate::pareto::front::{best_candidate, pareto_front};
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Spea2`].
#[derive(Debug, Clone)]
pub struct Spea2Config {
    /// Constant population size carried across generations.
    pub population_size: usize,
    /// Constant archive size; SPEA2 grows or shrinks it to this exact target.
    pub archive_size: usize,
    /// Number of generations to run.
    pub generations: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for Spea2Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            archive_size: 100,
            generations: 250,
            seed: 42,
        }
    }
}

/// SPEA2 optimizer.
#[derive(Debug, Clone)]
pub struct Spea2<I, V> {
    /// Algorithm configuration.
    pub config: Spea2Config,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Spea2<I, V> {
    /// Construct a `Spea2` optimizer.
    pub fn new(config: Spea2Config, initializer: I, variation: V) -> Self {
        Self {
            config,
            initializer,
            variation,
        }
    }
}

impl<P, I, V> Optimizer<P> for Spea2<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "Spea2 population_size must be greater than 0",
        );
        assert!(
            self.config.archive_size > 0,
            "Spea2 archive_size must be greater than 0",
        );
        let n_pop = self.config.population_size;
        let n_arc = self.config.archive_size;
        let objectives = problem.objectives();
        let mut rng = rng_from_seed(self.config.seed);

        // Initial population.
        let initial_decisions = self.initializer.initialize(n_pop, &mut rng);
        assert_eq!(
            initial_decisions.len(),
            n_pop,
            "SPEA2 initializer must return exactly population_size decisions",
        );
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();
        let mut archive: Vec<Candidate<P::Decision>> = Vec::new();

        for _ in 0..self.config.generations {
            // --- Combine pool, compute fitness ---
            let mut pool: Vec<Candidate<P::Decision>> =
                Vec::with_capacity(population.len() + archive.len());
            pool.append(&mut population);
            pool.append(&mut archive);
            let fitness = compute_fitness(&pool, &objectives);

            // --- Build the next archive ---
            archive = build_archive(&pool, &fitness, &objectives, n_arc);

            // --- Generate offspring from the archive (mating pool) ---
            let archive_fitness = compute_fitness(&archive, &objectives);
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n_pop);
            while offspring_decisions.len() < n_pop {
                let p1 = binary_tournament(&archive_fitness, &mut rng);
                let p2 = binary_tournament(&archive_fitness, &mut rng);
                let parents = vec![archive[p1].decision.clone(), archive[p2].decision.clone()];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(!children.is_empty(), "SPEA2 variation returned no children");
                for child_decision in children {
                    if offspring_decisions.len() >= n_pop {
                        break;
                    }
                    offspring_decisions.push(child_decision);
                }
            }
            let new_population = evaluate_batch(problem, offspring_decisions);
            evaluations += new_population.len();
            population = new_population;
        }

        let front = pareto_front(&archive, &objectives);
        let best = best_candidate(&archive, &objectives);
        OptimizationResult::new(
            Population::new(archive),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

/// SPEA2 fitness: `R(i) + D(i)`, where lower is better.
///
/// `R(i)` is the sum of `S(j)` over all `j` that dominate `i`. `S(j)` is the
/// count of members `j` dominates. `D(i) = 1 / (σ_k + 2)` where `σ_k` is the
/// distance to the k-th nearest neighbor (k = floor(sqrt(N))) in
/// minimization-oriented objective space.
fn compute_fitness<D>(pool: &[Candidate<D>], objectives: &ObjectiveSpace) -> Vec<f64> {
    let n = pool.len();
    if n == 0 {
        return Vec::new();
    }
    let oriented: Vec<Vec<f64>> = pool
        .iter()
        .map(|c| objectives.as_minimization(&c.evaluation.objectives))
        .collect();

    // Strength S(i) = number of members i dominates.
    let mut strength = vec![0_usize; n];
    let mut dominators_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if matches!(
                pareto_compare(&pool[i].evaluation, &pool[j].evaluation, objectives),
                Dominance::Dominates
            ) {
                strength[i] += 1;
                dominators_of[j].push(i);
            }
        }
    }

    // Raw fitness R(i) = sum of S(j) over j that dominate i.
    let raw: Vec<f64> = (0..n)
        .map(|i| dominators_of[i].iter().map(|&j| strength[j] as f64).sum())
        .collect();

    // Density D(i) = 1 / (σ_k + 2). Use kth_nearest distances.
    let k = (n as f64).sqrt() as usize;
    let density: Vec<f64> = (0..n)
        .map(|i| {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean(&oriented[i], &oriented[j]))
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // SPEA2's σ_k is the distance to the k-th nearest neighbor (1-indexed).
            // With k = floor(sqrt(N)), use index (k-1).clamp(0, len-1).
            let idx = if dists.is_empty() {
                return 0.0;
            } else {
                k.saturating_sub(1).min(dists.len() - 1)
            };
            1.0 / (dists[idx] + 2.0)
        })
        .collect();

    raw.into_iter().zip(density).map(|(r, d)| r + d).collect()
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Build the next archive of exactly `target_size` members.
///
/// All non-dominated members of the pool (`fitness < 1.0`) are taken first.
/// If too many, prune by iteratively removing the member with the smallest
/// distance to its nearest neighbor (ties broken by next-nearest, etc.). If
/// too few, fill from the rest sorted by fitness ascending.
fn build_archive<D: Clone>(
    pool: &[Candidate<D>],
    fitness: &[f64],
    objectives: &ObjectiveSpace,
    target_size: usize,
) -> Vec<Candidate<D>> {
    let mut nondom: Vec<usize> = (0..pool.len()).filter(|&i| fitness[i] < 1.0).collect();

    if nondom.len() == target_size {
        return nondom.into_iter().map(|i| pool[i].clone()).collect();
    }

    if nondom.len() < target_size {
        // Fill from dominated members ordered by ascending fitness.
        let mut dominated: Vec<usize> = (0..pool.len()).filter(|&i| fitness[i] >= 1.0).collect();
        dominated.sort_by(|&a, &b| {
            fitness[a]
                .partial_cmp(&fitness[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let needed = target_size - nondom.len();
        nondom.extend(dominated.into_iter().take(needed));
        return nondom.into_iter().map(|i| pool[i].clone()).collect();
    }

    // Truncation: while too large, drop the member with the smallest distance
    // to its nearest neighbor in the current archive.
    let oriented: Vec<Vec<f64>> = nondom
        .iter()
        .map(|&i| objectives.as_minimization(&pool[i].evaluation.objectives))
        .collect();
    let mut alive: Vec<bool> = vec![true; nondom.len()];
    let mut alive_count = nondom.len();
    while alive_count > target_size {
        // Compute per-member sorted distances to other alive members.
        let mut neighbor_dists: Vec<Vec<f64>> = vec![Vec::new(); nondom.len()];
        for i in 0..nondom.len() {
            if !alive[i] {
                continue;
            }
            for j in 0..nondom.len() {
                if !alive[j] || i == j {
                    continue;
                }
                neighbor_dists[i].push(euclidean(&oriented[i], &oriented[j]));
            }
            neighbor_dists[i].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        // Find the alive member whose neighbor-distance vector is lex-smallest.
        let mut victim = usize::MAX;
        for i in 0..nondom.len() {
            if !alive[i] {
                continue;
            }
            if victim == usize::MAX {
                victim = i;
                continue;
            }
            // Lex-compare neighbor distances.
            let cmp = neighbor_dists[i]
                .iter()
                .zip(neighbor_dists[victim].iter())
                .find_map(|(a, b)| {
                    let c = a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
                    if c != std::cmp::Ordering::Equal {
                        Some(c)
                    } else {
                        None
                    }
                })
                .unwrap_or(std::cmp::Ordering::Equal);
            if cmp == std::cmp::Ordering::Less {
                victim = i;
            }
        }
        alive[victim] = false;
        alive_count -= 1;
    }

    nondom
        .into_iter()
        .enumerate()
        .filter_map(|(local, idx)| {
            if alive[local] {
                Some(pool[idx].clone())
            } else {
                None
            }
        })
        .collect()
}

fn binary_tournament(fitness: &[f64], rng: &mut Rng) -> usize {
    let n = fitness.len();
    let a = rng.random_range(0..n);
    let b = rng.random_range(0..n);
    if fitness[a] < fitness[b] {
        a
    } else if fitness[a] > fitness[b] {
        b
    } else if rng.random_bool(0.5) {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{GaussianMutation, RealBounds};
    use crate::tests_support::SchafferN1;

    #[test]
    fn produces_pareto_front() {
        let mut opt = Spea2::new(
            Spea2Config {
                population_size: 30,
                archive_size: 30,
                generations: 10,
                seed: 1,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.3 },
        );
        let r = opt.run(&SchafferN1);
        assert!(!r.pareto_front.is_empty());
        assert_eq!(r.population.len(), 30);
        assert_eq!(r.generations, 10);
    }

    #[test]
    fn archive_size_respected() {
        let mut opt = Spea2::new(
            Spea2Config {
                population_size: 40,
                archive_size: 20,
                generations: 15,
                seed: 2,
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
            GaussianMutation { sigma: 0.3 },
        );
        let r = opt.run(&SchafferN1);
        assert_eq!(r.population.len(), 20);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let make = || {
            Spea2::new(
                Spea2Config {
                    population_size: 20,
                    archive_size: 20,
                    generations: 10,
                    seed: 99,
                },
                RealBounds::new(vec![(-5.0, 5.0)]),
                GaussianMutation { sigma: 0.2 },
            )
        };
        let mut a = make();
        let mut b = make();
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
    #[should_panic(expected = "population_size must be greater than 0")]
    fn zero_population_size_panics() {
        let mut opt = Spea2::new(
            Spea2Config {
                population_size: 0,
                archive_size: 10,
                generations: 1,
                seed: 0,
            },
            RealBounds::new(vec![(-1.0, 1.0)]),
            GaussianMutation { sigma: 0.1 },
        );
        let _ = opt.run(&SchafferN1);
    }
}
