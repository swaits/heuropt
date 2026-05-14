//! SPEA2 — Strength Pareto Evolutionary Algorithm 2 (Zitzler, Laumanns, Thiele 2001).

use rand::Rng as _;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
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
///
/// Strength Pareto Evolutionary Algorithm 2: combines a strength-based
/// dominance score with a k-th nearest-neighbor density estimate. Maintains
/// an external archive separate from the working population.
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
/// let mut opt = Spea2::new(
///     Spea2Config { population_size: 30, archive_size: 30, generations: 20, seed: 42 },
///     RealBounds::new(bounds.clone()),
///     CompositeVariation {
///         crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
///         mutation:  PolynomialMutation::new(bounds, 20.0, 1.0),
///     },
/// );
/// let r = opt.run(&Schaffer);
/// assert_eq!(r.population.len(), 30);
/// assert!(!r.pareto_front.is_empty());
/// ```
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

#[cfg(feature = "async")]
impl<I, V> Spea2<I, V> {
    /// Async version of [`Optimizer::run`] — drives evaluations through
    /// the user-chosen async runtime. Available only with the `async`
    /// feature.
    ///
    /// `concurrency` bounds in-flight evaluations per batch.
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

        let initial_decisions = self.initializer.initialize(n_pop, &mut rng);
        assert_eq!(
            initial_decisions.len(),
            n_pop,
            "SPEA2 initializer must return exactly population_size decisions",
        );
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch_async(problem, initial_decisions, concurrency).await;
        let mut evaluations = population.len();
        let mut archive: Vec<Candidate<P::Decision>> = Vec::new();

        for _ in 0..self.config.generations {
            let mut pool: Vec<Candidate<P::Decision>> =
                Vec::with_capacity(population.len() + archive.len());
            pool.append(&mut population);
            pool.append(&mut archive);
            let fitness = compute_fitness(&pool, &objectives);

            archive = build_archive(&pool, &fitness, &objectives, n_arc);

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
            let new_population =
                evaluate_batch_async(problem, offspring_decisions, concurrency).await;
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
    let feasible: Vec<bool> = pool.iter().map(|c| c.evaluation.is_feasible()).collect();
    let violation: Vec<f64> = pool
        .iter()
        .map(|c| c.evaluation.constraint_violation)
        .collect();
    let m = objectives.len();

    // Strength S(i) = number of members i dominates. Inline `pareto_compare`
    // against the cached oriented/feasibility arrays — the by-pair call into
    // `pareto_compare` would otherwise allocate two fresh `Vec<f64>`s per
    // pair via `as_minimization`, dominating per-generation cost on
    // population sizes ≥ 80.
    let mut strength = vec![0_usize; n];
    let mut dominators_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let ai_feasible = feasible[i];
        let ai_violation = violation[i];
        let ai = &oriented[i];
        for j in 0..n {
            if i == j {
                continue;
            }
            let bi_feasible = feasible[j];
            let i_dominates_j = match (ai_feasible, bi_feasible) {
                (true, false) => true,
                (false, true) => false,
                (false, false) => ai_violation < violation[j],
                (true, true) => {
                    let bj = &oriented[j];
                    let mut a_better_anywhere = false;
                    let mut b_better_anywhere = false;
                    for k in 0..m {
                        let av = ai[k];
                        let bv = bj[k];
                        if av < bv {
                            a_better_anywhere = true;
                        } else if av > bv {
                            b_better_anywhere = true;
                        }
                    }
                    a_better_anywhere && !b_better_anywhere
                }
            };
            if i_dominates_j {
                strength[i] += 1;
                dominators_of[j].push(i);
            }
        }
    }

    // Raw fitness R(i) = sum of S(j) over j that dominate i.
    let raw: Vec<f64> = (0..n)
        .map(|i| dominators_of[i].iter().map(|&j| strength[j] as f64).sum())
        .collect();

    // Density D(i) = 1 / (σ_k + 2) where σ_k is the distance to the k-th
    // nearest neighbor (k = floor(sqrt(N))). Build a symmetric distance
    // matrix once instead of recomputing each row independently — that
    // halves the euclidean calls (which dominate at higher M) and keeps
    // the σ_k value bit-identical.
    let mut dist: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean(&oriented[i], &oriented[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    let k = (n as f64).sqrt() as usize;
    let density: Vec<f64> = (0..n)
        .map(|i| {
            let mut dists: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| dist[i][j]).collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
    // to its nearest neighbor in the current archive (ties broken by next-
    // nearest, etc. via lex order on each member's sorted neighbor vector).
    //
    // Implementation: compute the pairwise distance matrix once, plus each
    // member's sorted neighbor-distance vector. Each iteration drops one
    // dead victim's entry from every survivor's sorted vector via
    // binary-search-remove, instead of resorting from scratch. That cuts
    // truncation cost from O(K³ log K) to O(K² log K) overall while
    // producing the identical victim choice every step (the sorted vector
    // post-removal is bit-equal to a fresh sort over the smaller set).
    let n = nondom.len();
    let oriented: Vec<Vec<f64>> = nondom
        .iter()
        .map(|&i| objectives.as_minimization(&pool[i].evaluation.objectives))
        .collect();
    let mut dist: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean(&oriented[i], &oriented[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    let mut sorted_dists: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut v: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| dist[i][j]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v
        })
        .collect();
    let mut alive: Vec<bool> = vec![true; n];
    let mut alive_count = n;
    while alive_count > target_size {
        // Find the alive member whose sorted-neighbor-distance vector is
        // lex-smallest (= the most crowded member).
        let mut victim = usize::MAX;
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            if victim == usize::MAX {
                victim = i;
                continue;
            }
            let cmp = sorted_dists[i]
                .iter()
                .zip(sorted_dists[victim].iter())
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
        // Update every still-alive member's sorted neighbor vector by
        // removing the entry corresponding to the dead victim. Binary-
        // search-remove on the (still-)sorted vector is O(log K + K) per
        // survivor — we tolerate the linear shift because K is tiny.
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            let d = dist[i][victim];
            if let Ok(pos) = sorted_dists[i]
                .binary_search_by(|x| x.partial_cmp(&d).unwrap_or(std::cmp::Ordering::Equal))
            {
                sorted_dists[i].remove(pos);
            }
        }
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

impl<I, V> crate::traits::AlgorithmInfo for Spea2<I, V> {
    fn name(&self) -> &'static str {
        "SPEA2"
    }
    fn full_name(&self) -> &'static str {
        "Strength Pareto Evolutionary Algorithm 2"
    }
    fn seed(&self) -> Option<u64> {
        Some(self.config.seed)
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

    // ---- Mutation-test pinned helpers --------------------------------------

    #[test]
    fn euclidean_distance_basics() {
        // (0,0) to (3,4) = 5.
        assert!((euclidean(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-12);
        // symmetric and zero-to-self.
        assert!((euclidean(&[3.0, 4.0], &[0.0, 0.0]) - 5.0).abs() < 1e-12);
        assert_eq!(euclidean(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn binary_tournament_prefers_lower_fitness() {
        // SPEA2 fitness is "lower is better" — index 1 here is the best.
        use crate::core::rng::rng_from_seed;
        let fitness = vec![5.0_f64, 0.5];
        let mut wins1 = 0;
        for seed in 0..200 {
            let mut rng = rng_from_seed(seed);
            if binary_tournament(&fitness, &mut rng) == 1 {
                wins1 += 1;
            }
        }
        assert!(wins1 > 130, "lower-fitness index won only {wins1}/200");
    }
}
