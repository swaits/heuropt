//! NSGA-III — Deb & Jain 2014, the canonical many-objective MOEA.

use rand::Rng as _;
use rand::seq::IndexedRandom;

use crate::algorithms::parallel_eval::evaluate_batch;
use crate::core::candidate::Candidate;
use crate::core::objective::ObjectiveSpace;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::core::rng::{Rng, rng_from_seed};
use crate::pareto::front::{best_candidate, pareto_front};
use crate::pareto::reference_points::das_dennis;
use crate::pareto::sort::non_dominated_sort;
use crate::traits::{Initializer, Optimizer, Variation};

/// Configuration for [`Nsga3`].
#[derive(Debug, Clone)]
pub struct Nsga3Config {
    /// Constant population size carried across generations.
    pub population_size: usize,
    /// Number of generations to run.
    pub generations: usize,
    /// Number of divisions `H` for Das–Dennis reference points.
    /// Final reference set has `binomial(H + M - 1, M - 1)` points for
    /// `M = objectives`. Typical: `H = 12` for `M = 3` (91 points),
    /// `H = 6` for `M = 5` (210 points).
    pub reference_divisions: usize,
    /// Seed for the deterministic RNG.
    pub seed: u64,
}

impl Default for Nsga3Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 250,
            reference_divisions: 12,
            seed: 42,
        }
    }
}

/// NSGA-III optimizer.
#[derive(Debug, Clone)]
pub struct Nsga3<I, V> {
    /// Algorithm configuration.
    pub config: Nsga3Config,
    /// Initial-decision sampler.
    pub initializer: I,
    /// Offspring-producing variation operator.
    pub variation: V,
}

impl<I, V> Nsga3<I, V> {
    /// Construct an `Nsga3` optimizer.
    pub fn new(config: Nsga3Config, initializer: I, variation: V) -> Self {
        Self { config, initializer, variation }
    }
}

impl<P, I, V> Optimizer<P> for Nsga3<I, V>
where
    P: Problem + Sync,
    P::Decision: Send,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(
            self.config.population_size > 0,
            "Nsga3 population_size must be greater than 0",
        );
        let n = self.config.population_size;
        let objectives = problem.objectives();
        let m = objectives.len();
        let reference_points = das_dennis(m, self.config.reference_divisions);
        assert!(
            !reference_points.is_empty(),
            "Nsga3 reference set is empty — check reference_divisions",
        );
        let mut rng = rng_from_seed(self.config.seed);

        // Initial population.
        let initial_decisions = self.initializer.initialize(n, &mut rng);
        assert_eq!(
            initial_decisions.len(),
            n,
            "NSGA-III initializer must return exactly population_size decisions",
        );
        let mut population: Vec<Candidate<P::Decision>> =
            evaluate_batch(problem, initial_decisions);
        let mut evaluations = population.len();

        for _ in 0..self.config.generations {
            // --- Random parent selection + variation ---
            let mut offspring_decisions: Vec<P::Decision> = Vec::with_capacity(n);
            while offspring_decisions.len() < n {
                let p1 = rng.random_range(0..population.len());
                let p2 = rng.random_range(0..population.len());
                let parents =
                    vec![population[p1].decision.clone(), population[p2].decision.clone()];
                let children = self.variation.vary(&parents, &mut rng);
                assert!(
                    !children.is_empty(),
                    "NSGA-III variation returned no children",
                );
                for child_decision in children {
                    if offspring_decisions.len() >= n {
                        break;
                    }
                    offspring_decisions.push(child_decision);
                }
            }
            let offspring = evaluate_batch(problem, offspring_decisions);
            evaluations += offspring.len();

            // --- Combine + survival selection ---
            let mut combined: Vec<Candidate<P::Decision>> =
                Vec::with_capacity(2 * n);
            combined.extend(population);
            combined.extend(offspring);
            population = environmental_selection(&combined, &objectives, &reference_points, n, &mut rng);
        }

        let front = pareto_front(&population, &objectives);
        let best = best_candidate(&population, &objectives);
        OptimizationResult::new(
            Population::new(population),
            front,
            best,
            evaluations,
            self.config.generations,
        )
    }
}

/// NSGA-III environmental selection: front-by-front + reference-point niching
/// on the splitting front.
fn environmental_selection<D: Clone>(
    combined: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    reference_points: &[Vec<f64>],
    n: usize,
    rng: &mut Rng,
) -> Vec<Candidate<D>> {
    let fronts = non_dominated_sort(combined, objectives);
    let mut selected: Vec<usize> = Vec::with_capacity(n);
    let mut splitting: &[usize] = &[];
    for front in &fronts {
        if selected.len() + front.len() <= n {
            selected.extend(front.iter().copied());
        } else {
            splitting = front;
            break;
        }
        if selected.len() == n {
            break;
        }
    }

    if selected.len() == n {
        return selected.into_iter().map(|i| combined[i].clone()).collect();
    }

    // The "working pool" is everything that might end up in the next pop:
    // already-selected plus the splitting front. Normalization and
    // association are computed on this pool only.
    let mut working: Vec<usize> = selected.clone();
    working.extend(splitting.iter().copied());
    let normalized = normalize(combined, &working, objectives);
    let m = objectives.len();
    let (assoc, dist): (Vec<usize>, Vec<f64>) = associate(&normalized, reference_points, m);

    // Niche counts over already-selected members only.
    let mut niche_count = vec![0_usize; reference_points.len()];
    for k in 0..selected.len() {
        niche_count[assoc[k]] += 1;
    }

    // Set of reference indices still available; we won't actually drop them
    // permanently — instead we track which references currently have any
    // candidate in F_l associated.
    let f_l_offset = selected.len();
    let mut available_in_fl: Vec<Vec<usize>> = vec![Vec::new(); reference_points.len()];
    for k in 0..splitting.len() {
        let working_idx = f_l_offset + k;
        available_in_fl[assoc[working_idx]].push(k); // store F_l-local index
    }

    while selected.len() < n {
        // Find min niche count among references with at least one F_l candidate.
        let mut min_count = usize::MAX;
        for j in 0..reference_points.len() {
            if !available_in_fl[j].is_empty() && niche_count[j] < min_count {
                min_count = niche_count[j];
            }
        }
        if min_count == usize::MAX {
            // No more F_l candidates anywhere. Should not happen if we still
            // need members, but guard anyway.
            break;
        }
        let candidate_refs: Vec<usize> = (0..reference_points.len())
            .filter(|&j| !available_in_fl[j].is_empty() && niche_count[j] == min_count)
            .collect();
        let &chosen_ref = candidate_refs.choose(rng).expect("non-empty by construction");

        let pool = &available_in_fl[chosen_ref];
        let pick_local = if niche_count[chosen_ref] == 0 {
            // Take the F_l member closest to the reference direction.
            *pool
                .iter()
                .min_by(|&&a, &&b| {
                    let da = dist[f_l_offset + a];
                    let db = dist[f_l_offset + b];
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        } else {
            *pool.choose(rng).unwrap()
        };

        let combined_idx = splitting[pick_local];
        selected.push(combined_idx);
        niche_count[chosen_ref] += 1;

        // Remove pick_local from available_in_fl[chosen_ref].
        let pos = available_in_fl[chosen_ref]
            .iter()
            .position(|&v| v == pick_local)
            .unwrap();
        available_in_fl[chosen_ref].swap_remove(pos);
    }

    selected.into_iter().map(|i| combined[i].clone()).collect()
}

/// Translate by ideal, compute extreme points + intercepts, return per-member
/// normalized objective vectors. Falls back to per-axis range when the
/// extreme-point hyperplane is degenerate.
fn normalize<D>(
    combined: &[Candidate<D>],
    working: &[usize],
    objectives: &ObjectiveSpace,
) -> Vec<Vec<f64>> {
    let m = objectives.len();
    let mut oriented: Vec<Vec<f64>> = working
        .iter()
        .map(|&i| objectives.as_minimization(&combined[i].evaluation.objectives))
        .collect();

    // Ideal point z*: per-axis min over `working`.
    let mut ideal = vec![f64::INFINITY; m];
    for o in &oriented {
        for (k, &v) in o.iter().enumerate() {
            if v < ideal[k] {
                ideal[k] = v;
            }
        }
    }
    // Translate.
    for o in oriented.iter_mut() {
        for (k, v) in o.iter_mut().enumerate() {
            *v -= ideal[k];
        }
    }

    // Extreme points by Achievement Scalarizing Function:
    //   ASF_k(x) = max_i(x[i] / w_k[i]),  w_k[i] = 1 if i==k else 1e-6
    let extremes: Vec<usize> = (0..m)
        .map(|axis| {
            let mut best = 0usize;
            let mut best_asf = f64::INFINITY;
            for (idx, o) in oriented.iter().enumerate() {
                let asf = o
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        let w = if k == axis { 1.0 } else { 1e-6 };
                        v / w
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                if asf < best_asf {
                    best_asf = asf;
                    best = idx;
                }
            }
            best
        })
        .collect();

    // Intercepts: solve A * a = 1 where rows of A are the extreme points.
    // If the system is singular or yields non-positive intercepts, fall back
    // to per-axis range (max value per axis in `oriented`).
    let intercepts = solve_intercepts(&oriented, &extremes).unwrap_or_else(|| {
        (0..m)
            .map(|k| {
                oriented
                    .iter()
                    .map(|o| o[k])
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(1e-12)
            })
            .collect()
    });

    for o in oriented.iter_mut() {
        for (k, v) in o.iter_mut().enumerate() {
            *v /= intercepts[k].max(1e-12);
        }
    }
    oriented
}

/// Try to compute axis intercepts from M extreme points by Gaussian
/// elimination. Returns `None` if singular or degenerate.
fn solve_intercepts(oriented: &[Vec<f64>], extremes: &[usize]) -> Option<Vec<f64>> {
    let m = extremes.len();
    if m == 0 {
        return None;
    }
    // Build the M×M matrix of extreme points (each row = one extreme).
    let mut a: Vec<Vec<f64>> = extremes.iter().map(|&i| oriented[i].clone()).collect();
    let mut b: Vec<f64> = vec![1.0; m];
    // Forward elimination with partial pivoting.
    #[allow(clippy::needless_range_loop)] // Body indexes both `a` and `b` by row.
    for k in 0..m {
        let mut pivot = k;
        for i in (k + 1)..m {
            if a[i][k].abs() > a[pivot][k].abs() {
                pivot = i;
            }
        }
        if a[pivot][k].abs() < 1e-12 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in (k + 1)..m {
            let factor = a[i][k] / a[k][k];
            #[allow(clippy::needless_range_loop)] // Body indexes both `a[i]` and `a[k]`.
            for j in k..m {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    // Back-substitution.
    let mut x = vec![0.0_f64; m];
    for i in (0..m).rev() {
        let mut sum = b[i];
        for j in (i + 1)..m {
            sum -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-12 {
            return None;
        }
        x[i] = sum / a[i][i];
    }
    // Intercept along axis k is 1 / x[k].
    let intercepts: Vec<f64> = x
        .into_iter()
        .map(|v| if v.abs() < 1e-12 { f64::NAN } else { 1.0 / v })
        .collect();
    if intercepts.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }
    Some(intercepts)
}

/// Associate each normalized point with the closest reference direction by
/// perpendicular distance. Returns parallel `(ref_index, perp_dist)` vectors.
fn associate(
    normalized: &[Vec<f64>],
    reference_points: &[Vec<f64>],
    _m: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut assoc = vec![0_usize; normalized.len()];
    let mut dist = vec![0.0_f64; normalized.len()];
    let ref_norms: Vec<f64> = reference_points
        .iter()
        .map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12))
        .collect();
    for (i, x) in normalized.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (j, r) in reference_points.iter().enumerate() {
            // Perpendicular distance from x to the line spanned by r:
            //   t = (x · r) / ||r||²
            //   d = ||x - t·r||
            let dot: f64 = x.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
            let t = dot / (ref_norms[j] * ref_norms[j]);
            let mut sq = 0.0_f64;
            for (a, b) in x.iter().zip(r.iter()) {
                let proj = t * b;
                let diff = a - proj;
                sq += diff * diff;
            }
            let d = sq.sqrt();
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        assoc[i] = best;
        dist[i] = best_d;
    }
    (assoc, dist)
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
    ) -> Nsga3<
        RealBounds,
        CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation>,
    > {
        let bounds = vec![(-5.0, 5.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        Nsga3::new(
            Nsga3Config {
                population_size: 20,
                generations: 8,
                reference_divisions: 12,
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
        assert_eq!(r.population.len(), 20);
        assert!(!r.pareto_front.is_empty());
        assert_eq!(r.generations, 8);
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
    #[should_panic(expected = "population_size must be greater than 0")]
    fn zero_population_size_panics() {
        let bounds = vec![(0.0, 1.0)];
        let initializer = RealBounds::new(bounds.clone());
        let variation = CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0),
        };
        let mut opt = Nsga3::new(
            Nsga3Config {
                population_size: 0,
                generations: 1,
                reference_divisions: 4,
                seed: 0,
            },
            initializer,
            variation,
        );
        let _ = opt.run(&SchafferN1);
    }
}
