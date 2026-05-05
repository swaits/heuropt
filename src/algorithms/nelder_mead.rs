//! `NelderMead` — Nelder & Mead 1965 simplex direct-search optimizer.

use crate::core::candidate::Candidate;
use crate::core::evaluation::Evaluation;
use crate::core::objective::Direction;
use crate::core::population::Population;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::operators::real::RealBounds;
use crate::traits::Optimizer;

/// Configuration for [`NelderMead`].
#[derive(Debug, Clone)]
pub struct NelderMeadConfig {
    /// Number of iterations.
    pub iterations: usize,
    /// Reflection coefficient `α` (canonical 1.0).
    pub reflection: f64,
    /// Expansion coefficient `γ` (canonical 2.0).
    pub expansion: f64,
    /// Contraction coefficient `ρ` (canonical 0.5).
    pub contraction: f64,
    /// Shrinkage coefficient `σ` (canonical 0.5).
    pub shrinkage: f64,
    /// Initial simplex edge length (added to each axis from the start point).
    pub initial_step: f64,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            iterations: 1_000,
            reflection: 1.0,
            expansion: 2.0,
            contraction: 0.5,
            shrinkage: 0.5,
            initial_step: 0.5,
        }
    }
}

/// Classical Nelder-Mead simplex method.
///
/// Maintains a simplex of `n+1` vertices in `n`-D, replacing the worst
/// vertex each iteration via reflection / expansion / contraction /
/// shrinkage relative to the centroid of the rest.
///
/// `Vec<f64>` decisions only. Single-objective only. Initial simplex is
/// built around the midpoint of the configured bounds; every new vertex
/// is clamped to those bounds.
#[derive(Debug, Clone)]
pub struct NelderMead {
    /// Algorithm configuration.
    pub config: NelderMeadConfig,
    /// Per-variable bounds — used to seed the simplex midpoint and to clamp
    /// every reflected/expanded vertex.
    pub bounds: RealBounds,
}

impl NelderMead {
    /// Construct a `NelderMead`.
    pub fn new(config: NelderMeadConfig, bounds: RealBounds) -> Self {
        Self { config, bounds }
    }
}

impl<P> Optimizer<P> for NelderMead
where
    P: Problem<Decision = Vec<f64>> + Sync,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        assert!(self.config.reflection > 0.0, "NelderMead reflection must be > 0");
        assert!(
            self.config.expansion > 1.0,
            "NelderMead expansion must be > 1",
        );
        assert!(
            self.config.contraction > 0.0 && self.config.contraction < 1.0,
            "NelderMead contraction must be in (0, 1)",
        );
        assert!(
            self.config.shrinkage > 0.0 && self.config.shrinkage < 1.0,
            "NelderMead shrinkage must be in (0, 1)",
        );
        assert!(
            self.config.initial_step > 0.0,
            "NelderMead initial_step must be > 0",
        );
        let objectives = problem.objectives();
        assert!(
            objectives.is_single_objective(),
            "NelderMead requires exactly one objective",
        );
        let direction = objectives.objectives[0].direction;
        let n = self.bounds.bounds.len();

        // Seed the simplex: start at the bounds midpoint, then build n
        // additional vertices by stepping `initial_step` along each axis.
        let mut vertices: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        let start: Vec<f64> = self
            .bounds
            .bounds
            .iter()
            .map(|&(lo, hi)| 0.5 * (lo + hi))
            .collect();
        vertices.push(start.clone());
        for j in 0..n {
            let mut v = start.clone();
            let (lo, hi) = self.bounds.bounds[j];
            let step = self.config.initial_step.min(0.5 * (hi - lo));
            v[j] = (v[j] + step).clamp(lo, hi);
            vertices.push(v);
        }
        let mut evals: Vec<Evaluation> =
            vertices.iter().map(|v| problem.evaluate(v)).collect();
        let mut evaluations = evals.len();

        for _ in 0..self.config.iterations {
            // Sort vertices best → worst.
            let mut order: Vec<usize> = (0..vertices.len()).collect();
            order.sort_by(|&a, &b| compare(&evals[a], &evals[b], direction));
            let best_idx = order[0];
            let worst_idx = order[order.len() - 1];
            let second_worst_idx = order[order.len() - 2];

            // Centroid of all vertices except the worst.
            let mut centroid = vec![0.0_f64; n];
            for &idx in &order[..order.len() - 1] {
                for j in 0..n {
                    centroid[j] += vertices[idx][j];
                }
            }
            for c in centroid.iter_mut() {
                *c /= (order.len() - 1) as f64;
            }

            // Reflection.
            let reflected = self.reflect(&centroid, &vertices[worst_idx], self.config.reflection);
            let r_eval = problem.evaluate(&reflected);
            evaluations += 1;

            if better(&r_eval, &evals[best_idx], direction) {
                // Reflection beat the best — try expansion.
                let expanded =
                    self.reflect(&centroid, &vertices[worst_idx], self.config.expansion);
                let e_eval = problem.evaluate(&expanded);
                evaluations += 1;
                if better(&e_eval, &r_eval, direction) {
                    vertices[worst_idx] = expanded;
                    evals[worst_idx] = e_eval;
                } else {
                    vertices[worst_idx] = reflected;
                    evals[worst_idx] = r_eval;
                }
            } else if better(&r_eval, &evals[second_worst_idx], direction) {
                // Reflection at least beat the second-worst — accept.
                vertices[worst_idx] = reflected;
                evals[worst_idx] = r_eval;
            } else {
                // Reflection didn't help — try contraction.
                let contraction_target = if better(&r_eval, &evals[worst_idx], direction) {
                    // Outside contraction (between centroid and reflected).
                    self.contract(&centroid, &reflected, self.config.contraction)
                } else {
                    // Inside contraction (between centroid and worst).
                    self.contract(&centroid, &vertices[worst_idx], self.config.contraction)
                };
                let c_eval = problem.evaluate(&contraction_target);
                evaluations += 1;
                if better(&c_eval, &evals[worst_idx], direction) {
                    vertices[worst_idx] = contraction_target;
                    evals[worst_idx] = c_eval;
                } else {
                    // Shrink: move every non-best vertex toward the best.
                    let best_pt = vertices[best_idx].clone();
                    for &idx in &order {
                        if idx == best_idx {
                            continue;
                        }
                        for j in 0..n {
                            vertices[idx][j] = best_pt[j]
                                + self.config.shrinkage
                                    * (vertices[idx][j] - best_pt[j]);
                        }
                        // Clamp to bounds.
                        for j in 0..n {
                            let (lo, hi) = self.bounds.bounds[j];
                            vertices[idx][j] = vertices[idx][j].clamp(lo, hi);
                        }
                        evals[idx] = problem.evaluate(&vertices[idx]);
                        evaluations += 1;
                    }
                }
            }
        }

        // Find the best vertex.
        let mut best_idx = 0;
        for i in 1..vertices.len() {
            if better(&evals[i], &evals[best_idx], direction) {
                best_idx = i;
            }
        }
        let best = Candidate::new(vertices[best_idx].clone(), evals[best_idx].clone());
        let population = Population::new(vec![best.clone()]);
        let front = vec![best.clone()];
        OptimizationResult::new(
            population,
            front,
            Some(best),
            evaluations,
            self.config.iterations,
        )
    }
}

impl NelderMead {
    fn reflect(&self, centroid: &[f64], worst: &[f64], coefficient: f64) -> Vec<f64> {
        let n = centroid.len();
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let v = centroid[j] + coefficient * (centroid[j] - worst[j]);
            let (lo, hi) = self.bounds.bounds[j];
            out.push(v.clamp(lo, hi));
        }
        out
    }

    fn contract(&self, centroid: &[f64], target: &[f64], coefficient: f64) -> Vec<f64> {
        let n = centroid.len();
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let v = centroid[j] + coefficient * (target[j] - centroid[j]);
            let (lo, hi) = self.bounds.bounds[j];
            out.push(v.clamp(lo, hi));
        }
        out
    }
}

fn compare(a: &Evaluation, b: &Evaluation, direction: Direction) -> std::cmp::Ordering {
    match (a.is_feasible(), b.is_feasible()) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a
            .constraint_violation
            .partial_cmp(&b.constraint_violation)
            .unwrap_or(std::cmp::Ordering::Equal),
        (true, true) => match direction {
            Direction::Minimize => a.objectives[0]
                .partial_cmp(&b.objectives[0])
                .unwrap_or(std::cmp::Ordering::Equal),
            Direction::Maximize => b.objectives[0]
                .partial_cmp(&a.objectives[0])
                .unwrap_or(std::cmp::Ordering::Equal),
        },
    }
}

fn better(a: &Evaluation, b: &Evaluation, direction: Direction) -> bool {
    compare(a, b, direction) == std::cmp::Ordering::Less
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Objective, ObjectiveSpace};
    use crate::tests_support::{SchafferN1, Sphere1D};

    /// 2-D Rosenbrock for shape exercise.
    struct Rosenbrock2D;
    impl Problem for Rosenbrock2D {
        type Decision = Vec<f64>;

        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }

        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            Evaluation::new(vec![a * a + 100.0 * b * b])
        }
    }

    #[test]
    fn finds_minimum_of_sphere() {
        let mut opt = NelderMead::new(
            NelderMeadConfig {
                iterations: 200,
                ..NelderMeadConfig::default()
            },
            RealBounds::new(vec![(-5.0, 5.0)]),
        );
        let r = opt.run(&Sphere1D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-8,
            "got f = {}",
            best.evaluation.objectives[0],
        );
    }

    #[test]
    fn finds_minimum_of_2d_rosenbrock() {
        let mut opt = NelderMead::new(
            NelderMeadConfig {
                iterations: 500,
                initial_step: 0.5,
                ..NelderMeadConfig::default()
            },
            RealBounds::new(vec![(-2.0, 2.0); 2]),
        );
        let r = opt.run(&Rosenbrock2D);
        let best = r.best.unwrap();
        assert!(
            best.evaluation.objectives[0] < 1e-3,
            "got f = {}",
            best.evaluation.objectives[0],
        );
    }

    #[test]
    fn deterministic_no_rng() {
        // Nelder-Mead is purely deterministic — same bounds + same iters
        // → same result, no seed needed.
        let make = || {
            NelderMead::new(
                NelderMeadConfig {
                    iterations: 100,
                    ..NelderMeadConfig::default()
                },
                RealBounds::new(vec![(-5.0, 5.0)]),
            )
        };
        let mut a = make();
        let mut b = make();
        let ra = a.run(&Sphere1D);
        let rb = b.run(&Sphere1D);
        assert_eq!(
            ra.best.unwrap().evaluation.objectives,
            rb.best.unwrap().evaluation.objectives,
        );
    }

    #[test]
    #[should_panic(expected = "exactly one objective")]
    fn multi_objective_panics() {
        let mut opt = NelderMead::new(
            NelderMeadConfig::default(),
            RealBounds::new(vec![(-5.0, 5.0)]),
        );
        let _ = opt.run(&SchafferN1);
    }
}
