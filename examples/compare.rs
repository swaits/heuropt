//! Multi-seed algorithm comparison harness.
//!
//! Runs every applicable optimizer on each test problem across N seeds and
//! prints aggregate quality metrics. Adding a new algorithm to the
//! comparison is a single-line edit to the runner table — see the bottom
//! of this file.
//!
//! ```bash
//! cargo run --release --example compare
//! ```

use std::f64::consts::PI;
use std::time::Instant;

use heuropt::metrics::{hypervolume::hypervolume_2d, spacing::spacing};
use heuropt::prelude::*;

const SEEDS: u64 = 10;

const ZDT1_DIM: usize = 30;
const ZDT1_BUDGET: usize = 25_000;
// Standard ZDT1 reference point. Using [11, 11] (rather than the
// near-front [1.1, 1.1]) so under-converged algorithms with large `g`
// values still register a meaningful — if poor — hypervolume.
const ZDT1_REFERENCE: [f64; 2] = [11.0, 11.0];

const RASTRIGIN_DIM: usize = 5;
const RASTRIGIN_BUDGET: usize = 50_000;

const DTLZ2_OBJECTIVES: usize = 3;
const DTLZ2_K: usize = 10;
const DTLZ2_DIM: usize = DTLZ2_OBJECTIVES + DTLZ2_K - 1; // 12
const DTLZ2_BUDGET: usize = 30_000;

const ROSENBROCK_DIM: usize = 5;
const ROSENBROCK_BUDGET: usize = 30_000;

const ACKLEY_DIM: usize = 5;
const ACKLEY_BUDGET: usize = 30_000;

const ZDT3_DIM: usize = 30;
const ZDT3_BUDGET: usize = 25_000;
const ZDT3_REFERENCE: [f64; 2] = [11.0, 11.0];

const DTLZ1_OBJECTIVES: usize = 3;
const DTLZ1_K: usize = 5;
const DTLZ1_DIM: usize = DTLZ1_OBJECTIVES + DTLZ1_K - 1;
const DTLZ1_BUDGET: usize = 30_000;

// -----------------------------------------------------------------------------
// Test problems
// -----------------------------------------------------------------------------

struct Zdt1 {
    dim: usize,
}

impl Problem for Zdt1 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let f1 = x[0];
        let tail_sum: f64 = x[1..].iter().sum();
        let g = 1.0 + 9.0 * tail_sum / (self.dim as f64 - 1.0);
        let f2 = g * (1.0 - (f1 / g).sqrt());
        Evaluation::new(vec![f1, f2])
    }
}

struct Dtlz2 {
    num_objectives: usize,
    dim: usize,
}

impl Problem for Dtlz2 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(
            (0..self.num_objectives)
                .map(|i| Objective::minimize(format!("f{}", i + 1)))
                .collect(),
        )
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let m = self.num_objectives;
        let g: f64 = x[(m - 1)..self.dim].iter().map(|v| (v - 0.5).powi(2)).sum();
        let scale = 1.0 + g;
        let mut f = vec![0.0_f64; m];
        for i in 0..m {
            let mut prod = scale;
            #[allow(clippy::needless_range_loop)] // Body indexes `x[j]`.
            for j in 0..(m - i - 1) {
                prod *= (x[j] * std::f64::consts::FRAC_PI_2).cos();
            }
            if i > 0 {
                prod *= (x[m - i - 1] * std::f64::consts::FRAC_PI_2).sin();
            }
            f[i] = prod;
        }
        Evaluation::new(f)
    }
}

struct Rosenbrock {
    dim: usize,
}

impl Problem for Rosenbrock {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let f: f64 = (0..self.dim - 1)
            .map(|i| {
                let a = 1.0 - x[i];
                let b = x[i + 1] - x[i] * x[i];
                a * a + 100.0 * b * b
            })
            .sum();
        Evaluation::new(vec![f])
    }
}

struct Ackley {
    dim: usize,
}

impl Problem for Ackley {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let n = self.dim as f64;
        let sum_sq: f64 = x.iter().map(|v| v * v).sum();
        let sum_cos: f64 = x.iter().map(|v| (2.0 * PI * v).cos()).sum();
        let f = -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp()
            - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E;
        Evaluation::new(vec![f])
    }
}

struct Zdt3 {
    dim: usize,
}

impl Problem for Zdt3 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let f1 = x[0];
        let tail_sum: f64 = x[1..].iter().sum();
        let g = 1.0 + 9.0 * tail_sum / (self.dim as f64 - 1.0);
        let r = f1 / g;
        let f2 = g * (1.0 - r.sqrt() - r * (10.0 * PI * f1).sin());
        Evaluation::new(vec![f1, f2])
    }
}

struct Dtlz1 {
    num_objectives: usize,
    dim: usize,
}

impl Problem for Dtlz1 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(
            (0..self.num_objectives)
                .map(|i| Objective::minimize(format!("f{}", i + 1)))
                .collect(),
        )
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let m = self.num_objectives;
        let k = self.dim - (m - 1);
        let g_term: f64 = x[(m - 1)..self.dim]
            .iter()
            .map(|v| (v - 0.5).powi(2) - (20.0 * PI * (v - 0.5)).cos())
            .sum();
        let g = 100.0 * (k as f64 + g_term);
        let mut f = vec![0.0_f64; m];
        for i in 0..m {
            let mut prod = 0.5 * (1.0 + g);
            #[allow(clippy::needless_range_loop)] // body indexes x[j].
            for j in 0..(m - i - 1) {
                prod *= x[j];
            }
            if i > 0 {
                prod *= 1.0 - x[m - i - 1];
            }
            f[i] = prod;
        }
        Evaluation::new(f)
    }
}

struct Rastrigin {
    dim: usize,
}

impl Problem for Rastrigin {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let n = self.dim as f64;
        let value = 10.0 * n
            + x.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>();
        Evaluation::new(vec![value])
    }
}

// -----------------------------------------------------------------------------
// Run results + metrics aggregation
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct MoRun {
    front: Vec<Candidate<Vec<f64>>>,
    wall_ms: u128,
}

#[derive(Clone)]
struct SoRun {
    best_value: f64,
    wall_ms: u128,
}

fn mean_l2_to_zdt1_front(front: &[Candidate<Vec<f64>>]) -> f64 {
    if front.is_empty() {
        return f64::INFINITY;
    }
    let samples: Vec<(f64, f64)> = (0..=1000)
        .map(|i| {
            let f1 = i as f64 / 1000.0;
            (f1, 1.0 - f1.sqrt())
        })
        .collect();
    let mut total = 0.0;
    for c in front {
        let f1 = c.evaluation.objectives[0];
        let f2 = c.evaluation.objectives[1];
        let mut best = f64::INFINITY;
        for &(rf1, rf2) in &samples {
            let d = ((rf1 - f1).powi(2) + (rf2 - f2).powi(2)).sqrt();
            if d < best {
                best = d;
            }
        }
        total += best;
    }
    total / front.len() as f64
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

// -----------------------------------------------------------------------------
// ZDT1 algorithm runners
// -----------------------------------------------------------------------------

fn zdt1_random(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let initializer = RealBounds::new(vec![(0.0, 1.0); ZDT1_DIM]);
    let config = RandomSearchConfig {
        iterations: ZDT1_BUDGET,
        batch_size: 1,
        seed,
    };
    let mut opt = RandomSearch::new(config, initializer);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_paes(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let initializer = RealBounds::new(vec![(0.0, 1.0); ZDT1_DIM]);
    let variation = BoundedGaussianMutation::new(0.05, vec![(0.0, 1.0); ZDT1_DIM]);
    let config = PaesConfig {
        iterations: ZDT1_BUDGET,
        archive_size: 100,
        seed,
    };
    let mut opt = Paes::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_spea2(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    let pop = 100;
    let arc = 100;
    // SPEA2 evaluates `pop_size` per generation after the initial population.
    let gens = (ZDT1_BUDGET - pop) / pop;
    let config = Spea2Config {
        population_size: pop,
        archive_size: arc,
        generations: gens,
        seed,
    };
    let mut opt = Spea2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_nsga2(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    let pop = 100;
    let gens = ZDT1_BUDGET / pop;
    let config = Nsga2Config { population_size: pop, generations: gens, seed };
    let mut opt = Nsga2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_sms_emoa(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    // SMS-EMOA is steady-state and computes O(N²) hypervolumes per
    // iteration, so we run it on a smaller population for a smaller
    // total budget to keep wall time tractable.
    let config = SmsEmoaConfig {
        population_size: 40,
        generations: 4_000,
        reference_point: vec![11.0, 11.0],
        seed,
    };
    let mut opt = SmsEmoa::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_hype(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    let pop = 80;
    // Each generation does multiple HV-estimation passes (parent selection +
    // survival truncation), each with mc_samples × N work. Keep budget
    // smaller than NSGA-II's so wall time is tractable.
    let gens = 80;
    let config = HypeConfig {
        population_size: pop,
        generations: gens,
        reference_point: vec![11.0, 11.0],
        mc_samples: 1_000,
        seed,
    };
    let mut opt = Hype::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_rvea(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    let pop = 100;
    let gens = ZDT1_BUDGET / pop;
    let config = RveaConfig {
        population_size: pop,
        generations: gens,
        reference_divisions: 99,
        alpha: 2.0,
        seed,
    };
    let mut opt = Rvea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_pesa2(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    let pop = 50;
    let gens = (ZDT1_BUDGET - pop) / pop;
    let config = PesaIIConfig {
        population_size: pop,
        archive_size: 100,
        generations: gens,
        grid_divisions: 20,
        seed,
    };
    let mut opt = PesaII::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_epsilon_moea(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    let config = EpsilonMoeaConfig {
        population_size: 50,
        evaluations: ZDT1_BUDGET,
        epsilon: vec![0.01, 0.01],
        seed,
    };
    let mut opt = EpsilonMoea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_mopso(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = RealBounds::new(vec![(0.0, 1.0); ZDT1_DIM]);
    let swarm = 100;
    let gens = (ZDT1_BUDGET - 2 * swarm) / swarm;
    let config = MopsoConfig {
        swarm_size: swarm,
        generations: gens,
        archive_size: 100,
        inertia: 0.7,
        cognitive: 1.5,
        social: 1.5,
        seed,
    };
    let mut opt = Mopso::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_ibea(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    let pop = 100;
    let gens = ZDT1_BUDGET / pop;
    let config = IbeaConfig { population_size: pop, generations: gens, kappa: 0.05, seed };
    let mut opt = Ibea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_moead(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    // 99 divisions → 100 weights for 2 obj. Each generation evaluates one
    // child per weight (so `n_weights` evals/gen).
    let pop = 100;
    let gens = (ZDT1_BUDGET - pop) / pop;
    let config = MoeadConfig {
        generations: gens,
        reference_divisions: 99,
        neighborhood_size: 20,
        seed,
    };
    let mut opt = Moead::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt1_nsga3(seed: u64) -> MoRun {
    let problem = Zdt1 { dim: ZDT1_DIM };
    let bounds = vec![(0.0, 1.0); ZDT1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT1_DIM as f64),
    };
    let pop = 100;
    let gens = ZDT1_BUDGET / pop;
    let config = Nsga3Config {
        population_size: pop,
        generations: gens,
        // 99 ref points for 2 objectives — same density as the population.
        reference_divisions: 99,
        seed,
    };
    let mut opt = Nsga3::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

// -----------------------------------------------------------------------------
// DTLZ2 algorithm runners (3-objective)
// -----------------------------------------------------------------------------

fn dtlz2_problem() -> Dtlz2 {
    Dtlz2 { num_objectives: DTLZ2_OBJECTIVES, dim: DTLZ2_DIM }
}

fn dtlz2_random(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let initializer = RealBounds::new(vec![(0.0, 1.0); DTLZ2_DIM]);
    let config = RandomSearchConfig {
        iterations: DTLZ2_BUDGET,
        batch_size: 1,
        seed,
    };
    let mut opt = RandomSearch::new(config, initializer);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_nsga2(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    let pop = 92; // close to the 91-ref-point NSGA-III pop, for fairness
    let gens = DTLZ2_BUDGET / pop;
    let config = Nsga2Config { population_size: pop, generations: gens, seed };
    let mut opt = Nsga2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_spea2(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    let pop = 92;
    let arc = 92;
    let gens = (DTLZ2_BUDGET - pop) / pop;
    let config = Spea2Config {
        population_size: pop,
        archive_size: arc,
        generations: gens,
        seed,
    };
    let mut opt = Spea2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_sms_emoa(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    // 3-D HV-by-slicing is significantly more expensive than 2-D; smaller
    // pop and gen budget here to keep the comparison tractable.
    let pop = 40;
    let gens = 4_000;
    let config = SmsEmoaConfig {
        population_size: pop,
        generations: gens,
        reference_point: vec![3.0, 3.0, 3.0],
        seed,
    };
    let mut opt = SmsEmoa::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_hype(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    let pop = 80;
    let gens = 80;
    let config = HypeConfig {
        population_size: pop,
        generations: gens,
        reference_point: vec![3.0, 3.0, 3.0],
        mc_samples: 1_000,
        seed,
    };
    let mut opt = Hype::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_rvea(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    let pop = 92;
    let gens = DTLZ2_BUDGET / pop;
    let config = RveaConfig {
        population_size: pop,
        generations: gens,
        reference_divisions: 12,
        alpha: 2.0,
        seed,
    };
    let mut opt = Rvea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_pesa2(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    let pop = 50;
    let gens = (DTLZ2_BUDGET - pop) / pop;
    let config = PesaIIConfig {
        population_size: pop,
        archive_size: 100,
        generations: gens,
        grid_divisions: 12,
        seed,
    };
    let mut opt = PesaII::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_epsilon_moea(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    let config = EpsilonMoeaConfig {
        population_size: 50,
        evaluations: DTLZ2_BUDGET,
        epsilon: vec![0.05, 0.05, 0.05],
        seed,
    };
    let mut opt = EpsilonMoea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_mopso(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = RealBounds::new(vec![(0.0, 1.0); DTLZ2_DIM]);
    let swarm = 92;
    let gens = (DTLZ2_BUDGET - 2 * swarm) / swarm;
    let config = MopsoConfig {
        swarm_size: swarm,
        generations: gens,
        archive_size: 100,
        inertia: 0.7,
        cognitive: 1.5,
        social: 1.5,
        seed,
    };
    let mut opt = Mopso::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_ibea(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    let pop = 92;
    let gens = DTLZ2_BUDGET / pop;
    let config = IbeaConfig { population_size: pop, generations: gens, kappa: 0.05, seed };
    let mut opt = Ibea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_moead(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    // 12 divisions for 3 objectives = 91 weights — same density as NSGA-III.
    let pop = 91;
    let gens = (DTLZ2_BUDGET - pop) / pop;
    let config = MoeadConfig {
        generations: gens,
        reference_divisions: 12,
        neighborhood_size: 20,
        seed,
    };
    let mut opt = Moead::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz2_nsga3(seed: u64) -> MoRun {
    let problem = dtlz2_problem();
    let bounds = vec![(0.0, 1.0); DTLZ2_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ2_DIM as f64),
    };
    // H=12 → 91 reference points (the canonical NSGA-III 3-objective set).
    // Population is sized to match: the spec recommends pop ≈ #refs.
    let pop = 92;
    let gens = DTLZ2_BUDGET / pop;
    let config = Nsga3Config {
        population_size: pop,
        generations: gens,
        reference_divisions: 12,
        seed,
    };
    let mut opt = Nsga3::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

/// DTLZ2's analytical Pareto front is the unit sphere octant in objective
/// space (`Σ f_i² = 1`, all `f_i ≥ 0`). The closest-point distance from
/// `f` to that surface is `|‖f‖ - 1|`.
fn mean_distance_to_dtlz2_front(front: &[Candidate<Vec<f64>>]) -> f64 {
    if front.is_empty() {
        return f64::INFINITY;
    }
    let total: f64 = front
        .iter()
        .map(|c| {
            let norm: f64 = c.evaluation.objectives.iter().map(|v| v * v).sum::<f64>().sqrt();
            (norm - 1.0).abs()
        })
        .sum();
    total / front.len() as f64
}

// -----------------------------------------------------------------------------
// Rastrigin algorithm runners
// -----------------------------------------------------------------------------

fn rastrigin_random(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let initializer = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let config = RandomSearchConfig {
        iterations: RASTRIGIN_BUDGET,
        batch_size: 1,
        seed,
    };
    let mut opt = RandomSearch::new(config, initializer);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_paes(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let initializer = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let variation = BoundedGaussianMutation::new(0.3, vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let config = PaesConfig {
        iterations: RASTRIGIN_BUDGET,
        archive_size: 32,
        seed,
    };
    let mut opt = Paes::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_nsga2(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let bounds = vec![(-5.12, 5.12); RASTRIGIN_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / RASTRIGIN_DIM as f64),
    };
    let pop = 50;
    let gens = RASTRIGIN_BUDGET / pop;
    let config = Nsga2Config { population_size: pop, generations: gens, seed };
    let mut opt = Nsga2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_de(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let bounds = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let pop = 50;
    let gens = (RASTRIGIN_BUDGET - pop) / pop; // initial pop also evaluates
    let config = DifferentialEvolutionConfig {
        population_size: pop,
        generations: gens,
        differential_weight: 0.5,
        crossover_probability: 0.9,
        seed,
    };
    let mut opt = DifferentialEvolution::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_hill_climber(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let initializer = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let variation = BoundedGaussianMutation::new(0.3, vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let config = HillClimberConfig { iterations: RASTRIGIN_BUDGET, seed };
    let mut opt = HillClimber::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_simulated_annealing(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let initializer = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let variation = BoundedGaussianMutation::new(0.5, vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let config = SimulatedAnnealingConfig {
        iterations: RASTRIGIN_BUDGET,
        initial_temperature: 5.0,
        final_temperature: 1e-3,
        seed,
    };
    let mut opt = SimulatedAnnealing::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_genetic_algorithm(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let bounds = vec![(-5.12, 5.12); RASTRIGIN_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / RASTRIGIN_DIM as f64),
    };
    let pop = 50;
    let gens = (RASTRIGIN_BUDGET - pop) / pop;
    let config = GeneticAlgorithmConfig {
        population_size: pop,
        generations: gens,
        tournament_size: 2,
        elitism: 2,
        seed,
    };
    let mut opt = GeneticAlgorithm::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_particle_swarm(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let bounds = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let swarm = 40;
    // PSO does swarm + swarm·gens + final evaluations. Approximate budget.
    let gens = (RASTRIGIN_BUDGET - 2 * swarm) / swarm;
    let config = ParticleSwarmConfig {
        swarm_size: swarm,
        generations: gens,
        inertia: 0.7,
        cognitive: 1.5,
        social: 1.5,
        seed,
    };
    let mut opt = ParticleSwarm::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_cma_es(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let bounds = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let pop = 16;
    let gens = RASTRIGIN_BUDGET / pop;
    let config = CmaEsConfig {
        population_size: pop,
        generations: gens,
        initial_sigma: 1.0,
        eigen_decomposition_period: 1,
        seed,
    };
    let mut opt = CmaEs::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

// -----------------------------------------------------------------------------
// Rosenbrock + Ackley runners (a curated SO subset on each)
// -----------------------------------------------------------------------------

fn rosenbrock_problem() -> Rosenbrock {
    Rosenbrock { dim: ROSENBROCK_DIM }
}
fn ackley_problem() -> Ackley {
    Ackley { dim: ACKLEY_DIM }
}

macro_rules! so_run_de {
    ($problem_expr:expr, $dim:expr, $bounds_lo:expr, $bounds_hi:expr, $budget:expr, $seed:expr) => {{
        let problem = $problem_expr;
        let bounds = RealBounds::new(vec![($bounds_lo, $bounds_hi); $dim]);
        let pop = 50;
        let gens = ($budget - pop) / pop;
        let config = DifferentialEvolutionConfig {
            population_size: pop,
            generations: gens,
            differential_weight: 0.5,
            crossover_probability: 0.9,
            seed: $seed,
        };
        let mut opt = DifferentialEvolution::new(config, bounds);
        let t0 = Instant::now();
        let result = opt.run(&problem);
        SoRun {
            best_value: result.best.unwrap().evaluation.objectives[0],
            wall_ms: t0.elapsed().as_millis(),
        }
    }};
}

macro_rules! so_run_cma {
    ($problem_expr:expr, $dim:expr, $bounds_lo:expr, $bounds_hi:expr, $budget:expr, $seed:expr) => {{
        let problem = $problem_expr;
        let bounds = RealBounds::new(vec![($bounds_lo, $bounds_hi); $dim]);
        let pop = 16;
        let config = CmaEsConfig {
            population_size: pop,
            generations: $budget / pop,
            initial_sigma: 1.0,
            eigen_decomposition_period: 1,
            seed: $seed,
        };
        let mut opt = CmaEs::new(config, bounds);
        let t0 = Instant::now();
        let result = opt.run(&problem);
        SoRun {
            best_value: result.best.unwrap().evaluation.objectives[0],
            wall_ms: t0.elapsed().as_millis(),
        }
    }};
}

macro_rules! so_run_pso {
    ($problem_expr:expr, $dim:expr, $bounds_lo:expr, $bounds_hi:expr, $budget:expr, $seed:expr) => {{
        let problem = $problem_expr;
        let bounds = RealBounds::new(vec![($bounds_lo, $bounds_hi); $dim]);
        let swarm = 40;
        let config = ParticleSwarmConfig {
            swarm_size: swarm,
            generations: ($budget - 2 * swarm) / swarm,
            inertia: 0.7,
            cognitive: 1.5,
            social: 1.5,
            seed: $seed,
        };
        let mut opt = ParticleSwarm::new(config, bounds);
        let t0 = Instant::now();
        let result = opt.run(&problem);
        SoRun {
            best_value: result.best.unwrap().evaluation.objectives[0],
            wall_ms: t0.elapsed().as_millis(),
        }
    }};
}

macro_rules! so_run_tlbo {
    ($problem_expr:expr, $dim:expr, $bounds_lo:expr, $bounds_hi:expr, $budget:expr, $seed:expr) => {{
        let problem = $problem_expr;
        let bounds = RealBounds::new(vec![($bounds_lo, $bounds_hi); $dim]);
        let pop = 30;
        // TLBO does ~2N evaluations per generation.
        let gens = ($budget - pop) / (2 * pop);
        let config = TlboConfig { population_size: pop, generations: gens, seed: $seed };
        let mut opt = Tlbo::new(config, bounds);
        let t0 = Instant::now();
        let result = opt.run(&problem);
        SoRun {
            best_value: result.best.unwrap().evaluation.objectives[0],
            wall_ms: t0.elapsed().as_millis(),
        }
    }};
}

fn rosenbrock_de(seed: u64) -> SoRun { so_run_de!(rosenbrock_problem(), ROSENBROCK_DIM, -5.0, 10.0, ROSENBROCK_BUDGET, seed) }
fn rosenbrock_cma(seed: u64) -> SoRun { so_run_cma!(rosenbrock_problem(), ROSENBROCK_DIM, -5.0, 10.0, ROSENBROCK_BUDGET, seed) }
fn rosenbrock_pso(seed: u64) -> SoRun { so_run_pso!(rosenbrock_problem(), ROSENBROCK_DIM, -5.0, 10.0, ROSENBROCK_BUDGET, seed) }
fn rosenbrock_tlbo(seed: u64) -> SoRun { so_run_tlbo!(rosenbrock_problem(), ROSENBROCK_DIM, -5.0, 10.0, ROSENBROCK_BUDGET, seed) }

fn ackley_de(seed: u64) -> SoRun { so_run_de!(ackley_problem(), ACKLEY_DIM, -32.768, 32.768, ACKLEY_BUDGET, seed) }
fn ackley_cma(seed: u64) -> SoRun { so_run_cma!(ackley_problem(), ACKLEY_DIM, -32.768, 32.768, ACKLEY_BUDGET, seed) }
fn ackley_pso(seed: u64) -> SoRun { so_run_pso!(ackley_problem(), ACKLEY_DIM, -32.768, 32.768, ACKLEY_BUDGET, seed) }
fn ackley_tlbo(seed: u64) -> SoRun { so_run_tlbo!(ackley_problem(), ACKLEY_DIM, -32.768, 32.768, ACKLEY_BUDGET, seed) }

// -----------------------------------------------------------------------------
// ZDT3 runners (curated MO subset)
// -----------------------------------------------------------------------------

fn zdt3_problem() -> Zdt3 { Zdt3 { dim: ZDT3_DIM } }

fn zdt3_nsga2(seed: u64) -> MoRun {
    let problem = zdt3_problem();
    let bounds = vec![(0.0, 1.0); ZDT3_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT3_DIM as f64),
    };
    let pop = 100;
    let config = Nsga2Config { population_size: pop, generations: ZDT3_BUDGET / pop, seed };
    let mut opt = Nsga2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt3_moead(seed: u64) -> MoRun {
    let problem = zdt3_problem();
    let bounds = vec![(0.0, 1.0); ZDT3_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT3_DIM as f64),
    };
    let pop = 100;
    let config = MoeadConfig {
        generations: (ZDT3_BUDGET - pop) / pop,
        reference_divisions: 99,
        neighborhood_size: 20,
        seed,
    };
    let mut opt = Moead::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt3_ibea(seed: u64) -> MoRun {
    let problem = zdt3_problem();
    let bounds = vec![(0.0, 1.0); ZDT3_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT3_DIM as f64),
    };
    let pop = 100;
    let config = IbeaConfig { population_size: pop, generations: ZDT3_BUDGET / pop, kappa: 0.05, seed };
    let mut opt = Ibea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn zdt3_age_moea(seed: u64) -> MoRun {
    let problem = zdt3_problem();
    let bounds = vec![(0.0, 1.0); ZDT3_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT3_DIM as f64),
    };
    let pop = 100;
    let config = AgeMoeaConfig { population_size: pop, generations: ZDT3_BUDGET / pop, seed };
    let mut opt = AgeMoea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

// -----------------------------------------------------------------------------
// DTLZ1 runners (curated many-obj subset)
// -----------------------------------------------------------------------------

fn dtlz1_problem() -> Dtlz1 {
    Dtlz1 { num_objectives: DTLZ1_OBJECTIVES, dim: DTLZ1_DIM }
}

fn dtlz1_nsga3(seed: u64) -> MoRun {
    let problem = dtlz1_problem();
    let bounds = vec![(0.0, 1.0); DTLZ1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ1_DIM as f64),
    };
    let pop = 92;
    let config = Nsga3Config {
        population_size: pop,
        generations: DTLZ1_BUDGET / pop,
        reference_divisions: 12,
        seed,
    };
    let mut opt = Nsga3::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz1_moead(seed: u64) -> MoRun {
    let problem = dtlz1_problem();
    let bounds = vec![(0.0, 1.0); DTLZ1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ1_DIM as f64),
    };
    let pop = 91;
    let config = MoeadConfig {
        generations: (DTLZ1_BUDGET - pop) / pop,
        reference_divisions: 12,
        neighborhood_size: 20,
        seed,
    };
    let mut opt = Moead::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz1_age_moea(seed: u64) -> MoRun {
    let problem = dtlz1_problem();
    let bounds = vec![(0.0, 1.0); DTLZ1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ1_DIM as f64),
    };
    let pop = 92;
    let config = AgeMoeaConfig { population_size: pop, generations: DTLZ1_BUDGET / pop, seed };
    let mut opt = AgeMoea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

fn dtlz1_grea(seed: u64) -> MoRun {
    let problem = dtlz1_problem();
    let bounds = vec![(0.0, 1.0); DTLZ1_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / DTLZ1_DIM as f64),
    };
    let pop = 92;
    let config = GreaConfig {
        population_size: pop,
        generations: DTLZ1_BUDGET / pop,
        grid_divisions: 8,
        seed,
    };
    let mut opt = Grea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun { front: result.pareto_front, wall_ms: t0.elapsed().as_millis() }
}

/// Mean L2 distance from each front point to the analytical DTLZ1 front
/// (`Σf_i = 0.5`, all `f_i ≥ 0`). Closed-form: signed distance from the
/// hyperplane projected to non-negative.
fn mean_distance_to_dtlz1_front(front: &[Candidate<Vec<f64>>]) -> f64 {
    if front.is_empty() {
        return f64::INFINITY;
    }
    let total: f64 = front
        .iter()
        .map(|c| {
            let s: f64 = c.evaluation.objectives.iter().sum();
            (s - 0.5).abs()
        })
        .sum();
    total / front.len() as f64
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

fn run_zdt1_comparison() {
    println!(
        "== ZDT1 (dim={ZDT1_DIM}, {ZDT1_BUDGET} evals/run × {SEEDS} seeds) =="
    );
    println!("metric arrows: hypervolume↑ (higher better), others↓ (lower better)");
    println!();
    println!(
        "{:<14} {:>16} {:>14} {:>14} {:>10} {:>10}",
        "algorithm", "hypervolume", "spacing", "mean L2", "front", "ms",
    );
    println!("{}", "-".repeat(82));

    let zdt1 = Zdt1 { dim: ZDT1_DIM };
    let zdt1_objs = zdt1.objectives();

    type Runner = fn(u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", zdt1_random),
        ("PAES", zdt1_paes),
        ("MOPSO", zdt1_mopso),
        ("SPEA2", zdt1_spea2),
        ("PESA-II", zdt1_pesa2),
        ("ε-MOEA", zdt1_epsilon_moea),
        ("IBEA", zdt1_ibea),
        ("HypE", zdt1_hype),
        ("SMS-EMOA", zdt1_sms_emoa),
        ("RVEA", zdt1_rvea),
        ("NSGA-II", zdt1_nsga2),
        ("NSGA-III", zdt1_nsga3),
        ("MOEA/D", zdt1_moead),
    ];

    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(runner).collect();
        let hv: Vec<f64> = runs
            .iter()
            .map(|r| hypervolume_2d(&r.front, &zdt1_objs, ZDT1_REFERENCE))
            .collect();
        let sp: Vec<f64> =
            runs.iter().map(|r| spacing(&r.front, &zdt1_objs)).collect();
        let l2: Vec<f64> =
            runs.iter().map(|r| mean_l2_to_zdt1_front(&r.front)).collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();

        let (hv_m, hv_s) = mean_std(&hv);
        let (sp_m, sp_s) = mean_std(&sp);
        let (l2_m, l2_s) = mean_std(&l2);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);

        println!(
            "{:<14} {:>16} {:>14} {:>14} {:>10} {:>10}",
            name,
            format!("{hv_m:.4}±{hv_s:.4}"),
            format!("{sp_m:.4}±{sp_s:.4}"),
            format!("{l2_m:.4}±{l2_s:.4}"),
            format!("{fs_m:.0}"),
            format!("{ms_m:.0}"),
        );
    }
}

fn run_dtlz2_comparison() {
    println!();
    println!(
        "== DTLZ2 (3-obj, dim={DTLZ2_DIM}, {DTLZ2_BUDGET} evals/run × {SEEDS} seeds) =="
    );
    println!("Pareto front: unit sphere octant (Σf²=1, all f≥0); 'mean dist' is |‖f‖−1|");
    println!();
    println!(
        "{:<14} {:>16} {:>14} {:>10} {:>10}",
        "algorithm", "mean dist↓", "spacing↓", "front", "ms",
    );
    println!("{}", "-".repeat(70));

    let dtlz2 = dtlz2_problem();
    let dtlz2_objs = dtlz2.objectives();

    type Runner = fn(u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", dtlz2_random),
        ("MOPSO", dtlz2_mopso),
        ("NSGA-II", dtlz2_nsga2),
        ("SPEA2", dtlz2_spea2),
        ("PESA-II", dtlz2_pesa2),
        ("ε-MOEA", dtlz2_epsilon_moea),
        ("IBEA", dtlz2_ibea),
        ("HypE", dtlz2_hype),
        ("SMS-EMOA", dtlz2_sms_emoa),
        ("RVEA", dtlz2_rvea),
        ("NSGA-III", dtlz2_nsga3),
        ("MOEA/D", dtlz2_moead),
    ];

    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(runner).collect();
        let dist: Vec<f64> =
            runs.iter().map(|r| mean_distance_to_dtlz2_front(&r.front)).collect();
        let sp: Vec<f64> =
            runs.iter().map(|r| spacing(&r.front, &dtlz2_objs)).collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();

        let (d_m, d_s) = mean_std(&dist);
        let (sp_m, sp_s) = mean_std(&sp);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);

        println!(
            "{:<14} {:>16} {:>14} {:>10} {:>10}",
            name,
            format!("{d_m:.4}±{d_s:.4}"),
            format!("{sp_m:.4}±{sp_s:.4}"),
            format!("{fs_m:.0}"),
            format!("{ms_m:.0}"),
        );
    }
}

fn run_rastrigin_comparison() {
    println!();
    println!(
        "== Rastrigin (dim={RASTRIGIN_DIM}, {RASTRIGIN_BUDGET} evals/run × {SEEDS} seeds) =="
    );
    println!("global minimum: f = 0  (lower is better)");
    println!();
    println!("{:<14} {:>20} {:>10}", "algorithm", "best f", "ms");
    println!("{}", "-".repeat(48));

    type Runner = fn(u64) -> SoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", rastrigin_random),
        ("HillClimber", rastrigin_hill_climber),
        ("SimulatedAnneal", rastrigin_simulated_annealing),
        ("PAES", rastrigin_paes),
        ("GA", rastrigin_genetic_algorithm),
        ("PSO", rastrigin_particle_swarm),
        ("NSGA-II", rastrigin_nsga2),
        ("DE", rastrigin_de),
        ("CMA-ES", rastrigin_cma_es),
    ];

    for (name, runner) in runners {
        let runs: Vec<SoRun> = (0..SEEDS).map(runner).collect();
        let best: Vec<f64> = runs.iter().map(|r| r.best_value).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();

        let (b_m, b_s) = mean_std(&best);
        let (ms_m, _) = mean_std(&ms);

        println!(
            "{:<14} {:>20} {:>10}",
            name,
            format!("{b_m:.4e} ± {b_s:.2e}"),
            format!("{ms_m:.0}"),
        );
    }
}

fn run_rosenbrock_comparison() {
    println!();
    println!("== Rosenbrock (dim={ROSENBROCK_DIM}, {ROSENBROCK_BUDGET} evals × {SEEDS} seeds) ==");
    println!("smooth non-convex valley; global minimum f = 0 at all-ones");
    println!();
    println!("{:<14} {:>20} {:>10}", "algorithm", "best f", "ms");
    println!("{}", "-".repeat(48));
    type Runner = fn(u64) -> SoRun;
    let runners: &[(&str, Runner)] = &[
        ("DE", rosenbrock_de),
        ("PSO", rosenbrock_pso),
        ("CMA-ES", rosenbrock_cma),
        ("TLBO", rosenbrock_tlbo),
    ];
    for (name, runner) in runners {
        let runs: Vec<SoRun> = (0..SEEDS).map(runner).collect();
        let best: Vec<f64> = runs.iter().map(|r| r.best_value).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (b_m, b_s) = mean_std(&best);
        let (ms_m, _) = mean_std(&ms);
        println!(
            "{:<14} {:>20} {:>10}",
            name,
            format!("{b_m:.4e} ± {b_s:.2e}"),
            format!("{ms_m:.0}"),
        );
    }
}

fn run_ackley_comparison() {
    println!();
    println!("== Ackley (dim={ACKLEY_DIM}, {ACKLEY_BUDGET} evals × {SEEDS} seeds) ==");
    println!("smoother multimodal landscape than Rastrigin; global minimum f = 0 at origin");
    println!();
    println!("{:<14} {:>20} {:>10}", "algorithm", "best f", "ms");
    println!("{}", "-".repeat(48));
    type Runner = fn(u64) -> SoRun;
    let runners: &[(&str, Runner)] = &[
        ("DE", ackley_de),
        ("PSO", ackley_pso),
        ("CMA-ES", ackley_cma),
        ("TLBO", ackley_tlbo),
    ];
    for (name, runner) in runners {
        let runs: Vec<SoRun> = (0..SEEDS).map(runner).collect();
        let best: Vec<f64> = runs.iter().map(|r| r.best_value).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (b_m, b_s) = mean_std(&best);
        let (ms_m, _) = mean_std(&ms);
        println!(
            "{:<14} {:>20} {:>10}",
            name,
            format!("{b_m:.4e} ± {b_s:.2e}"),
            format!("{ms_m:.0}"),
        );
    }
}

fn run_zdt3_comparison() {
    println!();
    println!("== ZDT3 (dim={ZDT3_DIM}, {ZDT3_BUDGET} evals × {SEEDS} seeds) ==");
    println!("disconnected Pareto front (not contiguous); spread across gaps matters");
    println!();
    println!(
        "{:<14} {:>16} {:>14} {:>10} {:>10}",
        "algorithm", "hypervolume↑", "spacing↓", "front", "ms",
    );
    println!("{}", "-".repeat(70));
    let problem = zdt3_problem();
    let objs = problem.objectives();
    type Runner = fn(u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("NSGA-II", zdt3_nsga2),
        ("MOEA/D", zdt3_moead),
        ("IBEA", zdt3_ibea),
        ("AGE-MOEA", zdt3_age_moea),
    ];
    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(runner).collect();
        let hv: Vec<f64> = runs.iter().map(|r| hypervolume_2d(&r.front, &objs, ZDT3_REFERENCE)).collect();
        let sp: Vec<f64> = runs.iter().map(|r| spacing(&r.front, &objs)).collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (hv_m, hv_s) = mean_std(&hv);
        let (sp_m, sp_s) = mean_std(&sp);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);
        println!(
            "{:<14} {:>16} {:>14} {:>10} {:>10}",
            name,
            format!("{hv_m:.4}±{hv_s:.4}"),
            format!("{sp_m:.4}±{sp_s:.4}"),
            format!("{fs_m:.0}"),
            format!("{ms_m:.0}"),
        );
    }
}

fn run_dtlz1_comparison() {
    println!();
    println!("== DTLZ1 (3-obj, dim={DTLZ1_DIM}, {DTLZ1_BUDGET} evals × {SEEDS} seeds) ==");
    println!("Pareto front: linear simplex Σf=0.5 in the positive octant");
    println!();
    println!(
        "{:<14} {:>16} {:>14} {:>10} {:>10}",
        "algorithm", "mean dist↓", "spacing↓", "front", "ms",
    );
    println!("{}", "-".repeat(70));
    let problem = dtlz1_problem();
    let objs = problem.objectives();
    type Runner = fn(u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("NSGA-III", dtlz1_nsga3),
        ("MOEA/D", dtlz1_moead),
        ("AGE-MOEA", dtlz1_age_moea),
        ("GrEA", dtlz1_grea),
    ];
    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(runner).collect();
        let dist: Vec<f64> = runs.iter().map(|r| mean_distance_to_dtlz1_front(&r.front)).collect();
        let sp: Vec<f64> = runs.iter().map(|r| spacing(&r.front, &objs)).collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (d_m, d_s) = mean_std(&dist);
        let (sp_m, sp_s) = mean_std(&sp);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);
        println!(
            "{:<14} {:>16} {:>14} {:>10} {:>10}",
            name,
            format!("{d_m:.4}±{d_s:.4}"),
            format!("{sp_m:.4}±{sp_s:.4}"),
            format!("{fs_m:.0}"),
            format!("{ms_m:.0}"),
        );
    }
}

fn main() {
    run_zdt1_comparison();
    run_zdt3_comparison();
    run_dtlz2_comparison();
    run_dtlz1_comparison();
    run_rastrigin_comparison();
    run_rosenbrock_comparison();
    run_ackley_comparison();
}
