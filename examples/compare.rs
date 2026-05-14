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

use rand::Rng as _;

use heuropt::core::rng::Rng;
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
        let f = -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
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
            + x.iter()
                .map(|v| v * v - 10.0 * (2.0 * PI * v).cos())
                .sum::<f64>();
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

/// Print an aligned text table: column 0 left-justified, the rest
/// right-justified. Column widths are derived from the actual cell
/// contents (header *and* every row), so the separator and all rows line
/// up no matter how the value magnitudes vary.
///
/// All cells must be ASCII — width is measured with `str::len`, so a
/// multi-byte character (e.g. `±`, `ε`) would silently break alignment on
/// terminals that render it at a different column width. Callers format
/// `mean +/- std` rather than `mean ± std` for exactly this reason.
fn print_table(header: &[&str], rows: &[Vec<String>]) {
    let ncols = header.len();
    let mut widths: Vec<usize> = header.iter().map(|h| h.len()).collect();
    for row in rows {
        for c in 0..ncols {
            widths[c] = widths[c].max(row[c].len());
        }
    }
    let fmt_row = |cells: &[String]| -> String {
        let mut out = String::new();
        for c in 0..ncols {
            if c > 0 {
                out.push_str("  ");
            }
            let w = widths[c];
            if c == 0 {
                out.push_str(&format!("{:<w$}", cells[c]));
            } else {
                out.push_str(&format!("{:>w$}", cells[c]));
            }
        }
        out
    };
    let header_owned: Vec<String> = header.iter().map(|s| s.to_string()).collect();
    let header_line = fmt_row(&header_owned);
    println!("{header_line}");
    println!("{}", "-".repeat(header_line.len()));
    for row in rows {
        println!("{}", fmt_row(row));
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    let config = Nsga2Config {
        population_size: pop,
        generations: gens,
        seed,
    };
    let mut opt = Nsga2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    let config = IbeaConfig {
        population_size: pop,
        generations: gens,
        kappa: 0.05,
        seed,
    };
    let mut opt = Ibea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

// -----------------------------------------------------------------------------
// DTLZ2 algorithm runners (3-objective)
// -----------------------------------------------------------------------------

fn dtlz2_problem() -> Dtlz2 {
    Dtlz2 {
        num_objectives: DTLZ2_OBJECTIVES,
        dim: DTLZ2_DIM,
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    let config = Nsga2Config {
        population_size: pop,
        generations: gens,
        seed,
    };
    let mut opt = Nsga2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    let config = IbeaConfig {
        population_size: pop,
        generations: gens,
        kappa: 0.05,
        seed,
    };
    let mut opt = Ibea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
            let norm: f64 = c
                .evaluation
                .objectives
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
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
    let config = Nsga2Config {
        population_size: pop,
        generations: gens,
        seed,
    };
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
    let config = HillClimberConfig {
        iterations: RASTRIGIN_BUDGET,
        seed,
    };
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
        initial_mean: None,
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

fn rastrigin_ipop_cma_es(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let bounds = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let pop = 16;
    let config = IpopCmaEsConfig {
        initial_population_size: pop,
        total_generations: RASTRIGIN_BUDGET / pop,
        initial_sigma: 1.0,
        eigen_decomposition_period: 1,
        stall_generations: None,
        seed,
    };
    let mut opt = IpopCmaEs::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rastrigin_one_plus_one_es(seed: u64) -> SoRun {
    let problem = Rastrigin { dim: RASTRIGIN_DIM };
    let bounds = RealBounds::new(vec![(-5.12, 5.12); RASTRIGIN_DIM]);
    let config = OnePlusOneEsConfig {
        iterations: RASTRIGIN_BUDGET,
        initial_sigma: 1.0,
        adaptation_period: 50,
        step_increase: 1.22,
        seed,
    };
    let mut opt = OnePlusOneEs::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rosenbrock_nelder_mead(_seed: u64) -> SoRun {
    let problem = rosenbrock_problem();
    let bounds = RealBounds::new(vec![(-5.0, 10.0); ROSENBROCK_DIM]);
    let config = NelderMeadConfig {
        iterations: ROSENBROCK_BUDGET / 4,
        ..NelderMeadConfig::default()
    };
    let mut opt = NelderMead::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rosenbrock_one_plus_one_es(seed: u64) -> SoRun {
    let problem = rosenbrock_problem();
    let bounds = RealBounds::new(vec![(-5.0, 10.0); ROSENBROCK_DIM]);
    let config = OnePlusOneEsConfig {
        iterations: ROSENBROCK_BUDGET,
        initial_sigma: 1.0,
        adaptation_period: 30,
        step_increase: 1.22,
        seed,
    };
    let mut opt = OnePlusOneEs::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn rosenbrock_bo(seed: u64) -> SoRun {
    let problem = rosenbrock_problem();
    let bounds = RealBounds::new(vec![(-5.0, 10.0); ROSENBROCK_DIM]);
    let config = BayesianOptConfig {
        initial_samples: 10,
        iterations: 50,
        length_scales: None,
        signal_variance: 1.0,
        noise_variance: 1e-6,
        acquisition_samples: 1_000,
        seed,
    };
    let mut opt = BayesianOpt::new(config, bounds);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn ackley_bo(seed: u64) -> SoRun {
    let problem = ackley_problem();
    let bounds = RealBounds::new(vec![(-32.768, 32.768); ACKLEY_DIM]);
    let config = BayesianOptConfig {
        initial_samples: 10,
        iterations: 50,
        length_scales: None,
        signal_variance: 1.0,
        noise_variance: 1e-6,
        acquisition_samples: 1_000,
        seed,
    };
    let mut opt = BayesianOpt::new(config, bounds);
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
    Rosenbrock {
        dim: ROSENBROCK_DIM,
    }
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
            initial_mean: None,
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
        let config = TlboConfig {
            population_size: pop,
            generations: gens,
            seed: $seed,
        };
        let mut opt = Tlbo::new(config, bounds);
        let t0 = Instant::now();
        let result = opt.run(&problem);
        SoRun {
            best_value: result.best.unwrap().evaluation.objectives[0],
            wall_ms: t0.elapsed().as_millis(),
        }
    }};
}

fn rosenbrock_de(seed: u64) -> SoRun {
    so_run_de!(
        rosenbrock_problem(),
        ROSENBROCK_DIM,
        -5.0,
        10.0,
        ROSENBROCK_BUDGET,
        seed
    )
}
fn rosenbrock_cma(seed: u64) -> SoRun {
    so_run_cma!(
        rosenbrock_problem(),
        ROSENBROCK_DIM,
        -5.0,
        10.0,
        ROSENBROCK_BUDGET,
        seed
    )
}
fn rosenbrock_pso(seed: u64) -> SoRun {
    so_run_pso!(
        rosenbrock_problem(),
        ROSENBROCK_DIM,
        -5.0,
        10.0,
        ROSENBROCK_BUDGET,
        seed
    )
}
fn rosenbrock_tlbo(seed: u64) -> SoRun {
    so_run_tlbo!(
        rosenbrock_problem(),
        ROSENBROCK_DIM,
        -5.0,
        10.0,
        ROSENBROCK_BUDGET,
        seed
    )
}

fn ackley_de(seed: u64) -> SoRun {
    so_run_de!(
        ackley_problem(),
        ACKLEY_DIM,
        -32.768,
        32.768,
        ACKLEY_BUDGET,
        seed
    )
}
fn ackley_cma(seed: u64) -> SoRun {
    so_run_cma!(
        ackley_problem(),
        ACKLEY_DIM,
        -32.768,
        32.768,
        ACKLEY_BUDGET,
        seed
    )
}
fn ackley_pso(seed: u64) -> SoRun {
    so_run_pso!(
        ackley_problem(),
        ACKLEY_DIM,
        -32.768,
        32.768,
        ACKLEY_BUDGET,
        seed
    )
}
fn ackley_tlbo(seed: u64) -> SoRun {
    so_run_tlbo!(
        ackley_problem(),
        ACKLEY_DIM,
        -32.768,
        32.768,
        ACKLEY_BUDGET,
        seed
    )
}

// -----------------------------------------------------------------------------
// ZDT3 runners (curated MO subset)
// -----------------------------------------------------------------------------

fn zdt3_problem() -> Zdt3 {
    Zdt3 { dim: ZDT3_DIM }
}

fn zdt3_nsga2(seed: u64) -> MoRun {
    let problem = zdt3_problem();
    let bounds = vec![(0.0, 1.0); ZDT3_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT3_DIM as f64),
    };
    let pop = 100;
    let config = Nsga2Config {
        population_size: pop,
        generations: ZDT3_BUDGET / pop,
        seed,
    };
    let mut opt = Nsga2::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    let config = IbeaConfig {
        population_size: pop,
        generations: ZDT3_BUDGET / pop,
        kappa: 0.05,
        seed,
    };
    let mut opt = Ibea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    let config = AgeMoeaConfig {
        population_size: pop,
        generations: ZDT3_BUDGET / pop,
        seed,
    };
    let mut opt = AgeMoea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn zdt3_knea(seed: u64) -> MoRun {
    let problem = zdt3_problem();
    let bounds = vec![(0.0, 1.0); ZDT3_DIM];
    let initializer = RealBounds::new(bounds.clone());
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.5),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / ZDT3_DIM as f64),
    };
    let pop = 100;
    let config = KneaConfig {
        population_size: pop,
        generations: ZDT3_BUDGET / pop,
        seed,
    };
    let mut opt = Knea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

// -----------------------------------------------------------------------------
// DTLZ1 runners (curated many-obj subset)
// -----------------------------------------------------------------------------

fn dtlz1_problem() -> Dtlz1 {
    Dtlz1 {
        num_objectives: DTLZ1_OBJECTIVES,
        dim: DTLZ1_DIM,
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    let config = AgeMoeaConfig {
        population_size: pop,
        generations: DTLZ1_BUDGET / pop,
        seed,
    };
    let mut opt = AgeMoea::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
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
    println!("== ZDT1 (dim={ZDT1_DIM}, {ZDT1_BUDGET} evals/run × {SEEDS} seeds) ==");
    println!("Zitzler-Deb-Thiele 2-objective benchmark: {ZDT1_DIM} real variables, one smooth");
    println!("convex Pareto front f2 = 1 - sqrt(f1). Hard because 29 of 30 variables must");
    println!("collapse to 0 before the front is even reachable, and only then can the");
    println!("population spread along it. Optimum: mean L2 -> 0 (the front is known exactly).");
    println!("sorted best-first by hypervolume (higher better; spacing / mean L2 lower better)");
    println!();

    let zdt1 = Zdt1 { dim: ZDT1_DIM };
    let zdt1_objs = zdt1.objectives();

    type Runner = fn(u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", zdt1_random),
        ("PAES", zdt1_paes),
        ("MOPSO", zdt1_mopso),
        ("SPEA2", zdt1_spea2),
        ("PESA-II", zdt1_pesa2),
        ("eps-MOEA", zdt1_epsilon_moea),
        ("IBEA", zdt1_ibea),
        ("HypE", zdt1_hype),
        ("SMS-EMOA", zdt1_sms_emoa),
        ("RVEA", zdt1_rvea),
        ("NSGA-II", zdt1_nsga2),
        ("NSGA-III", zdt1_nsga3),
        ("MOEA/D", zdt1_moead),
    ];

    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(runner).collect();
        let hv: Vec<f64> = runs
            .iter()
            .map(|r| hypervolume_2d(&r.front, &zdt1_objs, ZDT1_REFERENCE))
            .collect();
        let sp: Vec<f64> = runs.iter().map(|r| spacing(&r.front, &zdt1_objs)).collect();
        let l2: Vec<f64> = runs
            .iter()
            .map(|r| mean_l2_to_zdt1_front(&r.front))
            .collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();

        let (hv_m, hv_s) = mean_std(&hv);
        let (sp_m, sp_s) = mean_std(&sp);
        let (l2_m, l2_s) = mean_std(&l2);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);

        rows.push((
            hv_m,
            vec![
                name.to_string(),
                format!("{hv_m:.4}+/-{hv_s:.4}"),
                format!("{sp_m:.4}+/-{sp_s:.4}"),
                format!("{l2_m:.4}+/-{l2_s:.4}"),
                format!("{fs_m:.0}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    // Higher hypervolume is better.
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(
        &[
            "algorithm",
            "hypervolume",
            "spacing",
            "mean L2",
            "front",
            "ms",
        ],
        &table,
    );
}

fn run_dtlz2_comparison() {
    println!();
    println!("== DTLZ2 (3-obj, dim={DTLZ2_DIM}, {DTLZ2_BUDGET} evals/run × {SEEDS} seeds) ==");
    println!("Deb-Thiele-Laumanns-Zitzler 3-objective; the Pareto front is the unit-sphere");
    println!("octant (Σf² = 1, all f >= 0) -- a curved 2-D surface embedded in 3-D objective");
    println!("space. Hard because crowding/spacing must work in a higher dimension.");
    println!("'mean dist' = |‖f‖ - 1|, so 0 means perfectly on the sphere (the known optimum).");
    println!("sorted best-first by mean dist (lower is better)");
    println!();

    let dtlz2 = dtlz2_problem();
    let dtlz2_objs = dtlz2.objectives();

    type Runner = fn(u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", dtlz2_random),
        ("MOPSO", dtlz2_mopso),
        ("NSGA-II", dtlz2_nsga2),
        ("SPEA2", dtlz2_spea2),
        ("PESA-II", dtlz2_pesa2),
        ("eps-MOEA", dtlz2_epsilon_moea),
        ("IBEA", dtlz2_ibea),
        ("HypE", dtlz2_hype),
        ("SMS-EMOA", dtlz2_sms_emoa),
        ("RVEA", dtlz2_rvea),
        ("NSGA-III", dtlz2_nsga3),
        ("MOEA/D", dtlz2_moead),
    ];

    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(runner).collect();
        let dist: Vec<f64> = runs
            .iter()
            .map(|r| mean_distance_to_dtlz2_front(&r.front))
            .collect();
        let sp: Vec<f64> = runs
            .iter()
            .map(|r| spacing(&r.front, &dtlz2_objs))
            .collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();

        let (d_m, d_s) = mean_std(&dist);
        let (sp_m, sp_s) = mean_std(&sp);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);

        rows.push((
            d_m,
            vec![
                name.to_string(),
                format!("{d_m:.4}+/-{d_s:.4}"),
                format!("{sp_m:.4}+/-{sp_s:.4}"),
                format!("{fs_m:.0}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    // Lower mean distance to the true front is better.
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(
        &["algorithm", "mean dist", "spacing", "front", "ms"],
        &table,
    );
}

fn run_rastrigin_comparison() {
    println!();
    println!("== Rastrigin (dim={RASTRIGIN_DIM}, {RASTRIGIN_BUDGET} evals/run × {SEEDS} seeds) ==");
    println!(
        "Highly multimodal trap: f = 10n + Σ(x_i² - 10·cos(2π·x_i)), {RASTRIGIN_DIM} dimensions."
    );
    println!(
        "Hard because a near-quadratic global bowl is overlaid with ~10^{RASTRIGIN_DIM} regularly"
    );
    println!("spaced local minima -- any greedy step lands in the nearest dimple.");
    println!("Global optimum: f = 0 at the origin.");
    println!("sorted best-first (lower is better)");
    println!();

    type Runner = fn(u64) -> SoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", rastrigin_random),
        ("HillClimber", rastrigin_hill_climber),
        ("(1+1)-ES", rastrigin_one_plus_one_es),
        ("SimulatedAnneal", rastrigin_simulated_annealing),
        ("PAES", rastrigin_paes),
        ("GA", rastrigin_genetic_algorithm),
        ("PSO", rastrigin_particle_swarm),
        ("NSGA-II", rastrigin_nsga2),
        ("DE", rastrigin_de),
        ("CMA-ES", rastrigin_cma_es),
        ("IPOP-CMA-ES", rastrigin_ipop_cma_es),
    ];

    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<SoRun> = (0..SEEDS).map(runner).collect();
        let best: Vec<f64> = runs.iter().map(|r| r.best_value).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();

        let (b_m, b_s) = mean_std(&best);
        let (ms_m, _) = mean_std(&ms);

        rows.push((
            b_m,
            vec![
                name.to_string(),
                format!("{b_m:.4e} +/- {b_s:.2e}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    // Lower best f is better.
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(&["algorithm", "best f", "ms"], &table);
}

fn run_rosenbrock_comparison() {
    println!();
    println!("== Rosenbrock (dim={ROSENBROCK_DIM}, {ROSENBROCK_BUDGET} evals × {SEEDS} seeds) ==");
    println!(
        "Rosenbrock's banana valley: f = Σ(100·(x_{{i+1}} - x_i²)² + (1 - x_i)²), {ROSENBROCK_DIM} dims."
    );
    println!("Hard because the minimum sits in a long, bent, near-flat valley -- easy to enter,");
    println!("very slow to crawl along to the tip. Global optimum: f = 0 at the all-ones point.");
    println!("sorted best-first (lower is better)");
    println!();
    type Runner = fn(u64) -> SoRun;
    let runners: &[(&str, Runner)] = &[
        ("DE", rosenbrock_de),
        ("PSO", rosenbrock_pso),
        ("CMA-ES", rosenbrock_cma),
        ("TLBO", rosenbrock_tlbo),
        ("(1+1)-ES", rosenbrock_one_plus_one_es),
        ("Nelder-Mead", rosenbrock_nelder_mead),
        ("BO (60 evals)", rosenbrock_bo),
    ];
    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<SoRun> = (0..SEEDS).map(runner).collect();
        let best: Vec<f64> = runs.iter().map(|r| r.best_value).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (b_m, b_s) = mean_std(&best);
        let (ms_m, _) = mean_std(&ms);
        rows.push((
            b_m,
            vec![
                name.to_string(),
                format!("{b_m:.4e} +/- {b_s:.2e}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(&["algorithm", "best f", "ms"], &table);
}

fn run_ackley_comparison() {
    println!();
    println!("== Ackley (dim={ACKLEY_DIM}, {ACKLEY_BUDGET} evals × {SEEDS} seeds) ==");
    println!("Ackley's function: a near-flat outer plateau with shallow ripples surrounding a");
    println!(
        "single deep, narrow global basin, {ACKLEY_DIM} dimensions. Hard because the gradient is"
    );
    println!("almost zero far from the optimum, giving local search little to follow.");
    println!("Global optimum: f = 0 at the origin.");
    println!("sorted best-first (lower is better)");
    println!();
    type Runner = fn(u64) -> SoRun;
    let runners: &[(&str, Runner)] = &[
        ("DE", ackley_de),
        ("PSO", ackley_pso),
        ("CMA-ES", ackley_cma),
        ("TLBO", ackley_tlbo),
        ("BO (60 evals)", ackley_bo),
    ];
    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<SoRun> = (0..SEEDS).map(runner).collect();
        let best: Vec<f64> = runs.iter().map(|r| r.best_value).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (b_m, b_s) = mean_std(&best);
        let (ms_m, _) = mean_std(&ms);
        rows.push((
            b_m,
            vec![
                name.to_string(),
                format!("{b_m:.4e} +/- {b_s:.2e}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(&["algorithm", "best f", "ms"], &table);
}

fn run_zdt3_comparison() {
    println!();
    println!("== ZDT3 (dim={ZDT3_DIM}, {ZDT3_BUDGET} evals × {SEEDS} seeds) ==");
    println!("Zitzler-Deb-Thiele 2-objective with a DISCONNECTED front: five separate arcs");
    println!("rather than one curve. Hard because an algorithm has to discover and populate");
    println!("every arc while not stranding solutions in the dominated gaps between them.");
    println!("Optimum: cover all five arcs; scored by hypervolume vs [11, 11].");
    println!("sorted best-first by hypervolume (higher better; spacing lower better)");
    println!();
    let problem = zdt3_problem();
    let objs = problem.objectives();
    type Runner = fn(u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("NSGA-II", zdt3_nsga2),
        ("MOEA/D", zdt3_moead),
        ("IBEA", zdt3_ibea),
        ("AGE-MOEA", zdt3_age_moea),
        ("KnEA", zdt3_knea),
    ];
    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(runner).collect();
        let hv: Vec<f64> = runs
            .iter()
            .map(|r| hypervolume_2d(&r.front, &objs, ZDT3_REFERENCE))
            .collect();
        let sp: Vec<f64> = runs.iter().map(|r| spacing(&r.front, &objs)).collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (hv_m, hv_s) = mean_std(&hv);
        let (sp_m, sp_s) = mean_std(&sp);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);
        rows.push((
            hv_m,
            vec![
                name.to_string(),
                format!("{hv_m:.4}+/-{hv_s:.4}"),
                format!("{sp_m:.4}+/-{sp_s:.4}"),
                format!("{fs_m:.0}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(
        &["algorithm", "hypervolume", "spacing", "front", "ms"],
        &table,
    );
}

fn run_dtlz1_comparison() {
    println!();
    println!("== DTLZ1 (3-obj, dim={DTLZ1_DIM}, {DTLZ1_BUDGET} evals × {SEEDS} seeds) ==");
    println!("Deb-Thiele-Laumanns-Zitzler 3-objective; the Pareto front is the linear simplex");
    println!("Σf = 0.5 in the positive octant. Hard because a deceptive multimodal 'g' term");
    println!("riddles the approach with a huge number of local fronts -- only fully-converged");
    println!("runs land on the simplex. Optimum: mean dist -> 0.");
    println!("sorted best-first by mean dist (lower is better)");
    println!();
    let problem = dtlz1_problem();
    let objs = problem.objectives();
    type Runner = fn(u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("NSGA-III", dtlz1_nsga3),
        ("MOEA/D", dtlz1_moead),
        ("AGE-MOEA", dtlz1_age_moea),
        ("GrEA", dtlz1_grea),
    ];
    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(runner).collect();
        let dist: Vec<f64> = runs
            .iter()
            .map(|r| mean_distance_to_dtlz1_front(&r.front))
            .collect();
        let sp: Vec<f64> = runs.iter().map(|r| spacing(&r.front, &objs)).collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (d_m, d_s) = mean_std(&dist);
        let (sp_m, sp_s) = mean_std(&sp);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);
        rows.push((
            d_m,
            vec![
                name.to_string(),
                format!("{d_m:.4}+/-{d_s:.4}"),
                format!("{sp_m:.4}+/-{sp_s:.4}"),
                format!("{fs_m:.0}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(
        &["algorithm", "mean dist", "spacing", "front", "ms"],
        &table,
    );
}

// =============================================================================
// Combinatorial / sequencing problems
//
// These use a different decision encoding — `Vec<usize>` permutations and
// `Vec<bool>` bitstrings — than the continuous problems above, so the
// applicable-algorithm roster differs: the real-vector methods (CMA-ES, DE,
// PSO, …) cannot run here, while permutation-native methods (Ant Colony,
// Tabu Search) and the EDA-style ones can.
// =============================================================================

const TSP_CITIES: usize = 15;
const TSP_BUDGET: usize = 8_000;
const JSS_BUDGET: usize = 8_000;
const KNAPSACK_BUDGET: usize = 20_000;

// ---- Single-objective TSP: a regular polygon on the unit circle -------------

/// `TSP_CITIES` points equally spaced on the unit circle. The optimal tour
/// just visits them in angular order; its length is the regular-polygon
/// perimeter `2·n·sin(π/n)`, which gives an exact known-best baseline.
struct RingTsp {
    distances: Vec<Vec<f64>>,
}
impl RingTsp {
    fn new() -> Self {
        let pts: Vec<(f64, f64)> = (0..TSP_CITIES)
            .map(|i| {
                let a = 2.0 * PI * (i as f64) / (TSP_CITIES as f64);
                (a.cos(), a.sin())
            })
            .collect();
        let distances = (0..TSP_CITIES)
            .map(|i| {
                (0..TSP_CITIES)
                    .map(|j| {
                        let (xi, yi) = pts[i];
                        let (xj, yj) = pts[j];
                        ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt()
                    })
                    .collect()
            })
            .collect();
        Self { distances }
    }
    /// Known optimal tour length — the regular-polygon perimeter.
    fn optimal_length() -> f64 {
        2.0 * TSP_CITIES as f64 * (PI / TSP_CITIES as f64).sin()
    }
}
impl Problem for RingTsp {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("tour_length")])
    }
    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        let n = tour.len();
        let mut total = 0.0;
        for i in 0..n {
            total += self.distances[tour[i]][tour[(i + 1) % n]];
        }
        Evaluation::new(vec![total])
    }
}

fn tsp_random(seed: u64) -> SoRun {
    let problem = RingTsp::new();
    let mut opt = RandomSearch::new(
        RandomSearchConfig {
            iterations: TSP_BUDGET,
            batch_size: 1,
            seed,
        },
        ShuffledPermutation { n: TSP_CITIES },
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn tsp_hill_climber(seed: u64) -> SoRun {
    let problem = RingTsp::new();
    let mut opt = HillClimber::new(
        HillClimberConfig {
            iterations: TSP_BUDGET,
            seed,
        },
        ShuffledPermutation { n: TSP_CITIES },
        InversionMutation,
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn tsp_simulated_annealing(seed: u64) -> SoRun {
    let problem = RingTsp::new();
    let mut opt = SimulatedAnnealing::new(
        SimulatedAnnealingConfig {
            iterations: TSP_BUDGET,
            initial_temperature: 1.0,
            final_temperature: 1e-3,
            seed,
        },
        ShuffledPermutation { n: TSP_CITIES },
        InversionMutation,
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn tsp_tabu_search(seed: u64) -> SoRun {
    let problem = RingTsp::new();
    // Each step considers 16 inversion (2-opt-style) neighbours.
    let neighbors = |tour: &Vec<usize>, rng: &mut Rng| {
        let mut m = InversionMutation;
        (0..16)
            .map(|_| m.vary(std::slice::from_ref(tour), rng).pop().unwrap())
            .collect()
    };
    let mut opt = TabuSearch::new(
        TabuSearchConfig {
            iterations: TSP_BUDGET / 16,
            tabu_tenure: 20,
            seed,
        },
        ShuffledPermutation { n: TSP_CITIES },
        neighbors,
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn tsp_genetic_algorithm(seed: u64) -> SoRun {
    let problem = RingTsp::new();
    let mut opt = GeneticAlgorithm::new(
        GeneticAlgorithmConfig {
            population_size: 80,
            generations: TSP_BUDGET / 80,
            tournament_size: 3,
            elitism: 2,
            seed,
        },
        ShuffledPermutation { n: TSP_CITIES },
        CompositeVariation {
            crossover: OrderCrossover,
            mutation: InversionMutation,
        },
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn tsp_ant_colony(seed: u64) -> SoRun {
    let problem = RingTsp::new();
    let distances = problem.distances.clone();
    let mut opt = AntColonyTsp::new(
        AntColonyTspConfig {
            ants: 20,
            generations: TSP_BUDGET / 20,
            alpha: 1.0,
            beta: 3.0,
            evaporation: 0.5,
            deposit: 1.0,
            initial_pheromone: 1.0,
            seed,
        },
        distances,
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

// ---- Single-objective job-shop scheduling: Fisher & Thompson FT06 -----------

const JSS_JOBS: usize = 6;
const JSS_MACHINES: usize = 6;

/// FT06 routing — machine id of the k-th operation of job j.
const FT06_MACHINE: [[usize; JSS_MACHINES]; JSS_JOBS] = [
    [2, 0, 1, 3, 5, 4],
    [1, 2, 4, 5, 0, 3],
    [2, 3, 5, 0, 1, 4],
    [1, 0, 2, 3, 4, 5],
    [2, 1, 4, 5, 0, 3],
    [1, 3, 5, 0, 4, 2],
];
/// FT06 processing times — duration of the k-th operation of job j.
const FT06_TIME: [[f64; JSS_MACHINES]; JSS_JOBS] = [
    [1.0, 3.0, 6.0, 7.0, 3.0, 6.0],
    [8.0, 5.0, 10.0, 10.0, 10.0, 4.0],
    [5.0, 4.0, 8.0, 9.0, 1.0, 7.0],
    [5.0, 5.0, 5.0, 3.0, 8.0, 9.0],
    [9.0, 3.0, 5.0, 4.0, 3.0, 1.0],
    [3.0, 3.0, 9.0, 10.0, 4.0, 1.0],
];
/// FT06's optimal makespan is a long-settled benchmark value.
const FT06_OPTIMAL_MAKESPAN: f64 = 55.0;

/// FT06 job-shop, makespan objective. Operation-string encoding: a length-36
/// multiset where job id `j` appears `JSS_MACHINES` times; the k-th
/// occurrence of `j` is its k-th operation.
struct Ft06Makespan;
impl Problem for Ft06Makespan {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("makespan")])
    }
    fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
        let mut job_next = [0_usize; JSS_JOBS];
        let mut job_clock = [0.0_f64; JSS_JOBS];
        let mut machine_clock = [0.0_f64; JSS_MACHINES];
        for &job in schedule {
            let k = job_next[job];
            let m = FT06_MACHINE[job][k];
            let t = FT06_TIME[job][k];
            let start = job_clock[job].max(machine_clock[m]);
            let end = start + t;
            job_clock[job] = end;
            machine_clock[m] = end;
            job_next[job] = k + 1;
        }
        let makespan = machine_clock.iter().cloned().fold(0.0_f64, f64::max);
        Evaluation::new(vec![makespan])
    }
}

/// Precedence-Order Crossover — multiset-preserving recombination for the
/// operation-string encoding (the strict-permutation crossovers would
/// corrupt the multiset).
#[derive(Debug, Clone, Copy, Default)]
struct PrecedenceOrderCrossover;
impl Variation<Vec<usize>> for PrecedenceOrderCrossover {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        assert!(parents.len() >= 2, "POX requires 2 parents");
        let (p1, p2) = (&parents[0], &parents[1]);
        let mut in_j1 = [false; JSS_JOBS];
        loop {
            for slot in &mut in_j1 {
                *slot = rng.random_bool(0.5);
            }
            let c = in_j1.iter().filter(|&&b| b).count();
            if c > 0 && c < JSS_JOBS {
                break;
            }
        }
        vec![pox_child(p1, p2, &in_j1), pox_child(p2, p1, &in_j1)]
    }
}
fn pox_child(donor: &[usize], filler: &[usize], in_donor_set: &[bool]) -> Vec<usize> {
    let n = donor.len();
    let mut child = vec![usize::MAX; n];
    for k in 0..n {
        if in_donor_set[donor[k]] {
            child[k] = donor[k];
        }
    }
    let mut fill_idx = 0;
    for &v in filler {
        if !in_donor_set[v] {
            while fill_idx < n && child[fill_idx] != usize::MAX {
                fill_idx += 1;
            }
            child[fill_idx] = v;
            fill_idx += 1;
        }
    }
    child
}

fn jss_initializer() -> ShuffledMultisetPermutation {
    ShuffledMultisetPermutation::new(vec![JSS_MACHINES; JSS_JOBS])
}

fn jss_random(seed: u64) -> SoRun {
    let mut opt = RandomSearch::new(
        RandomSearchConfig {
            iterations: JSS_BUDGET,
            batch_size: 1,
            seed,
        },
        jss_initializer(),
    );
    let t0 = Instant::now();
    let result = opt.run(&Ft06Makespan);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn jss_hill_climber(seed: u64) -> SoRun {
    let mut opt = HillClimber::new(
        HillClimberConfig {
            iterations: JSS_BUDGET,
            seed,
        },
        jss_initializer(),
        InsertionMutation,
    );
    let t0 = Instant::now();
    let result = opt.run(&Ft06Makespan);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn jss_simulated_annealing(seed: u64) -> SoRun {
    let mut opt = SimulatedAnnealing::new(
        SimulatedAnnealingConfig {
            iterations: JSS_BUDGET,
            initial_temperature: 5.0,
            final_temperature: 1e-2,
            seed,
        },
        jss_initializer(),
        InsertionMutation,
    );
    let t0 = Instant::now();
    let result = opt.run(&Ft06Makespan);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn jss_tabu_search(seed: u64) -> SoRun {
    let neighbors = |schedule: &Vec<usize>, rng: &mut Rng| {
        let mut m = InsertionMutation;
        (0..16)
            .map(|_| m.vary(std::slice::from_ref(schedule), rng).pop().unwrap())
            .collect()
    };
    let mut opt = TabuSearch::new(
        TabuSearchConfig {
            iterations: JSS_BUDGET / 16,
            tabu_tenure: 20,
            seed,
        },
        jss_initializer(),
        neighbors,
    );
    let t0 = Instant::now();
    let result = opt.run(&Ft06Makespan);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn jss_genetic_algorithm(seed: u64) -> SoRun {
    let mut opt = GeneticAlgorithm::new(
        GeneticAlgorithmConfig {
            population_size: 80,
            generations: JSS_BUDGET / 80,
            tournament_size: 3,
            elitism: 2,
            seed,
        },
        jss_initializer(),
        CompositeVariation {
            crossover: PrecedenceOrderCrossover,
            mutation: InsertionMutation,
        },
    );
    let t0 = Instant::now();
    let result = opt.run(&Ft06Makespan);
    SoRun {
        best_value: result.best.unwrap().evaluation.objectives[0],
        wall_ms: t0.elapsed().as_millis(),
    }
}

// ---- Bi-objective 0/1 knapsack: Zitzler & Thiele style ----------------------

const KNAPSACK_N: usize = 30;
const KP_PROFIT_A: [f64; KNAPSACK_N] = [
    61.0, 17.0, 92.0, 49.0, 73.0, 28.0, 84.0, 36.0, 55.0, 78.0, 23.0, 91.0, 12.0, 67.0, 45.0, 58.0,
    33.0, 71.0, 14.0, 26.0, 87.0, 42.0, 19.0, 65.0, 30.0, 51.0, 79.0, 22.0, 47.0, 88.0,
];
const KP_PROFIT_B: [f64; KNAPSACK_N] = [
    24.0, 81.0, 16.0, 67.0, 29.0, 73.0, 41.0, 60.0, 52.0, 19.0, 77.0, 34.0, 95.0, 22.0, 71.0, 88.0,
    56.0, 27.0, 64.0, 90.0, 18.0, 43.0, 79.0, 31.0, 85.0, 25.0, 38.0, 92.0, 70.0, 13.0,
];
const KP_WEIGHT: [f64; KNAPSACK_N] = [
    35.0, 58.0, 22.0, 71.0, 14.0, 86.0, 31.0, 53.0, 78.0, 19.0, 44.0, 16.0, 67.0, 88.0, 25.0, 51.0,
    33.0, 74.0, 12.0, 47.0, 63.0, 28.0, 91.0, 36.0, 55.0, 17.0, 82.0, 41.0, 24.0, 68.0,
];
/// Hypervolume reference point for the knapsack front. Both objectives are
/// maximized, so in the minimization-oriented frame any positive profit is
/// below 0 — `[0, 0]` is dominated by every feasible solution.
const KNAPSACK_REFERENCE: [f64; 2] = [0.0, 0.0];

/// Bi-objective 0/1 knapsack: two profit vectors, one shared capacity.
/// Weight overruns are penalized in both objectives so the Pareto front is
/// composed of feasible solutions.
struct BiKnapsack {
    capacity: f64,
}
impl BiKnapsack {
    fn new() -> Self {
        Self {
            capacity: 0.5 * KP_WEIGHT.iter().sum::<f64>(),
        }
    }
}
impl Problem for BiKnapsack {
    type Decision = Vec<bool>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::maximize("profit_a"),
            Objective::maximize("profit_b"),
        ])
    }
    fn evaluate(&self, take: &Vec<bool>) -> Evaluation {
        let (mut pa, mut pb, mut w) = (0.0, 0.0, 0.0);
        for (i, &t) in take.iter().enumerate() {
            if t {
                pa += KP_PROFIT_A[i];
                pb += KP_PROFIT_B[i];
                w += KP_WEIGHT[i];
            }
        }
        let penalty = 1000.0 * (w - self.capacity).max(0.0);
        Evaluation::new(vec![pa - penalty, pb - penalty])
    }
}

/// Random binary initializer — each bit 50/50 independently.
#[derive(Debug, Clone, Copy)]
struct RandomBinary {
    n: usize,
}
impl Initializer<Vec<bool>> for RandomBinary {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<Vec<bool>> {
        (0..size)
            .map(|_| (0..self.n).map(|_| rng.random_bool(0.5)).collect())
            .collect()
    }
}

/// One-point crossover for binary chromosomes.
#[derive(Debug, Clone, Copy, Default)]
struct OnePointCrossoverBool;
impl Variation<Vec<bool>> for OnePointCrossoverBool {
    fn vary(&mut self, parents: &[Vec<bool>], rng: &mut Rng) -> Vec<Vec<bool>> {
        assert!(
            parents.len() >= 2,
            "OnePointCrossoverBool requires 2 parents"
        );
        let (p1, p2) = (&parents[0], &parents[1]);
        let n = p1.len();
        if n < 2 {
            return vec![p1.clone(), p2.clone()];
        }
        let cut = rng.random_range(1..n);
        let mut c1 = Vec::with_capacity(n);
        let mut c2 = Vec::with_capacity(n);
        c1.extend_from_slice(&p1[..cut]);
        c1.extend_from_slice(&p2[cut..]);
        c2.extend_from_slice(&p2[..cut]);
        c2.extend_from_slice(&p1[cut..]);
        vec![c1, c2]
    }
}

fn knap_binary_variation() -> CompositeVariation<OnePointCrossoverBool, BitFlipMutation> {
    CompositeVariation {
        crossover: OnePointCrossoverBool,
        mutation: BitFlipMutation {
            probability: 1.0 / KNAPSACK_N as f64,
        },
    }
}

/// One run's knapsack metrics. The bi-objective front lives over `Vec<bool>`
/// decisions, so `MoRun` (which fixes `Vec<f64>`) does not fit — we keep
/// only the aggregate quality numbers.
#[derive(Clone)]
struct KnapRun {
    hypervolume: f64,
    front_size: usize,
    wall_ms: u128,
}

fn knap_run(front: &[Candidate<Vec<bool>>], problem: &BiKnapsack, wall_ms: u128) -> KnapRun {
    KnapRun {
        hypervolume: hypervolume_2d(front, &problem.objectives(), KNAPSACK_REFERENCE),
        front_size: front.len(),
        wall_ms,
    }
}

fn knapsack_random(seed: u64) -> KnapRun {
    let problem = BiKnapsack::new();
    let mut opt = RandomSearch::new(
        RandomSearchConfig {
            iterations: KNAPSACK_BUDGET,
            batch_size: 1,
            seed,
        },
        RandomBinary { n: KNAPSACK_N },
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    let ms = t0.elapsed().as_millis();
    knap_run(&result.pareto_front, &problem, ms)
}

fn knapsack_nsga2(seed: u64) -> KnapRun {
    let problem = BiKnapsack::new();
    let mut opt = Nsga2::new(
        Nsga2Config {
            population_size: 100,
            generations: KNAPSACK_BUDGET / 100,
            seed,
        },
        RandomBinary { n: KNAPSACK_N },
        knap_binary_variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    let ms = t0.elapsed().as_millis();
    knap_run(&result.pareto_front, &problem, ms)
}

fn knapsack_spea2(seed: u64) -> KnapRun {
    let problem = BiKnapsack::new();
    let mut opt = Spea2::new(
        Spea2Config {
            population_size: 100,
            archive_size: 100,
            generations: KNAPSACK_BUDGET / 100,
            seed,
        },
        RandomBinary { n: KNAPSACK_N },
        knap_binary_variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    let ms = t0.elapsed().as_millis();
    knap_run(&result.pareto_front, &problem, ms)
}

fn knapsack_nsga3(seed: u64) -> KnapRun {
    let problem = BiKnapsack::new();
    let mut opt = Nsga3::new(
        Nsga3Config {
            population_size: 100,
            generations: KNAPSACK_BUDGET / 100,
            reference_divisions: 20,
            seed,
        },
        RandomBinary { n: KNAPSACK_N },
        knap_binary_variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    let ms = t0.elapsed().as_millis();
    knap_run(&result.pareto_front, &problem, ms)
}

fn knapsack_ibea(seed: u64) -> KnapRun {
    let problem = BiKnapsack::new();
    let mut opt = Ibea::new(
        IbeaConfig {
            population_size: 100,
            generations: KNAPSACK_BUDGET / 100,
            kappa: 0.05,
            seed,
        },
        RandomBinary { n: KNAPSACK_N },
        knap_binary_variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    let ms = t0.elapsed().as_millis();
    knap_run(&result.pareto_front, &problem, ms)
}

// ---- Combinatorial comparison runners --------------------------------------

fn run_tsp_comparison() {
    println!();
    println!("== TSP ring-{TSP_CITIES} ({TSP_BUDGET} evals/run × {SEEDS} seeds) ==");
    println!(
        "{TSP_CITIES} equally-spaced cities on the unit circle; minimize the closed tour length."
    );
    println!(
        "The space is ({TSP_CITIES}-1)!/2 distinct tours, but cities in convex position have no"
    );
    println!("2-opt local optima -- so this instance cleanly separates methods with good");
    println!("neighbourhood moves (inversion = 2-opt) from blind recombination / sampling.");
    println!(
        "Known optimum (the polygon perimeter): {:.4}",
        RingTsp::optimal_length()
    );
    println!("sorted best-first by tour length (lower is better)");
    println!();

    type Runner = fn(u64) -> SoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", tsp_random),
        ("HillClimber", tsp_hill_climber),
        ("SimulatedAnneal", tsp_simulated_annealing),
        ("TabuSearch", tsp_tabu_search),
        ("GA", tsp_genetic_algorithm),
        ("AntColony", tsp_ant_colony),
    ];

    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<SoRun> = (0..SEEDS).map(runner).collect();
        let best: Vec<f64> = runs.iter().map(|r| r.best_value).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (b_m, b_s) = mean_std(&best);
        let (ms_m, _) = mean_std(&ms);
        rows.push((
            b_m,
            vec![
                name.to_string(),
                format!("{b_m:.4}+/-{b_s:.4}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(&["algorithm", "tour length", "ms"], &table);
}

fn run_jss_comparison() {
    println!();
    println!("== JSS FT06 ({JSS_BUDGET} evals/run × {SEEDS} seeds) ==");
    println!("Fisher & Thompson 1963 6-job × 6-machine job-shop; minimize makespan.");
    println!("Hard because every job has a fixed machine order, so swapping two");
    println!(
        "operations can ripple delays across the whole schedule. Known optimum: {FT06_OPTIMAL_MAKESPAN:.0}"
    );
    println!("sorted best-first by makespan (lower is better)");
    println!();

    type Runner = fn(u64) -> SoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", jss_random),
        ("HillClimber", jss_hill_climber),
        ("SimulatedAnneal", jss_simulated_annealing),
        ("TabuSearch", jss_tabu_search),
        ("GA", jss_genetic_algorithm),
    ];

    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<SoRun> = (0..SEEDS).map(runner).collect();
        let best: Vec<f64> = runs.iter().map(|r| r.best_value).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (b_m, b_s) = mean_std(&best);
        let (ms_m, _) = mean_std(&ms);
        rows.push((
            b_m,
            vec![
                name.to_string(),
                format!("{b_m:.4}+/-{b_s:.4}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(&["algorithm", "makespan", "ms"], &table);
}

fn run_knapsack_comparison() {
    println!();
    println!(
        "== Knapsack ({KNAPSACK_N} items, bi-objective, {KNAPSACK_BUDGET} evals/run × {SEEDS} seeds) =="
    );
    println!("Zitzler-Thiele style 0/1 knapsack: two profit vectors, one capacity");
    println!("(half the total weight). Hard because the two profit objectives");
    println!("conflict and the capacity constraint carves feasible regions out of");
    println!("the 2^{KNAPSACK_N} bitstrings. No closed-form optimum; scored by hypervolume");
    println!("vs reference {KNAPSACK_REFERENCE:?} (higher is better).");
    println!("sorted best-first by hypervolume");
    println!();

    type Runner = fn(u64) -> KnapRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", knapsack_random),
        ("NSGA-II", knapsack_nsga2),
        ("SPEA2", knapsack_spea2),
        ("NSGA-III", knapsack_nsga3),
        ("IBEA", knapsack_ibea),
    ];

    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<KnapRun> = (0..SEEDS).map(runner).collect();
        let hv: Vec<f64> = runs.iter().map(|r| r.hypervolume).collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front_size as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (hv_m, hv_s) = mean_std(&hv);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);
        rows.push((
            hv_m,
            vec![
                name.to_string(),
                format!("{hv_m:.1}+/-{hv_s:.1}"),
                format!("{fs_m:.0}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(&["algorithm", "hypervolume", "front", "ms"], &table);
}

// =============================================================================
// Many-objective problems (4+ objectives)
//
// The "curse of dimensionality" for multi-objective optimizers: as the
// objective count climbs, the fraction of mutually non-dominated solution
// pairs rushes toward 1, so Pareto rank alone stops discriminating.
// NSGA-II's whole population collapses into front 0 and only crowding
// distance is left to steer. Reference-point (NSGA-III), decomposition
// (MOEA/D), reference-vector (RVEA), grid (GrEA) and indicator (IBEA,
// HypE) methods are built to survive this regime.
//
// Both DTLZ structs above are already parameterised by objective count,
// and the DTLZ1/DTLZ2 distance metrics generalise to any M, so a single
// `ManySpec` + generic runners cover every objective count.
// =============================================================================

/// A DTLZ instance at an arbitrary objective count, plus the budget and
/// the reference-set sizing the many-objective algorithms need.
#[derive(Clone, Copy)]
struct ManySpec {
    objectives: usize,
    dim: usize,
    budget: usize,
    /// Population for the fixed-population algorithms; also the
    /// Das-Dennis weight count MOEA/D derives from `reference_divisions`.
    population: usize,
    /// Das-Dennis divisions for NSGA-III / RVEA / MOEA/D reference sets.
    reference_divisions: usize,
    /// `true` = DTLZ1 (deceptive multimodal, linear-simplex front);
    /// `false` = DTLZ2 (unit-hypersphere-octant front).
    is_dtlz1: bool,
    /// Per-axis HypE hypervolume reference coordinate.
    hype_ref: f64,
}

/// One DTLZ problem type so the generic runners have a single `Problem`
/// to hand to `run` regardless of which front geometry is in play.
enum ManyDtlz {
    D1(Dtlz1),
    D2(Dtlz2),
}
impl Problem for ManyDtlz {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        match self {
            ManyDtlz::D1(p) => p.objectives(),
            ManyDtlz::D2(p) => p.objectives(),
        }
    }
    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        match self {
            ManyDtlz::D1(p) => p.evaluate(x),
            ManyDtlz::D2(p) => p.evaluate(x),
        }
    }
}

impl ManySpec {
    fn problem(&self) -> ManyDtlz {
        if self.is_dtlz1 {
            ManyDtlz::D1(Dtlz1 {
                num_objectives: self.objectives,
                dim: self.dim,
            })
        } else {
            ManyDtlz::D2(Dtlz2 {
                num_objectives: self.objectives,
                dim: self.dim,
            })
        }
    }
    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.dim]
    }
    fn variation(&self) -> CompositeVariation<SimulatedBinaryCrossover, PolynomialMutation> {
        let b = self.bounds();
        CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(b.clone(), 30.0, 1.0),
            mutation: PolynomialMutation::new(b, 20.0, 1.0 / self.dim as f64),
        }
    }
    fn mean_dist(&self, front: &[Candidate<Vec<f64>>]) -> f64 {
        if self.is_dtlz1 {
            mean_distance_to_dtlz1_front(front)
        } else {
            mean_distance_to_dtlz2_front(front)
        }
    }
}

fn many_random(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = RandomSearch::new(
        RandomSearchConfig {
            iterations: spec.budget,
            batch_size: 1,
            seed,
        },
        RealBounds::new(spec.bounds()),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn many_nsga2(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = Nsga2::new(
        Nsga2Config {
            population_size: spec.population,
            generations: spec.budget / spec.population,
            seed,
        },
        RealBounds::new(spec.bounds()),
        spec.variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn many_nsga3(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = Nsga3::new(
        Nsga3Config {
            population_size: spec.population,
            generations: spec.budget / spec.population,
            reference_divisions: spec.reference_divisions,
            seed,
        },
        RealBounds::new(spec.bounds()),
        spec.variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn many_moead(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = Moead::new(
        MoeadConfig {
            generations: spec.budget / spec.population,
            reference_divisions: spec.reference_divisions,
            neighborhood_size: 20,
            seed,
        },
        RealBounds::new(spec.bounds()),
        spec.variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn many_rvea(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = Rvea::new(
        RveaConfig {
            population_size: spec.population,
            generations: spec.budget / spec.population,
            reference_divisions: spec.reference_divisions,
            alpha: 2.0,
            seed,
        },
        RealBounds::new(spec.bounds()),
        spec.variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn many_grea(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = Grea::new(
        GreaConfig {
            population_size: spec.population,
            generations: spec.budget / spec.population,
            grid_divisions: 10,
            seed,
        },
        RealBounds::new(spec.bounds()),
        spec.variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn many_ibea(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = Ibea::new(
        IbeaConfig {
            population_size: spec.population,
            generations: spec.budget / spec.population,
            kappa: 0.05,
            seed,
        },
        RealBounds::new(spec.bounds()),
        spec.variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn many_hype(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = Hype::new(
        HypeConfig {
            population_size: spec.population,
            generations: spec.budget / spec.population,
            reference_point: vec![spec.hype_ref; spec.objectives],
            mc_samples: 1_000,
            seed,
        },
        RealBounds::new(spec.bounds()),
        spec.variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn many_age_moea(spec: ManySpec, seed: u64) -> MoRun {
    let problem = spec.problem();
    let mut opt = AgeMoea::new(
        AgeMoeaConfig {
            population_size: spec.population,
            generations: spec.budget / spec.population,
            seed,
        },
        RealBounds::new(spec.bounds()),
        spec.variation(),
    );
    let t0 = Instant::now();
    let result = opt.run(&problem);
    MoRun {
        front: result.pareto_front,
        wall_ms: t0.elapsed().as_millis(),
    }
}

fn run_many_objective_comparison(title: &str, blurb: &[&str], spec: ManySpec) {
    println!();
    println!("== {title} ==");
    for line in blurb {
        println!("{line}");
    }
    println!("sorted best-first by mean dist to the true front (lower is better)");
    println!();

    type Runner = fn(ManySpec, u64) -> MoRun;
    let runners: &[(&str, Runner)] = &[
        ("RandomSearch", many_random),
        ("NSGA-II", many_nsga2),
        ("NSGA-III", many_nsga3),
        ("MOEA/D", many_moead),
        ("RVEA", many_rvea),
        ("GrEA", many_grea),
        ("IBEA", many_ibea),
        ("HypE", many_hype),
        ("AGE-MOEA", many_age_moea),
    ];

    let mut rows: Vec<(f64, Vec<String>)> = Vec::new();
    for (name, runner) in runners {
        let runs: Vec<MoRun> = (0..SEEDS).map(|s| runner(spec, s)).collect();
        let dist: Vec<f64> = runs.iter().map(|r| spec.mean_dist(&r.front)).collect();
        let fs: Vec<f64> = runs.iter().map(|r| r.front.len() as f64).collect();
        let ms: Vec<f64> = runs.iter().map(|r| r.wall_ms as f64).collect();
        let (d_m, d_s) = mean_std(&dist);
        let (fs_m, _) = mean_std(&fs);
        let (ms_m, _) = mean_std(&ms);
        rows.push((
            d_m,
            vec![
                name.to_string(),
                format!("{d_m:.4}+/-{d_s:.4}"),
                format!("{fs_m:.0}"),
                format!("{ms_m:.0}"),
            ],
        ));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let table: Vec<Vec<String>> = rows.into_iter().map(|(_, r)| r).collect();
    print_table(&["algorithm", "mean dist", "front", "ms"], &table);
}

fn main() {
    run_zdt1_comparison();
    run_zdt3_comparison();
    run_dtlz2_comparison();
    run_dtlz1_comparison();
    run_rastrigin_comparison();
    run_rosenbrock_comparison();
    run_ackley_comparison();
    run_tsp_comparison();
    run_jss_comparison();
    run_knapsack_comparison();

    // ---- Many-objective (4+) ----
    run_many_objective_comparison(
        "DTLZ2 4-objective (dim=13, 40000 evals/run × 10 seeds)",
        &[
            "DTLZ2 scaled to 4 objectives -- the entry point to many-objective.",
            "Front is still the unit-hypersphere octant (Σf² = 1). Already hard:",
            "with 4 objectives most random pairs of solutions are mutually",
            "non-dominated, so Pareto rank alone barely discriminates. Optimum:",
            "mean dist -> 0.",
        ],
        ManySpec {
            objectives: 4,
            dim: 13,
            budget: 40_000,
            population: 56,
            reference_divisions: 5,
            is_dtlz1: false,
            hype_ref: 3.0,
        },
    );
    run_many_objective_comparison(
        "DTLZ2 10-objective (dim=19, 40000 evals/run × 10 seeds)",
        &[
            "DTLZ2 scaled to 10 objectives -- the curse of dimensionality in full.",
            "In 10-D objective space almost EVERY pair of solutions is mutually",
            "non-dominated, so NSGA-II's whole population collapses into front 0",
            "and crowding distance is the only signal left. Reference-point,",
            "decomposition and indicator methods are built for exactly this.",
            "Watch the 'front' column: it pins to the population size because",
            "nothing dominates anything. Optimum: mean dist -> 0.",
        ],
        ManySpec {
            objectives: 10,
            dim: 19,
            budget: 40_000,
            population: 55,
            reference_divisions: 2,
            is_dtlz1: false,
            hype_ref: 3.0,
        },
    );
    run_many_objective_comparison(
        "DTLZ1 8-objective (dim=12, 40000 evals/run × 10 seeds)",
        &[
            "DTLZ1 scaled to 8 objectives -- the brutal one. Stacks the",
            "many-objective dominance collapse on top of DTLZ1's deceptive",
            "multimodal g-term (a huge number of local fronts). The true front",
            "is the linear simplex Σf = 0.5; reaching it at all is the",
            "achievement. Expect large mean-dist values and wide spreads -- this",
            "is near the edge of what the catalogue does at this budget.",
        ],
        ManySpec {
            objectives: 8,
            dim: 12,
            budget: 40_000,
            population: 120,
            reference_divisions: 3,
            is_dtlz1: true,
            hype_ref: 1.0,
        },
    );
}
