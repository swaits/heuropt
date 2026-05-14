//! Crossover showdown on the bi-objective TSP from `btsp_kroab.rs`.
//!
//! Runs NSGA-II four times on the same KroAB-25 instance, holding everything
//! constant except the **crossover** operator. The mutation
//! ([`InversionMutation`]), initializer, population, generations, and seed
//! are identical across runs.
//!
//! Operators compared:
//! - [`OrderCrossover`] (OX)
//! - [`PartiallyMappedCrossover`] (PMX)
//! - [`CycleCrossover`] (CX)
//! - [`EdgeRecombinationCrossover`] (ERX)
//!
//! Each run is ranked by **hypervolume** (the standard Pareto-front quality
//! metric), not by single-objective fitness — for a Pareto search, "best
//! length on A" or "best length on B" alone is a misleading scoreboard.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example tsp_operators_compare
//! ```

use heuropt::metrics::hypervolume_2d;
use heuropt::prelude::*;
use std::time::Instant;

/// First 25 cities of TSPLIB KroA100 (EUC_2D).
const KROA_25: [(f64, f64); 25] = [
    (1380.0, 939.0),
    (2848.0, 96.0),
    (3510.0, 1671.0),
    (457.0, 334.0),
    (3888.0, 666.0),
    (984.0, 965.0),
    (2721.0, 1482.0),
    (1286.0, 525.0),
    (2716.0, 1432.0),
    (738.0, 1325.0),
    (1251.0, 1832.0),
    (2728.0, 1698.0),
    (3815.0, 169.0),
    (3683.0, 1533.0),
    (1247.0, 1945.0),
    (123.0, 862.0),
    (1234.0, 1946.0),
    (252.0, 1240.0),
    (611.0, 673.0),
    (2576.0, 1676.0),
    (928.0, 1700.0),
    (53.0, 857.0),
    (1807.0, 1711.0),
    (274.0, 1420.0),
    (2574.0, 946.0),
];

/// First 25 cities of TSPLIB KroB100 (EUC_2D).
const KROB_25: [(f64, f64); 25] = [
    (3140.0, 1401.0),
    (556.0, 1056.0),
    (3675.0, 1522.0),
    (1182.0, 1853.0),
    (3595.0, 1340.0),
    (1936.0, 953.0),
    (2722.0, 1311.0),
    (2839.0, 2055.0),
    (2253.0, 1242.0),
    (3142.0, 1591.0),
    (627.0, 1336.0),
    (936.0, 211.0),
    (4014.0, 471.0),
    (1376.0, 1452.0),
    (3289.0, 593.0),
    (1453.0, 67.0),
    (1014.0, 1944.0),
    (2811.0, 1080.0),
    (3010.0, 1290.0),
    (1817.0, 1517.0),
    (510.0, 458.0),
    (1717.0, 1693.0),
    (1252.0, 1633.0),
    (1693.0, 1374.0),
    (539.0, 1378.0),
];

const N_CITIES: usize = 25;
const REF_POINT: [f64; 2] = [40_000.0, 40_000.0];

fn euc2d_matrix(coords: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let n = coords.len();
    let mut d = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i].0 - coords[j].0;
            let dy = coords[i].1 - coords[j].1;
            let dij = (dx * dx + dy * dy).sqrt().round();
            d[i][j] = dij;
            d[j][i] = dij;
        }
    }
    d
}

struct BTsp {
    dist_a: Vec<Vec<f64>>,
    dist_b: Vec<Vec<f64>>,
}

impl BTsp {
    fn new() -> Self {
        Self {
            dist_a: euc2d_matrix(&KROA_25),
            dist_b: euc2d_matrix(&KROB_25),
        }
    }
    fn tour_length(d: &[Vec<f64>], tour: &[usize]) -> f64 {
        let n = tour.len();
        let mut total = 0.0;
        for i in 0..n {
            total += d[tour[i]][tour[(i + 1) % n]];
        }
        total
    }
}

impl Problem for BTsp {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("length_A"),
            Objective::minimize("length_B"),
        ])
    }
    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        Evaluation::new(vec![
            Self::tour_length(&self.dist_a, tour),
            Self::tour_length(&self.dist_b, tour),
        ])
    }
}

struct RunSummary {
    name: &'static str,
    front_size: usize,
    front_unique: usize,
    corner_a: (f64, f64),
    corner_b: (f64, f64),
    hypervolume: f64,
    seconds: f64,
}

fn run_once<C>(name: &'static str, problem: &BTsp, crossover: C) -> RunSummary
where
    C: Variation<Vec<usize>>,
{
    let mut optimizer = Nsga2::new(
        Nsga2Config {
            population_size: 200,
            generations: 500,
            seed: 11,
        },
        ShuffledPermutation { n: N_CITIES },
        CompositeVariation {
            crossover,
            mutation: InversionMutation,
        },
    );
    let t0 = Instant::now();
    let result = optimizer.run(problem);
    let seconds = t0.elapsed().as_secs_f64();

    let mut front: Vec<&Candidate<Vec<usize>>> = result.pareto_front.iter().collect();
    front.sort_by(|a, b| {
        a.evaluation.objectives[0]
            .partial_cmp(&b.evaluation.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen: Vec<(i64, i64)> = Vec::new();
    for c in &front {
        let o = &c.evaluation.objectives;
        let k = (o[0] as i64, o[1] as i64);
        if !seen.contains(&k) {
            seen.push(k);
        }
    }

    let corner_a = front
        .first()
        .map(|c| (c.evaluation.objectives[0], c.evaluation.objectives[1]))
        .unwrap_or((f64::NAN, f64::NAN));
    let corner_b = front
        .last()
        .map(|c| (c.evaluation.objectives[0], c.evaluation.objectives[1]))
        .unwrap_or((f64::NAN, f64::NAN));

    let owned: Vec<Candidate<Vec<usize>>> = result.pareto_front.to_vec();
    let hv = hypervolume_2d(&owned, &problem.objectives(), REF_POINT);

    RunSummary {
        name,
        front_size: result.pareto_front.len(),
        front_unique: seen.len(),
        corner_a,
        corner_b,
        hypervolume: hv,
        seconds,
    }
}

fn main() {
    let problem = BTsp::new();
    println!("Bi-objective TSP (KroAB-25): NSGA-II crossover showdown");
    println!("Same population, generations, seed across all runs.");
    println!("Mutation held constant at InversionMutation.");
    println!(
        "Reference point for hypervolume: ({:.0}, {:.0})",
        REF_POINT[0], REF_POINT[1]
    );
    println!();

    let runs = vec![
        run_once("Order (OX)", &problem, OrderCrossover),
        run_once("PartiallyMapped (PMX)", &problem, PartiallyMappedCrossover),
        run_once("Cycle (CX)", &problem, CycleCrossover),
        run_once("EdgeRecomb (ERX)", &problem, EdgeRecombinationCrossover),
    ];

    println!(
        "  {:<24} | {:>5} {:>5} | {:>17} | {:>17} | {:>14} | {:>6}",
        "crossover", "size", "uniq", "A-corner (A, B)", "B-corner (A, B)", "hypervolume", "time"
    );
    println!("  {}", "-".repeat(106));
    for r in &runs {
        println!(
            "  {:<24} | {:>5} {:>5} | ({:>6.0},{:>6.0}) | ({:>6.0},{:>6.0}) | {:>14.0} | {:>5.2}s",
            r.name,
            r.front_size,
            r.front_unique,
            r.corner_a.0,
            r.corner_a.1,
            r.corner_b.0,
            r.corner_b.1,
            r.hypervolume,
            r.seconds,
        );
    }
    println!();

    // Pick the winner by hypervolume (largest dominated area = best front).
    let winner = runs
        .iter()
        .max_by(|a, b| {
            a.hypervolume
                .partial_cmp(&b.hypervolume)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("non-empty runs");
    println!(
        "Best by hypervolume: {} ({:.0})",
        winner.name, winner.hypervolume
    );
}
