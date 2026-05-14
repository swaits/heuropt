//! Solve the Ulysses16 TSP benchmark from TSPLIB using a Genetic Algorithm
//! with the new permutation-toolkit operators.
//!
//! - **Benchmark**: Ulysses16 (Groetschel/Padberg "Odyssey of Ulysses"),
//!   16 cities, GEO distance metric (TSPLIB-95).
//! - **Known optimum**: tour length **6859**.
//! - **Algorithm**: [`GeneticAlgorithm`] with elitism.
//! - **Variation**: [`OrderCrossover`] (OX) → [`InversionMutation`], piped
//!   via [`CompositeVariation`].
//! - **Initializer**: [`ShuffledPermutation`].
//!
//! Source: TSPLIB95
//! <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/>
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example tsp_ulysses16
//! ```
//!
//! The GA reliably converges to within a few percent of the known optimum on
//! this instance; on most seeds it hits 6859 exactly.

use heuropt::prelude::*;

/// TSPLIB Ulysses16 coordinates as `(lat, lon)` in TSPLIB DD.MM format.
///
/// The "decimal" part is *minutes* (out of 60), not a true decimal fraction;
/// the GEO distance formula handles the conversion.
const ULYSSES16: [(f64, f64); 16] = [
    (38.24, 20.42),
    (39.57, 26.15),
    (40.56, 25.32),
    (36.26, 23.12),
    (33.48, 10.54),
    (37.56, 12.19),
    (38.42, 13.11),
    (37.52, 20.44),
    (41.23, 9.10),
    (41.17, 13.05),
    (36.08, -5.21),
    (38.47, 15.13),
    (38.15, 15.35),
    (37.51, 15.17),
    (35.49, 14.32),
    (39.36, 19.56),
];

const KNOWN_OPTIMUM: f64 = 6859.0;

/// TSPLIB-95 GEO distance metric.
///
/// Coordinates are interpreted as latitude/longitude in DD.MM (decimal-degrees
/// with the fractional part being minutes/100), converted to radians, and the
/// arc length between the two points on a sphere of radius `RRR = 6378.388`
/// is rounded to the next integer (`floor(d + 1)`).
fn geo_distance_matrix(coords: &[(f64, f64)]) -> Vec<Vec<f64>> {
    const RRR: f64 = 6378.388;
    let to_radians = |x: f64| {
        let deg = x.trunc();
        let min = x - deg;
        std::f64::consts::PI * (deg + 5.0 * min / 3.0) / 180.0
    };
    let radians: Vec<(f64, f64)> = coords
        .iter()
        .map(|&(la, lo)| (to_radians(la), to_radians(lo)))
        .collect();
    let n = radians.len();
    let mut d = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (la_i, lo_i) = radians[i];
            let (la_j, lo_j) = radians[j];
            let q1 = (lo_i - lo_j).cos();
            let q2 = (la_i - la_j).cos();
            let q3 = (la_i + la_j).cos();
            let dij = (RRR * (0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)).acos() + 1.0).trunc();
            d[i][j] = dij;
            d[j][i] = dij;
        }
    }
    d
}

struct Ulysses16Tsp {
    dist: Vec<Vec<f64>>,
}

impl Ulysses16Tsp {
    fn new() -> Self {
        Self {
            dist: geo_distance_matrix(&ULYSSES16),
        }
    }

    fn tour_length(&self, tour: &[usize]) -> f64 {
        let n = tour.len();
        let mut total = 0.0;
        for i in 0..n {
            let a = tour[i];
            let b = tour[(i + 1) % n];
            total += self.dist[a][b];
        }
        total
    }
}

impl Problem for Ulysses16Tsp {
    type Decision = Vec<usize>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("tour_length")])
    }

    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        Evaluation::new(vec![self.tour_length(tour)])
    }

    fn decision_schema(&self) -> Vec<DecisionVariable> {
        (0..ULYSSES16.len())
            .map(|k| DecisionVariable::new(format!("tour_position_{k}")))
            .collect()
    }
}

fn main() {
    let problem = Ulysses16Tsp::new();
    let n = ULYSSES16.len();

    let mut optimizer = GeneticAlgorithm::new(
        GeneticAlgorithmConfig {
            population_size: 150,
            generations: 1500,
            tournament_size: 3,
            elitism: 4,
            seed: 42,
        },
        ShuffledPermutation { n },
        CompositeVariation {
            crossover: OrderCrossover,
            mutation: InversionMutation,
        },
    );
    let result = optimizer.run(&problem);

    let best = result.best.expect("GA always returns a best candidate");
    let best_len = best.evaluation.objectives[0];
    let gap_abs = best_len - KNOWN_OPTIMUM;
    let gap_pct = 100.0 * gap_abs / KNOWN_OPTIMUM;

    println!("TSPLIB Ulysses16 — single-objective TSP via Genetic Algorithm");
    println!("Source: TSPLIB95 (Groetschel/Padberg)");
    println!();
    println!("Known optimum:    {:>8.0}", KNOWN_OPTIMUM);
    println!(
        "GA best found:    {:>8.0}   (gap {:+.0}, {:+.2}%)",
        best_len, gap_abs, gap_pct
    );
    println!();
    println!("Total evaluations: {}", result.evaluations);
    println!("Final population:  {}", result.population.len());
    println!();
    println!("Tour (city indices, returning to start):");
    for (i, c) in best.decision.iter().enumerate() {
        print!("{:>3}", c);
        if i + 1 < best.decision.len() {
            print!(" → ");
        }
    }
    println!(" → {}", best.decision[0]);
}
