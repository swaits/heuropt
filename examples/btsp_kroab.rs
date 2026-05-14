//! Bi-objective TSP using NSGA-II on the **Kroak/Krobk** instance family
//! (Lust & Teghem, 2010).
//!
//! Two TSP instances over the **same** set of cities define two distance
//! matrices A and B; the search trades off tour length under A versus tour
//! length under B. This is the canonical multi-objective combinatorial
//! benchmark, and it gives a rich Pareto front because the geographies
//! disagree.
//!
//! The instance embedded here is **KroAB-25**: the first 25 cities of
//! TSPLIB KroA100 and KroB100 (both EUC_2D). Same city *indices*, two
//! coordinate listings.
//!
//! - **Algorithm**: [`Nsga2`].
//! - **Variation**: [`EdgeRecombinationCrossover`] (the gold-standard TSP
//!   crossover) piped into [`InversionMutation`] via [`CompositeVariation`].
//! - **Initializer**: [`ShuffledPermutation`].
//! - **Encoding**: strict permutation of `[0..25)`.
//!
//! Sources:
//! - TSPLIB95 KroA100 / KroB100 (Reinelt, 1991).
//! - Lust & Teghem (2010), "The Multiobjective Traveling Salesman Problem:
//!   A Survey and a New Approach."
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example btsp_kroab
//! ```

use heuropt::metrics::hypervolume_2d;
use heuropt::prelude::*;

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

/// TSPLIB EUC_2D distance: rounded Euclidean.
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

struct BTspKroAB {
    dist_a: Vec<Vec<f64>>,
    dist_b: Vec<Vec<f64>>,
}

impl BTspKroAB {
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

impl Problem for BTspKroAB {
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

    fn decision_schema(&self) -> Vec<DecisionVariable> {
        (0..N_CITIES)
            .map(|k| DecisionVariable::new(format!("tour_position_{k}")))
            .collect()
    }
}

fn main() {
    let problem = BTspKroAB::new();

    let mut optimizer = Nsga2::new(
        Nsga2Config {
            population_size: 200,
            generations: 600,
            seed: 11,
        },
        ShuffledPermutation { n: N_CITIES },
        CompositeVariation {
            crossover: EdgeRecombinationCrossover,
            mutation: InversionMutation,
        },
    );
    let result = optimizer.run(&problem);

    println!("bTSP KroAB-25 — bi-objective TSP via NSGA-II");
    println!("Source: TSPLIB95 KroA100/KroB100 (first 25 cities), Lust & Teghem bTSP family");
    println!();
    println!("Total evaluations: {}", result.evaluations);
    println!("Pareto-front size: {}", result.pareto_front.len());
    println!();

    let mut front: Vec<&Candidate<Vec<usize>>> = result.pareto_front.iter().collect();
    front.sort_by(|a, b| {
        a.evaluation.objectives[0]
            .partial_cmp(&b.evaluation.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Print a spread sample of the front (no more than 12 rows).
    let stride = (front.len() / 12).max(1);
    println!("    length_A     length_B");
    let mut printed = 0_usize;
    for (i, c) in front.iter().enumerate() {
        if i % stride == 0 || i + 1 == front.len() {
            let o = &c.evaluation.objectives;
            println!("    {:>8.0}     {:>8.0}", o[0], o[1]);
            printed += 1;
            if printed >= 12 {
                break;
            }
        }
    }
    println!();

    if let (Some(corner_a), Some(corner_b)) = (front.first(), front.last()) {
        println!(
            "A-corner: A={:.0}, B={:.0}",
            corner_a.evaluation.objectives[0], corner_a.evaluation.objectives[1]
        );
        println!(
            "B-corner: A={:.0}, B={:.0}",
            corner_b.evaluation.objectives[0], corner_b.evaluation.objectives[1]
        );
    }

    // Hypervolume vs. a generous reference point. Pick a reference well past
    // the worst values likely to appear so different runs can be compared.
    let ref_point = [40_000.0, 40_000.0];
    let owned: Vec<Candidate<Vec<usize>>> = result.pareto_front.to_vec();
    let hv = hypervolume_2d(&owned, &problem.objectives(), ref_point);
    println!();
    println!(
        "Hypervolume vs. reference ({}, {}): {:.0}",
        ref_point[0], ref_point[1], hv
    );
}
