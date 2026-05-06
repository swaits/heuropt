//! `pick_a_car` — designing a car along four objectives at once.
//!
//! Three decision variables (engine displacement, curb weight,
//! aerodynamic drag) and four objectives (price, 0-60 acceleration,
//! fuel consumption, idle noise) coupled by non-linear cost
//! relationships, so the Pareto front is a real surface in 3D
//! decision space — not a 1D sweep that any human could enumerate.
//!
//! Run it:
//!
//! ```text
//! cargo run --release --example pick_a_car --features serde
//! ```
//!
//! It writes a `pick_a_car.json` file in the current directory that
//! you can drop into <https://swaits.github.io/heuropt-explorer/> to
//! filter, brush, pin, and rank the 100-car Pareto front
//! interactively.

use heuropt::prelude::*;

struct PickACar;

impl Problem for PickACar {
    type Decision = Vec<f64>; // [engine_liters, weight_kg, drag_cd]

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("price")
                .with_label("Price")
                .with_unit("$k"),
            Objective::minimize("zero_to_sixty")
                .with_label("0-60 mph")
                .with_unit("s"),
            Objective::minimize("fuel")
                .with_label("Fuel")
                .with_unit("gal/100mi"),
            Objective::minimize("noise")
                .with_label("Idle noise")
                .with_unit("dB"),
        ])
    }

    fn decision_schema(&self) -> Vec<DecisionVariable> {
        vec![
            DecisionVariable::new("displacement")
                .with_label("Engine size")
                .with_unit("L")
                .with_bounds(1.0, 6.0),
            DecisionVariable::new("weight")
                .with_label("Curb weight")
                .with_unit("kg")
                .with_bounds(1100.0, 2200.0),
            DecisionVariable::new("drag")
                .with_label("Drag coefficient")
                .with_unit("Cd")
                .with_bounds(0.20, 0.40),
        ]
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let displacement = x[0];
        let weight = x[1];
        let drag = x[2];

        // Price ($k): engine cost grows superlinearly; weight reduction
        // below 1500 kg and drag reduction below 0.35 Cd both cost extra.
        let engine_cost = 3.0 * displacement.powf(1.6);
        let weight_cost = ((1500.0 - weight).max(0.0) / 100.0).powi(2) * 2.0;
        let aero_cost = ((0.35 - drag).max(0.0) * 100.0).powf(1.5) * 0.4;
        let price = 10.0 + engine_cost + weight_cost + aero_cost;

        // 0-60 (s): heavier = slower; bigger engine = quicker but
        // with diminishing returns.
        let weight_factor = (weight - 1100.0) / 1000.0;
        let engine_factor = ((displacement - 1.0) / 5.0).max(0.0).powf(0.7);
        let zero_to_sixty = 5.0 + 5.0 * weight_factor - 4.0 * engine_factor;

        // Fuel consumption (gal/100 mi): all three decision vars matter.
        let fuel = 0.5 + 0.5 * displacement + 0.5 * weight / 1000.0 + 4.0 * drag;

        // Idle noise (dB): engine dominates, mildly non-linear.
        let noise = 60.0 + 3.0 * displacement.powf(1.2);

        Evaluation::new(vec![price, zero_to_sixty, fuel, noise])
    }
}

fn main() {
    let bounds = vec![
        (1.0_f64, 6.0_f64),       // engine
        (1100.0_f64, 2200.0_f64), // weight
        (0.20_f64, 0.40_f64),     // drag
    ];

    let started = std::time::Instant::now();

    let mut optimizer = Nsga3::new(
        Nsga3Config {
            population_size: 100,
            generations: 200,
            reference_divisions: 5,
            seed: 42,
        },
        RealBounds::new(bounds.clone()),
        CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.9),
            mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / 3.0),
        },
    );
    let result = optimizer.run(&PickACar);

    let elapsed = started.elapsed().as_secs_f64();

    // Print a short summary across the front so the user can see what
    // they got without leaving the terminal.
    let mut front: Vec<_> = result.pareto_front.iter().collect();
    front.sort_by(|a, b| {
        a.evaluation.objectives[0]
            .partial_cmp(&b.evaluation.objectives[0])
            .unwrap()
    });
    println!(
        "Pareto front: {} cars  (took {:.3} s)\n",
        front.len(),
        elapsed,
    );
    println!(
        "{:>5}  {:>5}  {:>4}    {:>6}  {:>5}  {:>5}  {:>5}",
        "L", "kg", "Cd", "$k", "0-60", "fuel", "dB"
    );
    let n = front.len();
    let sample_indices = if n <= 6 {
        (0..n).collect::<Vec<_>>()
    } else {
        // Six representative rows: first, ~20%, ~40%, ~60%, ~80%, last
        vec![0, n / 5, (2 * n) / 5, (3 * n) / 5, (4 * n) / 5, n - 1]
    };
    for &i in &sample_indices {
        let c = front[i];
        let d = &c.decision;
        let o = &c.evaluation.objectives;
        println!(
            "{:>5.2}  {:>5.0}  {:>4.2}    {:>6.1}  {:>5.1}  {:>5.2}  {:>5.1}",
            d[0], d[1], d[2], o[0], o[1], o[2], o[3]
        );
    }

    // Write the explorer JSON. With the metadata the Problem provides
    // (objective labels + units + decision schema) plus the algorithm's
    // own AlgorithmInfo, this is genuinely zero-config: one call.
    let path = "pick_a_car.json";
    let export = heuropt::explorer::ExplorerExport::from_result(&PickACar, &result)
        .with_algorithm_info(&optimizer)
        .with_problem_name("Pick a car")
        .with_wall_clock(elapsed);
    export.to_file(path).expect("failed to write JSON");

    println!(
        "\nWrote {} candidates to {} ({}/{} on the Pareto front).",
        result.population.candidates.len(),
        path,
        result.pareto_front.len(),
        result.population.candidates.len(),
    );
    println!("Drop it into https://swaits.github.io/heuropt-explorer/ to explore.");
}
