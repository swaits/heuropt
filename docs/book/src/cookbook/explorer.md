# Explore your results in a webapp

Real Pareto fronts have 50–200+ candidates spanning 2–7+ objectives.
Reading them as a wall of numbers in a terminal scales badly. Drop
the result into [heuropt-explorer](https://swaits.github.io/heuropt-explorer/)
to filter, brush, pin, and rank candidates interactively in the
browser — parallel coordinates, scatter plots, sortable table, range
filters, weighted ranking, knee-point detection.

This recipe shows the export side. The webapp is a static page; no
install needed beyond a browser.

## Enable the `serde` feature

```toml
[dependencies]
heuropt = { version = "0.10", features = ["serde"] }
```

The export uses `serde_json` under the hood, so the explorer module
is gated on the existing `serde` feature.

## Enrich your `Problem` (optional but worth it)

Two places to add display metadata that flows through to the
explorer's axis labels and tooltips:

```rust
use heuropt::prelude::*;

struct PickACar;

impl Problem for PickACar {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            // `name` is the canonical short ID; `label` and `unit`
            // are display-only. The explorer renders axes as
            // `Price ($k)` instead of just `price`.
            Objective::minimize("price").with_label("Price").with_unit("$k"),
            Objective::minimize("zero_to_sixty").with_label("0-60 mph").with_unit("s"),
            Objective::minimize("fuel").with_label("Fuel").with_unit("gal/100mi"),
            Objective::minimize("noise").with_label("Idle noise").with_unit("dB"),
        ])
    }

    fn decision_schema(&self) -> Vec<DecisionVariable> {
        // Optional: provide name/label/unit/bounds per decision-variable
        // slot. If you skip this, the exporter falls back to `x[0]`,
        // `x[1]`, … with no units or bounds.
        vec![
            DecisionVariable::new("displacement")
                .with_label("Engine size").with_unit("L").with_bounds(1.0, 6.0),
            DecisionVariable::new("weight")
                .with_label("Curb weight").with_unit("kg").with_bounds(1100.0, 2200.0),
            DecisionVariable::new("drag")
                .with_label("Drag coefficient").with_unit("Cd").with_bounds(0.20, 0.40),
        ]
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        // ... compute objectives ...
#       Evaluation::new(vec![0.0, 0.0, 0.0, 0.0])
    }
}
```

Both `Objective::with_label` / `with_unit` and `Problem::decision_schema`
are entirely optional — the rest of heuropt doesn't read them. They
exist so the exported JSON describes itself well enough for a
display tool to render readable axes.

## Run the optimizer and write the JSON

The simplest call (no algorithm metadata in the export):

```rust,ignore
use heuropt::prelude::*;

let result = optimizer.run(&problem);
heuropt::explorer::ExplorerExport::from_result(&problem, &result)
    .to_file("results.json")
    .unwrap();
```

The richer call — pulls algorithm name + seed automatically from
the `AlgorithmInfo` trait that every built-in algorithm implements:

```rust,ignore
use heuropt::prelude::*;

let started = std::time::Instant::now();
let result = optimizer.run(&problem);

let export = heuropt::explorer::ExplorerExport::from_result(&problem, &result)
    .with_algorithm_info(&optimizer)
    .with_problem_name("Pick a car")
    .with_wall_clock(started.elapsed().as_secs_f64());
export.to_file("results.json").unwrap();
```

There's also a one-liner if you don't need to set extra metadata:

```rust,ignore
heuropt::explorer::to_file("results.json", &problem, &optimizer, &result).unwrap();
```

## Open it in the explorer

Visit <https://swaits.github.io/heuropt-explorer/> and drag the JSON
file onto the page. The explorer reads the units and labels you
attached and renders parallel-coordinates / scatter / table views
that respect them. Brushing on any axis filters the others; pinned
candidates stay highlighted; the weight sliders let you rank the
front by your priorities.

## What's in the file

The full schema is documented in
[`heuropt::explorer::ExplorerExport`](https://docs.rs/heuropt/latest/heuropt/explorer/struct.ExplorerExport.html).
The shape:

```json
{
  "schema_version": 1,
  "run": {
    "problem_name": "Pick a car",
    "algorithm": "Nsga3",
    "seed": 42,
    "wall_clock_seconds": 0.097,
    "evaluations": 20100,
    "generations": 200
  },
  "objectives": [
    { "name": "price", "direction": "Minimize", "label": "Price", "unit": "$k" },
    ...
  ],
  "decision_variables": [
    { "name": "displacement", "label": "Engine size", "unit": "L", "min": 1.0, "max": 6.0 },
    ...
  ],
  "candidates": [
    {
      "decision": [1.0, 1505.0, 0.35],
      "objectives": [13.0, 7.0, 3.17, 63.0],
      "constraint_violation": 0.0,
      "feasible": true,
      "front_rank": 0,
      "in_pareto_front": true
    },
    ...
  ]
}
```

`front_rank` is computed by `non_dominated_sort` once at export
time — `0` means on the Pareto front, higher numbers indicate
deeper layers.

## Custom decision types

Out of the box, `Vec<f64>`, `Vec<bool>`, `Vec<usize>`, and `Vec<i64>`
work as decisions. For a custom decision type, implement
`heuropt::explorer::ToDecisionValues`:

```rust,ignore
struct MyDecision { color: String, count: u32 }

impl heuropt::explorer::ToDecisionValues for MyDecision {
    fn to_decision_values(&self) -> Vec<serde_json::Value> {
        vec![
            serde_json::Value::String(self.color.clone()),
            serde_json::Value::Number(self.count.into()),
        ]
    }
}
```

The explorer renders strings as categorical axes and numbers as
continuous.

## Worked example

`examples/pick_a_car.rs` ships with the crate. It implements the
problem above, runs NSGA-III for 200 generations, and writes
`pick_a_car.json` ready to load:

```text
cargo run --release --example pick_a_car --features serde
```
