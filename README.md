# heuropt

[![Crates.io](https://img.shields.io/crates/v/heuropt.svg)](https://crates.io/crates/heuropt)
[![Documentation](https://docs.rs/heuropt/badge.svg)](https://docs.rs/heuropt)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A practical Rust toolkit for implementing heuristic single-objective,
multi-objective, and many-objective optimization algorithms.

`heuropt` is **not** a research framework full of abstract machinery — it is a
small set of concrete types, a handful of simple traits, and a few reference
algorithms. The goal: an entry-level Rust engineer can define a problem, run a
built-in optimizer, or implement a new optimizer without learning any
framework concepts.

## Installation

```toml
[dependencies]
heuropt = "0.1"

# Optional features:
# - "serde":     derive Serialize/Deserialize on the core data types.
# - "parallel":  evaluate populations across rayon's thread pool.
#                Seeded runs stay bit-identical to serial mode.
# heuropt = { version = "0.1", features = ["serde", "parallel"] }
```

## Define a problem

```rust
use heuropt::prelude::*;

struct SchafferN1;

impl Problem for SchafferN1 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
        ])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let v = x[0];
        Evaluation::new(vec![v * v, (v - 2.0).powi(2)])
    }
}
```

## Run NSGA-II

```rust
use heuropt::prelude::*;

# struct SchafferN1;
# impl Problem for SchafferN1 {
#     type Decision = Vec<f64>;
#     fn objectives(&self) -> ObjectiveSpace {
#         ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
#     }
#     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
#         Evaluation::new(vec![x[0] * x[0], (x[0] - 2.0).powi(2)])
#     }
# }
let initializer = RealBounds::new(vec![(-5.0, 5.0)]);
let variation = GaussianMutation { sigma: 0.2 };
let config = Nsga2Config { population_size: 60, generations: 80, seed: 42 };
let mut optimizer = Nsga2::new(config, initializer, variation);
let result = optimizer.run(&SchafferN1);

println!("Pareto front size: {}", result.pareto_front.len());
```

See `examples/toy_nsga2.rs` for the full version.

## Implement a custom optimizer

A new optimizer is just an implementation of `Optimizer<P>`:

```rust
use heuropt::prelude::*;

struct MyOptimizer { /* state */ }

impl<P> Optimizer<P> for MyOptimizer
where
    P: Problem<Decision = Vec<f64>>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        // Generate candidates.
        // Evaluate them with `problem.evaluate(...)`.
        // Keep the best, or maintain a Pareto archive.
        // Return an OptimizationResult.
        # OptimizationResult::new(
        #     Population::new(Vec::new()),
        #     Vec::new(),
        #     None,
        #     0,
        #     0,
        # )
    }
}
```

A complete worked example is in `examples/custom_optimizer.rs`.

## Current algorithms

- `RandomSearch` — sample-evaluate-keep baseline.
- `Paes` — a small (1+1) Pareto Archived Evolution Strategy.
- `Nsga2` — the canonical Pareto-based evolutionary algorithm.
- `DifferentialEvolution` — DE/rand/1/bin for single-objective real-valued
  problems.

Plus reusable utilities: `pareto_compare`, `pareto_front`, `best_candidate`,
`non_dominated_sort`, `crowding_distance`, `ParetoArchive`, and the metrics
`spacing` and `hypervolume_2d`.

## Design philosophy

- **Concrete data, small trait surface.** `Problem`, `Optimizer`, `Initializer`,
  `Variation` are the only traits a user interacts with day-to-day. Everything
  else is plain structs.
- **No type hell.** No trait objects in the core path, no GATs, no HRTBs in
  user-facing APIs, no generic-RNG plumbing — `Rng` is a single concrete type
  alias.
- **Readable algorithms.** Built-ins are written for clarity, not maximum
  abstraction reuse. `RandomSearch` is the recommended file to read before
  writing your own optimizer.
- **One crate first.** No premature splitting into `-core`/`-algorithms`/
  `-operators`. Split later if the crate grows.
- **Panic on programmer error.** Invalid configuration panics with a clear
  message in v1; the API may grow `Result`-returning variants later if the
  base API proves useful.

See `docs/heuropt_tech_design_spec.md` for the full design rationale.

## License

MIT — see [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
