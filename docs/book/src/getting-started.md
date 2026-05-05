# Five-minute walkthrough

The shortest path from a fresh project to a working optimizer.

## 1. Add heuropt to your `Cargo.toml`

```toml
[dependencies]
heuropt = "0.5"
```

The default feature set is small. Optional features:

- `parallel` — rayon-backed parallel population evaluation.
- `serde` — `Serialize` / `Deserialize` derives on the core data
  types.

```toml
heuropt = { version = "0.5", features = ["parallel"] }
```

## 2. Define a problem

A problem is a struct that implements the [`Problem`] trait. You tell
heuropt what kind of decision your problem takes (`Vec<f64>`,
`Vec<bool>`, …), what objectives it has (minimize or maximize), and
how to score one decision.

```rust,no_run
use heuropt::prelude::*;

struct Sphere;

impl Problem for Sphere {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let f: f64 = x.iter().map(|v| v * v).sum();
        Evaluation::new(vec![f])
    }
}
```

The Sphere function is a single-objective continuous problem: minimize
`f(x) = Σ xᵢ²`. The optimum is `x = 0`, `f = 0`.

## 3. Pick an algorithm and run it

For a smooth single-objective continuous problem, [`CmaEs`] is a
strong default. Configure it, build it, run it.

```rust,no_run
# use heuropt::prelude::*;
# struct Sphere;
# impl Problem for Sphere {
#     type Decision = Vec<f64>;
#     fn objectives(&self) -> ObjectiveSpace {
#         ObjectiveSpace::new(vec![Objective::minimize("f")])
#     }
#     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
#         Evaluation::new(vec![x.iter().map(|v| v * v).sum::<f64>()])
#     }
# }
let bounds = RealBounds::new(vec![(-5.0, 5.0); 5]); // 5-dim search box

let mut opt = CmaEs::new(
    CmaEsConfig {
        population_size: 12,
        generations: 80,
        initial_sigma: 1.0,
        eigen_decomposition_period: 1,
        initial_mean: None,
        seed: 42,
    },
    bounds,
);

let result = opt.run(&Sphere);

let best = result.best.expect("at least one feasible candidate");
println!("best f = {:.3e} at x = {:?}", best.evaluation.objectives[0], best.decision);
```

Run with `cargo run --release` — heuristic optimization is allergic
to debug builds. Expect output like:

```text
best f = 1.4e-29 at x = [-1.6e-15, 4.5e-16, ...]
```

CMA-ES drops to machine epsilon on the Sphere in well under 80
generations.

## 4. What just happened

- [`Problem`] is the **what** you're optimizing.
- [`CmaEs`] (or any other optimizer) is the **how**.
- [`CmaEsConfig`] is a plain public-field struct: there are no
  builders, no chained setters, just public fields you set
  directly.
- [`Optimizer::run`] returns an [`OptimizationResult`] containing the
  full final `population`, the `pareto_front` (just the best for
  single-objective), the `best` candidate, the total `evaluations`,
  and the number of `generations`.

## 5. Where to go next

- **Multi-objective:** see [Defining a problem](./defining-problems.md)
  for how to express two or more objectives, and
  [Choosing an algorithm](./choosing-an-algorithm.md) for which
  optimizer fits.
- **Want to know which algorithm to pick:** read the README's
  decision tree, or jump straight to the [choosing-an-algorithm](./choosing-an-algorithm.md)
  chapter for the long form.
- **Production patterns:** the [cookbook](./cookbook.md) has recipes
  for parallelism, expensive evaluations, comparing algorithms, and
  more.

[`Problem`]: https://docs.rs/heuropt/latest/heuropt/core/problem/trait.Problem.html
[`Optimizer::run`]: https://docs.rs/heuropt/latest/heuropt/traits/trait.Optimizer.html
[`OptimizationResult`]: https://docs.rs/heuropt/latest/heuropt/core/result/struct.OptimizationResult.html
[`CmaEs`]: https://docs.rs/heuropt/latest/heuropt/algorithms/cma_es/struct.CmaEs.html
[`CmaEsConfig`]: https://docs.rs/heuropt/latest/heuropt/algorithms/cma_es/struct.CmaEsConfig.html
