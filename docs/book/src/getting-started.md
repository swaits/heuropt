# Five-minute walkthrough

The shortest path from a fresh project to a working optimizer.

## 1. Add heuropt to your `Cargo.toml`

```toml
[dependencies]
heuropt = "0.10"
```

The default feature set is small. Optional features:

- `parallel` — rayon-backed parallel population evaluation.
- `serde` — `Serialize` / `Deserialize` derives on the core data
  types, plus the `heuropt::explorer` JSON export module for the
  [heuropt-explorer](https://swaits.github.io/heuropt-explorer/)
  webapp.
- `async` — `AsyncProblem` trait + per-algorithm `run_async` for
  IO-bound evaluations.

```toml
heuropt = { version = "0.10", features = ["parallel"] }
```

## 2. Define a problem and run an optimizer

A problem is a struct that implements the [`Problem`] trait. You tell
heuropt what kind of decision your problem takes (`Vec<f64>`,
`Vec<bool>`, …), what objectives it has (minimize or maximize), and
how to score one decision.

We'll fit a straight line to a handful of `(x, y)` data points by
finding the slope and intercept that minimize the sum of squared
errors — same objective as least-squares regression. For a smooth
single-objective continuous problem like this, [CMA-ES][CmaEs] is a strong
default.

```rust,no_run
use heuropt::prelude::*;

struct LineFit {
    points: Vec<(f64, f64)>,
}

impl Problem for LineFit {
    type Decision = Vec<f64>; // [slope, intercept]

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("sum_squared_error")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let (slope, intercept) = (x[0], x[1]);
        let sse: f64 = self
            .points
            .iter()
            .map(|(px, py)| (py - (slope * px + intercept)).powi(2))
            .sum();
        Evaluation::new(vec![sse])
    }
}

fn main() {
    // Five noisy points roughly on the line y = 2x + 1.
    let problem = LineFit {
        points: vec![(0.0, 1.1), (1.0, 2.9), (2.0, 5.1), (3.0, 6.8), (4.0, 9.2)],
    };

    // Search box: slope and intercept each in [-10, 10].
    let bounds = RealBounds::new(vec![(-10.0, 10.0); 2]);

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

    let result = opt.run(&problem);
    let best = result.best.expect("at least one feasible candidate");
    let (slope, intercept) = (best.decision[0], best.decision[1]);
    println!(
        "best fit: y = {:.4} x + {:.4}   (sse = {:.4e}, evaluations = {})",
        slope, intercept, best.evaluation.objectives[0], result.evaluations,
    );

    println!();
    println!("predictions vs actual:");
    for (px, py) in &problem.points {
        let pred = slope * px + intercept;
        println!(
            "  x = {:.1}   actual = {:.2}   predicted = {:.4}   residual = {:+.4}",
            px, py, pred, py - pred,
        );
    }
}
```

Run with `cargo run --release` — heuristic optimization is allergic
to debug builds. The actual output:

```text
best fit: y = 2.0100 x + 1.0000   (sse = 1.0700e-1, evaluations = 960)

predictions vs actual:
  x = 0.0   actual = 1.10   predicted = 1.0000   residual = +0.1000
  x = 1.0   actual = 2.90   predicted = 3.0100   residual = -0.1100
  x = 2.0   actual = 5.10   predicted = 5.0200   residual = +0.0800
  x = 3.0   actual = 6.80   predicted = 7.0300   residual = -0.2300
  x = 4.0   actual = 9.20   predicted = 9.0400   residual = +0.1600
```

### Reading the result

CMA-ES recovered **slope ≈ 2.01, intercept ≈ 1.00** — within
hundredths of the underlying line `y = 2x + 1` that the data was
sampled from. The residuals are evenly distributed in sign (3
positive, 2 negative) and small in magnitude (the largest is 0.23
at `x = 3`), which means the fit is balancing the noise rather than
chasing any single point.

The total **sum of squared errors is 0.107** — that is the value
the optimizer was actually minimizing, and it matches the answer
you'd get from running `numpy.polyfit` or solving the normal
equations directly. CMA-ES is overkill for a two-parameter problem
(closed-form least-squares does it in one step), but the **same
code shape** scales straight up to nonlinear models, robust loss
functions, or constrained variants where there is no closed form.

It used 960 evaluations to get there. That's `population_size × generations`
= 12 × 80 = 960, and CMA-ES converges to machine epsilon on
problems this clean in well under that budget.

## 4. What just happened

- [`Problem`] is the **what** you're optimizing.
- [CMA-ES][CmaEs] (or any other optimizer) is the **how**.
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
[CmaEs]: https://docs.rs/heuropt/latest/heuropt/algorithms/cma_es/struct.CmaEs.html
[`CmaEsConfig`]: https://docs.rs/heuropt/latest/heuropt/algorithms/cma_es/struct.CmaEsConfig.html
