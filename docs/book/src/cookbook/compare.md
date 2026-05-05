# Compare two algorithms on your problem

The harness in `examples/compare.rs` runs every applicable algorithm
against every test problem with N seeds and reports mean ± std.
You can lift the same pattern for your own problem in ~30 lines.

## The pattern

1. Wrap your problem in a struct that implements [`Problem`].
2. Pick a few candidate algorithms.
3. For each algorithm × seed, run and record the metric you care about.
4. Print mean ± std.

## Worked example

```rust,no_run
use heuropt::prelude::*;
use std::time::Instant;

struct MyProblem;
impl Problem for MyProblem {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }
    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        // your problem here
        Evaluation::new(vec![x.iter().map(|v| v * v).sum::<f64>()])
    }
}

const SEEDS: u64 = 10;
const DIM: usize = 5;
const BUDGET: usize = 30_000;

fn main() {
    let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); DIM];

    let mut best_de = vec![];
    let mut best_cmaes = vec![];
    let mut best_ipop = vec![];
    let mut t_de = vec![];
    let mut t_cmaes = vec![];
    let mut t_ipop = vec![];

    for seed in 0..SEEDS {
        // Differential Evolution
        let t = Instant::now();
        let mut de = DifferentialEvolution::new(
            DifferentialEvolutionConfig {
                population_size: 30,
                generations: BUDGET / 30,
                differential_weight: 0.5,
                crossover_probability: 0.9,
                seed,
            },
            RealBounds::new(bounds.clone()),
        );
        let r = de.run(&MyProblem);
        t_de.push(t.elapsed().as_millis() as f64);
        best_de.push(r.best.unwrap().evaluation.objectives[0]);

        // CMA-ES
        let t = Instant::now();
        let mut cma = CmaEs::new(
            CmaEsConfig {
                population_size: 12,
                generations: BUDGET / 12,
                initial_sigma: 1.0,
                eigen_decomposition_period: 1,
                initial_mean: None,
                seed,
            },
            RealBounds::new(bounds.clone()),
        );
        let r = cma.run(&MyProblem);
        t_cmaes.push(t.elapsed().as_millis() as f64);
        best_cmaes.push(r.best.unwrap().evaluation.objectives[0]);

        // IPOP-CMA-ES
        let t = Instant::now();
        let mut ipop = IpopCmaEs::new(
            IpopCmaEsConfig {
                base: CmaEsConfig {
                    population_size: 12,
                    generations: BUDGET / 12 / 4,
                    initial_sigma: 1.0,
                    eigen_decomposition_period: 1,
                    initial_mean: None,
                    seed,
                },
                max_restarts: 3,
                population_factor: 2.0,
                seed,
            },
            RealBounds::new(bounds.clone()),
        );
        let r = ipop.run(&MyProblem);
        t_ipop.push(t.elapsed().as_millis() as f64);
        best_ipop.push(r.best.unwrap().evaluation.objectives[0]);
    }

    println!("{:<12} {:>14} {:>10}", "algorithm", "best f (mean±std)", "ms");
    print_row("DE",          &best_de,    &t_de);
    print_row("CMA-ES",      &best_cmaes, &t_cmaes);
    print_row("IPOP-CMA-ES", &best_ipop,  &t_ipop);
}

fn print_row(name: &str, values: &[f64], times: &[f64]) {
    let (m, s) = mean_std(values);
    let (t, _) = mean_std(times);
    println!("{:<12} {:>10.3e} ± {:>5.2e} {:>6.0}", name, m, s, t);
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    (m, v.sqrt())
}
```

## What to record

- **`best.evaluation.objectives[0]`** for single-objective.
- **`hypervolume_2d(&result.pareto_front, &space, ref_point)`** for
  2-objective.
- **`spacing(&result.pareto_front, &space)`** for front uniformity.
- **`result.evaluations`** to cross-check that every algorithm got
  the same evaluation budget.
- Wall-clock `Instant::now()` deltas for runtime comparison.

## Pitfalls

- **Population size matters.** Different algorithms have very
  different sweet spots. Don't just give them all the same
  population — the README's algorithm pages note typical defaults.
- **Different algorithms count "generations" differently.** What
  matters is the total `evaluations` count. Set
  `generations = BUDGET / population_size` to match across
  algorithms (with caveats for steady-state algorithms like SMS-EMOA
  that evaluate one offspring per generation).
- **One seed is not a comparison.** Always run ≥ 5 seeds; ≥ 10 is
  better. Single-seed comparisons are noise.
- **The harness in `examples/compare.rs` is the canonical version.**
  When in doubt, copy from there.

[`Problem`]: https://docs.rs/heuropt/latest/heuropt/core/problem/trait.Problem.html
