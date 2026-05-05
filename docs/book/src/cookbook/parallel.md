# Parallelize evaluation with rayon

If a single call to your `evaluate` takes more than ~50 µs, enabling
the `parallel` feature usually pays for itself immediately on
population-based algorithms. Each generation evaluates an entire
population, and rayon parallelizes that batch.

## Enable the feature

```toml
[dependencies]
heuropt = { version = "0.5", features = ["parallel"] }
```

There's nothing else to opt into in your code. The
population-evaluation helper is feature-gated; with `parallel` on it
uses `rayon::into_par_iter` internally, with `parallel` off it falls
back to plain `into_iter`.

## Determinism still holds

Seeded runs are bit-identical between the serial and parallel modes.
The trick is that population members are evaluated in parallel but
*assembled* back into the same order. Variation, selection, and the
RNG are all driven by the main thread, so seed-stability tests still
pass.

## Which algorithms benefit

Algorithms with a per-generation `evaluate_batch`:

- [`RandomSearch`], [`Nsga2`], [`Nsga3`], [`Spea2`], [`Moead`],
  [`Mopso`], [`Ibea`], [`SmsEmoa`], [`HypE`], [`PesaII`],
  [`EpsilonMoea`], [`AgeMoea`], [`Knea`], [`Grea`], [`Rvea`].
- [`DifferentialEvolution`] and [`GeneticAlgorithm`] benefit on the
  initial population and offspring batches.

Steady-state algorithms ([`Paes`], [`SimulatedAnnealing`],
[`HillClimber`], [`OnePlusOneEs`]) only evaluate one or a few
candidates per iteration, so the parallel feature gives them
nothing — leave it off if those are your primary optimizers.

## Worked example

The Sphere problem is too cheap to actually benefit from parallelism
— this example just shows the shape. In real workloads `evaluate` is
the expensive bit (a simulation, a model fit, an HTTP call).

```rust,no_run
use heuropt::prelude::*;

struct ExpensiveSphere;
impl Problem for ExpensiveSphere {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }
    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        // Pretend this is a 5 ms simulation.
        std::thread::sleep(std::time::Duration::from_millis(5));
        Evaluation::new(vec![x.iter().map(|v| v * v).sum::<f64>()])
    }
}

fn main() {
    let bounds = vec![(-1.0_f64, 1.0_f64); 5];
    let mut opt = DifferentialEvolution::new(
        DifferentialEvolutionConfig {
            population_size: 16,
            generations: 50,
            differential_weight: 0.5,
            crossover_probability: 0.9,
            seed: 42,
        },
        RealBounds::new(bounds),
    );
    let r = opt.run(&ExpensiveSphere);
    println!("best f = {}", r.best.unwrap().evaluation.objectives[0]);
}
```

With the `parallel` feature on, each generation's 16 evaluations run
across rayon's worker threads. On a 16-core machine the wall-clock
cost per generation drops from `16 × 5 ms = 80 ms` to roughly
`5 ms + scheduling overhead`.

## Sizing your thread pool

heuropt uses rayon's global thread pool. Override the size with:

```rust,ignore
rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();
```

Run this **before** any heuropt call, or use rayon's `install` API
to scope it.

## When parallelism *doesn't* help

- Your `evaluate` is sub-microsecond (Sphere, Rastrigin, Ackley
  unweighted) — the rayon scheduling overhead exceeds the work.
- You're already running multiple seeds in parallel at the harness
  level (see [Compare two algorithms](./compare.md)). Stacking
  parallelism rarely helps.
- The algorithm is steady-state (Paes, SA, hill climber).

[`RandomSearch`]: https://docs.rs/heuropt/latest/heuropt/algorithms/random_search/struct.RandomSearch.html
[`Nsga2`]: https://docs.rs/heuropt/latest/heuropt/algorithms/nsga2/struct.Nsga2.html
[`Nsga3`]: https://docs.rs/heuropt/latest/heuropt/algorithms/nsga3/struct.Nsga3.html
[`Spea2`]: https://docs.rs/heuropt/latest/heuropt/algorithms/spea2/struct.Spea2.html
[`Moead`]: https://docs.rs/heuropt/latest/heuropt/algorithms/moead/struct.Moead.html
[`Mopso`]: https://docs.rs/heuropt/latest/heuropt/algorithms/mopso/struct.Mopso.html
[`Ibea`]: https://docs.rs/heuropt/latest/heuropt/algorithms/ibea/struct.Ibea.html
[`SmsEmoa`]: https://docs.rs/heuropt/latest/heuropt/algorithms/sms_emoa/struct.SmsEmoa.html
[`HypE`]: https://docs.rs/heuropt/latest/heuropt/algorithms/hype/struct.Hype.html
[`PesaII`]: https://docs.rs/heuropt/latest/heuropt/algorithms/pesa2/struct.PesaII.html
[`EpsilonMoea`]: https://docs.rs/heuropt/latest/heuropt/algorithms/epsilon_moea/struct.EpsilonMoea.html
[`AgeMoea`]: https://docs.rs/heuropt/latest/heuropt/algorithms/age_moea/struct.AgeMoea.html
[`Knea`]: https://docs.rs/heuropt/latest/heuropt/algorithms/knea/struct.Knea.html
[`Grea`]: https://docs.rs/heuropt/latest/heuropt/algorithms/grea/struct.Grea.html
[`Rvea`]: https://docs.rs/heuropt/latest/heuropt/algorithms/rvea/struct.Rvea.html
[`DifferentialEvolution`]: https://docs.rs/heuropt/latest/heuropt/algorithms/differential_evolution/struct.DifferentialEvolution.html
[`GeneticAlgorithm`]: https://docs.rs/heuropt/latest/heuropt/algorithms/genetic_algorithm/struct.GeneticAlgorithm.html
[`Paes`]: https://docs.rs/heuropt/latest/heuropt/algorithms/paes/struct.Paes.html
[`SimulatedAnnealing`]: https://docs.rs/heuropt/latest/heuropt/algorithms/simulated_annealing/struct.SimulatedAnnealing.html
[`HillClimber`]: https://docs.rs/heuropt/latest/heuropt/algorithms/hill_climber/struct.HillClimber.html
[`OnePlusOneEs`]: https://docs.rs/heuropt/latest/heuropt/algorithms/one_plus_one_es/struct.OnePlusOneEs.html
