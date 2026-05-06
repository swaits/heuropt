# Async evaluation

When your `evaluate` does **IO** — calls an HTTP service, sends an
RPC, spawns a subprocess — `await`-ing it from the optimizer is
much more efficient than blocking a thread per evaluation. heuropt
ships first-class async support behind the `async` feature flag.

This is the differentiating capability vs pymoo / hyperopt /
optuna / DEAP / MOEA Framework — none of those have a native async
evaluation path.

## Enable the feature

```toml
[dependencies]
heuropt = { version = "0.8", features = ["async"] }

# Pick whatever async runtime you want; heuropt itself depends only on
# `futures`. The example below uses tokio.
tokio = { version = "1", features = ["rt-multi-thread", "macros", "time"] }
```

## Implement `AsyncProblem`

It mirrors the regular [`Problem`] trait one-for-one — same
`Decision` type, same `objectives()`, but `evaluate` is replaced
with `evaluate_async` returning a future.

```rust,no_run
use heuropt::core::async_problem::AsyncProblem;
use heuropt::prelude::*;

struct RemoteService;

impl AsyncProblem for RemoteService {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("loss")])
    }

    async fn evaluate_async(&self, x: &Vec<f64>) -> Evaluation {
        // Real workload: HTTP call to a model-scoring service, an RPC,
        // a subprocess. Here we just sleep to model 20 ms latency.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let loss: f64 = x.iter().map(|v| v * v).sum();
        Evaluation::new(vec![loss])
    }
}
```

## Run the optimizer with `run_async`

`run_async(&problem, concurrency).await` is provided by **every**
algorithm in the catalog as of v0.8. `concurrency` caps how many
evaluations are in-flight at once.

```rust,no_run
# use heuropt::core::async_problem::AsyncProblem;
# use heuropt::prelude::*;
# struct RemoteService;
# impl AsyncProblem for RemoteService {
#     type Decision = Vec<f64>;
#     fn objectives(&self) -> ObjectiveSpace {
#         ObjectiveSpace::new(vec![Objective::minimize("loss")])
#     }
#     async fn evaluate_async(&self, x: &Vec<f64>) -> Evaluation {
#         Evaluation::new(vec![x.iter().map(|v| v * v).sum::<f64>()])
#     }
# }
#[tokio::main]
async fn main() {
    let bounds = vec![(-1.0_f64, 1.0_f64); 4];
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
    let r = opt.run_async(&RemoteService, /* concurrency */ 8).await;
    println!("best: {}", r.best.unwrap().evaluation.objectives[0]);
}
```

## Picking `concurrency`

Concurrency is the maximum in-flight evaluation count. Tradeoffs:

| Setting | Effect |
|---|---|
| `1` | Sequential; equivalent to a sync run with extra overhead |
| `pop_size` | Full per-generation parallelism; fastest if your service tolerates it |
| `< pop_size` | Bounded — useful if your downstream service has a rate limit or finite worker pool |

The bigger you go, the more memory the in-flight futures hold and
the more load you put on the downstream service. A reasonable
starting point is `min(pop_size, 16)` and increase only if the
downstream service is comfortable.

## Determinism

Same seed produces the same final result whether you use `run` or
`run_async`, **provided your async `evaluate_async` is itself
deterministic**. heuropt drives the RNG and selection on the main
task; only the evaluations are concurrent, and the
`evaluate_batch_async` helper preserves input order before feeding
results back to the algorithm.

## What the worked example shows

`examples/async_eval.rs` runs `RandomSearch` (200 evaluations × 20 ms
each) at `concurrency = 1, 4, 16` and `DifferentialEvolution` at
`concurrency = 8`. On a recent machine:

```text
RandomSearch with 200 evaluations (20 ms each)

concurrency =  1  elapsed ≈ 4250 ms     (sequential 200 × 20 ms)
concurrency =  4  elapsed ≈ 2100 ms     (2× speedup, batch_size=2 caps it)
concurrency = 16  elapsed ≈ 2100 ms     (same — batch_size dominates)

DifferentialEvolution at concurrency=8
elapsed ≈ 230 ms     (8 ants run in parallel each generation)
```

Run it yourself: `cargo run --release --features async --example async_eval`.

## Which algorithms support `run_async`?

**All 33** algorithms in the catalog. The shape of the async path
depends on the algorithm:

- **Population-based / batch-evaluating** — NSGA-II, NSGA-III, SPEA2,
  MOEA/D, IBEA, SMS-EMOA, HypE, ε-MOEA, PESA-II, AGE-MOEA, KnEA,
  GrEA, RVEA, MOPSO, GA, DE, PSO, CMA-ES, IPOP-CMA-ES, sNES, TLBO,
  UMDA, Ant Colony, Random Search. Each generation's offspring
  evaluations are fanned out concurrently up to `concurrency`.
- **Steady-state (one-eval-per-step)** — Hill Climber, Simulated
  Annealing, (1+1)-ES, PAES, Nelder-Mead. The `concurrency`
  parameter is accepted for API uniformity but evaluation order is
  inherently sequential.
- **Tabu Search** — fans out the K-neighbor batch each step.
- **Surrogate (BO, TPE)** — fans out the initial design batch, then
  awaits per-iteration acquisitions sequentially (the surrogate
  must update before the next point is chosen).
- **Hyperband** — uses the separate
  [`AsyncPartialProblem`](https://docs.rs/heuropt/latest/heuropt/core/async_problem/trait.AsyncPartialProblem.html)
  trait (multi-fidelity); each Successive-Halving rung's evaluations
  fan out concurrently.

## Async vs `parallel`

| If your `evaluate` is… | Use |
|---|---|
| CPU-bound (math, simulation) | `parallel` feature → see [Parallelize evaluation](./parallel.md) |
| IO-bound (HTTP, RPC, subprocess) | `async` feature (this recipe) |

Both can be on at once if your evaluation does *both* substantial
CPU work *and* IO. The two features are independent.

[`Problem`]: https://docs.rs/heuropt/latest/heuropt/core/problem/trait.Problem.html
