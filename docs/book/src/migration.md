# Migration guides

Per-release notes for upgrading between heuropt versions. Skip the
sections that don't apply to your starting version.

## To 0.8

### From 0.5.x

**Additive feature only.** Bumping `heuropt = "0.8"` is enough for
any code that doesn't need async evaluation. To opt into async,
enable the new feature flag:

```toml
heuropt = { version = "0.8", features = ["async"] }
```

What changed:

- New `async` feature flag, gated on the
  [`futures`](https://crates.io/crates/futures) crate.
- New `core::async_problem::AsyncProblem` trait — mirrors `Problem`
  but with `async fn evaluate_async`.
- New `core::async_problem::AsyncPartialProblem` trait — mirrors
  `PartialProblem` for multi-fidelity (Hyperband) workloads.
- `run_async(&problem, concurrency).await` on **every** algorithm in
  the catalog (33 of them) for IO-bound evaluations.
- New cookbook recipe: [Async evaluation](./cookbook/async.md).

### From 0.7.x

`0.7.0` introduced an experimental observability layer (`Snapshot`,
`Observer`, `run_with`, `MaxTime`, `TargetFitness`, `Stagnation`,
`Periodic`, `AnyOf`, `AllOf`, `TracingObserver`) and three
additional Pareto metrics (`igd`, `igd_plus`, `r2`). All of those
were rolled back in `0.8.0` — the design didn't bake long enough
and they shipped half-wired (`run_with` was overridden on only 3 of
35 algorithms). The `tracing` feature flag is also gone.

If your code uses any of those APIs, the migration is:

- Remove all `run_with(&problem, &mut observer)` calls and replace
  with `run(&problem)`.
- Remove all uses of `Observer`, `Snapshot`, `ControlFlow`,
  `MaxTime`, `MaxIterations`, `TargetFitness`, `Stagnation`,
  `Periodic`, `AnyOf`, `AllOf`, `TracingObserver`.
- Remove all uses of `metrics::igd::igd`, `metrics::igd::igd_plus`,
  `metrics::r2::r2`.
- Remove `Population::as_slice()` calls (the method is gone).
- Drop the `tracing` feature from your `Cargo.toml` if you had it.

Stop conditions can still be implemented by wrapping `run` in a
loop with a custom RNG-driven termination, or by wrapping
the algorithm yourself; observers may return as a public API in a
future release once the design has settled.

The async work introduced in 0.7.0 (`AsyncProblem` + `run_async`)
**survived** and is broadened in 0.8: every algorithm in the catalog
now has a `run_async` (0.7.0 only had it on three of them), and
multi-fidelity problems get a parallel `AsyncPartialProblem` trait
that Hyperband's `run_async` consumes. Existing call sites continue
to work unchanged.

## To 0.5

### From 0.4.x

**No public-API changes.** v0.5 is a documentation-and-polish release.
Bumping `heuropt = "0.5"` in your Cargo.toml is enough.

What changed:

- Added a comprehensive mdbook user guide (this book).
- Added runnable rustdoc examples on every public algorithm,
  operator, metric, and Pareto utility.
- Added real-world `examples/portfolio.rs`,
  `examples/hyperparam_tuning.rs`, and `examples/scheduling.rs`.
- Added `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
  (Builder's Code of Conduct), GitHub issue templates, and PR
  template.

The full list is in CHANGELOG.md.

### From earlier than 0.4

If you're coming from 0.3.x or earlier, also read the older sections
below.

## To 0.4

### From 0.3.x

**No public-API changes.** v0.4 was a testing-infrastructure
expansion + perf pass. Same `cargo update` story.

The compare-harness wall-clock got 3.27× faster on v0.4 with
bit-identical quality metrics, so any benchmark numbers you have
from v0.3 are still numerically accurate but will run faster.

## To 0.3

### From 0.2.x

**Additive only.** New algorithms (`BayesianOpt`, `Tpe`,
`OnePlusOneEs`, `IpopCmaEs`, `SeparableNes`, `NelderMead`,
`Hyperband`), new operators (`LevyMutation`, `ClampToBounds`,
`ProjectToSimplex`), new traits (`PartialProblem`, `Repair<D>`).

`CmaEsConfig` gained an `initial_mean: Option<Vec<f64>>` field;
existing call sites need a `.. CmaEsConfig { initial_mean: None,
.. }` update.

## To 0.2

### From 0.1.x

**Additive.** New algorithms across the catalog (HillClimber, SA,
GA, PSO, CMA-ES, TabuSearch, AntColonyTsp, Umda, TLBO, MOPSO, IBEA,
SMS-EMOA, HypE, RVEA, PESA-II, ε-MOEA, AGE-MOEA, GrEA, KnEA), new
operators (`SimulatedBinaryCrossover`, `PolynomialMutation`,
`CompositeVariation`, `BoundedGaussianMutation`), and the
`hypervolume_nd` metric.

`Optimizer<P>` impls now require `P: Sync` and `P::Decision: Send`
(this enables the `parallel` feature without changing the public
trait surface). Any normal `Problem` you've written satisfies these
bounds automatically.
