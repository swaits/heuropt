# Migration guides

Per-release notes for upgrading between heuropt versions. Skip the
sections that don't apply to your starting version.

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
