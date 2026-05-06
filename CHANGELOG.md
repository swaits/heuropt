# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] — 2026-05-06

Theme: async evaluation, plus the docs / governance / CI catch-up
that came with finalizing the release.

heuropt now supports problems where each evaluation is a
`.await`-able operation — HTTP services, RPC clients, spawned
subprocesses. This is the differentiating capability vs.
pymoo / hyperopt / optuna / DEAP / MOEA Framework, none of which
ship first-class async support at the *evaluation* level.

No public-API breaks for synchronous users. The new surface is
gated behind a new `async` feature flag.

### Added

#### Async evaluation (the headline feature)

- New optional feature `async`, gated on
  [`futures`](https://crates.io/crates/futures).
- `core::async_problem::AsyncProblem` trait — mirrors `Problem` but
  with `async fn evaluate_async(&self, decision)`. Adapt an
  existing sync `Problem` with a one-line wrapper.
- `core::async_problem::AsyncPartialProblem` trait — mirrors
  `PartialProblem` for multi-fidelity (Hyperband) workloads with
  `async fn evaluate_at_budget_async(decision, budget)`.
- Per-algorithm `run_async(&problem, concurrency).await` methods on
  **every** algorithm in the catalog — all 33 of them — driving
  evaluations through whichever async runtime the caller is using
  (typically tokio). `concurrency` bounds in-flight evaluations.
  Population-based algorithms (NSGA-II, NSGA-III, SPEA2, MOEA/D,
  CMA-ES, DE, GA, PSO, IBEA, SMS-EMOA, HypE, ε-MOEA, PESA-II,
  AGE-MOEA, KnEA, GrEA, RVEA, MOPSO, TLBO, IPOP-CMA-ES, sNES, UMDA,
  Ant Colony, GA, Random Search) fan out per generation. Steady-state
  algorithms (Hill Climber, SA, (1+1)-ES, PAES, Nelder-Mead, Tabu
  Search) await each step sequentially. Surrogate algorithms (BO,
  TPE) batch the initial design and then await per-iteration
  acquisitions. Hyperband fans out each Successive-Halving rung
  through `AsyncPartialProblem`.
- Internal `algorithms::parallel_eval_async::evaluate_batch_async`
  and `evaluate_batch_at_budget_async` helpers — use
  `futures::stream::FuturesOrdered` with concurrency-bounded chunks,
  preserve input order so seeded determinism is preserved when
  evaluations are themselves deterministic.
- `examples/async_eval.rs` — worked example with a simulated 20 ms
  remote service. At concurrency = 1 it's serial; at concurrency = 4
  it's 2× faster; demonstrates `DifferentialEvolution` under tokio.

#### Documentation

- New cookbook recipe **[Async evaluation](docs/book/src/cookbook/async.md)**
  — implementing `AsyncProblem`, picking concurrency, determinism
  guarantees, async vs. `parallel`.
- Comparison-with-other-libraries chapter updated: `heuropt 0.8`
  row, `Async ✅ AsyncProblem + run_async` column, "When to pick
  heuropt" gains an explicit IO-bound bullet.
- Stability chapter rewritten: removes the speculative "Observer /
  Checkpoint planned" bullet (those didn't ship), documents the new
  `async` feature flag.
- Migration guide: new "To 0.8" section covering both
  `0.5.x → 0.8` (feature-additive — opt in by enabling the `async`
  feature) and `0.7 → 0.8` (the partial async surface from 0.7 is
  superseded by complete coverage; existing `run_async` callers
  keep working).
- Runnable `cargo test --doc` examples added to every public
  operator (10), metric (3), and Pareto utility (7) — every
  public item across the crate now ships with at least one
  example. 55 doctests in total (was 45).

#### CI / build

- `.github/workflows/docs.yml` builds the mdbook user guide on
  every push and deploys to GitHub Pages on `main` /
  tag pushes.
- `mdbook` book now uses `[rust] edition = "2021"` to satisfy
  `mdbook 0.4.40`.
- `clamp_to_bounds` cargo-fuzz target tolerance loosened to
  `1e-4 · max(simplex_total, max_abs_x, 1)` so the fuzzer doesn't
  flag ULP-level slop in the simplex projection's
  `max(x_i − τ, 0)` clamp boundary.

[0.8.0]: https://github.com/swaits/heuropt/releases/tag/v0.8.0

## [0.5.0] — 2026-05-05

Theme: comprehensive documentation and project polish. No public-API
changes — bumping `heuropt = "0.5"` in your `Cargo.toml` is enough.

### Added

#### User guide (mdbook)

A new mdbook user guide at `docs/book/`, deployed to
<https://swaits.github.io/heuropt/> via a CI workflow on tag pushes.
Chapters:

- **Introduction** — what heuropt is, who it's for, what's in the box.
- **Five-minute walkthrough** — install, define a problem, run an
  optimizer, look at the result.
- **Defining a problem** — the `Problem` trait in depth: single- vs
  multi-objective, constraints, custom decision types
  (`Vec<f64>`, `Vec<bool>`, `Vec<usize>`, custom structs).
- **Choosing an algorithm** — the README's decision tree, expanded
  to a full chapter with the reasoning behind every branch.
- **Cookbook** — seven recipes covering parallelism, expensive
  evaluations, comparison harnesses, permutation problems,
  constraint repair, picking one answer off a Pareto front, and
  writing your own optimizer.
- **Comparison with other libraries** — heuropt vs pymoo, hyperopt,
  optuna, MOEA Framework, metaheuristics-rs, argmin. Honest about
  when *not* to pick heuropt.
- **Stability and SemVer** — explicit guarantees about which surfaces
  are stable; what's likely to change before 1.0; bit-identical
  determinism contract.
- **Migration guides** — per-release upgrade notes.

#### Runnable rustdoc examples

Every algorithm now has a runnable ` ```rust ` example block in its
rustdoc — 35 algorithms, all exercised by `cargo test --doc`. Plus
the existing crate-level example in `lib.rs` and the
`CompositeVariation` operator example.

#### Real-world examples

Three new polished examples covering distinct domains:

- `examples/portfolio.rs` — multi-objective portfolio optimization
  with budget constraint via `ProjectToSimplex`. Pareto front of
  return-vs-risk trade-offs, plus a-posteriori weighted decision.
- `examples/hyperparam_tuning.rs` — sample-efficient hyperparameter
  tuning with `BayesianOpt` and `Tpe`, demonstrating mixed-scale
  decoding (log-uniform learning rate, integer depth) and a 60-eval
  budget.
- `examples/scheduling.rs` — single-machine weighted-completion-time
  scheduling: permutation decisions optimized via
  `SimulatedAnnealing` + `SwapMutation`, comparing against the
  Smith's-rule oracle.

#### Governance docs

- `CONTRIBUTING.md` — local-test checklist, conventional-commits
  requirement, contribution areas that land easily vs. those that
  need prior discussion.
- `SECURITY.md` — disclosure policy, supported versions, what counts
  as a security issue.
- `CODE_OF_CONDUCT.md` — adopts the
  [Builder's Code of Conduct](https://builderscode.org/) (CC0).
- `.github/ISSUE_TEMPLATE/` — bug, feature, docs templates plus a
  `config.yml` that points security reports to the private
  vulnerability-disclosure flow.
- `.github/PULL_REQUEST_TEMPLATE.md` — short, opinionated PR
  template.

#### CI / tooling

- `.github/workflows/docs.yml` — builds the mdbook user guide and
  deploys it to GitHub Pages on `main` pushes and tag pushes.

### Changed

- README hero block expanded with badges and a punchier opening;
  added explicit links to the user guide, the docs.rs API reference,
  and the testing-coverage breakdown.
- `lib.rs` crate-level docs polished — better intro, points readers
  at the user guide and the design spec.

[0.5.0]: https://github.com/swaits/heuropt/releases/tag/v0.5.0

## [0.4.0] — 2026-05-05

Theme: testing infrastructure, two real bug fixes surfaced by that
infrastructure, and a CPU-time optimization pass that made the
comparison harness 3.27× faster end-to-end. No breaking changes to
the v0.3.0 public API.

### Performance

A focused, measure-and-iterate optimization pass on the Pareto-based
multi-objective hot paths. Every change verified bit-identical against
the v0.3.0 comparison-harness snapshot — quality metrics
(hypervolume, spacing, mean L2, mean dist, front size) match to the
last decimal in every benchmark.

**Cumulative wall-clock impact (compare harness, 10-seed mean):**

| Algorithm / Problem  | v0.3.0  | v0.4.0  | Speedup |
|----------------------|--------:|--------:|--------:|
| AGE-MOEA / DTLZ1     | 2299 ms |  229 ms |  10×    |
| SPEA2 / DTLZ2        | 4304 ms |  513 ms |  8.4×   |
| AGE-MOEA / ZDT3      |  932 ms |  193 ms |  4.8×   |
| NSGA-II / ZDT1       |  268 ms |   65 ms |  4.1×   |
| NSGA-II / ZDT3       |  267 ms |   65 ms |  4.1×   |
| SMS-EMOA / DTLZ2     | 5643 ms | 1369 ms |  4.1×   |
| NSGA-II / Rastrigin  |  260 ms |   71 ms |  3.7×   |
| NSGA-II / DTLZ2      |  344 ms |  106 ms |  3.2×   |
| NSGA-III / DTLZ2     |  318 ms |  122 ms |  2.6×   |
| NSGA-III / DTLZ1     |  303 ms |  122 ms |  2.5×   |
| HypE / DTLZ2         |   80 ms |   44 ms |  1.8×   |
| **Total compare**    | **18 629 ms** | **5688 ms** | **3.27×** |

**Hot-path instruction counts (gungraun):**

| Benchmark               | v0.3.0      | v0.4.0   | Speedup |
|-------------------------|------------:|---------:|--------:|
| `hypervolume_nd_3d` n=100 | 13 523 760 | 367 767 |    37×  |
| `hypervolume_nd_3d` n=30  |    676 902 |  70 334 |   9.6×  |
| `non_dominated_sort_2d` n=200 | 13 513 271 | 2 601 813 | 5.2× |
| `non_dominated_sort_2d` n=50  |    852 317 |   198 574 | 4.3× |
| `spea2_short`             |    179 113 | 133 783 |   1.34× |

**Changes (in commit order):**

- `perf(hypervolume)` — Rewrote the M≥3 HSO recursion in
  `hypervolume_nd`. The original cloned the active set into a fresh
  Vec<Vec<f64>> at the top of every recursive call, used a linear-scan
  `position` lookup to remove the just-processed point each band, and
  re-projected onto M-1 axes inside every band. Now: sort-by-index,
  pre-project once, slice prefixes for the active set, and skip
  `non_dominated_projection` when recursing into the M=2 base case
  (whose sweep already filters dominated points internally).
- `perf(non_dominated_sort)` — Cache `as_minimization` /
  feasibility / violation per individual once at the top of the
  Deb fast-non-dominated-sort, then inline the dominance test against
  those arrays. The naïve formulation called `pareto_compare` twice
  per pair, each call allocating two fresh Vec<f64>s — 4N(N-1)
  allocations per sort. Propagates to every Pareto-based MOEA.
- `perf(age_moea)` — Cache `lp_norm(translated[i], p)` once per
  candidate at function entry; maintain a `nearest[]` array updated
  incrementally on each pick (single `min` per remaining instead of
  a fresh full scan over the keep list). Cuts the splitting-front
  scoring loop from O(R · K · M) per iteration to O(R · M).
- `perf(spea2)` — Two wins. (1) `compute_fitness` (called twice per
  generation): inline dominance against cached oriented arrays,
  symmetric distance matrix built once. (2) `build_archive` truncation:
  compute pairwise distances + sorted neighbor vectors once, then on
  victim removal use binary-search-remove on every survivor's
  still-sorted vector — total truncation cost O(K³ log K) → O(K² log K).
- `perf(hypervolume)` — Index-sort instead of cloning point vectors
  in the M≥3 recursion. The N inner-Vec clones per HV call were
  redundant once we'd already sorted by last-axis. Big bench win
  (32×→37× cumulative on n=100/3D), modest wall-clock impact because
  SMS-EMOA's worst-front HV calls operate on small fronts.
- `build(release)` — Enable thin LTO + codegen-units=1 in the
  release profile. Worth ~150 ms across the harness; only applies
  when heuropt is the workspace root, so downstream consumers see
  whatever profile their own Cargo.toml configures.
- `perf(pareto_archive)` — Cache the candidate's oriented +
  feasibility once per `insert`, build each member's oriented vector
  once, and inline the two-pass dominance checks. Used by PESA-II
  (most impact), PAES, ε-MOEA, and any user code working through the
  archive directly.

### Added

- **Decision tree update** in README to cover all v0.3.0 algorithms,
  with a new top-level branch on "is each evaluation expensive?" so
  `BayesianOpt` / `Tpe` / `Hyperband` have a clear home.
- **Comparison results snapshot** at `examples/compare-results.md` —
  reference output of the harness across 7 benchmark problems and ~20
  algorithms, captured after v0.3.0 landed.
- **Instruction-count benchmarks** via `gungraun` (the Rust 2026
  rename of `iai-callgrind`) at `benches/hot_paths.rs`. Covers
  `non_dominated_sort`, `crowding_distance`, `hypervolume_2d`,
  `hypervolume_nd` (HSO), and one-generation costs of NSGA-II and
  CMA-ES, plus a short-run bench for every algorithm. Stable across
  machines via callgrind.
- **Property-based test suite expansion**: `tests/properties.rs`
  (Pareto-comparison antisymmetry, partitioning, operator bounds),
  `tests/algorithm_properties.rs` (per-algorithm determinism +
  population-size invariants — 32 tests, one per algorithm),
  `tests/operator_properties.rs` (every `Variation` / `Initializer` /
  `Repair` impl), `tests/metric_properties.rs` (HV / spacing
  invariants), and `tests/numerical_stability.rs` (empty / singleton /
  duplicate / flat-fitness / zero-width-bounds populations).
- **Coverage-guided fuzz harness** at `fuzz/` (cargo-fuzz +
  libFuzzer). Eight targets covering `pareto_compare`,
  `non_dominated_sort`, `hypervolume_2d`, `ParetoArchive`,
  `crowding_distance`, `spacing`, SBX/PolyMut, and the `Repair`
  operators. Runs in CI for a short soak per PR; longer runs locally
  via `cargo +nightly fuzz run <target>`.
- **cargo-mutants config** at `.cargo/mutants.toml` for advisory
  mutation testing. Not gated in CI; run with `cargo mutants` to
  surface tests that don't actually check the behavior they look like
  they do.
- **GitHub Actions CI** at `.github/workflows/ci.yml` with fmt /
  clippy / test (4-feature matrix) / doc / MSRV / fuzz-smoke jobs,
  all gated on `-D warnings`.

### Fixed

- `pareto::sort::non_dominated_sort` previously dropped indices when
  the dominance graph contained a cycle (which arises when objectives
  contain NaN — `pareto_compare` becomes intransitive). Fuzzing the
  partition invariant surfaced the bug; orphans now go into a final
  residual front.
- `operators::repair::ProjectToSimplex` could silently return the
  all-zero vector when the input vector's magnitude dwarfed `total`
  (the standard Duchi/Held-Wolfe τ computation lost precision and
  τ ≈ max(x), so `max(x_i - τ, 0)` rounded to zero everywhere).
  Detected by the `clamp_to_bounds` fuzzer; now falls through to a
  degenerate "all mass on argmax" projection above a 1e15 magnitude
  ratio, and is robust to floating-point precision loss in the
  algorithm's inner loop.

[0.4.0]: https://github.com/swaits/heuropt/releases/tag/v0.4.0

## [0.3.0] — 2026-05-05

Theme: filling heuropt's expensive-evaluation, gradient-free, and
constraint-handling gaps. No breaking changes to the v0.2.0 public API.

### Added

#### New algorithms (9)

**Sample-efficient / surrogate-based:**

- `BayesianOpt` — Gaussian-process Bayesian Optimization with Expected
  Improvement acquisition. heuropt's first sample-efficient algorithm:
  targets the 50–500 evaluation regime.
- `Tpe` — Bergstra et al. 2011 Tree-structured Parzen Estimator
  (workhorse of Hyperopt and Optuna). KDE-based surrogate; cheaper
  per-step than BO and more robust without hyperparameter tuning.

**Classical and modern evolution strategies:**

- `OnePlusOneEs` — Rechenberg 1973 (1+1)-ES with the one-fifth success
  rule. Smallest possible self-adapting evolution strategy.
- `IpopCmaEs` — Auger & Hansen 2005 increasing-population CMA-ES with
  restart. Specifically fixes vanilla CMA-ES's known weakness on
  multimodal problems.
- `SeparableNes` — Wierstra et al. 2008/2014 Natural Evolution Strategy
  with diagonal covariance (sNES). Different theoretical foundation
  than CMA-ES; cheaper per-step at the cost of being unable to model
  rotated landscapes.

**Direct search:**

- `NelderMead` — Nelder & Mead 1965 simplex method. Classical gradient-
  free local optimizer; superb on low-dim smooth problems
  (Rosenbrock 5-D: f = 0 exactly).

**Multi-fidelity:**

- `Hyperband` — Li et al. 2017 multi-fidelity hyperparameter optimizer
  built on Successive Halving. Operates on a new `PartialProblem`
  trait so configurations can be evaluated at adjustable fidelity
  budgets.

#### New operators

- `LevyMutation` — heavy-tailed Lévy-flight mutation via Mantegna's
  algorithm. The actual algorithmic contribution from Cuckoo Search
  packaged as a reusable `Variation` operator.

#### New traits + impls

- `PartialProblem` — multi-fidelity problem contract:
  `evaluate_at_budget(decision, budget) -> Evaluation`. Used by
  `Hyperband`. Intentionally not a sub-trait of `Problem`.
- `Repair<D>` — in-place projection trait for restoring decisions to
  feasibility. Pair with `Variation` operators to get bounds-aware
  variants. Provided impls:
  - `ClampToBounds` for `Vec<f64>` per-axis clamping
  - `ProjectToSimplex` for L1-budget / probability-simplex projection

#### New selection helpers

- `stochastic_ranking_select` — Runarsson & Yao 2000 stochastic
  ranking. Better than strict feasibility-first tournament selection
  on heavily-constrained problems.

#### Internal helpers

- `internal::cholesky` — Cholesky factorization + triangular solves
  for SPD matrices, used by the GP posterior in `BayesianOpt`.

### Changed

- `CmaEsConfig` gained `initial_mean: Option<Vec<f64>>`. `None`
  preserves the existing midpoint-of-bounds default; `IpopCmaEs` sets
  it to inject restart diversity without shrinking the search box.

[0.3.0]: https://github.com/swaits/heuropt/releases/tag/v0.3.0

## [0.2.0] — 2026-05-05

A substantial expansion of the algorithm catalog (21 new algorithms),
five new operators, an n-D hypervolume utility, an algorithm-selection
guide in the README, and a multi-seed comparison harness covering seven
benchmark problems. No breaking changes to the v0.1.0 public API.

### Added

#### New algorithms

**Single-objective:**

- `HillClimber` — simplest greedy local search.
- `SimulatedAnnealing` — Kirkpatrick et al. 1983, generic over decision type.
- `GeneticAlgorithm` — generational SO GA with tournament selection + elitism.
- `ParticleSwarm` — Eberhart & Kennedy 1995 PSO for `Vec<f64>`.
- `CmaEs` — Hansen & Ostermeier 2001 covariance-matrix adaptation.
- `TabuSearch` — Glover 1986, with a user-supplied neighbor generator.
- `AntColonyTsp` — Dorigo Ant System for permutation problems.
- `Umda` — Mühlenbein 1997 univariate marginal-distribution EDA for
  `Vec<bool>`.
- `Tlbo` — Rao 2011 Teaching-Learning-Based Optimization (parameter-free).

**Multi-objective:**

- `Mopso` — Coello, Pulido & Lechuga 2004 multi-objective PSO.
- `Ibea` — Zitzler & Künzli 2004 indicator-based EA.
- `SmsEmoa` — Beume, Naujoks & Emmerich 2007 S-metric selection EMOA.
- `Hype` — Bader & Zitzler 2011 Hypervolume Estimation Algorithm.
- `Rvea` — Cheng et al. 2016 Reference Vector-guided EA.
- `PesaII` — Corne et al. 2001 Pareto Envelope-based Selection II.
- `EpsilonMoea` — Deb, Mohan & Mishra 2003 ε-dominance MOEA.
- `AgeMoea` — Panichella 2019 Adaptive Geometry Estimation MOEA.
- `Grea` — Yang et al. 2013 Grid-based EA.
- `Knea` — Zhang, Tian & Jin 2015 Knee point-driven EA.

#### New operators

- `BoundedGaussianMutation` — Gaussian noise + per-axis clamping.
- `SimulatedBinaryCrossover` (SBX) — Deb & Agrawal 1995 canonical
  real-valued crossover.
- `PolynomialMutation` — Deb's polynomial mutation, the standard NSGA-II
  pair to SBX.
- `CompositeVariation` — pipeline two `Variation` operators
  (typically crossover → mutation).
- `LevyMutation` — heavy-tailed Lévy-flight mutation via Mantegna's
  algorithm.

#### New metrics / utilities

- `hypervolume_nd` — exact N-dimensional dominated hypervolume via the
  Hypervolume-by-Slicing-Objectives (HSO) algorithm, plus an internal
  Jacobi symmetric eigendecomposition helper used by CMA-ES.

#### New examples

- `compare` — multi-seed comparison harness running every applicable
  algorithm across ZDT1, ZDT3, DTLZ1, DTLZ2 (multi/many-objective) and
  Rastrigin, Rosenbrock, Ackley (single-objective). Reports
  hypervolume, spacing, mean L2/dist, front size, and wall-clock ms.
- `benchmarks` — canonical reference runs of NSGA-II on ZDT1 and DE on
  Rastrigin.
- `jiggly_tuning` — real-world 4-objective NSGA-III firmware tuning
  for the [`jiggly`](https://github.com/swaits/jiggly) USB-mouse-jiggler,
  with an a-posteriori weighted-decision step that picks one
  recommendation off the Pareto front.

#### New optional feature

- `parallel` — rayon-backed parallel population evaluation in
  `RandomSearch`, `Nsga2`, `DifferentialEvolution`, `Spea2`, `Ibea`,
  `Mopso`, and most other algorithms with batchable inner loops.
  Seeded runs stay bit-identical to serial mode.

#### Documentation

- README gained an explanatory algorithm-selection decision tree that
  walks newcomers through choosing an optimizer, defining the
  terminology (multi-objective, Pareto front, dominance, multimodality,
  evaluation cost) as it goes.

### Changed

- Minimum supported Rust version remains 1.85 (edition 2024).
- Algorithm impls now require `P: Sync` and `P::Decision: Send` so the
  same impl serves both `parallel` and serial feature builds. Any
  `Problem` / decision type without exotic interior mutability already
  satisfies these.

[0.2.0]: https://github.com/swaits/heuropt/releases/tag/v0.2.0

## [0.1.0] — 2026-05-04

Initial release.

### Core types and traits

- `Direction`, `Objective`, `ObjectiveSpace` (with `as_minimization` direction
  conversion).
- `Evaluation` with feasibility (`constraint_violation <= 0.0`).
- `Candidate<D>`, `Population<D>`, `OptimizationResult<D>`.
- `type Rng = rand::rngs::StdRng` and `rng_from_seed` so no public trait is
  generic over the RNG.
- `Problem`, `Optimizer<P>`, `Initializer<D>`, `Variation<D>`.

### Pareto utilities

- `pareto_compare`, `pareto_front`, `best_candidate`,
  `non_dominated_sort` (Deb fast non-dominated sort), `crowding_distance`,
  `ParetoArchive<D>`, `das_dennis` (structured reference points for NSGA-III
  and MOEA/D).

### Operators

- Real: `RealBounds`, `GaussianMutation`, `BoundedGaussianMutation`,
  `SimulatedBinaryCrossover` (SBX), `PolynomialMutation`.
- Binary: `BitFlipMutation`.
- Permutation: `SwapMutation`.
- `CompositeVariation` pipeline (typically crossover → mutation).

### Selection helpers

- `select_random`, `tournament_select_single_objective`.

### Reference algorithms

- `RandomSearch` — sample-evaluate-keep baseline.
- `Paes` — small (1+1) Pareto Archived Evolution Strategy.
- `Nsga2` — canonical Pareto-based EA with crowding distance.
- `Nsga3` — many-objective NSGA-III with reference-point niching.
- `Spea2` — Strength Pareto Evolutionary Algorithm 2.
- `Moead` — decomposition-based MOEA/D with the Tchebycheff scalar.
- `DifferentialEvolution` — single-objective DE/rand/1/bin.

### Metrics

- `spacing` (Schott), `hypervolume_2d` (exact 2-D dominated hypervolume).

### Examples

- `random_search`, `toy_nsga2`, `custom_optimizer` — minimum-viable
  walkthroughs.
- `benchmarks` — ZDT1 and Rastrigin reference runs.
- `compare` — multi-seed comparison harness running every applicable
  algorithm on ZDT1 (2-obj), DTLZ2 (3-obj), and Rastrigin (single-obj),
  reporting hypervolume, spacing, mean L2, front size, and wall time.
- `jiggly_tuning` — 4-objective NSGA-III tuning of the
  [`jiggly`](https://github.com/swaits/jiggly) USB-mouse-jiggler firmware
  with an a-posteriori weighted-decision step that picks one
  recommendation off the Pareto front.

### Optional features

- `serde` — `Serialize` / `Deserialize` derives on the core data types.
- `parallel` — rayon-backed parallel population evaluation in
  `RandomSearch`, `Nsga2`, and `DifferentialEvolution`. Seeded runs stay
  bit-identical to serial mode.

[Unreleased]: https://github.com/swaits/heuropt/compare/v0.8.0...HEAD
[0.1.0]: https://github.com/swaits/heuropt/releases/tag/v0.1.0
