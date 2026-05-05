# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/swaits/heuropt/compare/v0.3.0...HEAD
[0.1.0]: https://github.com/swaits/heuropt/releases/tag/v0.1.0
