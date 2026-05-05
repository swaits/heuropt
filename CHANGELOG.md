# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/swaits/heuropt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/swaits/heuropt/releases/tag/v0.1.0
