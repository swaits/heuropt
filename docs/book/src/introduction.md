# Introduction

heuropt is a practical Rust toolkit for **heuristic optimization** — the
art of searching for good answers when the problem is too gnarly to
solve analytically.

The kinds of problems heuropt is built for:

- **Single-objective:** "find the parameters that minimize the loss of
  this model." Hyperparameter tuning. Curve fitting. Calibration.
- **Multi-objective:** "find the trade-off curve between cost and
  accuracy." Engineering design. Portfolio optimization. Fleet
  scheduling.
- **Many-objective (4+):** the same idea but with enough objectives
  that classical Pareto methods break down. Power-grid planning.
  Airfoil design. Multi-criteria recommendation.

If your problem is differentiable and convex, you don't need this
crate — use a gradient solver. heuropt is for the *messy* problems:
landscapes with lots of local minima, decisions that aren't continuous
(permutations, bit vectors), or evaluations that are noisy / expensive
/ black-box.

## Why heuropt

There are other Rust optimization crates and many more in Python (pymoo,
hyperopt, optuna, DEAP). heuropt's design priorities:

1. **Approachable code.** No trait objects in the public API. No
   GATs, HRTBs, generic-RNG plumbing. A junior Rust engineer should
   be able to read `RandomSearch` and write a new optimizer by
   implementing only the `Optimizer<P>` trait.
2. **One concrete RNG type.** Seeded determinism is a property tested
   across the crate; identical inputs always produce identical
   outputs.
3. **Algorithms that work.** Every algorithm is benchmarked against
   the canonical test problems (ZDT, DTLZ, Rastrigin, Rosenbrock,
   Ackley) and the results are checked into [examples/compare-results.md](https://github.com/swaits/heuropt/blob/main/examples/compare-results.md)
   so you can see what each algorithm's strengths actually are.
4. **Testing as a first-class concern.** 316+ unit / integration /
   property tests, eight cargo-fuzz targets in CI, gungraun
   instruction-count benchmarks. The fuzzers find real bugs and the
   property tests check actual invariants.

## What's in the box

heuropt v0.8 ships **33 algorithms** spanning:

- Single-objective continuous: `RandomSearch`, `HillClimber`,
  `OnePlusOneEs`, `SimulatedAnnealing`, `GeneticAlgorithm`,
  `ParticleSwarm`, `DifferentialEvolution`, `Tlbo`, `CmaEs`,
  `IpopCmaEs`, `SeparableNes`, `NelderMead`.
- Single-objective other types: `Umda` (binary), `TabuSearch`
  (any), `AntColonyTsp` (permutation).
- Multi-objective (2–3): `Paes`, `Nsga2`, `Spea2`, `Mopso`, `Ibea`,
  `SmsEmoa`, `HypE`, `EpsilonMoea`, `PesaII`, `AgeMoea`, `Knea`,
  `Moead`.
- Many-objective (4+): `Nsga3`, `Rvea`, `Grea`.
- Sample-efficient / multi-fidelity: `BayesianOpt`, `Tpe`,
  `Hyperband`.

Plus the operators (SBX, PolynomialMutation, BoundedGaussianMutation,
LevyMutation, BitFlipMutation, SwapMutation, ClampToBounds,
ProjectToSimplex), the metrics (hypervolume, spacing), and the Pareto
utilities (dominance, fronts, crowding distance, Das–Dennis reference
points, the `ParetoArchive`) that you'd expect.

**Async evaluation** (since v0.8, behind the `async` feature flag):
when your `evaluate` function is IO-bound — calling an HTTP service,
an RPC, or a subprocess — implement [`AsyncProblem`] and use
`run_async(&problem, concurrency).await` on any algorithm in the
catalog. heuropt is the only mainstream optimization library with
first-class async support across its entire algorithm set.

[`AsyncProblem`]: https://docs.rs/heuropt/latest/heuropt/core/async_problem/trait.AsyncProblem.html

## How to use this guide

If you're new to heuropt, read it linearly:

1. [Five-minute walkthrough](./getting-started.md) — install, define
   a problem, run an optimizer, look at the result.
2. [Defining a problem](./defining-problems.md) — the `Problem`
   trait in depth: single- vs multi-objective, constraints, custom
   decision types.
3. [Choosing an algorithm](./choosing-an-algorithm.md) — the
   decision tree, expanded with the reasoning behind each branch.

If you're already up and running, jump into the [cookbook](./cookbook.md)
for recipes, or [comparison](./comparison.md) for how heuropt stacks
up against other libraries.
