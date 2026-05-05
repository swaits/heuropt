# Heuropt Technical Design Specification

**Project name:** `heuropt`  
**Crate type:** Rust library crate  
**Primary goal:** Make it obvious and easy for Rust developers to implement heuristic single-objective, multi-objective, and many-objective optimization algorithms.

---

## 1. What We Are Building

`heuropt` is a Rust toolkit for implementing heuristic optimization algorithms.

It is not intended to be a research framework full of abstract machinery. It is a practical crate with a small set of concrete types, simple traits, and reusable utilities.

The crate should help users do three things:

1. Define an optimization problem.
2. Run a built-in optimizer.
3. Implement their own optimizer with minimal framework knowledge.

The crate supports:

- Single-objective optimization.
- Multi-objective optimization.
- Many-objective optimization.
- Heuristic, evolutionary, swarm, and local-search style algorithms.
- Pareto-front utilities for multi-objective optimization.

The crate does **not** attempt to support every optimization style. In particular, version 1 does not need to support Bayesian optimization, async studies, trial databases, distributed execution, or experiment-management abstractions.

---

## 2. How We Know We Got It Right

We got the design right if an entry-level Rust engineer can do the following without asking framework-design questions:

### 2.1 Define a Problem

They should be able to implement a problem in under 20 lines:

```rust
use heuropt::prelude::*;

struct ToyProblem;

impl Problem for ToyProblem {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
        ])
    }

    fn evaluate(&self, x: &Self::Decision) -> Evaluation {
        Evaluation::new(vec![
            x[0] * x[0],
            (x[0] - 2.0).powi(2),
        ])
    }
}
```

### 2.2 Run a Built-in Optimizer

They should be able to run an optimizer with straightforward code:

```rust
use heuropt::prelude::*;

let problem = ToyProblem;
let initializer = RealBounds::new(vec![(-5.0, 5.0)]);
let variation = GaussianMutation { sigma: 0.1 };

let config = Nsga2Config {
    population_size: 100,
    generations: 250,
    seed: 42,
};

let mut optimizer = Nsga2::new(config, initializer, variation);
let result = optimizer.run(&problem);

println!("Pareto front size: {}", result.pareto_front.len());
```

### 2.3 Implement a New Optimizer

They should be able to implement a custom optimizer by implementing one trait:

```rust
impl<P> Optimizer<P> for MyOptimizer
where
    P: Problem,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        // Generate candidates.
        // Evaluate them.
        // Keep the best or Pareto front.
        // Return OptimizationResult.
    }
}
```

### 2.4 Read the Source

A junior engineer should be able to read the implementations of `RandomSearch`, `PAES`, and `NSGA-II` and understand how to write another algorithm.

### 2.5 Avoid Rust Type Hell

The public API should avoid:

- Trait objects in the core path.
- Object-safety adapter traits.
- Generic associated types.
- Higher-ranked trait bounds in user-facing APIs.
- Generic RNG parameters everywhere.
- Excessive associated types.
- Deeply generic builder types.
- Macro-heavy APIs.

The API may use basic generics where they clearly help, such as `Candidate<D>` and `Initializer<D>`.

---

## 3. Non-Goals

The following are explicitly out of scope for version 1:

- Bayesian optimization.
- Asynchronous optimization.
- Distributed worker orchestration.
- Trial databases.
- Persistent experiment studies.
- Automatic checkpointing.
- Generic numeric objective types other than `f64`.
- Advanced constraint modeling with named equality and inequality constraints.
- Full benchmark suites such as all WFG or DTLZ variants.
- A plugin system.
- Dynamic dispatch APIs for operators.
- Full support for every metaheuristic family.

Some of these can be added later if the simple core proves stable.

---

## 4. Design Principles

### 4.1 Principle of Least Surprise

Types and method names should mean what an average Rust engineer expects.

Use names like:

- `Problem`
- `Optimizer`
- `Candidate`
- `Population`
- `Evaluation`
- `Objective`
- `ObjectiveSpace`
- `OptimizationResult`

Avoid highly academic names in the core API unless they are standard, such as `Pareto`.

### 4.2 Concrete Data, Small Trait Surface

Use concrete structs for data:

- `Evaluation`
- `Candidate<D>`
- `Population<D>`
- `Objective`
- `ObjectiveSpace`
- `OptimizationResult<D>`

Use traits only for obvious extension points:

- `Problem`
- `Optimizer<P>`
- `Initializer<D>`
- `Variation<D>`

Additional traits can be introduced later, but the initial crate should not make every concept a trait.

### 4.3 Algorithms Should Be Readable

Built-in algorithms should be written plainly. Prefer readable code over maximum abstraction reuse.

A built-in `NSGA-II` implementation should be usable as a tutorial for implementing `SPEA2`, `MOEA/D`, or another algorithm.

### 4.4 One Crate First

`heuropt` is the toolkit crate. It contains:

- Core types.
- Core traits.
- Pareto utilities.
- Selection utilities.
- Basic operators.
- Reference algorithms.
- Basic metrics.

Do not start with `heuropt-core`, `heuropt-algorithms`, and `heuropt-operators`. Split later only if the crate becomes too large.

---

## 5. Crate Layout

Recommended source layout:

```text
heuropt/
  Cargo.toml
  src/
    lib.rs
    prelude.rs

    core/
      mod.rs
      objective.rs
      evaluation.rs
      candidate.rs
      population.rs
      problem.rs
      result.rs
      rng.rs

    traits/
      mod.rs
      initializer.rs
      variation.rs
      optimizer.rs

    pareto/
      mod.rs
      dominance.rs
      front.rs
      sort.rs
      crowding.rs
      archive.rs

    selection/
      mod.rs
      random.rs
      tournament.rs

    operators/
      mod.rs
      real.rs
      binary.rs
      permutation.rs

    algorithms/
      mod.rs
      random_search.rs
      paes.rs
      nsga2.rs
      differential_evolution.rs

    metrics/
      mod.rs
      spacing.rs
      hypervolume.rs

    tests_support/
      mod.rs
```

Feature flags may be added, but the default feature set should make the crate useful immediately.

Recommended initial feature flags:

```toml
[features]
default = []
serde = ["dep:serde"]
```

Avoid splitting built-in algorithms behind feature flags until there is a clear compile-time or dependency reason.

---

## 6. Public API Overview

The public API should fit conceptually on one screen:

```rust
pub trait Problem {
    type Decision: Clone;

    fn objectives(&self) -> ObjectiveSpace;
    fn evaluate(&self, decision: &Self::Decision) -> Evaluation;
}

pub trait Optimizer<P>
where
    P: Problem,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision>;
}

pub trait Initializer<D> {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<D>;
}

pub trait Variation<D> {
    fn vary(&mut self, parents: &[D], rng: &mut Rng) -> Vec<D>;
}

pub struct Candidate<D> {
    pub decision: D,
    pub evaluation: Evaluation,
}

pub struct Population<D> {
    pub candidates: Vec<Candidate<D>>,
}

pub struct Evaluation {
    pub objectives: Vec<f64>,
    pub constraint_violation: f64,
}

pub struct OptimizationResult<D> {
    pub population: Population<D>,
    pub pareto_front: Vec<Candidate<D>>,
    pub best: Option<Candidate<D>>,
    pub evaluations: usize,
    pub generations: usize,
}
```

This is not a full code listing. It defines the intended shape of the crate.

---

## 7. Core Types

### 7.1 `Direction`

Location: `src/core/objective.rs`

Purpose: Defines whether an objective should be minimized or maximized.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}
```

Implementation notes:

- Keep this enum simple.
- Do not make objective direction a trait.

---

### 7.2 `Objective`

Location: `src/core/objective.rs`

Purpose: Names one objective and stores its direction.

Required fields:

```rust
pub struct Objective {
    pub name: String,
    pub direction: Direction,
}
```

Required constructors:

```rust
impl Objective {
    pub fn minimize(name: impl Into<String>) -> Self;
    pub fn maximize(name: impl Into<String>) -> Self;
}
```

Implementation requirements:

- `Objective::minimize("cost")` should create a minimize objective named `cost`.
- `Objective::maximize("accuracy")` should create a maximize objective named `accuracy`.

---

### 7.3 `ObjectiveSpace`

Location: `src/core/objective.rs`

Purpose: Holds all objectives for a problem.

Required fields:

```rust
pub struct ObjectiveSpace {
    pub objectives: Vec<Objective>,
}
```

Required methods:

```rust
impl ObjectiveSpace {
    pub fn new(objectives: Vec<Objective>) -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn is_single_objective(&self) -> bool;
    pub fn is_multi_objective(&self) -> bool;
    pub fn as_minimization(&self, values: &[f64]) -> Vec<f64>;
}
```

Behavior of `as_minimization`:

- For a minimize objective, return the value unchanged.
- For a maximize objective, return the negated value.

Example:

```rust
let space = ObjectiveSpace::new(vec![
    Objective::minimize("cost"),
    Objective::maximize("accuracy"),
]);

assert_eq!(space.as_minimization(&[10.0, 0.8]), vec![10.0, -0.8]);
```

Validation behavior:

- In v1, `as_minimization` may silently zip to the shortest length.
- However, internal algorithms should check that evaluation objective lengths match `ObjectiveSpace::len()` and panic or handle gracefully during development.
- Prefer debug assertions for length mismatches.

---

### 7.4 `Evaluation`

Location: `src/core/evaluation.rs`

Purpose: Stores objective values and a simple total constraint violation.

Required fields:

```rust
pub struct Evaluation {
    pub objectives: Vec<f64>,
    pub constraint_violation: f64,
}
```

Required constructors and methods:

```rust
impl Evaluation {
    pub fn new(objectives: Vec<f64>) -> Self;
    pub fn constrained(objectives: Vec<f64>, constraint_violation: f64) -> Self;
    pub fn is_feasible(&self) -> bool;
}
```

Behavior:

- `Evaluation::new(objectives)` sets `constraint_violation` to `0.0`.
- `is_feasible()` returns true when `constraint_violation <= 0.0`.
- Positive constraint violation means infeasible.

Notes:

- Do not model named constraints in v1.
- Do not use generic objective scalar types.
- Store objective values as `f64`.

---

### 7.5 `Candidate<D>`

Location: `src/core/candidate.rs`

Purpose: Pairs a decision with its evaluation.

Required fields:

```rust
pub struct Candidate<D> {
    pub decision: D,
    pub evaluation: Evaluation,
}
```

Required constructor:

```rust
impl<D> Candidate<D> {
    pub fn new(decision: D, evaluation: Evaluation) -> Self;
}
```

Design decision:

- Do not add generic metadata to `Candidate` in v1.
- Algorithm-specific metadata should live inside algorithm modules.

Example:

```rust
struct Nsga2Entry<D> {
    candidate: Candidate<D>,
    rank: usize,
    crowding_distance: f64,
}
```

This keeps the public candidate type simple.

---

### 7.6 `Population<D>`

Location: `src/core/population.rs`

Purpose: Friendly wrapper around `Vec<Candidate<D>>`.

Required fields:

```rust
pub struct Population<D> {
    pub candidates: Vec<Candidate<D>>,
}
```

Required methods:

```rust
impl<D> Population<D> {
    pub fn new(candidates: Vec<Candidate<D>>) -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn iter(&self) -> impl Iterator<Item = &Candidate<D>>;
    pub fn into_vec(self) -> Vec<Candidate<D>>;
}

impl<D> From<Vec<Candidate<D>>> for Population<D>;
```

Design notes:

- This is not a trait.
- Keep `candidates` public for simplicity.
- Algorithms may use `Vec<Candidate<D>>` internally and wrap it at the end.

---

### 7.7 `OptimizationResult<D>`

Location: `src/core/result.rs`

Purpose: Standard return type for optimizers.

Required fields:

```rust
pub struct OptimizationResult<D> {
    pub population: Population<D>,
    pub pareto_front: Vec<Candidate<D>>,
    pub best: Option<Candidate<D>>,
    pub evaluations: usize,
    pub generations: usize,
}
```

Required methods:

```rust
impl<D> OptimizationResult<D> {
    pub fn population(&self) -> &Population<D>;
    pub fn pareto_front(&self) -> &[Candidate<D>];
    pub fn best(&self) -> Option<&Candidate<D>>;
}
```

Optional convenience constructor:

```rust
impl<D> OptimizationResult<D> {
    pub fn new(
        population: Population<D>,
        pareto_front: Vec<Candidate<D>>,
        best: Option<Candidate<D>>,
        evaluations: usize,
        generations: usize,
    ) -> Self;
}
```

Semantics:

- `population` is the final population or all sampled candidates, depending on algorithm.
- `pareto_front` is the non-dominated subset of the final population or archive.
- `best` is only meaningful for single-objective problems.
- For multi-objective problems, `best` may be `None`.
- `evaluations` counts calls to `Problem::evaluate`.
- `generations` counts major optimizer iterations. For algorithms without generations, use iterations.

---

### 7.8 `Rng`

Location: `src/core/rng.rs`

Purpose: Provide a single RNG type so public traits do not require generic RNG parameters.

Required type alias:

```rust
pub type Rng = rand::rngs::StdRng;
```

Required helper:

```rust
pub fn rng_from_seed(seed: u64) -> Rng;
```

Implementation:

- Use `rand::SeedableRng`.
- Return `StdRng::seed_from_u64(seed)`.

Design note:

- Do not make `Initializer` or `Variation` generic over RNG.
- This is a deliberate DevUX tradeoff.

---

## 8. Core Traits

### 8.1 `Problem`

Location: `src/core/problem.rs`

Purpose: Defines an optimization problem.

Trait:

```rust
pub trait Problem {
    type Decision: Clone;

    fn objectives(&self) -> ObjectiveSpace;
    fn evaluate(&self, decision: &Self::Decision) -> Evaluation;
}
```

Rules:

- `Decision` must be `Clone` because many heuristic algorithms clone decisions.
- `objectives()` returns by value for simple API ergonomics.
- `evaluate()` must not mutate the problem.
- Expensive caching is out of scope for v1.

Examples of possible decision types:

- `Vec<f64>`
- `Vec<bool>`
- `Vec<i64>`
- custom domain structs
- permutations represented as `Vec<usize>`

---

### 8.2 `Optimizer<P>`

Location: `src/traits/optimizer.rs`

Purpose: The single trait users implement to add a new optimizer.

Trait:

```rust
pub trait Optimizer<P>
where
    P: Problem,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision>;
}
```

Rules:

- `run` owns the optimizer loop.
- Optimizers may keep internal state as struct fields.
- Optimizers should return `OptimizationResult`.
- Do not require an associated error type in v1.
- If an algorithm cannot run because of invalid config, it may panic with a clear message in v1.

Rationale:

- A result-returning `run` method is the easiest API to understand.
- More advanced step-by-step APIs can be added later.

---

### 8.3 `Initializer<D>`

Location: `src/traits/initializer.rs`

Purpose: Generates initial decisions.

Trait:

```rust
pub trait Initializer<D> {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<D>;
}
```

Rules:

- `size` is the number of decisions to return.
- Implementations should return exactly `size` decisions unless impossible.
- If impossible, panic with a clear message in v1.

---

### 8.4 `Variation<D>`

Location: `src/traits/variation.rs`

Purpose: Generates child decisions from parent decisions.

Trait:

```rust
pub trait Variation<D> {
    fn vary(&mut self, parents: &[D], rng: &mut Rng) -> Vec<D>;
}
```

Rules:

- The optimizer chooses how many parents to pass.
- The variation operator may return one or more children.
- Parent decisions are passed by value inside the slice. In practice, algorithms clone selected parents into a small `Vec<D>` before calling `vary`.
- This is intentionally simpler than `&[&D]` for v1.

Examples:

- Gaussian mutation uses one parent and returns one child.
- Simulated binary crossover may use two parents and return two children.
- Differential evolution style variation may use three or more parents and return one child.

---

## 9. Pareto Utilities

Pareto utilities should mostly be free functions. Do not start with a `DominanceComparator` trait unless needed later.

### 9.1 `Dominance`

Location: `src/pareto/dominance.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dominance {
    Dominates,
    DominatedBy,
    NonDominated,
    Equal,
}
```

### 9.2 `pareto_compare`

Location: `src/pareto/dominance.rs`

Signature:

```rust
pub fn pareto_compare(
    a: &Evaluation,
    b: &Evaluation,
    objectives: &ObjectiveSpace,
) -> Dominance;
```

Behavior:

1. Feasible candidates dominate infeasible candidates.
2. If both are infeasible, the candidate with smaller `constraint_violation` dominates.
3. If both are feasible, compare objective values after converting everything to minimization using `ObjectiveSpace::as_minimization`.
4. Return:
   - `Dominance::Dominates` if `a` is no worse in every objective and better in at least one.
   - `Dominance::DominatedBy` if `b` dominates `a`.
   - `Dominance::Equal` if all objective values are equal after orientation conversion.
   - `Dominance::NonDominated` otherwise.

### 9.3 `pareto_front`

Location: `src/pareto/front.rs`

Signature:

```rust
pub fn pareto_front<D: Clone>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
) -> Vec<Candidate<D>>;
```

Behavior:

- Return all candidates not dominated by any other candidate in the input.
- O(N² * M) is acceptable for v1.
- Preserve input order among returned candidates.

### 9.4 `best_candidate`

Location: `src/pareto/front.rs` or `src/metrics/best.rs`

Signature:

```rust
pub fn best_candidate<D: Clone>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
) -> Option<Candidate<D>>;
```

Behavior:

- If there is not exactly one objective, return `None`.
- Ignore infeasible candidates.
- Return the best feasible candidate according to the objective direction.
- If all candidates are infeasible, return `None`.

### 9.5 `non_dominated_sort`

Location: `src/pareto/sort.rs`

Signature:

```rust
pub fn non_dominated_sort<D>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
) -> Vec<Vec<usize>>;
```

Behavior:

- Return fronts as indices into the input population.
- `fronts[0]` is the non-dominated front.
- O(N² * M) implementation is acceptable.

### 9.6 `crowding_distance`

Location: `src/pareto/crowding.rs`

Signature:

```rust
pub fn crowding_distance<D>(
    population: &[Candidate<D>],
    front: &[usize],
    objectives: &ObjectiveSpace,
) -> Vec<f64>;
```

Behavior:

- Return a vector with the same length as `front`.
- Boundary points receive `f64::INFINITY`.
- Interior points receive normalized crowding distance.
- If a front has length 0, return an empty vector.
- If a front has length 1 or 2, return all `f64::INFINITY`.
- Use minimization-oriented objective values.

---

## 10. Selection Utilities

Selection utilities may be concrete functions or small structs.

### 10.1 Random Selection

Location: `src/selection/random.rs`

Function:

```rust
pub fn select_random<D: Clone>(
    population: &[Candidate<D>],
    count: usize,
    rng: &mut Rng,
) -> Vec<D>;
```

Behavior:

- Select with replacement.
- Return cloned decisions.
- Panic if population is empty and `count > 0`.

### 10.2 Tournament Selection

Location: `src/selection/tournament.rs`

For v1, implement simple single-objective tournament selection:

```rust
pub fn tournament_select_single_objective<D: Clone>(
    population: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    tournament_size: usize,
    count: usize,
    rng: &mut Rng,
) -> Vec<D>;
```

Behavior:

- Requires exactly one objective.
- Select with replacement.
- Each tournament samples `tournament_size` candidates uniformly.
- Feasible candidates beat infeasible candidates.
- Among feasible candidates, best objective wins.
- Among infeasible candidates, smaller `constraint_violation` wins.

### 10.3 NSGA-II Parent Selection Helper

Location: `src/algorithms/nsga2.rs` or `src/selection/tournament.rs`

For NSGA-II, use binary tournament selection based on:

1. Lower non-dominated rank.
2. Higher crowding distance.
3. Random tie-break.

This can be private to the NSGA-II module in v1.

---

## 11. Operators

### 11.1 `RealBounds`

Location: `src/operators/real.rs`

Purpose: Initialize real-valued vectors within bounds.

Struct:

```rust
pub struct RealBounds {
    pub bounds: Vec<(f64, f64)>,
}
```

Required methods:

```rust
impl RealBounds {
    pub fn new(bounds: Vec<(f64, f64)>) -> Self;
}
```

Trait implementation:

```rust
impl Initializer<Vec<f64>> for RealBounds;
```

Behavior:

- For each decision, sample each variable uniformly in its inclusive bound range.
- Panic if any bound has `lo > hi`.

### 11.2 `GaussianMutation`

Location: `src/operators/real.rs`

Purpose: Simple mutation for `Vec<f64>`.

Struct:

```rust
pub struct GaussianMutation {
    pub sigma: f64,
}
```

Trait implementation:

```rust
impl Variation<Vec<f64>> for GaussianMutation;
```

Behavior:

- Expects at least one parent.
- Clone first parent.
- Add Gaussian noise with mean `0.0` and standard deviation `sigma` to every variable.
- Return one child.
- Panic if `sigma <= 0.0`.

Note:

- This operator does not enforce bounds in v1.
- Users can write bounded variants later.

### 11.3 `BitFlipMutation`

Location: `src/operators/binary.rs`

Purpose: Mutation for `Vec<bool>`.

Struct:

```rust
pub struct BitFlipMutation {
    pub probability: f64,
}
```

Behavior:

- Expects at least one parent.
- Clone first parent.
- Flip each bit with probability `probability`.
- Return one child.
- Panic if probability is outside `[0.0, 1.0]`.

### 11.4 `SwapMutation`

Location: `src/operators/permutation.rs`

Purpose: Mutation for permutations represented as `Vec<usize>`.

Struct:

```rust
pub struct SwapMutation;
```

Behavior:

- Expects at least one parent.
- Clone first parent.
- If length is at least 2, choose two indices and swap them.
- Return one child.

---

## 12. Algorithms

Built-in algorithms are reference implementations. They should be readable and practical, not excessively abstract.

### 12.1 `RandomSearch`

Location: `src/algorithms/random_search.rs`

Purpose: Baseline optimizer and example implementation.

Config:

```rust
pub struct RandomSearchConfig {
    pub iterations: usize,
    pub batch_size: usize,
    pub seed: u64,
}
```

Default:

- `iterations = 100`
- `batch_size = 1`
- `seed = 42`

Struct:

```rust
pub struct RandomSearch<I> {
    pub config: RandomSearchConfig,
    pub initializer: I,
}
```

Constructor:

```rust
impl<I> RandomSearch<I> {
    pub fn new(config: RandomSearchConfig, initializer: I) -> Self;
}
```

Implementation:

```rust
impl<P, I> Optimizer<P> for RandomSearch<I>
where
    P: Problem,
    I: Initializer<P::Decision>,
```

Behavior:

1. Create RNG from config seed.
2. For each iteration:
   - Initialize `batch_size` decisions.
   - Evaluate each decision.
   - Push candidates into population.
3. Compute `pareto_front` using `pareto_front`.
4. Compute `best` using `best_candidate`.
5. Return `OptimizationResult`.

### 12.2 `PAES`

Location: `src/algorithms/paes.rs`

Purpose: Simple Pareto Archived Evolution Strategy. Good first multi-objective reference algorithm.

Config:

```rust
pub struct PaesConfig {
    pub iterations: usize,
    pub archive_size: usize,
    pub seed: u64,
}
```

Struct:

```rust
pub struct Paes<I, V> {
    pub config: PaesConfig,
    pub initializer: I,
    pub variation: V,
}
```

Behavior:

1. Initialize one starting decision.
2. Evaluate it as the current candidate.
3. Initialize a Pareto archive with the current candidate.
4. For each iteration:
   - Use variation with the current decision as the only parent.
   - Evaluate the first returned child.
   - Compare child and current with `pareto_compare`.
   - If child dominates current, make child current.
   - If non-dominated, insert child into archive and choose whether to move to child. For v1, moving to child on non-dominated comparison is acceptable.
   - Insert current into archive.
   - If archive exceeds `archive_size`, truncate using crowding or simple removal. For v1, simple truncation is acceptable but should be documented.
5. Return archive as population and Pareto front.

Implementation note:

- This is intentionally simple. It is a readable baseline, not a research-perfect PAES.

### 12.3 `NSGA-II`

Location: `src/algorithms/nsga2.rs`

Purpose: Canonical Pareto-based evolutionary algorithm.

Config:

```rust
pub struct Nsga2Config {
    pub population_size: usize,
    pub generations: usize,
    pub seed: u64,
}
```

Default:

- `population_size = 100`
- `generations = 250`
- `seed = 42`

Struct:

```rust
pub struct Nsga2<I, V> {
    pub config: Nsga2Config,
    pub initializer: I,
    pub variation: V,
}
```

Constructor:

```rust
impl<I, V> Nsga2<I, V> {
    pub fn new(config: Nsga2Config, initializer: I, variation: V) -> Self;
}
```

Implementation:

```rust
impl<P, I, V> Optimizer<P> for Nsga2<I, V>
where
    P: Problem,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
```

Behavior:

1. Create RNG from seed.
2. Initialize `population_size` decisions.
3. Evaluate each initial decision.
4. For each generation:
   - Generate offspring until offspring length equals `population_size`.
   - Parent selection uses binary tournament on rank and crowding distance.
   - Variation receives cloned parent decisions.
   - Evaluate children.
   - Combine parent population and offspring.
   - Run non-dominated sorting.
   - Fill next population front by front.
   - If the next front does not fully fit, sort that front by crowding distance descending and take the most diverse candidates.
5. Return final population, Pareto front, best candidate if single objective, evaluation count, and generation count.

Private helper structs inside module:

```rust
struct Nsga2Entry<D> {
    candidate: Candidate<D>,
    rank: usize,
    crowding_distance: f64,
}
```

Implementation details:

- Use `non_dominated_sort` and `crowding_distance` utilities.
- Keep metadata private.
- Do not expose `Nsga2Entry` publicly.
- Panic clearly if `population_size == 0`.

### 12.4 `DifferentialEvolution`

Location: `src/algorithms/differential_evolution.rs`

Purpose: Single-objective real-valued reference optimizer.

This is optional for the first release but useful.

Config:

```rust
pub struct DifferentialEvolutionConfig {
    pub population_size: usize,
    pub generations: usize,
    pub differential_weight: f64,
    pub crossover_probability: f64,
    pub seed: u64,
}
```

Scope:

- Only support `Vec<f64>` decisions in v1.
- Require `RealBounds` initializer or bounds field.
- Only support single-objective problems.

If this adds too much complexity, defer it.

---

## 13. Pareto Archive

Location: `src/pareto/archive.rs`

Provide a concrete archive, not a trait.

Struct:

```rust
pub struct ParetoArchive<D> {
    pub members: Vec<Candidate<D>>,
    pub objectives: ObjectiveSpace,
}
```

Required methods:

```rust
impl<D: Clone> ParetoArchive<D> {
    pub fn new(objectives: ObjectiveSpace) -> Self;
    pub fn insert(&mut self, candidate: Candidate<D>);
    pub fn extend<I>(&mut self, candidates: I)
    where
        I: IntoIterator<Item = Candidate<D>>;
    pub fn truncate(&mut self, max_size: usize);
    pub fn members(&self) -> &[Candidate<D>];
    pub fn into_vec(self) -> Vec<Candidate<D>>;
}
```

Behavior of `insert`:

- If any existing member dominates the new candidate, discard the new candidate.
- Remove all existing members dominated by the new candidate.
- If equal or non-dominated with all members, insert the new candidate.

Behavior of `truncate`:

- If length is less than or equal to `max_size`, do nothing.
- For v1, simple truncation is acceptable.
- Later, truncation can use crowding distance.

---

## 14. Metrics

Metrics are helpful but should not complicate the core.

### 14.1 Spacing

Location: `src/metrics/spacing.rs`

Purpose: Measure spread of a Pareto front.

May be deferred.

### 14.2 Hypervolume

Location: `src/metrics/hypervolume.rs`

Purpose: Compute dominated hypervolume.

For v1:

- Implement only 2D exact hypervolume.
- Name it clearly: `hypervolume_2d`.

Signature:

```rust
pub fn hypervolume_2d<D>(
    front: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    reference_point: [f64; 2],
) -> f64;
```

Rules:

- Require exactly two objectives.
- Convert to minimization orientation.
- Assume reference point is worse than all relevant front points.
- Panic clearly if objective count is not 2.

---

## 15. Prelude

Location: `src/prelude.rs`

The prelude should export the common user-facing types and functions.

Include:

```rust
pub use crate::core::{
    Candidate,
    Direction,
    Evaluation,
    Objective,
    ObjectiveSpace,
    OptimizationResult,
    Population,
    Problem,
    Rng,
    rng_from_seed,
};

pub use crate::traits::{
    Initializer,
    Optimizer,
    Variation,
};

pub use crate::pareto::{
    Dominance,
    ParetoArchive,
    pareto_compare,
    pareto_front,
    best_candidate,
    non_dominated_sort,
    crowding_distance,
};

pub use crate::operators::{
    RealBounds,
    GaussianMutation,
    BitFlipMutation,
    SwapMutation,
};

pub use crate::algorithms::{
    RandomSearch,
    RandomSearchConfig,
    Paes,
    PaesConfig,
    Nsga2,
    Nsga2Config,
};
```

Avoid exporting too much if names become noisy.

---

## 16. `lib.rs`

`src/lib.rs` should declare modules and re-export the prelude.

Required shape:

```rust
pub mod core;
pub mod traits;
pub mod pareto;
pub mod selection;
pub mod operators;
pub mod algorithms;
pub mod metrics;
pub mod prelude;
```

`core/mod.rs` should publicly re-export core types:

```rust
pub use objective::*;
pub use evaluation::*;
pub use candidate::*;
pub use population::*;
pub use problem::*;
pub use result::*;
pub use rng::*;
```

Do similar simple re-exports in other `mod.rs` files.

---

## 17. Error Handling Policy

For v1, prefer simple behavior:

- Return normal values for normal optimizer output.
- Panic with clear messages for invalid algorithm configuration.
- Do not add `Result` to `Optimizer::run` in v1.

Examples of acceptable panics:

- `population_size must be greater than 0`
- `GaussianMutation sigma must be positive`
- `NSGA-II requires variation to return at least one child`
- `hypervolume_2d requires exactly 2 objectives`

Rationale:

- This keeps the API approachable.
- Sophisticated error handling can be added after the basic API proves useful.

---

## 18. Testing Plan

Use ordinary Rust unit tests and integration tests.

### 18.1 Core Type Tests

Test:

- `Objective::minimize`
- `Objective::maximize`
- `ObjectiveSpace::as_minimization`
- `Evaluation::new`
- `Evaluation::constrained`
- `Evaluation::is_feasible`
- `Candidate::new`
- `Population::new`
- `OptimizationResult` accessors

### 18.2 Pareto Tests

Test `pareto_compare`:

- One candidate dominates another.
- One candidate is dominated by another.
- Two candidates are non-dominated.
- Two candidates are equal.
- Feasible beats infeasible.
- Among infeasible candidates, lower violation wins.
- Maximize objectives are handled correctly.

Test `pareto_front`:

- Empty population returns empty front.
- Single candidate returns that candidate.
- Dominated points are removed.
- Non-dominated points are preserved.

Test `non_dominated_sort`:

- Known small population produces expected fronts.

Test `crowding_distance`:

- Empty front returns empty.
- One or two points return infinity.
- Boundary points get infinity.
- Interior points get finite values.

### 18.3 Operator Tests

Test:

- `RealBounds` returns correct shape and values within bounds.
- `GaussianMutation` returns one child of same length.
- `BitFlipMutation` with probability 0 changes nothing.
- `BitFlipMutation` with probability 1 flips all bits.
- `SwapMutation` preserves permutation contents.

### 18.4 Algorithm Tests

Test `RandomSearch`:

- Evaluation count equals `iterations * batch_size`.
- Result population length equals evaluation count.
- Pareto front is not empty if evaluations > 0.

Test `PAES`:

- Produces at least one candidate.
- Archive does not exceed configured archive size.

Test `NSGA-II`:

- Final population length equals population size.
- Evaluation count is at least initial population size.
- Pareto front is not empty.
- Deterministic with same seed.

### 18.5 Example Tests

Put examples under `examples/`:

```text
examples/
  toy_nsga2.rs
  random_search.rs
  custom_optimizer.rs
```

Run examples in CI with `cargo test --examples`.

---

## 19. Documentation Plan

### 19.1 README Structure

The README should include:

1. What `heuropt` is.
2. Installation.
3. Define a problem.
4. Run NSGA-II.
5. Implement a custom optimizer.
6. Current algorithms.
7. Design philosophy.

### 19.2 Crate-Level Docs

Add `//!` docs in `lib.rs` explaining:

- The purpose of the crate.
- The minimum API users need.
- A short example.

### 19.3 Doc Comments

Every public type and function must have a short doc comment.

Bad:

```rust
pub struct Candidate<D> { ... }
```

Good:

```rust
/// A decision together with its evaluated objective values.
pub struct Candidate<D> { ... }
```

---

## 20. Implementation Order

Implement in this order:

### Phase 1: Core

1. `Direction`
2. `Objective`
3. `ObjectiveSpace`
4. `Evaluation`
5. `Candidate`
6. `Population`
7. `OptimizationResult`
8. `Rng` and `rng_from_seed`
9. `Problem`
10. `Optimizer`
11. `Initializer`
12. `Variation`
13. `prelude`

Run tests.

### Phase 2: Pareto Utilities

1. `Dominance`
2. `pareto_compare`
3. `pareto_front`
4. `best_candidate`
5. `non_dominated_sort`
6. `crowding_distance`
7. `ParetoArchive`

Run tests.

### Phase 3: Operators

1. `RealBounds`
2. `GaussianMutation`
3. `BitFlipMutation`
4. `SwapMutation`

Run tests.

### Phase 4: Algorithms

1. `RandomSearch`
2. `PAES`
3. `NSGA-II`

Run tests.

### Phase 5: Examples and Docs

1. README.
2. Crate docs.
3. `examples/toy_nsga2.rs`.
4. `examples/random_search.rs`.
5. `examples/custom_optimizer.rs`.

Run all tests and examples.

---

## 21. Acceptance Criteria

The crate is ready for an initial release when all of the following are true:

1. `cargo test` passes.
2. `cargo test --examples` passes.
3. A user can define a problem with only `Problem` and `Evaluation`.
4. A user can run `RandomSearch`.
5. A user can run `NSGA-II`.
6. A user can implement a custom optimizer by implementing only `Optimizer<P>`.
7. Pareto utilities work for small known examples.
8. The README includes a full working example.
9. Public docs exist for every public item.
10. No core public API requires trait objects.
11. No core public API uses generic associated types.
12. No core public API exposes object-safety concerns.
13. The crate can be understood from `use heuropt::prelude::*`.

---

# 22. Dialectical Review: Junior Engineer Questions and Resolutions

This section records repeated review passes from intentionally literal junior-engineer reviewers. The goal is to eliminate ambiguity from the spec.

---

## Review Round 1

### Junior Engineer A: “What is a decision?”

A decision is the thing the optimizer changes. For real-valued problems it might be `Vec<f64>`. For binary problems it might be `Vec<bool>`. For a custom engineering problem it might be a struct.

The crate does not define a universal decision type. The problem author chooses it through:

```rust
impl Problem for MyProblem {
    type Decision = Vec<f64>;
}
```

Resolution added: Section 8.1 now lists example decision types.

---

### Junior Engineer B: “What is the difference between a decision and a candidate?”

A decision is unevaluated input.

A candidate is a decision plus its evaluation.

Example:

```rust
let decision = vec![1.0, 2.0];
let evaluation = problem.evaluate(&decision);
let candidate = Candidate::new(decision, evaluation);
```

Resolution added: Section 7.5 explicitly says `Candidate` pairs a decision with its evaluation.

---

### Junior Engineer C: “Do I use `Candidate` in `Problem::evaluate`?”

No.

`Problem::evaluate` receives a decision and returns an evaluation:

```rust
fn evaluate(&self, decision: &Self::Decision) -> Evaluation;
```

The optimizer creates `Candidate` after evaluation.

Resolution added: Section 8.1 clarifies the flow.

---

### Junior Engineer D: “Why does `Problem::objectives` return by value instead of reference?”

Because this is simpler for users. They do not need to store an `ObjectiveSpace` field inside every problem unless they want to.

Returning by value may clone small data. This is acceptable because objective definitions are tiny compared with optimization work.

Resolution added: Section 8.1 explains this DevUX tradeoff.

---

## Review Round 2

### Junior Engineer A: “How do I count evaluations?”

Every call to `problem.evaluate(...)` counts as one evaluation.

If an optimizer evaluates 100 initial candidates and 100 offspring for 250 generations, the evaluation count is:

```text
100 + 100 * 250 = 25,100
```

Resolution added: Section 7.7 defines `evaluations`.

---

### Junior Engineer B: “What is `generations` for random search?”

For algorithms that do not naturally have generations, use iterations.

For `RandomSearch`, `generations` equals `iterations`.

Resolution added: Section 7.7 defines this.

---

### Junior Engineer C: “What is `best` for multi-objective problems?”

For multi-objective problems, there usually is no single best candidate unless the user defines a preference.

So `best` should be `None` when there is more than one objective.

Use `pareto_front` for multi-objective results.

Resolution added: Section 7.7 and Section 9.4 clarify this.

---

### Junior Engineer D: “What if every single-objective candidate is infeasible?”

`best_candidate` returns `None`.

Resolution added: Section 9.4 clarifies this.

---

## Review Round 3

### Junior Engineer A: “What should happen if objective lengths do not match?”

Example: the problem says there are two objectives but `evaluate` returns one value.

For v1:

- Utilities may use debug assertions.
- Algorithms should panic with a clear message if they detect this.
- Do not add complicated error handling to the public API yet.

Resolution added: Section 7.3 and Section 17.

---

### Junior Engineer B: “Why not make `Optimizer::run` return `Result`?”

Because DevUX is the top priority and invalid optimizer configuration is programmer error in v1.

A simple return type is easier:

```rust
let result = optimizer.run(&problem);
```

Instead of:

```rust
let result = optimizer.run(&problem)?;
```

Later versions can add fallible APIs if needed.

Resolution added: Section 17.

---

### Junior Engineer C: “What if my evaluation can fail?”

In v1, wrap failure into a bad evaluation yourself.

For example:

```rust
Evaluation::constrained(vec![f64::INFINITY], 1.0)
```

or panic inside your problem if failure means a bug.

A fallible `Problem` trait is out of scope for v1.

Resolution added: Section 3 and Section 17.

---

### Junior Engineer D: “Why no metadata on `Candidate`?”

Because metadata differs by algorithm and makes the core type harder to understand.

NSGA-II can use a private struct with rank and crowding distance.

MOPSO can use a private particle struct with velocity.

The public `Candidate` stays simple.

Resolution added: Section 7.5.

---

## Review Round 4

### Junior Engineer A: “How does `Variation` know how many parents it gets?”

It does not know ahead of time. The optimizer decides.

For example:

- Mutation-only algorithms pass one parent.
- Crossover algorithms pass two parents.
- Differential evolution may pass three or more parents.

The variation implementation should panic with a clear message if it requires more parents than it received.

Resolution added: Section 8.4.

---

### Junior Engineer B: “Why does `Variation` take `&[D]` instead of `&[&D]`?”

Because `&[D]` is easier for entry-level users.

The optimizer can clone selected parents into a small vector before calling `vary`.

This may allocate more than an advanced API, but it avoids confusing lifetime-heavy signatures.

Resolution added: Section 8.4.

---

### Junior Engineer C: “What if a variation operator returns zero children?”

Algorithms should panic with a clear message if they require children and get none.

Example message:

```text
NSGA-II variation returned no children
```

Resolution added: Section 17 and Section 12.3.

---

### Junior Engineer D: “How do I keep children inside bounds?”

In v1, `GaussianMutation` does not enforce bounds.

Options:

1. Write a bounded mutation operator.
2. Clamp inside your problem’s evaluation.
3. Add a repair operator later.

A general repair trait is out of scope for v1.

Resolution added: Section 11.2.

---

## Review Round 5

### Junior Engineer A: “Is `ParetoArchive` a trait?”

No. It is a concrete struct in v1.

Reason: one obvious archive implementation is enough at first. Traits should be added only when multiple archive implementations need a shared interface.

Resolution added: Section 13.

---

### Junior Engineer B: “What does `ParetoArchive::truncate` do?”

For v1, simple truncation is acceptable. It removes extra members after `max_size`.

This is not ideal for quality, but it is simple and documented.

Later, truncation can use crowding distance.

Resolution added: Section 13.

---

### Junior Engineer C: “Can `PAES` use simple truncation?”

Yes for v1. The spec explicitly says PAES is a readable baseline, not a research-perfect implementation.

Resolution added: Section 12.2.

---

### Junior Engineer D: “Does `NSGA-II` need an archive?”

No. NSGA-II uses elitist survival selection over combined parent and offspring populations.

The final Pareto front can be computed from the final population.

Resolution added: Section 12.3.

---

## Review Round 6

### Junior Engineer A: “What should be in the first release?”

The first release should include:

- Core types.
- Core traits.
- Pareto utilities.
- Real-valued initializer.
- Gaussian mutation.
- Random search.
- PAES.
- NSGA-II.
- Examples.
- Tests.

Differential evolution and hypervolume can be deferred if needed.

Resolution added: Section 20 and Section 21.

---

### Junior Engineer B: “What should I implement first?”

Implement in this order:

1. Core types and traits.
2. Pareto utilities.
3. Operators.
4. Random search.
5. PAES.
6. NSGA-II.
7. Examples and docs.

Resolution added: Section 20.

---

### Junior Engineer C: “How do I know if my code works?”

Run:

```bash
cargo test
cargo test --examples
```

Then check the acceptance criteria in Section 21.

Resolution added: Section 18 and Section 21.

---

### Junior Engineer D: “Do I need to implement all algorithms before publishing?”

No.

Minimum useful release:

- `RandomSearch`
- `PAES`
- `NSGA-II`

Differential evolution, MOPSO, SPEA2, MOEA/D, and NSGA-III can come later.

Resolution added: Section 12 and Section 20.

---

## Review Round 7

### Junior Engineer A: “Where do I put public exports?”

Use `mod.rs` files for module-level re-exports and `prelude.rs` for common user imports.

Example:

```rust
pub use objective::*;
pub use evaluation::*;
pub use candidate::*;
```

Resolution added: Section 16.

---

### Junior Engineer B: “What does `use heuropt::prelude::*` include?”

It includes common user-facing types, traits, algorithms, operators, and Pareto functions.

Resolution added: Section 15.

---

### Junior Engineer C: “Should every public thing be in the prelude?”

No.

The prelude should include common things. Specialized metrics and advanced helpers can stay under their modules.

Resolution added: Section 15.

---

### Junior Engineer D: “Should examples use module paths or prelude?”

Examples should use:

```rust
use heuropt::prelude::*;
```

This confirms that the prelude is useful.

Resolution added: Section 18.5 and Section 19.

---

## Final Junior Review Status

After the above iterations, the simulated junior engineers had no remaining blocking questions.

Remaining acceptable unknowns for future versions:

- Whether to add fallible problems.
- Whether to add step-by-step optimizer APIs.
- Whether to add bounded mutation or repair operators.
- Whether to split the crate into multiple crates later.
- Whether to add advanced metrics and benchmark suites.

These are intentionally future-version questions and do not block v1 implementation.

---

# 23. Final Implementation Summary

Build `heuropt` as a single Rust crate with a small, obvious public API.

The core should be:

```text
Problem
Optimizer
Initializer
Variation

Objective
ObjectiveSpace
Evaluation
Candidate
Population
OptimizationResult
```

The first useful algorithm set should be:

```text
RandomSearch
PAES
NSGA-II
```

The first utility set should be:

```text
pareto_compare
pareto_front
best_candidate
non_dominated_sort
crowding_distance
ParetoArchive
```

The first operator set should be:

```text
RealBounds
GaussianMutation
BitFlipMutation
SwapMutation
```

The final test of success is this:

> A junior engineer can implement a new optimizer by reading `RandomSearch` and implementing `Optimizer<P>` without needing to understand object safety, advanced generics, or framework internals.

