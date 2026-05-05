# heuropt

[![Crates.io](https://img.shields.io/crates/v/heuropt.svg)](https://crates.io/crates/heuropt)
[![Documentation](https://docs.rs/heuropt/badge.svg)](https://docs.rs/heuropt)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A practical Rust toolkit for implementing heuristic single-objective,
multi-objective, and many-objective optimization algorithms.

`heuropt` is **not** a research framework full of abstract machinery — it is a
small set of concrete types, a handful of simple traits, and a few reference
algorithms. The goal: an entry-level Rust engineer can define a problem, run a
built-in optimizer, or implement a new optimizer without learning any
framework concepts.

## Installation

```toml
[dependencies]
heuropt = "0.2"

# Optional features:
# - "serde":     derive Serialize/Deserialize on the core data types.
# - "parallel":  evaluate populations across rayon's thread pool.
#                Seeded runs stay bit-identical to serial mode.
# heuropt = { version = "0.2", features = ["serde", "parallel"] }
```

## Define a problem

```rust
use heuropt::prelude::*;

struct SchafferN1;

impl Problem for SchafferN1 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
        ])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let v = x[0];
        Evaluation::new(vec![v * v, (v - 2.0).powi(2)])
    }
}
```

## Run NSGA-II

```rust
use heuropt::prelude::*;

# struct SchafferN1;
# impl Problem for SchafferN1 {
#     type Decision = Vec<f64>;
#     fn objectives(&self) -> ObjectiveSpace {
#         ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")])
#     }
#     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
#         Evaluation::new(vec![x[0] * x[0], (x[0] - 2.0).powi(2)])
#     }
# }
let initializer = RealBounds::new(vec![(-5.0, 5.0)]);
let variation = GaussianMutation { sigma: 0.2 };
let config = Nsga2Config { population_size: 60, generations: 80, seed: 42 };
let mut optimizer = Nsga2::new(config, initializer, variation);
let result = optimizer.run(&SchafferN1);

println!("Pareto front size: {}", result.pareto_front.len());
```

See `examples/toy_nsga2.rs` for the full version.

## Implement a custom optimizer

A new optimizer is just an implementation of `Optimizer<P>`:

```rust
use heuropt::prelude::*;

struct MyOptimizer { /* state */ }

impl<P> Optimizer<P> for MyOptimizer
where
    P: Problem<Decision = Vec<f64>>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        // Generate candidates.
        // Evaluate them with `problem.evaluate(...)`.
        // Keep the best, or maintain a Pareto archive.
        // Return an OptimizationResult.
        # OptimizationResult::new(
        #     Population::new(Vec::new()),
        #     Vec::new(),
        #     None,
        #     0,
        #     0,
        # )
    }
}
```

A complete worked example is in `examples/custom_optimizer.rs`.

## Choosing an algorithm

Optimization is a noisy field with a lot of jargon. This section walks you
through picking a starting algorithm for a real problem, defining the terms
as they come up. If you already know the vocabulary, jump to the
[quick-reference table](#quick-reference) at the bottom.

### Step 1: What is your problem?

Three ingredients describe any optimization problem:

- A **decision** — the thing the algorithm is allowed to change. Examples:
  five real numbers (`Vec<f64>`), a yes/no flag for each of 100 features
  (`Vec<bool>`), or an ordering of cities to visit (`Vec<usize>`).
- One or more **objectives** — numbers you want to make small (or large).
  Examples: a model's prediction error, a tour's total length, a circuit's
  power draw.
- An optional set of **constraints** — conditions a decision must satisfy
  to be valid. Examples: "the budget cannot exceed $1M," or "every car
  must be visited exactly once."

Your job is to express the problem; heuropt's job is to search for
decisions that score well on the objectives without violating the
constraints.

### Step 2: How many objectives?

The biggest fork in the road. Algorithms specialize sharply by
objective count:

- **Single-objective (1)** — one number to optimize. There's a clear
  "best" answer. Examples: minimize loss, maximize throughput.
- **Multi-objective (2 or 3)** — several conflicting goals. There is no
  single best; instead there is a **Pareto front**: the set of decisions
  where you cannot improve any objective without sacrificing another.
  Each point on the front is a different tradeoff.
- **Many-objective (4+)** — same idea, but classical multi-objective
  algorithms break down because almost every pair of points is
  *non-dominated* (neither one is strictly better) once you have lots
  of objectives.

> **Dominance:** Decision A *dominates* decision B if A is at least as
> good as B on every objective and strictly better on at least one. The
> Pareto front is what you get after deleting every dominated decision.

If you found yourself staring at a single composite score that's a
weighted sum of conflicting goals, you probably actually have a
multi-objective problem in disguise.

### Step 3: What does the search space look like?

A few questions about the geometry of your problem:

- Is the **decision continuous** (real numbers), **discrete** (integers,
  bits), or a **permutation** (an ordering)?
- Is the landscape **unimodal** (one hill, easy to climb) or
  **multimodal** (lots of local optima that aren't the global one)?
  Rastrigin and Ackley are classic multimodal traps.
- How **smooth** is it? Smooth landscapes (e.g., a quadratic bowl)
  reward gradient-like methods (CMA-ES); jagged or noisy ones reward
  population-based methods (DE, GA).

If you don't know, treat it as multimodal — it's the cautious default.

### Step 4: How expensive is each evaluation?

Cheap evaluations (a few microseconds — pure math, simple simulation)
let you afford 100k+ evaluations per run. Expensive evaluations (a
training run, a CFD simulation, a real-world measurement that costs
money) force you to be sample-efficient: 50–500 evaluations total.

This decides whether you can afford a **population-based** algorithm
that throws hundreds of evaluations at each generation, or whether
you need a **sample-efficient** approach. heuropt's current toolkit
is mostly population-based; for budgets under a few hundred
evaluations you may want Bayesian optimization (out of scope today).

The `parallel` feature flag also matters here — if your `evaluate`
function takes more than ~50 µs, enabling rayon-backed parallel
population evaluation will speed runs up significantly.

### Step 5: Are there hard constraints?

heuropt models constraints as a single scalar **constraint violation**
on each `Evaluation`. The convention: `0.0` (or negative) means
feasible; positive means infeasible, and bigger numbers are worse
violations. Every Pareto-comparison and tournament-selection helper
in the crate prefers feasible candidates and breaks ties on
violation magnitude, so the rule "feasibility comes first" is
enforced automatically.

If your constraints are very tight and the search keeps hitting them,
consider also adding a **repair operator** (clamping, rounding, or a
greedy fix) inside your `Variation` impl so children come out feasible
in the first place. The example `BoundedGaussianMutation` does this for
real-valued bounds.

---

### The decision tree

A flow you can run mentally:

```
START
 │
 ├─ How many objectives?
 │   │
 │   ├─ 1 (single-objective)
 │   │    │
 │   │    ├─ Decision is Vec<f64> (continuous)
 │   │    │   ├─ Smooth, low-dim, expensive evals
 │   │    │   │     → CmaEs              (sample-efficient,
 │   │    │   │                            invariant to scale & rotation)
 │   │    │   ├─ Multimodal or jagged
 │   │    │   │     → DifferentialEvolution
 │   │    │   │     → ParticleSwarm
 │   │    │   │     → SimulatedAnnealing  (cheap & generic)
 │   │    │   ├─ Just want a strong default
 │   │    │   │     → DifferentialEvolution  (rarely beaten on cheap
 │   │    │   │                                evaluators)
 │   │    │   └─ Just want a baseline
 │   │    │         → RandomSearch
 │   │    │
 │   │    ├─ Decision is Vec<bool> (binary)
 │   │    │   ├─ Independent bits, smooth fitness
 │   │    │   │     → Umda            (estimates per-bit marginals)
 │   │    │   └─ Bit interactions matter
 │   │    │         → GeneticAlgorithm with BitFlipMutation +
 │   │    │           a bit-string crossover
 │   │    │
 │   │    ├─ Decision is Vec<usize> (permutation, e.g., TSP)
 │   │    │     → AntColonyTsp (with a distance matrix)
 │   │    │     → TabuSearch  (with your own neighbor function)
 │   │    │     → SimulatedAnnealing with SwapMutation
 │   │    │
 │   │    └─ Custom decision type (a struct, a tree, …)
 │   │          → SimulatedAnnealing or HillClimber
 │   │            with your own Variation impl
 │   │            (the trait is generic over D)
 │   │
 │   ├─ 2 or 3 (multi-objective)
 │   │    │
 │   │    ├─ Just want a strong default
 │   │    │     → Nsga2  (canonical, fast, well-understood)
 │   │    │
 │   │    ├─ Convergence quality matters more than speed
 │   │    │     → Spea2  (slower, comparable quality)
 │   │    │     → Ibea   (often beats Nsga2 on tough fronts)
 │   │    │
 │   │    ├─ Want decomposition / weight-vector style
 │   │    │     → Moead  (very fast per generation, scales well)
 │   │    │
 │   │    ├─ Real-valued and want swarm style
 │   │    │     → Mopso  (good on simple 2-obj fronts)
 │   │    │
 │   │    └─ Just one starting decision (no population budget)
 │   │          → Paes  (1+1 ES with a Pareto archive)
 │   │
 │   └─ 4+ (many-objective)
 │        │
 │        ├─ Just want a strong default
 │        │     → Nsga3  (reference-point niching, the canonical
 │        │              many-obj choice)
 │        │     → Moead  (also scales naturally past 3 objectives)
 │        │
 │        └─ Convergence vs diversity tradeoff matters
 │              → Ibea   (indicator-based, doesn't lose discrimination
 │                       at high obj count)
 │
 └─ Don't forget:
     - Set a seed for reproducibility (every Config has one).
     - Enable the `parallel` feature if your evaluate is expensive.
     - Use `examples/compare.rs` as a template for benchmarking
       multiple algorithms on your own problem.
```

### Quick reference

| Algorithm | Objectives | Decision type | Strengths |
|---|---|---|---|
| `RandomSearch`            | any   | any            | sanity baseline |
| `HillClimber`             | 1     | any            | simplest greedy local search |
| `SimulatedAnnealing`      | 1     | any            | escapes local optima, decision-type-agnostic |
| `TabuSearch`              | 1     | any            | combinatorial / discrete, you supply neighbors |
| `GeneticAlgorithm`        | 1     | any            | classic SO GA with elitism |
| `ParticleSwarm`           | 1     | `Vec<f64>`     | simple swarm, good baseline |
| `DifferentialEvolution`   | 1     | `Vec<f64>`     | strong default for cheap continuous problems |
| `CmaEs`                   | 1     | `Vec<f64>`     | sample-efficient, smooth landscapes |
| `Umda`                    | 1     | `Vec<bool>`    | independent-bit binary problems |
| `AntColonyTsp`            | 1     | `Vec<usize>`   | TSP / permutation problems |
| `Paes`                    | 2–3   | any (variation defines) | 1+1 ES with archive |
| `Nsga2`                   | 2–3   | any            | canonical multi-objective EA |
| `Spea2`                   | 2–3   | any            | strength + density-based MOEA |
| `Moead`                   | 2+    | any            | decomposition-based, fast per gen |
| `Mopso`                   | 2–3   | `Vec<f64>`     | multi-objective PSO with archive |
| `Ibea`                    | 2+    | any            | indicator-based, scales to many obj |
| `Nsga3`                   | 4+    | any            | reference-point niching for many-obj |

## Current algorithms

The full list with one-line descriptions:

**Single-objective:**

- `RandomSearch` — sample-evaluate-keep baseline.
- `HillClimber` — greedy single-step local search.
- `SimulatedAnnealing` — Kirkpatrick et al. 1983, generic over decision type.
- `TabuSearch` — Glover 1986, with a user-supplied neighbor generator.
- `GeneticAlgorithm` — generational GA with tournament selection + elitism.
- `ParticleSwarm` — Eberhart & Kennedy 1995 PSO for `Vec<f64>`.
- `DifferentialEvolution` — Storn & Price DE/rand/1/bin for `Vec<f64>`.
- `CmaEs` — Hansen & Ostermeier 2001 covariance-matrix adaptation.
- `Umda` — Mühlenbein 1997 univariate marginal-distribution EDA for `Vec<bool>`.
- `AntColonyTsp` — Dorigo Ant System for permutation problems.

**Multi-objective:**

- `Paes` — Knowles & Corne 1999 Pareto Archived Evolution Strategy.
- `Nsga2` — Deb et al. 2002, the canonical Pareto-based EA.
- `Spea2` — Zitzler, Laumanns & Thiele 2001 strength-Pareto EA.
- `Moead` — Zhang & Li 2007 decomposition-based MOEA with Tchebycheff scalarization.
- `Mopso` — Coello, Pulido & Lechuga 2004 multi-objective PSO.
- `Ibea` — Zitzler & Künzli 2004 indicator-based EA.

**Many-objective (4+):**

- `Nsga3` — Deb & Jain 2014 reference-point NSGA-III.

**Reusable utilities:** `pareto_compare`, `pareto_front`, `best_candidate`,
`non_dominated_sort`, `crowding_distance`, `ParetoArchive`, `das_dennis`,
and the metrics `spacing` and `hypervolume_2d`.

## Design philosophy

- **Concrete data, small trait surface.** `Problem`, `Optimizer`, `Initializer`,
  `Variation` are the only traits a user interacts with day-to-day. Everything
  else is plain structs.
- **No type hell.** No trait objects in the core path, no GATs, no HRTBs in
  user-facing APIs, no generic-RNG plumbing — `Rng` is a single concrete type
  alias.
- **Readable algorithms.** Built-ins are written for clarity, not maximum
  abstraction reuse. `RandomSearch` is the recommended file to read before
  writing your own optimizer.
- **One crate first.** No premature splitting into `-core`/`-algorithms`/
  `-operators`. Split later if the crate grows.
- **Panic on programmer error.** Invalid configuration panics with a clear
  message in v1; the API may grow `Result`-returning variants later if the
  base API proves useful.

See `docs/heuropt_tech_design_spec.md` for the full design rationale.

## License

MIT — see [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
