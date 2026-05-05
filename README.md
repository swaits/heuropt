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
heuropt = "0.3"

# Optional features:
# - "serde":     derive Serialize/Deserialize on the core data types.
# - "parallel":  evaluate populations across rayon's thread pool.
#                Seeded runs stay bit-identical to serial mode.
# heuropt = { version = "0.3", features = ["serde", "parallel"] }
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
you need a **sample-efficient** or **multi-fidelity** approach:

- **Cheap (1k+ evals affordable):** any of the population-based
  algorithms — DE, GA, CMA-ES, NSGA-II, etc.
- **Expensive (50–500 evals):** `BayesianOpt` (Gaussian-process
  surrogate + Expected Improvement) or `Tpe` (Parzen-density
  surrogate, cheaper per step, more robust without hyperparameter
  tuning).
- **Multi-fidelity (each eval has a tunable budget — epochs, sim
  steps, MC samples):** `Hyperband`. Implement the `PartialProblem`
  trait on your problem and Hyperband allocates compute aggressively
  across promising configs.

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
you have three options:

- **Repair**: implement the `Repair<D>` trait (or use the provided
  `ClampToBounds` / `ProjectToSimplex` impls) to in-place project
  infeasible decisions back into the feasible region. Pair with a
  `Variation` operator to get bounds-aware variants without writing a
  custom `Variation` impl.
- **Stochastic ranking**: use `stochastic_ranking_select` instead of
  `tournament_select_single_objective`. It probabilistically explores
  near-feasibility instead of strict feasibility-first ordering, which
  helps when feasible regions are narrow.
- **Penalty-only**: stick with `constraint_violation` — the simplest,
  works well when the feasible region is large and convex.

---

### The decision tree

A flow you can run mentally:

```
START
 │
 ├─ Is each evaluation EXPENSIVE (>1 sec) or BUDGETED (50–500 total)?
 │   │
 │   ├─ Yes → sample-efficient regime
 │   │    ├─ Standard expensive black-box, single-objective
 │   │    │     → BayesianOpt   (GP + Expected Improvement; gold
 │   │    │                      standard *with* per-problem kernel
 │   │    │                      tuning. The default RBF kernel at
 │   │    │                      60 evals is honestly bad — give it
 │   │    │                      more evals or tune the kernel.)
 │   │    │     → Tpe           (KDE-based; cheaper per-step,
 │   │    │                      more robust without tuning)
 │   │    │
 │   │    └─ Each eval has a tunable fidelity (epochs, sim steps, …)
 │   │          → Hyperband     (implement PartialProblem; allocates
 │   │                           compute across configs adaptively)
 │   │
 │   └─ No → continue to the population-based branches below
 │
 └─ How many objectives?
     │
     ├─ 1 (single-objective)
     │    │
     │    ├─ Decision is Vec<f64> (continuous)
     │    │   ├─ Smooth landscape (well-conditioned)
     │    │   │     → CmaEs        (full-cov adaptive Gaussian)
     │    │   │     → SeparableNes (cheaper diag-cov; high-dim)
     │    │   │     → NelderMead   (low-dim, deterministic, simple)
     │    │   ├─ Multimodal landscape
     │    │   │     → IpopCmaEs            (CMA-ES with restart;
     │    │   │                              fixes vanilla CMA-ES's
     │    │   │                              multimodal failure)
     │    │   │     → DifferentialEvolution (rarely beaten on cheap
     │    │   │                              multimodal continuous)
     │    │   │     → SimulatedAnnealing   (cheap & generic)
     │    │   ├─ Want parameter-free (no F, CR, w, σ to tune)
     │    │   │     → Tlbo
     │    │   ├─ Want minimum self-adapting baseline
     │    │   │     → OnePlusOneEs          (one-fifth rule,
     │    │   │                               smallest possible ES)
     │    │   ├─ Just want a strong default for cheap continuous
     │    │   │     → DifferentialEvolution
     │    │   └─ Just want a baseline
     │    │         → RandomSearch
     │    │
     │    ├─ Decision is Vec<bool> (binary)
     │    │   ├─ Independent bits, smooth fitness
     │    │   │     → Umda            (per-bit marginal EDA)
     │    │   └─ Bit interactions matter
     │    │         → GeneticAlgorithm with BitFlipMutation +
     │    │           a bit-string crossover
     │    │
     │    ├─ Decision is Vec<usize> (permutation, e.g., TSP)
     │    │     → AntColonyTsp (with a distance matrix)
     │    │     → TabuSearch  (with your own neighbor function)
     │    │     → SimulatedAnnealing with SwapMutation
     │    │
     │    └─ Custom decision type (a struct, a tree, …)
     │          → SimulatedAnnealing or HillClimber
     │            with your own Variation impl
     │
     ├─ 2 or 3 (multi-objective)
     │    │
     │    ├─ Strong default, fast, well-understood
     │    │     → Nsga2
     │    │
     │    ├─ Real-valued, smooth front, want best convergence
     │    │     → Mopso    (multi-objective PSO; on the benches
     │    │                 here it wins ZDT1 on both HV and
     │    │                 convergence by 100× over the
     │    │                 dominance-based methods)
     │    │
     │    ├─ Want better front quality than NSGA-II
     │    │     → Ibea     (indicator-based; consistently the best
     │    │                 of the dominance-based methods on these
     │    │                 benches — wins ZDT3 HV and DTLZ2 mean
     │    │                 dist by 24×)
     │    │     → Spea2    (strength + density)
     │    │     → SmsEmoa  (hypervolume-contribution selection;
     │    │                 elegant in theory but underperforms
     │    │                 NSGA-II on these benches at our budgets —
     │    │                 only worth its higher per-step cost on
     │    │                 fronts where exact HV-contribution is
     │    │                 the right discriminator)
     │    │
     │    ├─ Want decomposition / weight-vector style
     │    │     → Moead    (very fast per generation, scales well)
     │    │
     │    ├─ Disconnected or non-convex front
     │    │     → AgeMoea  (estimates front geometry adaptively)
     │    │     → Knea     (favors knee points)
     │    │     → Ibea
     │    │
     │    ├─ Want region-based diversity
     │    │     → PesaII   (grid hyperboxes drive selection)
     │    │     → EpsilonMoea (ε-grid archive,
     │    │                     archive size auto-limits)
     │    │
     │    └─ Just one starting decision (no population budget)
     │          → Paes  (1+1 ES with a Pareto archive)
     │
     └─ 4+ (many-objective)
          │
          ├─ Linear / simplex-shaped front (e.g., DTLZ1)
          │     → Grea       (grid coords drive ranking; on DTLZ1
          │                    here it beats NSGA-III by 3× and
          │                    AGE-MOEA by 2.5×)
          │     → Moead      (decomposition shines on linear fronts;
          │                    second on DTLZ1, also among the
          │                    fastest per generation)
          │
          ├─ Curved / unknown front geometry
          │     → Nsga3      (reference-point niching, canonical;
          │                    a strong default when the front
          │                    isn't simplex-shaped)
          │     → AgeMoea    (estimates L_p geometry per generation)
          │     → Rvea       (reference vectors with adaptive penalty)
          │
          ├─ Want indicator-based selection
          │     → Ibea       (additive ε-indicator; doesn't degrade
          │                   at high obj count)
          │     → Hype       (Monte Carlo HV estimation; scales
          │                   to arbitrary M)
```

### Quick reference

**Sample-efficient / expensive evaluation (50–500 evals):**

| Algorithm        | Objectives | Decision   | Strengths |
|---|---|---|---|
| `BayesianOpt`    | 1          | `Vec<f64>` | GP surrogate + EI; gold standard *with* per-problem kernel tuning (default RBF at 60 evals is honestly bad) |
| `Tpe`            | 1          | `Vec<f64>` | KDE surrogate; robust without hyperparameter tuning |
| `Hyperband`      | 1          | any        | multi-fidelity; needs `PartialProblem` |

**Single-objective continuous (`Vec<f64>`):**

| Algorithm                | Strengths |
|---|---|
| `RandomSearch`           | sanity baseline |
| `HillClimber`            | simplest greedy local search |
| `OnePlusOneEs`           | one-fifth-rule self-adapting baseline |
| `SimulatedAnnealing`     | escapes local optima |
| `GeneticAlgorithm`       | classic SO GA with elitism |
| `ParticleSwarm`          | simple swarm baseline |
| `DifferentialEvolution`  | strong default for cheap continuous |
| `Tlbo`                   | parameter-free (no F, CR, w, σ) |
| `CmaEs`                  | smooth landscapes; full covariance |
| `IpopCmaEs`              | CMA-ES + restart for multimodal |
| `SeparableNes`           | diagonal-cov NES; cheap per-step |
| `NelderMead`             | classical simplex; deterministic |

**Single-objective other decision types:**

| Algorithm        | Decision        | Strengths |
|---|---|---|
| `Umda`           | `Vec<bool>`     | independent-bit EDA |
| `TabuSearch`     | any             | discrete, you supply neighbors |
| `AntColonyTsp`   | `Vec<usize>`    | TSP / permutation |

**Multi-objective (2–3) and many-objective (4+):**

| Algorithm     | Objectives | Strengths |
|---|---|---|
| `Paes`        | 2–3        | 1+1 ES with Pareto archive |
| `Nsga2`       | 2–3        | canonical Pareto-based EA |
| `Spea2`       | 2–3        | strength + density |
| `Mopso`       | 2–3        | multi-objective PSO; best convergence on smooth real-valued 2-obj fronts |
| `Ibea`        | 2+         | indicator-based; consistently best of the dominance-based methods |
| `SmsEmoa`     | 2+         | exact HV-contribution selection; high per-step cost, modest gain |
| `Hype`        | 2+         | Monte Carlo HV estimation |
| `EpsilonMoea` | 2+         | ε-grid archive; auto-sized |
| `PesaII`      | 2+         | grid-based region selection |
| `AgeMoea`     | 2+         | adaptive front-geometry estimation |
| `Knea`        | 2+         | knee-point favored survival |
| `Moead`       | 2+         | decomposition; fast per-gen |
| `Nsga3`       | 4+         | reference-point niching; strong on curved fronts |
| `Rvea`        | 4+         | reference vectors with penalty |
| `Grea`        | 4+         | grid coords drive selection; particularly strong on linear/simplex fronts |

## Current algorithms

The full list with one-line descriptions:

**Sample-efficient / multi-fidelity:**

- `BayesianOpt` — Gaussian-process surrogate + Expected Improvement.
- `Tpe` — Bergstra et al. 2011 Tree-structured Parzen Estimator.
- `Hyperband` — Li et al. 2017 multi-fidelity (uses `PartialProblem`).

**Single-objective:**

- `RandomSearch` — sample-evaluate-keep baseline.
- `HillClimber` — greedy single-step local search.
- `OnePlusOneEs` — Rechenberg 1973 (1+1)-ES with one-fifth rule.
- `SimulatedAnnealing` — Kirkpatrick et al. 1983, generic over decision type.
- `TabuSearch` — Glover 1986, with a user-supplied neighbor generator.
- `GeneticAlgorithm` — generational GA with tournament selection + elitism.
- `ParticleSwarm` — Eberhart & Kennedy 1995 PSO for `Vec<f64>`.
- `DifferentialEvolution` — Storn & Price DE/rand/1/bin for `Vec<f64>`.
- `Tlbo` — Rao 2011 Teaching-Learning-Based Optimization (parameter-free).
- `CmaEs` — Hansen & Ostermeier 2001 covariance-matrix adaptation.
- `IpopCmaEs` — Auger & Hansen 2005 CMA-ES with restart, for multimodal.
- `SeparableNes` — Wierstra et al. 2008/2014 diagonal-cov NES.
- `NelderMead` — Nelder & Mead 1965 simplex direct search.
- `Umda` — Mühlenbein 1997 univariate marginal-distribution EDA for `Vec<bool>`.
- `AntColonyTsp` — Dorigo Ant System for permutation problems.

**Multi-objective:**

- `Paes` — Knowles & Corne 1999 Pareto Archived Evolution Strategy.
- `Nsga2` — Deb et al. 2002, the canonical Pareto-based EA.
- `Spea2` — Zitzler, Laumanns & Thiele 2001 strength-Pareto EA.
- `Moead` — Zhang & Li 2007 decomposition-based MOEA with Tchebycheff scalarization.
- `Mopso` — Coello, Pulido & Lechuga 2004 multi-objective PSO.
- `Ibea` — Zitzler & Künzli 2004 indicator-based EA.
- `SmsEmoa` — Beume, Naujoks & Emmerich 2007 hypervolume-selection EMOA.
- `Hype` — Bader & Zitzler 2011 Hypervolume Estimation Algorithm.
- `EpsilonMoea` — Deb, Mohan & Mishra 2003 ε-dominance MOEA.
- `PesaII` — Corne et al. 2001 Pareto Envelope Selection II.
- `AgeMoea` — Panichella 2019 Adaptive Geometry Estimation MOEA.
- `Knea` — Zhang, Tian & Jin 2015 Knee point-driven EA.

**Many-objective (4+):**

- `Nsga3` — Deb & Jain 2014 reference-point NSGA-III.
- `Rvea` — Cheng et al. 2016 Reference Vector-guided EA.
- `Grea` — Yang et al. 2013 Grid-based EA.

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

## Testing

heuropt is exhaustively tested across several layers:

- **Unit + integration tests** (`cargo test`) — 313 tests covering
  every algorithm, operator, metric, Pareto utility, and edge case
  (empty/singleton/duplicate populations, flat fitness, zero-width
  bounds, infeasible-only populations).
- **Property-based tests** (`proptest`) — bounds preservation,
  Pareto antisymmetry/reflexivity, partition correctness,
  determinism, and seed-stability checks for every algorithm.
- **Coverage-guided fuzzing** (`cargo +nightly fuzz run <target>`) —
  eight targets at `fuzz/fuzz_targets/`, soaked for 60 s per target
  in CI on every PR.
- **Instruction-count benchmarks** (`cargo bench`) — `gungraun`
  (callgrind) hot-path benchmarks for every algorithm and Pareto
  utility, machine-stable so PR-level regressions show up.
- **Mutation testing** (`cargo mutants`) — advisory; config at
  `.cargo/mutants.toml`.
- **CI** (`.github/workflows/ci.yml`) — fmt, clippy
  (`-D warnings`), test (4-feature matrix), doc, MSRV (1.85), fuzz.

## License

MIT — see [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
