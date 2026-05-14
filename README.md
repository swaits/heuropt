# heuropt

[![Crates.io](https://img.shields.io/crates/v/heuropt.svg)](https://crates.io/crates/heuropt)
[![Documentation](https://docs.rs/heuropt/badge.svg)](https://docs.rs/heuropt)
[![Book](https://img.shields.io/badge/book-online-blue.svg)](https://swaits.github.io/heuropt/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/swaits/heuropt/actions/workflows/ci.yml/badge.svg)](https://github.com/swaits/heuropt/actions/workflows/ci.yml)

**A practical Rust toolkit for heuristic optimization.** Single-objective.
Multi-objective. Many-objective. 33 algorithms — every one of them with a
sync `run` and an async `run_async`. One small set of traits. Bit-identical
seeded determinism. No trait objects, no GATs, no generic-RNG plumbing in
the public API.

If you can write a `Problem` impl and read Random Search, you can write your
own optimizer. That's the whole pitch.

Docs: [user guide](https://swaits.github.io/heuropt/) · [API reference](https://docs.rs/heuropt).

## Installation

```toml
[dependencies]
heuropt = "0.10"

# Optional features:
# - "serde":     derive Serialize/Deserialize on the core data types.
# - "parallel":  evaluate populations across rayon's thread pool.
#                Seeded runs stay bit-identical to serial mode.
# - "async":     AsyncProblem / AsyncPartialProblem traits and a
#                run_async(&problem, concurrency).await method on
#                every algorithm — for IO-bound evaluations.
# heuropt = { version = "0.10", features = ["serde", "parallel", "async"] }
```

## Define a problem and run an optimizer

You're designing a car. Three things you can pick: **engine
displacement** (1.0–6.0 L), **curb weight** (1100–2200 kg, where
going lighter requires aluminum/carbon and costs money), and
**aerodynamic drag** (Cd from 0.20 to 0.40, where slipperier needs
expensive aero R&D). Four things you want to optimize: **price**,
**0-60 acceleration**, **fuel consumption**, **idle noise** — all
in tension.

The relationships between decisions and objectives are nonlinear
and coupled: engine cost grows superlinearly with displacement,
weight reduction below 1500 kg costs a quadratic premium, drag
reduction below 0.35 Cd costs a 1.5-power premium, and 0-60 depends
on weight × engine in a non-trivial way. You can't just sweep one
slider — the Pareto front is a genuine surface in 3D decision space,
and finding it by hand is hopeless.

NSGA-III is the canonical many-objective (4+) optimizer; it uses
Das–Dennis reference points to keep the front well-spread.

```rust
use heuropt::prelude::*;

struct PickACar;

impl Problem for PickACar {
    type Decision = Vec<f64>; // [engine_liters, weight_kg, drag_cd]

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("price_thousand_dollars"),
            Objective::minimize("seconds_to_60mph"),
            Objective::minimize("fuel_gallons_per_100mi"),
            Objective::minimize("noise_db_at_idle"),
        ])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let displacement = x[0]; // liters
        let weight       = x[1]; // kg
        let drag         = x[2]; // dimensionless Cd

        // Price ($k): engine cost grows superlinearly; weight reduction
        // below 1500 kg and drag reduction below 0.35 Cd both cost extra.
        let engine_cost = 3.0 * displacement.powf(1.6);
        let weight_cost = ((1500.0 - weight).max(0.0) / 100.0).powi(2) * 2.0;
        let aero_cost   = ((0.35 - drag).max(0.0) * 100.0).powf(1.5) * 0.4;
        let price = 10.0 + engine_cost + weight_cost + aero_cost;

        // 0-60 (s): heavier = slower; bigger engine = quicker but with
        // diminishing returns.
        let weight_factor = (weight - 1100.0) / 1000.0;
        let engine_factor = ((displacement - 1.0) / 5.0).max(0.0).powf(0.7);
        let zero_to_sixty = 5.0 + 5.0 * weight_factor - 4.0 * engine_factor;

        // Fuel consumption (gal/100 mi): all three matter.
        let fuel = 0.5 + 0.5 * displacement + 0.5 * weight / 1000.0 + 4.0 * drag;

        // Idle noise (dB): engine dominates, mildly nonlinear.
        let noise = 60.0 + 3.0 * displacement.powf(1.2);

        Evaluation::new(vec![price, zero_to_sixty, fuel, noise])
    }
}

fn main() {
    let bounds = vec![
        (1.0_f64, 6.0_f64),       // engine
        (1100.0_f64, 2200.0_f64), // weight
        (0.20_f64, 0.40_f64),     // drag
    ];

    let mut optimizer = Nsga3::new(
        Nsga3Config {
            population_size: 100,
            generations: 200,
            reference_divisions: 5,
            seed: 42,
        },
        RealBounds::new(bounds.clone()),
        CompositeVariation {
            crossover: SimulatedBinaryCrossover::new(bounds.clone(), 15.0, 0.9),
            mutation:  PolynomialMutation::new(bounds, 20.0, 1.0 / 3.0),
        },
    );
    let result = optimizer.run(&PickACar);

    let mut front: Vec<_> = result.pareto_front.iter().collect();
    front.sort_by(|a, b| {
        a.evaluation.objectives[0]
            .partial_cmp(&b.evaluation.objectives[0]).unwrap()
    });
    println!("{:>5} {:>5} {:>4}    {:>6} {:>5} {:>5} {:>5}",
             "L", "kg", "Cd", "$k", "0-60", "fuel", "dB");
    for c in &front {
        let d = &c.decision;
        let o = &c.evaluation.objectives;
        println!("{:>5.2} {:>5.0} {:>4.2}    {:>6.1} {:>5.1} {:>5.2} {:>5.1}",
                 d[0], d[1], d[2], o[0], o[1], o[2], o[3]);
    }
}
```

Run it (`cargo run --release`) and you get 100 cars on the front.
A representative slice from the actual output, hand-picked across
the spectrum:

```text
    L    kg   Cd        $k  0-60  fuel    dB     ← role
 1.00  1505 0.35      13.0   7.0  3.17  63.0     cheap baseline
 2.00  1370 0.35      22.4   5.1  3.56  66.7     sensible sport sedan
 2.45  1330 0.38      28.5   4.5  3.92  68.8     quicker midprice
 1.00  1430 0.21      35.8   6.6  2.54  63.0     fuel-saver (small + slippery)
 3.50  1300 0.25      52.9   3.5  3.88  73.3     genuine sports car
 5.27  1100 0.20     108.1   1.4  4.48  82.0     hypercar corner
```

### Reading the result

Every row is **non-dominated** — no row is strictly better than
another on every metric. The interesting part is what each one does
*differently*:

- The **cheap baseline** ($13k) takes the path of least resistance:
  smallest engine, no weight reduction, average drag. Slow but
  affordable.
- The **sensible sedan** ($22k) trades $9k for **2 seconds off
  0-60** by running a 2.0L engine with mild weight reduction.
- The **fuel-saver** is interesting: it's a 1.0L econobox engine,
  but it spends $22k *just on aero* (0.21 Cd) to push fuel
  consumption down to **2.54 gal/100mi**. The optimizer figured
  out that aero matters more than displacement at this fuel point.
  No human would pick this combo by intuition.
- The **sports car** ($53k) doesn't blow money on the lightest
  possible weight — it picks 1300 kg, because dropping further
  costs disproportionately and the 3.5L engine is doing most of
  the acceleration work.
- The **hypercar corner** ($108k) is the optimizer pushing every
  decision to its ceiling: minimum weight (1100 kg), minimum
  drag (0.20 Cd), big engine (5.3L). Sub-1.5 second 0-60, but
  you pay for it on every other axis except fuel (because the
  weight + aero savings partly cancel the V8's thirst).

That last point is the kind of insight a Pareto front gives you
that no single-objective optimizer would: **the cheapest fuel-
efficient car is not the smallest engine alone**, it's a small
engine + aggressive aero. **The lightest sports car is not the
lightest possible**, it's the point where weight cost stops paying
back in 0-60. The optimizer doesn't tell you what to buy — it
hands you the frontier of *every defensible compromise* and lets
you pick by your own priorities.

### Explore it interactively

Six hand-picked rows out of a hundred is a sample, not a search.
With the `serde` feature enabled, the same result becomes one JSON
file you can drop into the [heuropt-explorer](https://swaits.github.io/heuropt-explorer/)
webapp to browse interactively — parallel coordinates, scatter,
range filters, weighted ranking:

```rust,ignore
heuropt::explorer::ExplorerExport::from_result(&PickACar, &result)
    .with_algorithm_info(&optimizer)
    .with_problem_name("Pick a car")
    .to_file("results.json")?;
```

The full worked example (which produces this output verbatim) is at
`examples/pick_a_car.rs`:

```text
cargo run --release --example pick_a_car --features serde
```

See the [Explore your results](https://swaits.github.io/heuropt/cookbook/explorer.html)
cookbook recipe for the export schema and how to enrich your `Problem`
with display labels and units.

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
        todo!()
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
- **Expensive (50–500 evals):** Bayesian Optimization (Gaussian-process
  surrogate + Expected Improvement) or TPE (Parzen-density
  surrogate, cheaper per step, more robust without hyperparameter
  tuning).
- **Multi-fidelity (each eval has a tunable budget — epochs, sim
  steps, MC samples):** Hyperband. Implement the `PartialProblem`
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
 │   │    │     → Bayesian Optimization (GP + Expected Improvement; gold
 │   │    │                      standard *with* per-problem kernel
 │   │    │                      tuning. The default RBF kernel at
 │   │    │                      60 evals is honestly bad — give it
 │   │    │                      more evals or tune the kernel.)
 │   │    │     → TPE           (KDE-based; cheaper per-step,
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
     │    │   │     → CMA-ES       (full-cov adaptive Gaussian)
     │    │   │     → sNES         (cheaper diag-cov; high-dim)
     │    │   │     → Nelder-Mead  (low-dim, deterministic, simple)
     │    │   ├─ Multimodal landscape
     │    │   │     → IPOP-CMA-ES          (CMA-ES with restart;
     │    │   │                              fixes vanilla CMA-ES's
     │    │   │                              multimodal failure)
     │    │   │     → Differential Evolution (rarely beaten on cheap
     │    │   │                              multimodal continuous)
     │    │   │     → Simulated Annealing  (cheap & generic)
     │    │   ├─ Want parameter-free (no F, CR, w, σ to tune)
     │    │   │     → TLBO
     │    │   ├─ Want minimum self-adapting baseline
     │    │   │     → (1+1)-ES              (one-fifth rule,
     │    │   │                               smallest possible ES)
     │    │   ├─ Just want a strong default for cheap continuous
     │    │   │     → Differential Evolution
     │    │   └─ Just want a baseline
     │    │         → Random Search
     │    │
     │    ├─ Decision is Vec<bool> (binary)
     │    │   ├─ Independent bits, smooth fitness
     │    │   │     → UMDA            (per-bit marginal EDA)
     │    │   └─ Bit interactions matter
     │    │         → GA with BitFlipMutation +
     │    │           a bit-string crossover
     │    │
     │    ├─ Decision is Vec<usize> (permutation: TSP, JSS, …)
     │    │     → Ant Colony (TSP, with a distance matrix)
     │    │     → Simulated Annealing / Tabu Search (strong on
     │    │        sequencing — they win the harness TSP and JSS
     │    │        tables — you supply the neighbour move)
     │    │     → GA + permutation toolkit (ERX for TSP-shaped
     │    │        instances)
     │    │
     │    └─ Custom decision type (a struct, a tree, …)
     │          → Simulated Annealing or Hill Climber
     │            with your own Variation impl
     │
     ├─ 2 or 3 (multi-objective)
     │    │
     │    ├─ Strong default, fast, well-understood
     │    │     → NSGA-II
     │    │
     │    ├─ Real-valued, smooth front, want best convergence
     │    │     → MOPSO    (multi-objective PSO; on the benches
     │    │                 here it wins ZDT1 on both HV and
     │    │                 convergence by 100× over the
     │    │                 dominance-based methods)
     │    │
     │    ├─ Want better front quality than NSGA-II
     │    │     → IBEA     (indicator-based; consistently the best
     │    │                 of the dominance-based methods on these
     │    │                 benches — wins ZDT3 HV and DTLZ2 mean
     │    │                 dist by 24×)
     │    │     → SPEA2    (strength + density)
     │    │     → SMS-EMOA (hypervolume-contribution selection;
     │    │                 elegant in theory but underperforms
     │    │                 NSGA-II on these benches at our budgets —
     │    │                 only worth its higher per-step cost on
     │    │                 fronts where exact HV-contribution is
     │    │                 the right discriminator)
     │    │
     │    ├─ Want decomposition / weight-vector style
     │    │     → MOEA/D   (very fast per generation, scales well)
     │    │
     │    ├─ Disconnected front (separate arcs, e.g. ZDT3)
     │    │     → IBEA     (wins ZDT3 hypervolume on the harness;
     │    │                 MOEA/D and NSGA-II follow. Geometry-aware
     │    │                 methods trail when the front is in pieces)
     │    │
     │    ├─ Non-convex but *contiguous* front
     │    │     → AGE-MOEA (estimates front geometry adaptively)
     │    │     → KnEA     (favors knee points)
     │    │
     │    ├─ Want region-based diversity
     │    │     → PESA-II  (grid hyperboxes drive selection)
     │    │     → ε-MOEA   (ε-grid archive,
     │    │                  archive size auto-limits)
     │    │
     │    └─ Just one starting decision (no population budget)
     │          → PAES  (1+1 ES with a Pareto archive)
     │
     └─ 4+ (many-objective)
          │
          ├─ Linear / simplex-shaped front (e.g., DTLZ1)
          │     → GrEA       (grid coords drive ranking; on DTLZ1
          │                    here it beats NSGA-III by 3× and
          │                    AGE-MOEA by 2.5×)
          │     → MOEA/D     (decomposition shines on linear fronts;
          │                    second on DTLZ1, also among the
          │                    fastest per generation)
          │
          ├─ Curved / unknown front geometry
          │     → NSGA-III   (reference-point niching, canonical;
          │                    a strong default when the front
          │                    isn't simplex-shaped)
          │     → AGE-MOEA   (estimates L_p geometry per generation)
          │     → RVEA       (reference vectors with adaptive penalty)
          │
          ├─ Want indicator-based selection
          │     → IBEA       (additive ε-indicator; doesn't degrade
          │                   at high obj count)
          │     → HypE       (Monte Carlo HV estimation; scales
          │                   to arbitrary M)
```

### Quick reference

**Sample-efficient / expensive evaluation (50–500 evals):**

| Algorithm        | Objectives | Decision   | Strengths |
|---|---|---|---|
| **Bayesian Optimization** | 1 | `Vec<f64>` | GP surrogate + EI; gold standard *with* per-problem kernel tuning (default RBF at 60 evals is honestly bad) |
| **TPE**          | 1          | `Vec<f64>` | KDE surrogate; robust without hyperparameter tuning |
| **Hyperband**    | 1          | any        | multi-fidelity; needs `PartialProblem` |

**Single-objective continuous (`Vec<f64>`):**

| Algorithm                | Strengths |
|---|---|
| **Random Search**        | sanity baseline |
| **Hill Climber**         | simplest greedy local search |
| **(1+1)-ES**             | one-fifth-rule self-adapting baseline |
| **Simulated Annealing**  | escapes local optima |
| **GA**                   | classic SO GA with elitism |
| **PSO**                  | simple swarm baseline |
| **Differential Evolution** | strong default for cheap continuous |
| **TLBO**                 | parameter-free (no F, CR, w, σ) |
| **CMA-ES**               | smooth landscapes; full covariance |
| **IPOP-CMA-ES**          | CMA-ES + restart for multimodal |
| **sNES**                 | diagonal-cov NES; cheap per-step |
| **Nelder-Mead**          | classical simplex; deterministic |

**Single-objective other decision types:**

| Algorithm        | Decision        | Strengths |
|---|---|---|
| **UMDA**         | `Vec<bool>`     | independent-bit EDA |
| **Tabu Search**  | any             | discrete, you supply neighbors |
| **Ant Colony**   | `Vec<usize>`    | TSP / permutation |

**Multi-objective (2–3) and many-objective (4+):**

| Algorithm     | Objectives | Strengths |
|---|---|---|
| **PAES**      | 2–3        | 1+1 ES with Pareto archive |
| **NSGA-II**   | 2–3        | canonical Pareto-based EA |
| **SPEA2**     | 2–3        | strength + density |
| **MOPSO**     | 2–3        | multi-objective PSO; best convergence on smooth real-valued 2-obj fronts |
| **IBEA**      | 2+         | indicator-based; consistently best of the dominance-based methods |
| **SMS-EMOA**  | 2+         | exact HV-contribution selection; high per-step cost, modest gain |
| **HypE**      | 2+         | Monte Carlo HV estimation |
| **ε-MOEA**    | 2+         | ε-grid archive; auto-sized |
| **PESA-II**   | 2+         | grid-based region selection |
| **AGE-MOEA**  | 2+         | adaptive front-geometry estimation |
| **KnEA**      | 2+         | knee-point favored survival |
| **MOEA/D**    | 2+         | decomposition; fast per-gen |
| **NSGA-III**  | 4+         | reference-point niching; strong on curved fronts |
| **RVEA**      | 4+         | reference vectors with penalty |
| **GrEA**      | 4+         | grid coords drive selection; particularly strong on linear/simplex fronts |

## Current algorithms

The full list with one-line descriptions:

**Sample-efficient / multi-fidelity:**

- **Bayesian Optimization** — Gaussian-process surrogate + Expected Improvement.
- **TPE** — Bergstra et al. 2011 Tree-structured Parzen Estimator.
- **Hyperband** — Li et al. 2017 multi-fidelity (uses `PartialProblem`).

**Single-objective:**

- **Random Search** — sample-evaluate-keep baseline.
- **Hill Climber** — greedy single-step local search.
- **(1+1)-ES** — Rechenberg 1973 (1+1)-ES with one-fifth rule.
- **Simulated Annealing** — Kirkpatrick et al. 1983, generic over decision type.
- **Tabu Search** — Glover 1986, with a user-supplied neighbor generator.
- **GA** — generational GA with tournament selection + elitism.
- **PSO** — Eberhart & Kennedy 1995 PSO for `Vec<f64>`.
- **Differential Evolution** — Storn & Price DE/rand/1/bin for `Vec<f64>`.
- **TLBO** — Rao 2011 Teaching-Learning-Based Optimization (parameter-free).
- **CMA-ES** — Hansen & Ostermeier 2001 covariance-matrix adaptation.
- **IPOP-CMA-ES** — Auger & Hansen 2005 CMA-ES with restart, for multimodal.
- **sNES** — Wierstra et al. 2008/2014 diagonal-cov NES.
- **Nelder-Mead** — Nelder & Mead 1965 simplex direct search.
- **UMDA** — Mühlenbein 1997 univariate marginal-distribution EDA for `Vec<bool>`.
- **Ant Colony** — Dorigo Ant System for permutation problems.

**Multi-objective:**

- **PAES** — Knowles & Corne 1999 Pareto Archived Evolution Strategy.
- **NSGA-II** — Deb et al. 2002, the canonical Pareto-based EA.
- **SPEA2** — Zitzler, Laumanns & Thiele 2001 strength-Pareto EA.
- **MOEA/D** — Zhang & Li 2007 decomposition-based MOEA with Tchebycheff scalarization.
- **MOPSO** — Coello, Pulido & Lechuga 2004 multi-objective PSO.
- **IBEA** — Zitzler & Künzli 2004 indicator-based EA.
- **SMS-EMOA** — Beume, Naujoks & Emmerich 2007 hypervolume-selection EMOA.
- **HypE** — Bader & Zitzler 2011 Hypervolume Estimation Algorithm.
- **ε-MOEA** — Deb, Mohan & Mishra 2003 ε-dominance MOEA.
- **PESA-II** — Corne et al. 2001 Pareto Envelope Selection II.
- **AGE-MOEA** — Panichella 2019 Adaptive Geometry Estimation MOEA.
- **KnEA** — Zhang, Tian & Jin 2015 Knee point-driven EA.

**Many-objective (4+):**

- **NSGA-III** — Deb & Jain 2014 reference-point NSGA-III.
- **RVEA** — Cheng et al. 2016 Reference Vector-guided EA.
- **GrEA** — Yang et al. 2013 Grid-based EA.

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
  abstraction reuse. Random Search is the recommended file to read before
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local-test checklist,
conventional-commits requirement, and project-governance docs.

This project follows the [Builder's Code of Conduct](CODE_OF_CONDUCT.md):
stay professional, stay technical, focus on the work and its merit.

For security disclosures, see [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
