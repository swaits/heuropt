# Comparison with other libraries

heuropt is one of many heuristic-optimization libraries. This chapter
is an honest, opinionated comparison to help you choose.

The columns:

- **Lang** — primary implementation language.
- **Algorithms** — rough catalog count.
- **Multi-obj** — built-in support for Pareto-based multi-objective
  optimization.
- **Surrogates** — built-in Bayesian / TPE / multi-fidelity.
- **Determinism** — seeded reproducibility as a first-class property.
- **Async / async-eval** — first-class async runtime support.

| Library | Lang | Algorithms | Multi-obj | Surrogates | Determinism | Async |
|---|---|---|---|---|---|---|
| **heuropt 0.10** | Rust | 33 | ✅ NSGA-II/III, SPEA2, IBEA, MOEA/D, MOPSO, SMS-EMOA, HypE, AGE-MOEA, GrEA, KnEA, RVEA, PESA-II, ε-MOEA, PAES | ✅ BO, TPE, Hyperband | ✅ bit-identical seeded | ✅ `AsyncProblem` + `run_async` on every algorithm |
| pymoo | Python | ~25 | ✅ extensive | partial (BO via plug-ins) | ✅ | ❌ |
| DEAP | Python | flexible toolbox | ✅ | ❌ | ✅ | ❌ |
| hyperopt | Python | TPE-focused | ❌ | ✅ TPE | partial | partial |
| optuna | Python | TPE / CMA-ES / NSGA-II | ✅ | ✅ TPE, BoTorch via plug-in | ✅ | partial (study-level, not eval-level) |
| MOEA Framework | Java | ~40 | ✅ very extensive | ❌ | ✅ | ❌ |
| metaheuristics-rs | Rust | ~10 | partial | ❌ | ✅ | ❌ |
| argmin | Rust | line-search / quasi-Newton | ❌ | ❌ | ✅ | ❌ |

## When to pick heuropt

- You're working in **Rust** and want a single, dependency-light crate
  for evolutionary / metaheuristic optimization.
- You need **multi-objective or many-objective** algorithms (12+
  Pareto-aware methods in the catalog) AND you don't want to glue
  Python into your Rust pipeline.
- You want **bit-identical determinism**: same seed produces same
  output, on every machine, across releases unless explicitly noted
  otherwise.
- You want a **small, readable codebase** — every algorithm is
  written for clarity, no trait-object plumbing, no GATs in user-
  facing APIs. Reading Random Search should be enough to write a
  new optimizer.
- You have **IO-bound evaluations** — calling an HTTP service, an
  RPC, or a subprocess — and want first-class `async fn evaluate`
  support. heuropt is the only mainstream optimization library that
  ships this (see [Async evaluation](./cookbook/async.md)).

## When *not* to pick heuropt

- You need **gradient-based** optimization. Use `argmin` (Rust) or
  `scipy.optimize` (Python) — heuropt is gradient-free by design.
- You need **GPU-accelerated** evaluations. heuropt's `evaluate`
  function runs on CPU; use Python (jax/torch) or roll your own
  GPU pipeline.
- You need **distributed multi-machine** evaluation. heuropt
  parallelizes within one process via rayon. Distribution is up to
  you (split the seeds across machines, aggregate).
- You're comfortable in Python and pymoo / optuna already cover
  your problem. heuropt's value-add over pymoo is mostly that it's
  Rust — if that doesn't matter to you, the Python ecosystem has more
  battle-tested integrations.

## Algorithm coverage at a glance

heuropt covers the same major Pareto MOEAs as pymoo and MOEA Framework:
NSGA-II/III, SPEA2, IBEA, MOEA/D, MOPSO, SMS-EMOA, HypE, AGE-MOEA,
GrEA, KnEA, RVEA, PESA-II, ε-MOEA, PAES.

The expensive-evaluation regime: Bayesian Optimization + TPE + Hyperband. This
is comparable to optuna's coverage but in pure Rust.

The single-objective continuous catalog (CMA-ES, IPOP-CMA-ES, sNES,
DE, PSO, GA, TLBO, (1+1)-ES, Nelder-Mead, Random Search, Hill Climber,
Simulated Annealing) covers the canonical baselines and several modern
variants.

What heuropt does **not** ship that some libraries do:

- **Re-themed metaphor metaheuristics** (Whale Optimization, Grey
  Wolf, Bat, Firefly, Harris Hawks, etc.). These are cut from the
  catalog deliberately — they are mostly DE/PSO with new names. If
  you specifically need one, please open an issue with citations.
- **Non-evolutionary global optimizers** like dual annealing or
  basin-hopping (use `scipy.optimize` for those).
- **A web UI / dashboard** like optuna's. heuropt is library-only.

## Speed

heuropt's hot paths (Pareto utilities, hypervolume, key inner loops)
are heavily optimized — see the perf entry in the v0.4.0 CHANGELOG.
On the comparison harness in `examples/compare.rs` (10-seed mean,
30 000 evaluations on DTLZ2), the total wall-clock time across 12
algorithms is ~5 seconds. Per-algorithm timings are in
[`examples/compare-results.md`](https://github.com/swaits/heuropt/blob/main/examples/compare-results.md).

For comparison-shopping speed against Python libraries, the gap is
typically 10×–100× in heuropt's favor for compute-bound
`evaluate` functions, because Rust skips the Python-loop overhead. If
your `evaluate` calls into NumPy/PyTorch and those are the bottleneck,
the gap shrinks substantially.

## Honest weakness: ecosystem

The biggest thing pymoo / optuna / DEAP have that heuropt doesn't:
**community + plug-ins + tutorials**. They've been around longer and
have rich third-party integrations (visualization, MLflow,
Hyperband+BO hybrids, distributed runners). heuropt is younger; the
core is solid but the ecosystem is small.

If you adopt heuropt and miss a thing, the project is small enough
that contributions land fast. See [CONTRIBUTING.md](https://github.com/swaits/heuropt/blob/main/CONTRIBUTING.md).
