# Choosing an algorithm

The README has a compact decision tree. This chapter expands it with
the *reasoning* behind each branch.

## Step 0: How expensive is one evaluation?

This is the first fork because it changes everything that comes
after it.

| Eval cost                  | Budget you can afford     | Algorithm family            |
|----------------------------|---------------------------|-----------------------------|
| Microseconds (pure math)   | 10 000 – 1 000 000 evals | Population-based            |
| Milliseconds (sim, IO)     | 1 000 – 10 000 evals     | Population-based            |
| Seconds (small training)   | 100 – 1 000 evals        | Sample-efficient (BO, TPE)  |
| Minutes+ (full training)   | 50 – 500 evals            | Sample-efficient + multi-fidelity |

For the cheap-eval branch, you have the run of the catalog. For the
expensive branch, classical evolutionary methods waste your evaluation
budget — go to [Bayesian Optimization][BayesianOpt] or [TPE]. For the *very* expensive
branch where each eval has a tunable budget (epochs, MC samples, sim
steps), [Hyperband] over the [`PartialProblem`] trait is the move.

## Step 1: How many objectives?

The biggest fork.

- **One** — there's a single best answer. Pick from the
  single-objective branch.
- **Two or three** — a Pareto front. Pick from the multi-objective
  branch.
- **Four or more** — a many-objective Pareto front; classical
  multi-objective methods break down here because almost every pair
  of points is non-dominated. Pick from the many-objective branch.

> **Pareto front:** the set of decisions where you cannot improve any
> objective without sacrificing another. In a 2-objective minimize
> problem, plot every solution; the Pareto front is the lower-left
> envelope.

If you found yourself staring at a single composite score that's a
weighted sum of conflicting goals, you probably have a multi-objective
problem in disguise. A weighted sum bakes in your preferences before
you've seen the trade-off; running a multi-objective optimizer first
and picking off the front later is almost always a better workflow
(see [Pick one answer off a Pareto front](./cookbook/pick-one.md)).

## Step 2 — single-objective continuous

These all take `Vec<f64>` decisions.

### Smooth, low-to-moderate dimension

[CMA-ES][CmaEs] is the strong default. It adapts the search distribution's
covariance to the local landscape. On the comparison harness it
hits machine epsilon on Rosenbrock at 30 000 evaluations.

For very low-dimensional smooth problems (≤ 5 dim), [Nelder-Mead][NelderMead] is
deterministic and converges to f = 0 exactly on Rosenbrock.

### High dimension, smooth

[sNES][SeparableNes] uses a diagonal covariance — cheaper per step than
CMA-ES at the cost of being unable to model rotated landscapes. Worth
trying when CMA-ES's `O(d²)` per-step cost hurts.

### Multimodal landscapes

Multimodal = many local minima that aren't the global one. Rastrigin
and Ackley are classic traps.

[IPOP-CMA-ES][IpopCmaEs] is CMA-ES with an increasing-population restart strategy
specifically designed for this. On the harness it drops vanilla CMA-ES's
Rastrigin score from f = 2.35 to f = 0.13.

[Differential Evolution][DifferentialEvolution] is rarely beaten on cheap multimodal
continuous problems. On Rastrigin it ties with `(1+1)-ES` at f = 0.

[Simulated Annealing][SimulatedAnnealing] is a cheap, generic baseline that escapes local
optima via temperature decay.

### Want parameter-free

[TLBO][Tlbo] (Teaching-Learning-Based Optimization) has no `F`, `CR`, `w`,
or `σ` to tune. Often a respectable middle-of-the-pack performer.

### Smallest possible self-adapting baseline

[(1+1)-ES][OnePlusOneEs] — Rechenberg's 1973 `(1+1)`-ES with the one-fifth
success rule. On the harness it hits f = 0 on Rastrigin in 50 000
evaluations.

### Just want a baseline

[Random Search][RandomSearch]. Useful as a sanity check: if your fancy optimizer
can't beat random search, something is wrong (with the fancy
optimizer or with the problem).

## Step 2 — single-objective other types

| Decision type | Algorithm | Notes |
|---|---|---|
| `Vec<bool>` | [UMDA][Umda] | Per-bit marginal EDA. Independent-bit assumption. |
| `Vec<bool>` | [GA][GeneticAlgorithm] + [`BitFlipMutation`] | When bit interactions matter. |
| `Vec<usize>` (permutation) | [Ant Colony][AntColonyTsp] | TSP-style with a distance matrix. |
| `Vec<usize>` (permutation) | [GA][GeneticAlgorithm] + [`ShuffledPermutation`] + [`OrderCrossover`] + [`InversionMutation`] | Generic permutation GA; use [`EdgeRecombinationCrossover`] for TSP-shaped instances. |
| `Vec<usize>` (JSS multiset) | [Simulated Annealing][SimulatedAnnealing] / [Tabu Search][TabuSearch] with [`InsertionMutation`], or [GA][GeneticAlgorithm] + [`ShuffledMultisetPermutation`] + local POX | Operation-string encoding. On the FT06 harness the local-search pair edges out the GA — see [Optimize a permutation](./cookbook/permutation.md). |
| `Vec<usize>` (permutation) | [Simulated Annealing][SimulatedAnnealing] + [`InversionMutation`] | Strong on sequencing, not just a baseline — wins the harness's FT06 job-shop table and ties for the TSP optimum. |
| `Vec<usize>` or custom | [Tabu Search][TabuSearch] | You supply the neighbor function; consistently near the top on the TSP and JSS tables. |
| Custom struct | [Simulated Annealing][SimulatedAnnealing] / [Hill Climber][HillClimber] | With your own `Variation` impl. |

heuropt's permutation operator toolkit covers four crossovers
([`OrderCrossover`], [`PartiallyMappedCrossover`], [`CycleCrossover`],
[`EdgeRecombinationCrossover`]) and four mutations ([`SwapMutation`],
[`InversionMutation`], [`InsertionMutation`], [`ScrambleMutation`]),
plus two initializers for strict and multiset permutations. See
[Optimize a permutation](./cookbook/permutation.md) for the full
picker.

## Step 2 — multi-objective (2 or 3)

### Strong default

[NSGA-II][Nsga2] is the canonical Pareto-based EA. Fast, well-understood,
maintains diversity via crowding distance. On the harness it lands
on the Pareto front of every test problem.

NSGA-II is generic over the decision type — drop in
[`ShuffledPermutation`] + a permutation crossover and it solves
bi-objective TSP; drop in a binary initializer and [`BitFlipMutation`]
and it solves bi-objective knapsack. See
[Multi-objective combinatorial problems](./cookbook/multi-objective-combinatorial.md).

### Real-valued, smooth front, want best convergence

[MOPSO][Mopso] (multi-objective PSO with archive). On ZDT1 it wins
hypervolume outright and converges 100× tighter than the
dominance-based methods.

### Better front quality than NSGA-II

[IBEA][Ibea] (indicator-based) is consistently the best of the
dominance-based methods on the harness — wins ZDT3 hypervolume and
DTLZ2 mean distance by 24×. It uses an additive ε-indicator for
selection rather than dominance + crowding.

[SPEA2][Spea2] (strength + density) — solid alternative; explicit external
archive separate from the population.

[SMS-EMOA][SmsEmoa] uses exact hypervolume contribution for selection. Elegant
in theory; in practice on the harness budgets here it underperforms
NSGA-II. Worth the higher per-step cost only when exact HV
contribution is the right discriminator.

### Decomposition / weight-vector style

[MOEA/D][Moead] decomposes the multi-objective problem into many scalar
sub-problems (Tchebycheff or weighted sum) and solves them in
parallel. Very fast per generation; scales naturally to many
objectives.

### Disconnected or non-convex front

A *disconnected* front (separate arcs, like ZDT3) and a *non-convex but
contiguous* front are different problems — don't conflate them.

For a **disconnected** front, [IBEA][Ibea] is the clear pick: on the
harness it wins ZDT3 — the disconnected-front benchmark — outright on
hypervolume, with [MOEA/D][Moead] and [NSGA-II][Nsga2] close behind.
Counter-intuitively the geometry-aware methods below *trail* here:
estimating a single front geometry or chasing knee points doesn't help
when the front is in pieces (on ZDT3, AGE-MOEA and KnEA finish last).

For a **non-convex but contiguous** front:

[AGE-MOEA][AgeMoea] estimates the front geometry adaptively (the L_p
parameter `p` is fit from data each generation).

[KnEA][Knea] favors knee points — the regions of the front where small
gains in one objective cost large losses in another.

### Region-based diversity

[PESA-II][PesaII] uses grid hyperboxes to drive selection — divide the
objective space into a grid, pick from the least-crowded boxes.

[ε-MOEA][EpsilonMoea] uses an ε-grid archive that auto-limits its size.

### Just one starting decision (no population budget)

[PAES][Paes] — `(1+1)`-ES with a Pareto archive. Cheap, simple, useful
when your evaluations are expensive enough that you can't afford a
population.

## Step 2 — many-objective (4+)

### Linear / simplex-shaped front (e.g., DTLZ1)

[GrEA][Grea] — grid coords drive ranking. On DTLZ1 it beats NSGA-III by
3× and AGE-MOEA by 2.5×.

[MOEA/D][Moead] — decomposition shines on linear fronts; second on DTLZ1
and among the fastest per generation.

### Curved / unknown front geometry

[NSGA-III][Nsga3] — reference-point niching; canonical many-objective method;
strong default when the front isn't simplex-shaped.

[AGE-MOEA][AgeMoea] — estimates L_p geometry per generation.

[RVEA][Rvea] — reference vectors with adaptive penalty.

### Indicator-based selection

[IBEA][Ibea] — additive ε-indicator; doesn't degrade at high obj count.

[HypE][Hype] — Monte Carlo hypervolume estimation; scales to arbitrary
objective count where exact HV is too expensive.

## Step 3: Are there hard constraints?

heuropt models constraints as a single scalar `constraint_violation`
on each `Evaluation`. Three escalations when the feasibility region
is hard to find:

1. **Penalty-only.** Just set `constraint_violation > 0` for
   infeasible decisions. The default tournament/Pareto comparisons
   prefer feasibles automatically.
2. **Repair.** Implement [`Repair<D>`] (or use the provided
   [`ClampToBounds`] / [`ProjectToSimplex`]) to project infeasible
   decisions back into the feasible region. Pair with a `Variation`
   in a [`CompositeVariation`] for bounds-aware variants.
3. **Stochastic ranking.** Use [`stochastic_ranking_select`] instead
   of `tournament_select_single_objective`. It probabilistically
   explores near-feasibility instead of strict feasibility-first
   ordering, which helps when feasible regions are narrow.

See [Constrain your search with `Repair`](./cookbook/constraints.md)
for worked examples.

## Step 4: Should you parallelize?

Enable the `parallel` feature flag if your `evaluate` takes more
than ~50 µs. Population-based algorithms ([Random Search][RandomSearch], [NSGA-II][Nsga2],
[Differential Evolution][DifferentialEvolution], [SPEA2][Spea2], [IBEA][Ibea], [MOPSO][Mopso], …) batch-
evaluate via rayon when the feature is on. **Seeded runs stay
bit-identical** to serial mode.

```toml
heuropt = { version = "0.10", features = ["parallel"] }
```

If your evaluation is **IO-bound** (HTTP request, RPC, subprocess)
rather than CPU-bound, use the `async` feature instead — it gives
you `AsyncProblem` and a `run_async(&problem, concurrency).await`
method on every algorithm in the catalog. See the
[Async evaluation cookbook recipe](./cookbook/async.md).

## TL;DR table

| Situation | Pick |
|---|---|
| Smooth single-objective continuous | [CMA-ES][CmaEs] |
| Multimodal single-objective continuous | [IPOP-CMA-ES][IpopCmaEs] or [Differential Evolution][DifferentialEvolution] |
| Expensive single-objective | [Bayesian Optimization][BayesianOpt] or [TPE] |
| Multi-fidelity single-objective | [Hyperband] |
| 2- or 3-objective default | [NSGA-II][Nsga2] |
| 2-objective real-valued smooth front | [MOPSO][Mopso] |
| Disconnected / non-convex front | [IBEA][Ibea] |
| Many-objective default (curved front) | [NSGA-III][Nsga3] |
| Many-objective linear / simplex front | [GrEA][Grea] |
| Permutation problem (TSP with distance matrix) | [Ant Colony][AntColonyTsp] |
| Generic permutation problem | [GA][GeneticAlgorithm] + permutation toolkit |
| Bi-objective combinatorial (TSP / scheduling / knapsack) | [NSGA-II][Nsga2] + matching encoding operators |
| 3-objective combinatorial | [NSGA-III][Nsga3] + matching encoding operators |
| Binary problem | [UMDA][Umda] |
| Custom decision type | [Simulated Annealing][SimulatedAnnealing] + your `Variation` |
| Sanity baseline | [Random Search][RandomSearch] |

[CmaEs]: https://docs.rs/heuropt/latest/heuropt/algorithms/cma_es/struct.CmaEs.html
[IpopCmaEs]: https://docs.rs/heuropt/latest/heuropt/algorithms/ipop_cma_es/struct.IpopCmaEs.html
[SeparableNes]: https://docs.rs/heuropt/latest/heuropt/algorithms/snes/struct.SeparableNes.html
[NelderMead]: https://docs.rs/heuropt/latest/heuropt/algorithms/nelder_mead/struct.NelderMead.html
[DifferentialEvolution]: https://docs.rs/heuropt/latest/heuropt/algorithms/differential_evolution/struct.DifferentialEvolution.html
[SimulatedAnnealing]: https://docs.rs/heuropt/latest/heuropt/algorithms/simulated_annealing/struct.SimulatedAnnealing.html
[Tlbo]: https://docs.rs/heuropt/latest/heuropt/algorithms/tlbo/struct.Tlbo.html
[OnePlusOneEs]: https://docs.rs/heuropt/latest/heuropt/algorithms/one_plus_one_es/struct.OnePlusOneEs.html
[RandomSearch]: https://docs.rs/heuropt/latest/heuropt/algorithms/random_search/struct.RandomSearch.html
[HillClimber]: https://docs.rs/heuropt/latest/heuropt/algorithms/hill_climber/struct.HillClimber.html
[BayesianOpt]: https://docs.rs/heuropt/latest/heuropt/algorithms/bayesian_opt/struct.BayesianOpt.html
[TPE]: https://docs.rs/heuropt/latest/heuropt/algorithms/tpe/struct.Tpe.html
[Hyperband]: https://docs.rs/heuropt/latest/heuropt/algorithms/hyperband/struct.Hyperband.html
[`PartialProblem`]: https://docs.rs/heuropt/latest/heuropt/core/partial_problem/trait.PartialProblem.html
[Umda]: https://docs.rs/heuropt/latest/heuropt/algorithms/umda/struct.Umda.html
[GeneticAlgorithm]: https://docs.rs/heuropt/latest/heuropt/algorithms/genetic_algorithm/struct.GeneticAlgorithm.html
[`BitFlipMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.BitFlipMutation.html
[AntColonyTsp]: https://docs.rs/heuropt/latest/heuropt/algorithms/ant_colony_tsp/struct.AntColonyTsp.html
[`SwapMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.SwapMutation.html
[`InversionMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.InversionMutation.html
[`InsertionMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.InsertionMutation.html
[`ScrambleMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ScrambleMutation.html
[`OrderCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.OrderCrossover.html
[`PartiallyMappedCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.PartiallyMappedCrossover.html
[`CycleCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.CycleCrossover.html
[`EdgeRecombinationCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.EdgeRecombinationCrossover.html
[`ShuffledPermutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ShuffledPermutation.html
[`ShuffledMultisetPermutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ShuffledMultisetPermutation.html
[TabuSearch]: https://docs.rs/heuropt/latest/heuropt/algorithms/tabu_search/struct.TabuSearch.html
[Nsga2]: https://docs.rs/heuropt/latest/heuropt/algorithms/nsga2/struct.Nsga2.html
[Nsga3]: https://docs.rs/heuropt/latest/heuropt/algorithms/nsga3/struct.Nsga3.html
[Mopso]: https://docs.rs/heuropt/latest/heuropt/algorithms/mopso/struct.Mopso.html
[Ibea]: https://docs.rs/heuropt/latest/heuropt/algorithms/ibea/struct.Ibea.html
[Spea2]: https://docs.rs/heuropt/latest/heuropt/algorithms/spea2/struct.Spea2.html
[SmsEmoa]: https://docs.rs/heuropt/latest/heuropt/algorithms/sms_emoa/struct.SmsEmoa.html
[Moead]: https://docs.rs/heuropt/latest/heuropt/algorithms/moead/struct.Moead.html
[AgeMoea]: https://docs.rs/heuropt/latest/heuropt/algorithms/age_moea/struct.AgeMoea.html
[Knea]: https://docs.rs/heuropt/latest/heuropt/algorithms/knea/struct.Knea.html
[PesaII]: https://docs.rs/heuropt/latest/heuropt/algorithms/pesa2/struct.PesaII.html
[EpsilonMoea]: https://docs.rs/heuropt/latest/heuropt/algorithms/epsilon_moea/struct.EpsilonMoea.html
[Paes]: https://docs.rs/heuropt/latest/heuropt/algorithms/paes/struct.Paes.html
[Grea]: https://docs.rs/heuropt/latest/heuropt/algorithms/grea/struct.Grea.html
[Rvea]: https://docs.rs/heuropt/latest/heuropt/algorithms/rvea/struct.Rvea.html
[Hype]: https://docs.rs/heuropt/latest/heuropt/algorithms/hype/struct.Hype.html
[`Repair<D>`]: https://docs.rs/heuropt/latest/heuropt/traits/trait.Repair.html
[`ClampToBounds`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ClampToBounds.html
[`ProjectToSimplex`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ProjectToSimplex.html
[`stochastic_ranking_select`]: https://docs.rs/heuropt/latest/heuropt/selection/tournament/fn.stochastic_ranking_select.html
[`CompositeVariation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.CompositeVariation.html
