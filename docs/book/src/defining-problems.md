# Defining a problem

Everything in heuropt starts with the [`Problem`] trait. This chapter
walks through every shape it can take.

## The trait

```rust,ignore
pub trait Problem {
    type Decision: Clone;
    fn objectives(&self) -> ObjectiveSpace;
    fn evaluate(&self, decision: &Self::Decision) -> Evaluation;
}
```

Three things you decide:

1. **`Decision`** — the type of the thing you're optimizing.
   `Vec<f64>` is by far the most common; `Vec<bool>` for binary
   search, `Vec<usize>` for permutations, your own struct for
   anything else.
2. **`objectives`** — how many objectives you have, what they're
   called, and whether each is minimized or maximized. Returned as
   an [`ObjectiveSpace`].
3. **`evaluate`** — given one decision, score it. Returns an
   [`Evaluation`] with a vector of objective values (and optionally
   a constraint-violation scalar).

`evaluate` takes `&self`, so caches and lookup tables are easy. It
is called many thousands of times during a typical run, so keep it
fast.

## Single-objective continuous

The Rosenbrock banana — minimize a smooth non-convex valley.

```rust,no_run
use heuropt::prelude::*;

struct Rosenbrock;

impl Problem for Rosenbrock {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let f: f64 = x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0].powi(2)).powi(2) + (1.0 - w[0]).powi(2))
            .sum();
        Evaluation::new(vec![f])
    }
}
```

## Multi-objective

ZDT1 — two objectives that conflict. The Pareto front is the set of
non-dominated trade-offs.

```rust,no_run
use heuropt::prelude::*;

struct Zdt1 { dim: usize }

impl Problem for Zdt1 {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("f1"),
            Objective::minimize("f2"),
        ])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let n = x.len() as f64;
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1.0);
        let h = 1.0 - (f1 / g).sqrt();
        let f2 = g * h;
        Evaluation::new(vec![f1, f2])
    }
}
```

For multi-objective problems, pick a Pareto-aware optimizer:
[`Nsga2`] is the canonical default; [`Mopso`] often wins on
smooth-front 2-objective problems; [`Ibea`] often wins on
disconnected fronts. See [choosing-an-algorithm](./choosing-an-algorithm.md).

## Maximizing instead of minimizing

heuropt's internals normalize everything to minimization, but you
declare your objective with the orientation that's natural for your
problem. A scoring problem might want to maximize:

```rust,no_run
use heuropt::prelude::*;
let space = ObjectiveSpace::new(vec![
    Objective::minimize("cost"),
    Objective::maximize("accuracy"),
]);
```

`Objective::maximize` is a convenience for `Direction::Maximize`. Mix
freely; the Pareto-comparison machinery handles the orientation.

## Constraints

heuropt models constraints as a single non-negative scalar
**`constraint_violation`** on each `Evaluation`. The convention:

- `0.0` (or negative) means **feasible**.
- Any positive value means **infeasible**, and bigger numbers are
  worse violations.

Pareto-comparison and tournament-selection helpers prefer feasible
candidates and break ties on the violation magnitude — so the rule
"feasibility comes first" is enforced automatically.

```rust,no_run
use heuropt::prelude::*;

struct Constrained;
impl Problem for Constrained {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let f: f64 = x.iter().map(|v| v * v).sum();

        // Constraint: x[0] + x[1] >= 1. Violation = how much we miss it by.
        let g1 = (1.0 - (x[0] + x[1])).max(0.0);
        let total_violation: f64 = g1; // sum of max(0, gᵢ) for each constraint

        Evaluation::constrained(vec![f], total_violation)
    }
}
```

If your constraints are very tight and the search keeps hitting them,
see [Constrain your search with `Repair`](./cookbook/constraints.md).

## Decision types beyond `Vec<f64>`

### Binary (`Vec<bool>`)

```rust,no_run
use heuropt::prelude::*;

struct OneMax { bits: usize }
impl Problem for OneMax {
    type Decision = Vec<bool>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::maximize("ones")])
    }
    fn evaluate(&self, x: &Vec<bool>) -> Evaluation {
        Evaluation::new(vec![x.iter().filter(|b| **b).count() as f64])
    }
}
```

For `Vec<bool>` problems, [`Umda`] is a parameter-free EDA;
[`GeneticAlgorithm`] with [`BitFlipMutation`] is the GA route.

### Permutations (`Vec<usize>`)

```rust,no_run
use heuropt::prelude::*;

struct Tsp { distances: Vec<Vec<f64>> }
impl Problem for Tsp {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("length")])
    }
    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        let mut len = 0.0;
        for w in tour.windows(2) {
            len += self.distances[w[0]][w[1]];
        }
        len += self.distances[*tour.last().unwrap()][tour[0]];
        Evaluation::new(vec![len])
    }
}
```

For permutations, [`AntColonyTsp`] specializes on TSP-style problems;
[`TabuSearch`] takes a user-supplied neighbor function for arbitrary
discrete neighborhoods; [`SimulatedAnnealing`] with [`SwapMutation`]
is the simplest baseline.

### Custom decision types

Any `Clone` type works. If you have a struct, just implement `Clone`
and you can use it. You'll need to write your own `Variation` impl
to mutate it; see [Write your own algorithm](./cookbook/custom-optimizer.md).

## What `Evaluation` carries

```rust,ignore
pub struct Evaluation {
    pub objectives: Vec<f64>,        // one entry per objective
    pub constraint_violation: f64,    // 0.0 = feasible
}
```

That's it. Construct with [`Evaluation::new`] for unconstrained
problems or [`Evaluation::constrained`] when you have a violation.

## Summary

- Implement [`Problem`] with your decision type.
- Declare objectives via [`ObjectiveSpace`] (mix minimize/maximize
  freely).
- Return an [`Evaluation`] from `evaluate`.
- For constraints, set `constraint_violation > 0` for infeasible
  decisions; heuropt's selection helpers prefer feasibles
  automatically.

Next: [Choosing an algorithm](./choosing-an-algorithm.md) walks
through the decision tree.

[`Problem`]: https://docs.rs/heuropt/latest/heuropt/core/problem/trait.Problem.html
[`ObjectiveSpace`]: https://docs.rs/heuropt/latest/heuropt/core/objective/struct.ObjectiveSpace.html
[`Evaluation`]: https://docs.rs/heuropt/latest/heuropt/core/evaluation/struct.Evaluation.html
[`Evaluation::new`]: https://docs.rs/heuropt/latest/heuropt/core/evaluation/struct.Evaluation.html#method.new
[`Evaluation::constrained`]: https://docs.rs/heuropt/latest/heuropt/core/evaluation/struct.Evaluation.html#method.constrained
[`Nsga2`]: https://docs.rs/heuropt/latest/heuropt/algorithms/nsga2/struct.Nsga2.html
[`Mopso`]: https://docs.rs/heuropt/latest/heuropt/algorithms/mopso/struct.Mopso.html
[`Ibea`]: https://docs.rs/heuropt/latest/heuropt/algorithms/ibea/struct.Ibea.html
[`Umda`]: https://docs.rs/heuropt/latest/heuropt/algorithms/umda/struct.Umda.html
[`GeneticAlgorithm`]: https://docs.rs/heuropt/latest/heuropt/algorithms/genetic_algorithm/struct.GeneticAlgorithm.html
[`BitFlipMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.BitFlipMutation.html
[`AntColonyTsp`]: https://docs.rs/heuropt/latest/heuropt/algorithms/ant_colony_tsp/struct.AntColonyTsp.html
[`TabuSearch`]: https://docs.rs/heuropt/latest/heuropt/algorithms/tabu_search/struct.TabuSearch.html
[`SimulatedAnnealing`]: https://docs.rs/heuropt/latest/heuropt/algorithms/simulated_annealing/struct.SimulatedAnnealing.html
[`SwapMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.SwapMutation.html
