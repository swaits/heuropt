# Constrain your search with `Repair`

heuropt models constraints with a single `constraint_violation` scalar
on each `Evaluation`. That works for soft penalties. When constraints
are *hard* and the search keeps generating infeasible decisions, the
better pattern is **repair**: project each candidate back into the
feasible region every time it leaves.

The [`Repair<D>`] trait is the abstraction. Two impls ship in the box;
you can write your own for arbitrary geometry.

## Built-in: `ClampToBounds`

For per-axis box constraints (`lo ≤ xᵢ ≤ hi`), pair `ClampToBounds`
with any `Variation` to get a bounds-aware variant for free.

```rust,no_run
use heuropt::prelude::*;

let bounds = vec![(-5.0, 5.0); 3];

// Without bounds, GaussianMutation can step outside the search box.
// ClampToBounds projects each variable back in.
let mut sigma = GaussianMutation { sigma: 0.5 };
let mut clamp = ClampToBounds::new(bounds.clone());

let mut rng = rng_from_seed(42);
let parent = vec![4.9, -4.9, 0.0];
let mut child = sigma.vary(std::slice::from_ref(&parent), &mut rng).pop().unwrap();
clamp.repair(&mut child);
// every entry of `child` is now within [-5, 5].
```

`ClampToBounds` is idempotent: applying it twice is the same as
applying it once.

For most real problems you'd just use [`BoundedGaussianMutation`]
which combines both in one operator.

## Built-in: `ProjectToSimplex`

For *budget* constraints — "the components must sum to a fixed
total and be non-negative" — `ProjectToSimplex` projects onto the
probability simplex (or any scaled simplex).

```rust,no_run
use heuropt::prelude::*;

let mut proj = ProjectToSimplex::new(1.0); // probability simplex
let mut x = vec![0.6, 0.5, -0.1, 0.3];     // sum 1.3, one negative
proj.repair(&mut x);
// x now sums to 1.0 and every entry is ≥ 0.
let s: f64 = x.iter().sum();
debug_assert!((s - 1.0).abs() < 1e-12);
debug_assert!(x.iter().all(|&v| v >= 0.0));
```

Use this for portfolio / resource-allocation problems where the
decision is a vector of weights that must sum to a budget.

## Custom repair

Anything that takes a `&mut Vec<f64>` (or any `&mut D` for your
custom decision type) and returns a feasible version is a valid
`Repair`. Implement the trait directly:

```rust,no_run
use heuropt::prelude::*;

/// Force the largest variable to be at least `min_largest`.
struct AtLeastOneActive { min_largest: f64 }

impl Repair<Vec<f64>> for AtLeastOneActive {
    fn repair(&mut self, x: &mut Vec<f64>) {
        let max_idx = x.iter()
            .enumerate()
            .fold(0, |best, (i, &v)| {
                if v > x[best] { i } else { best }
            });
        if x[max_idx] < self.min_largest {
            x[max_idx] = self.min_largest;
        }
    }
}
```

## Stochastic-ranking selection

When the feasible region is *narrow* — most of the search space is
infeasible — the strict "feasibles always beat infeasibles" rule
traps the search outside it. Runarsson & Yao's stochastic ranking
breaks the trap by, on each pairwise comparison, using a probabilistic
"compare by objective" instead of "compare by feasibility" with a
small probability `pf`:

```rust,ignore
use heuropt::selection::tournament::stochastic_ranking_select;

let picks = stochastic_ranking_select(
    &population,
    &objectives,
    0.45,             // pf — Runarsson & Yao's canonical value
    count,
    &mut rng,
);
```

This is a drop-in replacement for `tournament_select_single_objective`
in your custom optimizer or in a forked algorithm.

## When to use which

| Situation | Use |
|---|---|
| Box constraints | [`BoundedGaussianMutation`] (built-in mutation) |
| Manual repair after any mutation | [`ClampToBounds`] |
| Budget / probability-simplex constraints | [`ProjectToSimplex`] |
| Custom geometric constraints | Your own `Repair` impl |
| Narrow feasible region, frequent infeasibility | [`stochastic_ranking_select`] |
| Soft penalty, mostly feasible search | Set `constraint_violation` and let default tournament handle it |

[`Repair<D>`]: https://docs.rs/heuropt/latest/heuropt/traits/trait.Repair.html
[`ClampToBounds`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ClampToBounds.html
[`ProjectToSimplex`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ProjectToSimplex.html
[`BoundedGaussianMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.BoundedGaussianMutation.html
[`stochastic_ranking_select`]: https://docs.rs/heuropt/latest/heuropt/selection/tournament/fn.stochastic_ranking_select.html
