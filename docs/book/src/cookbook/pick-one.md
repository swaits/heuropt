# Pick one answer off a Pareto front

A multi-objective optimizer hands you a *front* — a Pareto-optimal
trade-off curve — not a single answer. Eventually you have to pick
*one* point off it. There are several principled ways to do that;
this recipe covers the most common: the **a-posteriori weighted
decision rule**.

The pattern: optimize *without* baking your preferences into the
search, then apply your preferences as a scoring function over the
front.

This is exactly the pattern from `examples/jiggly_tuning.rs` (the
USB-jiggler firmware tuning example).

## The shape

```rust,no_run
use heuropt::prelude::*;

# struct Cost;
# impl Problem for Cost {
#     type Decision = Vec<f64>;
#     fn objectives(&self) -> ObjectiveSpace {
#         ObjectiveSpace::new(vec![Objective::minimize("a"), Objective::minimize("b"), Objective::minimize("c")])
#     }
#     fn evaluate(&self, _x: &Vec<f64>) -> Evaluation { Evaluation::new(vec![0.0,0.0,0.0]) }
# }

let problem = Cost;
let mut opt = Nsga2::new(
    Nsga2Config { population_size: 100, generations: 200, seed: 42 },
    RealBounds::new(vec![(-1.0, 1.0); 4]),
    CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(vec![(-1.0, 1.0); 4], 15.0, 0.5),
        mutation:  PolynomialMutation::new(vec![(-1.0, 1.0); 4], 20.0, 1.0),
    },
);
let result = opt.run(&problem);

// 1. Get the Pareto front.
let front = &result.pareto_front;

// 2. Define your preferences as a scoring function over (oriented)
//    objective values. Lower score = preferred.
let space = problem.objectives();
let weights = [1.0, 2.0, 0.5];

let scored: Vec<(f64, &Candidate<Vec<f64>>)> = front.iter()
    .map(|c| {
        let oriented = space.as_minimization(&c.evaluation.objectives);
        let score: f64 = oriented.iter().zip(&weights)
            .map(|(v, w)| v * w)
            .sum();
        (score, c)
    })
    .collect();

// 3. Pick the lowest-scoring point.
let best = scored.iter()
    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    .unwrap();

println!("picked: {:?} with weighted score {:.3}",
    best.1.evaluation.objectives, best.0);
```

`as_minimization` returns the objective vector with maximized axes
flipped to negative — so a single set of *positive* weights does
the right thing whether each axis is min or max.

## Why a-posteriori vs a-priori weighting

If you know your weights up front, you could just optimize the
weighted sum directly with a single-objective algorithm. Why bother
with the multi-objective dance?

Two reasons:

1. **Weighted sum can't reach concave parts of the Pareto front.**
   Any single-objective optimization with a linear scalarization
   converges to a point at the boundary of the convex hull. Concave
   front segments are unreachable. The multi-objective optimizer
   finds them.
2. **Weights are usually wrong on the first try.** Optimizing the
   front first lets you see what's actually possible before deciding
   how much each axis is worth. Run once, look at the trade-offs,
   adjust weights.

## Penalty terms beyond linear weights

The jiggly example also adds a *hinge penalty* — a term that's zero
inside an acceptable region and grows quadratically once you exceed
some hard cap. Useful when one axis is "soft up to X, hard cap at Y":

```rust,no_run
fn hinge(x: f64, soft_cap: f64, hard_cap: f64) -> f64 {
    if x <= soft_cap { 0.0 }
    else if x >= hard_cap { f64::INFINITY }
    else {
        let t = (x - soft_cap) / (hard_cap - soft_cap);
        100.0 * t * t
    }
}
```

Compose linear weights + hinge penalties and you have a flexible
scoring function over the front without re-running the optimizer.

## Other strategies

- **Knee point.** Pick the point where small gains in one axis cost
  large losses in another — the "elbow" of the trade-off curve.
  [`Knea`] explicitly biases the search toward knees during the run.
- **Reference-direction.** Pick the point closest to a desired
  trade-off direction (a unit vector in objective space).
  [`Moead`] / [`Nsga3`] use this internally during search; you can
  apply it post-hoc the same way.
- **Random / interactive selection.** Show the front to a user
  (perhaps via a plotting library), let them pick.

The right pick depends on the problem; the front itself doesn't
prescribe one.

[`Knea`]: https://docs.rs/heuropt/latest/heuropt/algorithms/knea/struct.Knea.html
[`Moead`]: https://docs.rs/heuropt/latest/heuropt/algorithms/moead/struct.Moead.html
[`Nsga3`]: https://docs.rs/heuropt/latest/heuropt/algorithms/nsga3/struct.Nsga3.html
