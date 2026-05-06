# Write your own algorithm

Implement [`Optimizer<P>`] and you're done. There are no other traits
to think about, no internal hooks to register. The example walks
through a tiny hill-climber that reads almost identically to the
canonical pseudocode.

## The trait

```rust,ignore
pub trait Optimizer<P>
where
    P: Problem,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision>;
}
```

That's it. You own your config, your RNG, your main loop, and your
`OptimizationResult` construction.

## A minimal hill-climber

```rust,no_run
use heuropt::prelude::*;

pub struct MyHillClimber<I, V> {
    pub iterations: usize,
    pub seed: u64,
    pub initializer: I,
    pub variation: V,
}

impl<P, I, V> Optimizer<P> for MyHillClimber<I, V>
where
    P: Problem,
    P::Decision: Clone,
    I: Initializer<P::Decision>,
    V: Variation<P::Decision>,
{
    fn run(&mut self, problem: &P) -> OptimizationResult<P::Decision> {
        let mut rng = rng_from_seed(self.seed);
        let objectives = problem.objectives();
        assert!(objectives.is_single_objective(), "MyHillClimber is single-objective only");

        // Start with one initial decision.
        let init_decisions = self.initializer.initialize(1, &mut rng);
        let init = init_decisions.into_iter().next().unwrap();
        let mut current = Candidate::new(init.clone(), problem.evaluate(&init));
        let mut evaluations: usize = 1;

        for _ in 0..self.iterations {
            let children = self.variation.vary(std::slice::from_ref(&current.decision), &mut rng);
            for child_decision in children {
                let child_eval = problem.evaluate(&child_decision);
                evaluations += 1;
                let child = Candidate::new(child_decision, child_eval);
                if better(&child.evaluation, &current.evaluation, &objectives) {
                    current = child;
                }
            }
        }

        let pareto_front = vec![current.clone()];
        let best = Some(current.clone());
        OptimizationResult::new(
            Population::new(vec![current]),
            pareto_front,
            best,
            evaluations,
            self.iterations,
        )
    }
}

fn better(a: &Evaluation, b: &Evaluation, objectives: &ObjectiveSpace) -> bool {
    let am = objectives.as_minimization(&a.objectives);
    let bm = objectives.as_minimization(&b.objectives);
    am[0] < bm[0]
}
```

## Things to notice

- **`Rng` is one concrete type.** No generics — call
  [`rng_from_seed`] and pass `&mut rng` everywhere it's needed.
- **`Initializer<D>`** sources the starting point(s).
- **`Variation<D>`** generates children from parents. For the
  hill-climber it's called with one parent.
- **`OptimizationResult`** carries the final population, the Pareto
  front (just the best for single-objective), the best candidate,
  the total evaluations, and the iteration count.
- **`as_minimization`** flips maximize-axis values so your
  comparison logic only ever needs to deal with "lower is better."

## Adding parallel evaluation

If your algorithm batch-evaluates candidates per generation, use the
crate's internal helper. From inside heuropt source you can call
`evaluate_batch(problem, decisions)`; from outside you'd use rayon
directly behind a feature flag, the same way the built-in algorithms
do.

```rust,ignore
#[cfg(feature = "parallel")]
fn batch_eval<P>(problem: &P, decisions: Vec<P::Decision>) -> Vec<Candidate<P::Decision>>
where P: Problem + Sync, P::Decision: Send,
{
    use rayon::prelude::*;
    decisions.into_par_iter()
        .map(|d| Candidate::new(d.clone(), problem.evaluate(&d)))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn batch_eval<P>(problem: &P, decisions: Vec<P::Decision>) -> Vec<Candidate<P::Decision>>
where P: Problem,
{
    decisions.into_iter()
        .map(|d| Candidate::new(d.clone(), problem.evaluate(&d)))
        .collect()
}
```

To stay bit-identical between serial and parallel modes, keep the
RNG and selection on the main thread; only the *evaluations* run in
parallel.

## What's *not* in the trait

- **No iteration / step API.** The optimizer owns its loop.
- **No callbacks.** A future minor release may add an observer hook;
  for now you'd run the algorithm to completion and process the
  result.
- **No error type.** Invalid configuration panics with a clear
  message; this matches the style of the built-in algorithms.
- **No async on the trait.** `Optimizer<P>` is synchronous. For
  async evaluation, implement [`AsyncProblem`](https://docs.rs/heuropt/latest/heuropt/core/async_problem/trait.AsyncProblem.html)
  on your problem and use the `run_async(&problem, concurrency)`
  method that comes with the `async` feature. See the
  [Async evaluation cookbook recipe](./async.md).

The smallness is the point: you should be able to read a built-in
algorithm and write your own in an afternoon. See
`examples/custom_optimizer.rs` for a slightly more polished version
of the hill-climber above.

[`Optimizer<P>`]: https://docs.rs/heuropt/latest/heuropt/traits/trait.Optimizer.html
[`rng_from_seed`]: https://docs.rs/heuropt/latest/heuropt/core/rng/fn.rng_from_seed.html
