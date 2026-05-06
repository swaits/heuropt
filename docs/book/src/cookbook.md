# Cookbook

Short, focused recipes for the patterns that come up in practice.
Each recipe is self-contained and small enough to copy into your own
project.

## Recipes

- [Parallelize evaluation with rayon](./cookbook/parallel.md) — when
  your `evaluate` is non-trivial CPU work, the `parallel` feature
  pays for itself almost immediately.
- [Async evaluation](./cookbook/async.md) — when your `evaluate` is
  IO-bound (HTTP / RPC / subprocess), the `async` feature lets the
  optimizer await many evaluations concurrently. The differentiating
  feature vs other optimization libraries.
- [Tune a model with expensive evaluations](./cookbook/expensive-evaluations.md)
  — Bayesian Optimization, TPE, and Hyperband for the 50–500-eval
  regime.
- [Compare two algorithms on your problem](./cookbook/compare.md) —
  multi-seed harness pattern straight from `examples/compare.rs`.
- [Optimize a permutation (TSP-style)](./cookbook/permutation.md) —
  Ant Colony with a distance matrix.
- [Constrain your search with `Repair`](./cookbook/constraints.md) —
  bounds, simplex projection, custom repair.
- [Pick one answer off a Pareto front](./cookbook/pick-one.md) — the
  a-posteriori weighted-decision pattern from the `jiggly_tuning`
  example.
- [Explore your results in a webapp](./cookbook/explorer.md) — export
  an `OptimizationResult` to JSON and browse it interactively at
  [heuropt-explorer](https://swaits.github.io/heuropt-explorer/) —
  parallel coordinates, scatter, range filters, weighted ranking.
- [Write your own algorithm](./cookbook/custom-optimizer.md) —
  implement `Optimizer<P>` from scratch, à la the
  `examples/custom_optimizer.rs` walkthrough.
