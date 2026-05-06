# heuropt-plot

[![Crates.io](https://img.shields.io/crates/v/heuropt-plot.svg)](https://crates.io/crates/heuropt-plot)
[![Documentation](https://docs.rs/heuropt-plot/badge.svg)](https://docs.rs/heuropt-plot)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

Lightweight SVG plotting helpers for [`heuropt`](https://crates.io/crates/heuropt)
results.

Hand-rolled SVG output (no `plotters`, no `tiny-skia`, no
heavyweight dependency) so adding `heuropt-plot` to your project
costs ~20 KB of compiled code.

## What's in the box

- `pareto_front_svg` — render a 2-objective Pareto front as an SVG
  scatter plot with axes and labels.
- `convergence_svg` — render a "best fitness so far" trace as an
  SVG line plot.

Output is a `String` of valid SVG. Write it to a file, embed it in
HTML, or pipe it to a browser.

## Example

```rust
use heuropt::prelude::*;
use heuropt_plot::pareto_front_svg;

let space = ObjectiveSpace::new(vec![
    Objective::minimize("f1"),
    Objective::minimize("f2"),
]);
let front = vec![
    Candidate::new((), Evaluation::new(vec![0.0, 1.0])),
    Candidate::new((), Evaluation::new(vec![0.5, 0.5])),
    Candidate::new((), Evaluation::new(vec![1.0, 0.0])),
];

let svg = pareto_front_svg(&front, &space, 600, 400, "Sample front");
std::fs::write("front.svg", svg).unwrap();
```

## License

MIT — see [LICENSE](../LICENSE) at the repo root.
