//! `heuropt` — a practical Rust toolkit for heuristic single-,
//! multi-, and many-objective optimization.
//!
//! The crate aims to make three things obvious:
//!
//! 1. **Define a problem** by implementing [`Problem`](crate::core::Problem).
//! 2. **Run a built-in optimizer** — pick from 35 algorithms in
//!    [`algorithms`] covering single-objective continuous (CMA-ES,
//!    Differential Evolution, Nelder-Mead, …), multi-objective
//!    (NSGA-II, MOPSO, IBEA, MOEA/D, …), many-objective (NSGA-III,
//!    GrEA, RVEA, …), and sample-efficient regimes (Bayesian
//!    Optimization, TPE, Hyperband).
//! 3. **Or implement your own** by implementing
//!    [`Optimizer`](crate::traits::Optimizer). The trait is one
//!    method long.
//!
//! ## Where to read more
//!
//! - **User guide / cookbook / comparison vs pymoo & friends:**
//!   <https://swaits.github.io/heuropt/>.
//! - **Algorithm selection:** the README's decision tree, or the
//!   "Choosing an algorithm" book chapter.
//! - **Design rationale:** `docs/heuropt_tech_design_spec.md` in the
//!   repository.
//!
//! ## Optional features
//!
//! - `serde` — derives `Serialize` / `Deserialize` on the core data
//!   types ([`Candidate`](crate::core::Candidate),
//!   [`Population`](crate::core::Population),
//!   [`Evaluation`](crate::core::Evaluation), …).
//! - `parallel` — rayon-backed parallel population evaluation in
//!   every population-based algorithm. Seeded runs stay bit-
//!   identical to serial mode.
//!
//! # Quick example
//!
//! ```
//! use heuropt::prelude::*;
//!
//! struct Toy;
//!
//! impl Problem for Toy {
//!     type Decision = Vec<f64>;
//!
//!     fn objectives(&self) -> ObjectiveSpace {
//!         ObjectiveSpace::new(vec![
//!             Objective::minimize("f1"),
//!             Objective::minimize("f2"),
//!         ])
//!     }
//!
//!     fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
//!         Evaluation::new(vec![x[0] * x[0], (x[0] - 2.0).powi(2)])
//!     }
//! }
//!
//! let initializer = RealBounds::new(vec![(-5.0, 5.0)]);
//! let variation = GaussianMutation { sigma: 0.2 };
//! let config = Nsga2Config { population_size: 30, generations: 10, seed: 42 };
//! let mut opt = Nsga2::new(config, initializer, variation);
//! let result = opt.run(&Toy);
//! assert_eq!(result.population.len(), 30);
//! assert!(!result.pareto_front.is_empty());
//! ```

pub mod algorithms;
pub mod core;
pub(crate) mod internal;
pub mod metrics;
pub mod operators;
pub mod pareto;
pub mod prelude;
pub mod selection;
pub mod traits;

#[cfg(test)]
pub(crate) mod tests_support;
