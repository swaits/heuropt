//! `heuropt` — a practical Rust toolkit for implementing heuristic
//! single-objective, multi-objective, and many-objective optimization
//! algorithms.
//!
//! The crate aims to make three things obvious:
//!
//! 1. Define an optimization problem by implementing [`Problem`](crate::core::Problem).
//! 2. Run a built-in optimizer such as [`Nsga2`](crate::algorithms::Nsga2) or
//!    [`RandomSearch`](crate::algorithms::RandomSearch).
//! 3. Implement a new optimizer by implementing
//!    [`Optimizer`](crate::traits::Optimizer).
//!
//! See `docs/heuropt_tech_design_spec.md` for the full design rationale.
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
pub mod metrics;
pub mod operators;
pub mod pareto;
pub mod prelude;
pub mod selection;
pub mod traits;

#[cfg(test)]
pub(crate) mod tests_support;
