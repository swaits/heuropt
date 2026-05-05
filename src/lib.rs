//! `heuropt` — a practical Rust toolkit for heuristic single-, multi-, and
//! many-objective optimization. See `docs/heuropt_tech_design_spec.md` for the
//! full design.

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
