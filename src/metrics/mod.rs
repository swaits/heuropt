//! Quality metrics for Pareto fronts.

pub mod hypervolume;
pub mod igd;
pub mod r2;
pub mod spacing;

pub use hypervolume::*;
pub use igd::{igd, igd_plus};
pub use r2::r2;
pub use spacing::*;
