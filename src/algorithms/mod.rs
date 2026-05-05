//! Built-in reference optimizers.

pub mod differential_evolution;
pub mod nsga2;
pub mod paes;
pub(crate) mod parallel_eval;
pub mod random_search;

pub use differential_evolution::*;
pub use nsga2::*;
pub use paes::*;
pub use random_search::*;
