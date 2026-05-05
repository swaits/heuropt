//! Built-in reference optimizers.

pub mod nsga2;
pub mod paes;
pub mod random_search;

pub use nsga2::*;
pub use paes::*;
pub use random_search::*;
