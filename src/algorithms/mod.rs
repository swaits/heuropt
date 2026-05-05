//! Built-in reference optimizers.

pub mod differential_evolution;
pub mod moead;
pub mod nsga2;
pub mod nsga3;
pub mod paes;
pub(crate) mod parallel_eval;
pub mod random_search;
pub mod spea2;

pub use differential_evolution::*;
pub use moead::*;
pub use nsga2::*;
pub use nsga3::*;
pub use paes::*;
pub use random_search::*;
pub use spea2::*;
