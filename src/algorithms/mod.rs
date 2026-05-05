//! Built-in reference optimizers.

pub mod differential_evolution;
pub mod genetic_algorithm;
pub mod hill_climber;
pub mod moead;
pub mod nsga2;
pub mod nsga3;
pub mod paes;
pub(crate) mod parallel_eval;
pub mod random_search;
pub mod simulated_annealing;
pub mod spea2;

pub use differential_evolution::*;
pub use genetic_algorithm::*;
pub use hill_climber::*;
pub use moead::*;
pub use nsga2::*;
pub use nsga3::*;
pub use paes::*;
pub use random_search::*;
pub use simulated_annealing::*;
pub use spea2::*;
