//! Concrete data types and the `Problem` trait that the rest of the crate is built on.

#[cfg(feature = "async")]
pub mod async_problem;
pub mod candidate;
pub mod evaluation;
pub mod objective;
pub mod partial_problem;
pub mod population;
pub mod problem;
pub mod result;
pub mod rng;

#[cfg(feature = "async")]
pub use async_problem::AsyncProblem;
pub use candidate::*;
pub use evaluation::*;
pub use objective::*;
pub use partial_problem::*;
pub use population::*;
pub use problem::*;
pub use result::*;
pub use rng::*;
