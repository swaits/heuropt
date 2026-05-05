//! Concrete data types and the `Problem` trait that the rest of the crate is built on.

pub mod candidate;
pub mod evaluation;
pub mod objective;
pub mod population;
pub mod result;
pub mod rng;

pub use candidate::*;
pub use evaluation::*;
pub use objective::*;
pub use population::*;
pub use result::*;
pub use rng::*;
