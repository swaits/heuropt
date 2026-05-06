//! Lightweight metadata about an algorithm — its short canonical name
//! and the seed driving the current run.
//!
//! `AlgorithmInfo` is separate from [`Optimizer<P>`](super::Optimizer)
//! so multi-fidelity algorithms (which use `PartialProblem` instead of
//! `Problem`) can implement it uniformly. Every built-in algorithm in
//! `heuropt` implements `AlgorithmInfo`; the explorer JSON export reads
//! these methods to populate `algorithm` and `seed` fields in the
//! exported run metadata.

/// Algorithm metadata used by tooling such as the explorer JSON export.
///
/// Implementors return a short canonical name like `"Nsga3"` or
/// `"DifferentialEvolution"`, and the seed driving their current run
/// when applicable.
pub trait AlgorithmInfo {
    /// Short, canonical algorithm name — e.g. `"Nsga3"`,
    /// `"DifferentialEvolution"`, `"BayesianOpt"`.
    fn name(&self) -> &'static str;

    /// The deterministic seed driving this run, if the algorithm uses
    /// one. Default: `None`. Built-in algorithms return
    /// `Some(self.config.seed)`.
    fn seed(&self) -> Option<u64> {
        None
    }
}
