//! Lightweight metadata about an algorithm — its canonical short
//! name, an academic long name, and the seed driving the current
//! run.
//!
//! `AlgorithmInfo` is separate from [`Optimizer<P>`](super::Optimizer)
//! so multi-fidelity algorithms (which use `PartialProblem` instead
//! of `Problem`) can implement it uniformly. Every built-in
//! algorithm in `heuropt` implements `AlgorithmInfo`; the explorer
//! JSON export reads these methods to populate the `algorithm` and
//! `algorithm_full_name` fields in the exported run metadata.

/// Algorithm metadata used by tooling such as the explorer JSON
/// export.
///
/// Implementors return:
/// - **`name`** — the canonical short display name as it appears
///   in the literature: `"NSGA-II"`, `"MOEA/D"`, `"ε-MOEA"`,
///   `"CMA-ES"`. *Not* the Rust type name.
/// - **`full_name`** — the academic long form, e.g.
///   `"Non-dominated Sorting Genetic Algorithm II"`. Defaults to
///   `name()` when not overridden, which is the right answer for
///   algorithms whose short name *is* their full name (Random
///   Search, Hill Climber, Tabu Search, …).
/// - **`seed`** — the deterministic seed driving this run, when
///   the algorithm uses one. Defaults to `None`.
pub trait AlgorithmInfo {
    /// Canonical short algorithm name — e.g. `"NSGA-II"`,
    /// `"DE"`, `"CMA-ES"`. This is the form that should appear
    /// in tables, plot legends, and exported JSON metadata.
    fn name(&self) -> &'static str;

    /// Academic long name, expanded — e.g.
    /// `"Non-dominated Sorting Genetic Algorithm II"`. Defaults
    /// to `name()` for algorithms whose short and long forms
    /// coincide (Random Search, Hill Climber, Tabu Search,
    /// Hyperband, …).
    fn full_name(&self) -> &'static str {
        self.name()
    }

    /// The deterministic seed driving this run, if the algorithm
    /// uses one. Default: `None`. Built-in algorithms return
    /// `Some(self.config.seed)`.
    fn seed(&self) -> Option<u64> {
        None
    }
}
