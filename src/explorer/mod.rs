//! Explorer JSON export — serialize an `OptimizationResult` to a
//! self-describing JSON file that the
//! [heuropt-explorer](https://swaits.github.io/heuropt-explorer/)
//! webapp can load and explore interactively.
//!
//! ## Quick start
//!
//! ```ignore
//! use heuropt::prelude::*;
//!
//! let result = optimizer.run(&problem);
//!
//! // Zero-config — pulls metadata from `problem.objectives()`,
//! // `problem.decision_schema()`, and the algorithm's `AlgorithmInfo`.
//! heuropt::explorer::to_file("results.json", &problem, &optimizer, &result)?;
//! ```
//!
//! Drop the resulting `results.json` into the explorer at
//! <https://swaits.github.io/heuropt-explorer/> to filter, brush,
//! pin, and rank candidates.
//!
//! ## What's in the export
//!
//! The output contains:
//! - `schema_version` — an integer the explorer uses to detect
//!   incompatible files. Bump on breaking schema changes.
//! - `run` — algorithm name, seed, evaluations, generations, and
//!   optional problem name / wall-clock seconds.
//! - `objectives` — name, direction, and (if set) `label` and
//!   `unit` so the explorer can render axes like `Price ($k)`.
//! - `decision_variables` — name, label, unit, and bounds for each
//!   decision-variable slot. If `Problem::decision_schema()` returns
//!   fewer entries than the decision length, the exporter pads with
//!   fallback names like `x[0]`, `x[1]`.
//! - `candidates` — the full population, each tagged with its
//!   front rank (from `non_dominated_sort`), feasibility, and
//!   whether it sits on the Pareto front.
//!
//! Everything is gated on the `serde` feature, since the export
//! uses `serde_json`.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::core::candidate::Candidate;
use crate::core::decision_variable::DecisionVariable;
use crate::core::objective::Objective;
use crate::core::problem::Problem;
use crate::core::result::OptimizationResult;
use crate::pareto::sort::non_dominated_sort;
use crate::traits::AlgorithmInfo;

/// JSON schema version embedded in every export. The explorer
/// webapp checks this on load and rejects files with an unknown
/// version. Bump on breaking schema changes.
pub const SCHEMA_VERSION: u32 = 1;

/// Serialized envelope describing one optimization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerExport {
    /// Schema version (always equal to [`SCHEMA_VERSION`] when written).
    pub schema_version: u32,
    /// Run metadata — algorithm, seed, eval/generation counts.
    pub run: RunMeta,
    /// Objective definitions, with optional `label` / `unit` if set.
    pub objectives: Vec<Objective>,
    /// Decision-variable schemas, padded with fallback `x[i]` names
    /// when the user didn't override `Problem::decision_schema()`.
    pub decision_variables: Vec<DecisionVariable>,
    /// One row per candidate in the final population.
    pub candidates: Vec<ExplorerCandidate>,
}

/// Per-candidate row in [`ExplorerExport`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerCandidate {
    /// Decision values, one entry per decision variable. Numbers,
    /// booleans, integers, or strings — whatever the
    /// [`ToDecisionValues`] impl produces for the decision type.
    pub decision: Vec<serde_json::Value>,
    /// Objective values, parallel to the `objectives` array.
    pub objectives: Vec<f64>,
    /// Constraint violation magnitude (≤ 0 means feasible).
    pub constraint_violation: f64,
    /// Convenience: `true` iff `constraint_violation <= 0.0`.
    pub feasible: bool,
    /// Non-domination rank from `non_dominated_sort`. `0` means
    /// on the first front (Pareto front).
    pub front_rank: usize,
    /// `true` iff this candidate is on the first front. (Same as
    /// `front_rank == 0` for the rank-0 set, kept as an explicit
    /// field so downstream tools don't have to re-derive it.)
    pub in_pareto_front: bool,
}

/// Run-level metadata: algorithm name, seed, eval count, etc.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunMeta {
    /// Optional human-readable problem name (e.g. `"Pick a car"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem_name: Option<String>,
    /// Canonical algorithm name (e.g. `"Nsga3"`). Pulled from
    /// [`AlgorithmInfo::name`] when an algorithm is provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    /// Seed driving this run, if applicable. Pulled from
    /// [`AlgorithmInfo::seed`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Wall-clock duration of the run, in seconds. Optional —
    /// the user provides this if they timed the run externally.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_clock_seconds: Option<f64>,
    /// Total number of `Problem::evaluate` calls.
    pub evaluations: usize,
    /// Number of major optimizer iterations.
    pub generations: usize,
    /// Optional ISO-8601 timestamp recorded at export time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// Adapter trait that converts a decision value into a vector of
/// `serde_json::Value`s (one per element). Implemented for the
/// common decision types out of the box; users with custom
/// decision types implement it themselves.
pub trait ToDecisionValues {
    /// Convert the decision into one JSON value per decision-variable
    /// slot.
    fn to_decision_values(&self) -> Vec<serde_json::Value>;
}

impl ToDecisionValues for Vec<f64> {
    fn to_decision_values(&self) -> Vec<serde_json::Value> {
        self.iter()
            .map(|v| {
                serde_json::Number::from_f64(*v)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            })
            .collect()
    }
}

impl ToDecisionValues for Vec<bool> {
    fn to_decision_values(&self) -> Vec<serde_json::Value> {
        self.iter().map(|b| serde_json::Value::Bool(*b)).collect()
    }
}

impl ToDecisionValues for Vec<usize> {
    fn to_decision_values(&self) -> Vec<serde_json::Value> {
        self.iter()
            .map(|i| serde_json::Value::Number(serde_json::Number::from(*i as u64)))
            .collect()
    }
}

impl ToDecisionValues for Vec<i64> {
    fn to_decision_values(&self) -> Vec<serde_json::Value> {
        self.iter()
            .map(|i| serde_json::Value::Number(serde_json::Number::from(*i)))
            .collect()
    }
}

impl ExplorerExport {
    /// Build an `ExplorerExport` from a problem and its result.
    /// The run metadata is initially empty (no algorithm / seed);
    /// chain `with_algorithm_info` or the individual setters to
    /// populate it.
    pub fn from_result<P>(problem: &P, result: &OptimizationResult<P::Decision>) -> Self
    where
        P: Problem,
        P::Decision: ToDecisionValues,
    {
        let objective_space = problem.objectives();
        let n_obj = objective_space.objectives.len();

        let user_schema = problem.decision_schema();
        let decision_arity = result
            .population
            .candidates
            .first()
            .map(|c| c.decision.to_decision_values().len())
            .unwrap_or(user_schema.len());
        let decision_variables = pad_decision_schema(user_schema, decision_arity);

        let pop_slice: &[Candidate<P::Decision>] = &result.population.candidates;
        let fronts = non_dominated_sort(pop_slice, &objective_space);
        let mut rank_of: Vec<usize> = vec![0; pop_slice.len()];
        for (rank, front) in fronts.iter().enumerate() {
            for &idx in front {
                rank_of[idx] = rank;
            }
        }

        let candidates = pop_slice
            .iter()
            .enumerate()
            .map(|(i, c)| candidate_to_export(c, rank_of[i], n_obj))
            .collect();

        Self {
            schema_version: SCHEMA_VERSION,
            run: RunMeta {
                evaluations: result.evaluations,
                generations: result.generations,
                ..RunMeta::default()
            },
            objectives: objective_space.objectives,
            decision_variables,
            candidates,
        }
    }

    /// Populate `algorithm` and `seed` from anything implementing
    /// [`AlgorithmInfo`] — every built-in algorithm does.
    pub fn with_algorithm_info<A: AlgorithmInfo>(mut self, algorithm: &A) -> Self {
        self.run.algorithm = Some(algorithm.name().to_owned());
        self.run.seed = algorithm.seed();
        self
    }

    /// Override the problem name shown in the explorer header.
    pub fn with_problem_name(mut self, name: impl Into<String>) -> Self {
        self.run.problem_name = Some(name.into());
        self
    }

    /// Attach a wall-clock duration in seconds.
    pub fn with_wall_clock(mut self, seconds: f64) -> Self {
        self.run.wall_clock_seconds = Some(seconds);
        self
    }

    /// Attach an ISO-8601 timestamp string (the caller formats it).
    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.run.timestamp = Some(timestamp.into());
        self
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to any `Write` sink as pretty-printed JSON.
    pub fn to_writer<W: Write>(&self, writer: W) -> serde_json::Result<()> {
        serde_json::to_writer_pretty(writer, self)
    }

    /// Write the export to a file as pretty-printed JSON. Creates
    /// the file (truncating if it exists) and returns any I/O or
    /// serialization error.
    pub fn to_file<Q: AsRef<Path>>(&self, path: Q) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        self.to_writer(writer)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// Convenience: build an [`ExplorerExport`] from problem +
/// algorithm + result, with `algorithm` and `seed` populated from
/// the [`AlgorithmInfo`] trait, then serialize to a pretty JSON
/// string.
pub fn to_json<P, A>(
    problem: &P,
    algorithm: &A,
    result: &OptimizationResult<P::Decision>,
) -> serde_json::Result<String>
where
    P: Problem,
    P::Decision: ToDecisionValues,
    A: AlgorithmInfo,
{
    ExplorerExport::from_result(problem, result)
        .with_algorithm_info(algorithm)
        .to_json()
}

/// Convenience: same as [`to_json`] but writes to any `Write`.
pub fn to_writer<W, P, A>(
    writer: W,
    problem: &P,
    algorithm: &A,
    result: &OptimizationResult<P::Decision>,
) -> serde_json::Result<()>
where
    W: Write,
    P: Problem,
    P::Decision: ToDecisionValues,
    A: AlgorithmInfo,
{
    ExplorerExport::from_result(problem, result)
        .with_algorithm_info(algorithm)
        .to_writer(writer)
}

/// Convenience: same as [`to_json`] but writes directly to a
/// file path.
pub fn to_file<Q, P, A>(
    path: Q,
    problem: &P,
    algorithm: &A,
    result: &OptimizationResult<P::Decision>,
) -> std::io::Result<()>
where
    Q: AsRef<Path>,
    P: Problem,
    P::Decision: ToDecisionValues,
    A: AlgorithmInfo,
{
    ExplorerExport::from_result(problem, result)
        .with_algorithm_info(algorithm)
        .to_file(path)
}

fn candidate_to_export<D: ToDecisionValues>(
    c: &Candidate<D>,
    front_rank: usize,
    n_obj: usize,
) -> ExplorerCandidate {
    let objectives = if c.evaluation.objectives.len() == n_obj {
        c.evaluation.objectives.clone()
    } else {
        // Defensive: shouldn't happen in practice, but pad/truncate so
        // the export is well-formed even if a buggy algorithm produced
        // a mismatched evaluation.
        let mut v = c.evaluation.objectives.clone();
        v.resize(n_obj, f64::NAN);
        v
    };
    ExplorerCandidate {
        decision: c.decision.to_decision_values(),
        objectives,
        constraint_violation: c.evaluation.constraint_violation,
        feasible: c.evaluation.constraint_violation <= 0.0,
        front_rank,
        in_pareto_front: front_rank == 0,
    }
}

fn pad_decision_schema(
    mut schema: Vec<DecisionVariable>,
    decision_arity: usize,
) -> Vec<DecisionVariable> {
    if schema.len() < decision_arity {
        let start = schema.len();
        for i in start..decision_arity {
            schema.push(DecisionVariable::new(format!("x[{i}]")));
        }
    }
    schema
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::candidate::Candidate;
    use crate::core::evaluation::Evaluation;
    use crate::core::objective::{Direction, Objective, ObjectiveSpace};
    use crate::core::population::Population;
    use crate::core::problem::Problem;
    use crate::core::result::OptimizationResult;

    /// Two-objective minimize problem used for most explorer tests.
    /// f1 = decision[0], f2 = decision[1] — both minimize, so
    /// `(a, b)` dominates `(c, d)` iff `a ≤ c && b ≤ d` with at
    /// least one strict.
    struct TwoObjMin;
    impl Problem for TwoObjMin {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![
                Objective::minimize("a")
                    .with_label("Apples")
                    .with_unit("count"),
                Objective::maximize("b").with_unit("score"),
            ])
        }
        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            Evaluation::new(vec![x[0], x[1]])
        }
    }

    struct EnrichedProblem;
    impl Problem for EnrichedProblem {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("a")])
        }
        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            Evaluation::new(vec![x[0]])
        }
        fn decision_schema(&self) -> Vec<DecisionVariable> {
            vec![
                DecisionVariable::new("alpha")
                    .with_label("Alpha")
                    .with_unit("u")
                    .with_bounds(0.0, 1.0),
                DecisionVariable::new("beta"),
            ]
        }
    }

    struct DummyAlgo;
    impl AlgorithmInfo for DummyAlgo {
        fn name(&self) -> &'static str {
            "DummyAlgo"
        }
        fn seed(&self) -> Option<u64> {
            Some(123)
        }
    }

    /// Build a result whose evaluations match `objectives_per_candidate`.
    /// Each candidate's objective vector is the closure applied to the
    /// decision.
    fn make_result(
        decisions: Vec<Vec<f64>>,
        eval: impl Fn(&[f64]) -> Vec<f64>,
    ) -> OptimizationResult<Vec<f64>> {
        let cands: Vec<Candidate<Vec<f64>>> = decisions
            .into_iter()
            .map(|d| {
                let objs = eval(&d);
                Candidate::new(d, Evaluation::new(objs))
            })
            .collect();
        let n = cands.len();
        OptimizationResult::new(Population::new(cands.clone()), cands, None, n, 1)
    }

    #[test]
    fn schema_version_is_one() {
        assert_eq!(SCHEMA_VERSION, 1);
    }

    /// Single-objective minimize problem (used for tests where the
    /// problem only declares one objective).
    struct SingleObjMin;
    impl Problem for SingleObjMin {
        type Decision = Vec<f64>;
        fn objectives(&self) -> ObjectiveSpace {
            ObjectiveSpace::new(vec![Objective::minimize("f")])
        }
        fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
            Evaluation::new(vec![x[0]])
        }
    }

    #[test]
    fn zero_config_export_uses_fallback_decision_names() {
        let problem = TwoObjMin;
        // Two objectives — eval just maps decision to objective values.
        let result = make_result(vec![vec![0.0, 1.0], vec![1.0, 0.0]], |d| d.to_vec());

        let export = ExplorerExport::from_result(&problem, &result);
        assert_eq!(export.schema_version, SCHEMA_VERSION);
        assert_eq!(export.decision_variables.len(), 2);
        assert_eq!(export.decision_variables[0].name, "x[0]");
        assert_eq!(export.decision_variables[1].name, "x[1]");
        assert!(export.decision_variables[0].label.is_none());
    }

    #[test]
    fn objectives_carry_label_and_unit_through_export() {
        let problem = TwoObjMin;
        let result = make_result(vec![vec![0.0, 1.0]], |d| d.to_vec());
        let export = ExplorerExport::from_result(&problem, &result);
        assert_eq!(export.objectives.len(), 2);
        assert_eq!(export.objectives[0].label.as_deref(), Some("Apples"));
        assert_eq!(export.objectives[0].unit.as_deref(), Some("count"));
        assert_eq!(export.objectives[1].direction, Direction::Maximize);
    }

    #[test]
    fn enriched_decision_schema_passes_through() {
        let problem = EnrichedProblem; // 1 objective, 2-element decisions
        let result = make_result(vec![vec![0.5, 0.5]], |d| vec![d[0]]);
        let export = ExplorerExport::from_result(&problem, &result);
        assert_eq!(export.decision_variables.len(), 2);
        assert_eq!(export.decision_variables[0].name, "alpha");
        assert_eq!(export.decision_variables[0].label.as_deref(), Some("Alpha"));
        assert_eq!(export.decision_variables[0].min, Some(0.0));
        assert_eq!(export.decision_variables[1].name, "beta");
        assert!(export.decision_variables[1].min.is_none());
    }

    #[test]
    fn front_rank_zero_for_pareto_front_members() {
        // Use SingleObjMin (1 objective) to make dominance trivial:
        // among [3.0, 1.0, 2.0], only 1.0 is non-dominated.
        let problem = SingleObjMin;
        let result = make_result(vec![vec![3.0], vec![1.0], vec![2.0]], |d| vec![d[0]]);
        let export = ExplorerExport::from_result(&problem, &result);
        // Index 1 (decision = 1.0) is the unique minimum.
        assert_eq!(export.candidates[1].front_rank, 0);
        assert!(export.candidates[1].in_pareto_front);
        assert_eq!(export.candidates[2].front_rank, 1);
        assert!(!export.candidates[2].in_pareto_front);
        assert_eq!(export.candidates[0].front_rank, 2);
        assert!(!export.candidates[0].in_pareto_front);
    }

    #[test]
    fn algorithm_info_populates_run_meta() {
        let problem = TwoObjMin;
        let result = make_result(vec![vec![0.0, 1.0]], |d| d.to_vec());
        let export = ExplorerExport::from_result(&problem, &result).with_algorithm_info(&DummyAlgo);
        assert_eq!(export.run.algorithm.as_deref(), Some("DummyAlgo"));
        assert_eq!(export.run.seed, Some(123));
    }

    #[test]
    fn round_trip_serde() {
        let problem = TwoObjMin;
        let result = make_result(vec![vec![0.0, 1.0], vec![1.0, 0.0]], |d| d.to_vec());
        let export = ExplorerExport::from_result(&problem, &result)
            .with_algorithm_info(&DummyAlgo)
            .with_problem_name("Toy")
            .with_wall_clock(0.001);
        let json = export.to_json().unwrap();
        let back: ExplorerExport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, SCHEMA_VERSION);
        assert_eq!(back.run.algorithm.as_deref(), Some("DummyAlgo"));
        assert_eq!(back.candidates.len(), 2);
        assert_eq!(back.objectives.len(), 2);
    }

    #[test]
    fn vec_bool_decisions_serialize_as_bool_array() {
        let v: Vec<bool> = vec![true, false, true];
        let values = v.to_decision_values();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], serde_json::Value::Bool(true));
        assert_eq!(values[1], serde_json::Value::Bool(false));
    }

    #[test]
    fn vec_usize_decisions_serialize_as_int_array() {
        let v: Vec<usize> = vec![3, 1, 4];
        let values = v.to_decision_values();
        assert_eq!(values.len(), 3);
        assert_eq!(
            values[0],
            serde_json::Value::Number(serde_json::Number::from(3u64))
        );
    }

    #[test]
    fn nan_decision_renders_as_null() {
        let v: Vec<f64> = vec![1.0, f64::NAN, 2.0];
        let values = v.to_decision_values();
        assert_eq!(values[0].as_f64(), Some(1.0));
        assert_eq!(values[1], serde_json::Value::Null);
        assert_eq!(values[2].as_f64(), Some(2.0));
    }
}
