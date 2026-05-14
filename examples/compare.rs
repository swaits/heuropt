//! Multi-seed algorithm comparison harness.
//!
//! Runs every applicable optimizer on each test problem across N seeds and
//! prints aggregate quality metrics.
//!
//! ```bash
//! cargo run --release --example compare
//! ```
//!
//! The problem definitions, the ~88 algorithm runners, and the
//! table-printing presentation all live in the `compare_workload` module
//! so the gungraun profiling benchmark (`benches/compare_profile.rs`) can
//! reuse the exact same workload. This file is just the entry point.

#[path = "_shared/compare_workload.rs"]
mod workload;

fn main() {
    workload::run_all();
}
