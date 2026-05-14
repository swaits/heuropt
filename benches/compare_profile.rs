//! Whole-program callgrind profile of the `compare` example workload.
//!
//! Runs every algorithm runner once (seed 0) under callgrind via gungraun —
//! the same workload `examples/compare.rs` runs, minus the multi-seed
//! averaging and table printing. gungraun reports the total instruction
//! count and diffs it against the previous run; the saved `callgrind.out`
//! (`target/gungraun/compare_profile/compare_group/full_compare_workload/`)
//! carries the per-function breakdown — `callgrind_annotate` it to rank
//! functions by self-instruction cost.
//!
//! ```bash
//! cargo bench --bench compare_profile
//! ```

// The shared `compare_workload` module also carries the example's
// presentation layer (`run_all`, the `run_*_comparison` printers,
// `print_table`, …), which this profiling benchmark deliberately does not
// use — it drives only the runner functions via `profile_workload`. The
// runner functions themselves are *not* allow-listed, so a runner that
// `profile_workload` forgets to call still warns.
#![allow(dead_code)]

use std::hint::black_box;

use gungraun::prelude::*;

#[path = "../examples/_shared/compare_workload.rs"]
mod workload;

#[library_benchmark]
fn full_compare_workload() -> u64 {
    black_box(workload::profile_workload())
}

library_benchmark_group!(
    name = compare_group;
    benchmarks = full_compare_workload
);

main!(library_benchmark_groups = compare_group);
