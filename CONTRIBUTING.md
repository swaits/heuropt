# Contributing to heuropt

Thanks for considering a contribution. heuropt is a small, opinionated
crate, but careful additions are welcome.

## Quick checklist

Before opening a pull request:

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test` (default features) and `cargo test --all-features`
- [ ] `cargo doc --no-deps --all-features` with `RUSTDOCFLAGS="-D warnings"`
- [ ] If you touched algorithm output: re-run `cargo run --release --example compare`
      and confirm the quality metrics did not change. Speed-only changes
      are required to be **bit-identical** against the prior snapshot.

CI runs all of the above on every PR; the matrix covers MSRV (1.85),
the default / serde / parallel / serde+parallel feature combinations,
and a 60-second fuzz soak per target.

## Commit style

Conventional Commits (https://www.conventionalcommits.org/) are
required. The first line follows `<type>(<scope>): <summary>` where
`<type>` is one of `feat`, `fix`, `perf`, `refactor`, `docs`, `test`,
`chore`, `ci`, `build`, `style`. `<scope>` is the most specific module
the change touches (e.g. `nsga2`, `hypervolume`, `pareto_archive`).

Bad: `Phase 1.1: Add core data types`
Good: `feat(core): add data types and Rng alias`

Multiple logical changes in a single PR should be split into multiple
commits, each on a single concern.

## What kinds of contributions land easily

- **Bug fixes** with a regression test that fails on `main` and passes
  on the fix.
- **Performance wins** that preserve bit-identical output and include
  a `cargo bench` (gungraun) before/after, plus a `cargo run --release
  --example compare` diff confirming no quality regression.
- **Documentation improvements** — missing rustdoc examples, README
  clarifications, mdbook chapters.
- **New algorithms** that fit the established `Optimizer<P>` shape and
  ship with: a unit test, a property test (determinism + invariants),
  a comparison-harness entry, and rustdoc.
- **New operators / metrics / Pareto utilities** with the same
  hygiene.

## What needs prior discussion

Open an issue before starting on:

- New traits or breaking changes to the public API surface.
- A new optional feature flag.
- Anything that depends on a heavy new dependency.
- Restructuring of `src/algorithms/` or `src/pareto/`.

The crate intentionally keeps the trait surface small (`Problem`,
`Optimizer`, `Initializer`, `Variation`, `Repair`); changes there
are not refused but they need a clear motivation.

## Running the test suites locally

```sh
# unit + integration + property tests
cargo test

# all feature combinations
cargo test --features serde
cargo test --features parallel
cargo test --all-features

# instruction-count benchmarks (needs valgrind installed)
cargo bench

# coverage-guided fuzzing (needs nightly + cargo-fuzz)
cd fuzz
cargo +nightly fuzz run pareto_compare -- -max_total_time=60

# mutation testing (slow, optional)
cargo install cargo-mutants
cargo mutants
```

## Reporting bugs

Please include:

1. The smallest reproducing input you can produce — ideally a 20-line
   `examples/repro.rs`.
2. The exact command (`cargo run --release --example repro` etc.) and
   the observed vs expected output.
3. The Rust toolchain (`rustc --version`) and feature flags.
4. The heuropt version you saw the bug on.

Bugs that surface fuzz-target panics are particularly welcome; please
attach the failing artifact (`fuzz/artifacts/<target>/crash-...`) so
we can add it to the regression-test corpus.

## Security

For security concerns please follow the disclosure policy in
[SECURITY.md](SECURITY.md). Don't open public issues for security
bugs.

## Code of conduct

This project follows the [Builder's Code of Conduct](CODE_OF_CONDUCT.md).
The short version: stay professional, stay technical, focus on the
work and its merit.

## License

By submitting a contribution, you agree that your work is licensed
under the same MIT license as the rest of heuropt.
