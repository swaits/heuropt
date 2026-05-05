# Stability and SemVer

heuropt is pre-1.0. The public API may change between minor versions.
This page sets explicit expectations.

## What "public API" means in heuropt

The crate's public surface is everything re-exported from
[`heuropt::prelude`] plus the items reachable from `heuropt::core`,
`heuropt::traits`, `heuropt::operators`, `heuropt::algorithms`,
`heuropt::pareto`, `heuropt::metrics`, and `heuropt::selection`.

Items in `heuropt::internal` (e.g. the Cholesky / eigendecomposition
helpers) are **not** public API. They may change between any two
versions — use them at your own risk.

## SemVer in heuropt 0.x

While we are pre-1.0:

- **Minor bumps (`0.5 → 0.6`) may break the public API.** The
  CHANGELOG calls out everything that changed, and a **migration
  guide** in this book documents the move.
- **Patch bumps (`0.5.0 → 0.5.1`) only contain bug fixes,
  performance improvements, and additive non-breaking features.**
  No deprecations, no removals.

## What's actually likely to change before 1.0

In rough order of likelihood:

1. **`Optimizer<P>` may grow new optional methods** for callbacks,
   stop conditions, and save/resume support. These will land as
   methods with default implementations so existing trait impls
   keep compiling, but the trait shape will be different.
2. **Algorithm config structs may gain fields.** All current configs
   are public-field structs; adding a non-`Default` field is a
   breaking change. We may switch to builder patterns to avoid this
   class of break, or we may add `#[non_exhaustive]`.
3. **The `Snapshot`, `Observer`, and `Checkpoint` types** (planned
   for a future release) will land as new public surfaces.
4. **Some operators may move between `operators` and `pareto`** as
   the boundary between "things that produce candidates" and "Pareto
   utilities" gets clearer.

What is **not** likely to change:

- The `Problem` trait shape.
- The `Variation` / `Initializer` / `Repair` traits.
- The `Evaluation` / `Candidate` / `Population` / `OptimizationResult`
  data types.
- The seeded determinism property.

## What "bit-identical" means for stability

heuropt promises that a given algorithm + seed + config produces the
same numeric output on the same minor version of heuropt.

Across minor versions, output may change if an algorithm's
implementation changes (e.g. a perf rewrite that reorders
floating-point operations, or a new feature that changes the
RNG-consumption pattern). The CHANGELOG calls this out explicitly
when it happens. As of v0.5, the entire history of perf optimizations
has been bit-identical against the v0.3.0 reference.

## MSRV (minimum supported Rust version)

heuropt's MSRV is **1.85** as of v0.5. This is tested in CI against
every PR.

MSRV bumps are treated as patch-bump-eligible (they don't break the
public API). When the MSRV is bumped, the CHANGELOG entry for that
release will note the new MSRV.

## Feature-flag stability

The current optional features:

- `serde` — adds `Serialize` / `Deserialize` derives on the core data
  types.
- `parallel` — rayon-backed parallel population evaluation.

Features added in 0.x can be renamed or removed in any minor bump
that documents the change. Removing a feature is treated like a
breaking API change.

## How to track changes

- **CHANGELOG.md** — the canonical record of changes per release.
- **Migration guides** — per-release, in this book at
  [migration](./migration.md).
- **GitHub releases** — each tag has release notes.
- **Watch the repo** — https://github.com/swaits/heuropt — to be
  notified of new releases.

[`heuropt::prelude`]: https://docs.rs/heuropt/latest/heuropt/prelude/index.html
