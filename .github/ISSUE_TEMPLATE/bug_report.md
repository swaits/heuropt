---
name: Bug report
about: A correctness, performance, or panic bug in heuropt
title: "bug: <one-line summary>"
labels: bug
---

## What happened

<Concise description of the bug.>

## Reproducer

```rust
// Smallest example that demonstrates the bug. Ideally <30 lines and
// runnable as a fresh `examples/repro.rs`. Include the Cargo.toml
// `[features]` you used.
```

Command used:

```sh
cargo run --release --example repro
```

## Expected vs observed

- **Expected:** <what should happen>
- **Observed:** <what actually happens>

## Environment

- heuropt version:
- `rustc --version`:
- OS / arch:
- Feature flags enabled:

## Additional context

<Anything else — fuzz artifact path, screenshots, profiler output.>
