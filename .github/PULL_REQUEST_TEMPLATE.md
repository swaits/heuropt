<!--
Thanks for the contribution! Please skim CONTRIBUTING.md if you
haven't yet — it has the local-test checklist and the conventional-
commits requirement.
-->

## What

<One- or two-sentence summary. Focus on the *what* and *why*, not
the *how*.>

## Why

<Motivation. Link the issue this resolves with `Closes #N` if
applicable.>

## Checklist

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test` and `cargo test --all-features`
- [ ] `cargo doc --no-deps --all-features` (with `-D warnings`)
- [ ] Conventional-commit subject(s) (`<type>(<scope>): <summary>`)
- [ ] If touching algorithm output: confirmed bit-identical results
      via `cargo run --release --example compare`
- [ ] If perf change: included gungraun before/after numbers in the
      commit message
- [ ] Updated CHANGELOG.md under `[Unreleased]` if user-visible

## Anything else

<Caveats, follow-ups, screenshots, perf numbers, etc.>
