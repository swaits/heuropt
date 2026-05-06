# Security policy

## Supported versions

Security fixes are applied to the latest released minor version on
crates.io. Patch-level releases (`0.x.y` → `0.x.y+1`) are issued as
needed.

| Version | Supported          |
|---------|--------------------|
| 0.10.x  | ✅                  |
| ≤ 0.9.x | ❌ (please upgrade) |

heuropt is pre-1.0; the public API may change between minor versions.
Once 1.0.0 ships, the support window will be at least the latest two
minor versions.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security bug.
Instead use one of these channels:

- GitHub's [private vulnerability reporting](https://github.com/swaits/heuropt/security/advisories/new)
  on the repository.
- Email **steve@waits.net** with subject line `[heuropt security]
  <short summary>`.

Please include:

1. A description of the vulnerability and the affected versions.
2. The smallest reproducer you can produce — a `cargo run --example
   repro` is ideal.
3. Your assessment of impact and exploitability.
4. Any suggested mitigation if you have one.

## What I will do

- Acknowledge the report within **72 hours**.
- Confirm or refute reproducibility within **7 days**.
- Issue a fix in a patch release within **30 days** for confirmed
  high-severity issues; less urgent issues may roll into the next
  minor release.
- Credit the reporter in the CHANGELOG entry unless you ask
  otherwise.

## What counts as a security issue

heuropt is a numerical library, not a network service or sandbox. The
realistic security-relevant categories are:

- **Memory safety**: any unsafe-code-related UB or unwinds-across-FFI
  bug. heuropt itself uses no `unsafe`; this category covers
  dependencies it transitively pulls in.
- **Denial of service**: an input to a public API that causes
  unbounded memory growth, infinite loop, or panic outside its
  documented panic conditions. (Documented panics for invalid config
  are not bugs.)
- **Supply-chain compromise**: a published heuropt crate that doesn't
  match the source on the tagged commit.

Functional correctness bugs (an algorithm produces wrong
hypervolumes, etc.) are tracked as ordinary issues, not security.
