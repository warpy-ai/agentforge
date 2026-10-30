# Contributing

## Toolchain

The workspace pins Rust 1.90.0 in `rust-toolchain.toml` (rustup installs it on first build).
Quality gates, all of which CI runs:

```bash
cargo fmt --all -- --check
```

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

```bash
cargo test --workspace
```

## Adding a machine description

Add a TOML file under `crates/agentforge-machine/machines/`, register it in the `BUILTINS`
table in `crates/agentforge-machine/src/lib.rs`, and cite every number in `sources` and
`notes`. Mark vendor peaks as such. Capacities are in GB (10^9 bytes), bandwidths in TB/s for
device memory and GB/s for tiers and links.

## Adding a model preset

Extend `ModelSpec::preset` in `crates/agentforge-ir/src/model.rs`. Use the published total and
active parameter counts as overrides when the generic formula is too coarse (MoE, latent
attention), and add a test if the preset exposes a new attention kind.

## Research notes

Research lives under `docs/research/`. Notes are dated and cite URLs; update them rather than
rewriting them, and record the date of the refresh.

## Working on macOS exFAT volumes

Finder creates `._*` AppleDouble files next to every file. They are git-ignored, but one inside
`.git/objects/pack` makes git print "non-monotonic index" errors. Remove them with:

```bash
find . -name '._*' -delete
```
