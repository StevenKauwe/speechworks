#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
"$root/tools/build-web.sh"
export RUSTC_WRAPPER="${SPEECHWORKS_RUSTC_WRAPPER:-}"
cargo run --locked --release --manifest-path "$root/Cargo.toml" -p speechworks-assets -- --root "$root" "$@"
