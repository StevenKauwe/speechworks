#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
"$root/tools/build-web.sh"
if [ "${SPEECHWORKS_WASM_THREADS:-0}" = 1 ]; then
  SPEECHWORKS_THREAD_DEST="$root/dist/threaded" SPEECHWORKS_THREAD_RELAXED=1 \
    "$root/tools/build-web-threaded.sh"
  printf '%s\n' 'export const runtimeConfig = Object.freeze({threadedAvailable: true});' \
    > "$root/dist/runtime-config.js"
fi
export RUSTC_WRAPPER="${SPEECHWORKS_RUSTC_WRAPPER:-}"
cargo run --locked --release --manifest-path "$root/Cargo.toml" -p speechworks-assets -- --root "$root" "$@"
