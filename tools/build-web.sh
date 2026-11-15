#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
export RUSTC_WRAPPER="${SPEECHWORKS_RUSTC_WRAPPER:-}"
bindgen=${SPEECHWORKS_WASM_BINDGEN:-wasm-bindgen}
if [ "$("$bindgen" --version 2>/dev/null || true)" != 'wasm-bindgen 0.2.129' ]; then
  bindgen="$root/target/bindgen-0.2.129/bin/wasm-bindgen"
  if [ ! -x "$bindgen" ]; then
    cargo install --locked wasm-bindgen-cli --version 0.2.129 --root "$root/target/bindgen-0.2.129"
  fi
fi
# Plain wasm SIMD only: relaxed SIMD is not required for the CPU fallback.
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128,-relaxed-simd"
cargo build --locked --release --target wasm32-unknown-unknown -p speechworks-web
target=${CARGO_TARGET_DIR:-"$root/target"}
case "$target" in /*) ;; *) target="$root/$target" ;; esac
mkdir -p "$root/dist"
"$bindgen" "$target/wasm32-unknown-unknown/release/speechworks_web.wasm" --target web --out-dir "$root/dist" --out-name speechworks_web
cp "$root"/web/*.js "$root/dist/"
cp "$root/LICENSE" "$root/dist/LICENSE"
cp "$root/THIRD_PARTY_NOTICES.md" "$root/dist/THIRD_PARTY_NOTICES.md"
mkdir -p "$root/dist/licenses"
cp "$root"/licenses/*.txt "$root/dist/licenses/"
