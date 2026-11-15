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
target=${CARGO_TARGET_DIR:-"$root/target"}
case "$target" in /*) ;; *) target="$root/$target" ;; esac
mkdir -p "$root/dist"
# Never ship an optional shared-memory build left over from different source. The
# opt-in installer rebuilds it after both compatible single-thread artifacts.
rm -rf "$root/dist/threaded"
flags=${RUSTFLAGS:-}
for variant in plain relaxed; do
  feature=-relaxed-simd
  out="$root/dist"
  if [ "$variant" = relaxed ]; then
    feature=+relaxed-simd
    out="$root/dist/relaxed"
  fi
  RUSTFLAGS="$flags -C target-feature=+simd128,$feature" \
    cargo build --locked --release --target wasm32-unknown-unknown -p speechworks-web
  mkdir -p "$out"
  "$bindgen" "$target/wasm32-unknown-unknown/release/speechworks_web.wasm" \
    --target web --out-dir "$out" --out-name speechworks_web
done
cp "$root"/web/*.js "$root/dist/"
cp "$root/LICENSE" "$root/dist/LICENSE"
cp "$root/THIRD_PARTY_NOTICES.md" "$root/dist/THIRD_PARTY_NOTICES.md"
mkdir -p "$root/dist/licenses"
cp "$root"/licenses/*.txt "$root/dist/licenses/"
