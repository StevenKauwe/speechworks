#!/bin/sh
# Optional shared-memory runtime. Does not change the ordinary build or server headers.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
nightly=nightly-2026-11-14
bindgen=${SPEECHWORKS_WASM_BINDGEN:-wasm-bindgen}
if [ "$("$bindgen" --version 2>/dev/null || true)" != 'wasm-bindgen 0.2.129' ]; then
  # The ordinary build installs this pinned CLI locally when it is not on PATH.
  bindgen="$root/target/bindgen-0.2.129/bin/wasm-bindgen"
  if [ "$("$bindgen" --version 2>/dev/null || true)" != 'wasm-bindgen 0.2.129' ]; then
    echo 'Threaded build requires wasm-bindgen-cli 0.2.129; run tools/build-web.sh first.' >&2; exit 1
  fi
fi
# Install explicitly if absent; do not change the global/default toolchain:
# rustup toolchain install nightly-2026-11-14 --profile minimal --component rust-src --target wasm32-unknown-unknown
rustup run "$nightly" rustc --version
export RUSTC_WRAPPER="${SPEECHWORKS_RUSTC_WRAPPER:-}"
export CARGO_TARGET_DIR="${SPEECHWORKS_THREAD_TARGET:-$root/target-threaded}"
case "$CARGO_TARGET_DIR" in /*) ;; *) CARGO_TARGET_DIR="$root/$CARGO_TARGET_DIR" ;; esac
export CARGO_TARGET_DIR
out=${SPEECHWORKS_THREAD_DEST:-"$root/dist-threaded"}
case "$out" in /*) ;; *) out="$root/$out" ;; esac
relaxed=-relaxed-simd
if [ "${SPEECHWORKS_THREAD_RELAXED:-0}" = 1 ]; then relaxed=+relaxed-simd; fi
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128,$relaxed,+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"
cargo +"$nightly" build --locked --release --target wasm32-unknown-unknown -p speechworks-web --features wasm-threads -Z build-std=panic_abort,std
mkdir -p "$out"
"$bindgen" "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/speechworks_web.wasm" --target web --out-dir "$out" --out-name speechworks_web
python3 - "$root" "$out" "$relaxed" <<'PY'
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:3])
requires_relaxed=sys.argv[3]=="+relaxed-simd"
helpers=list(out.glob('snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js'))
if len(helpers)!=1:raise SystemExit('Expected exactly one pinned wasm-bindgen-rayon no-bundler helper')
helper=helpers[0]
helper.write_bytes((root/'tools/wasm-rayon-worker.js').read_bytes())
relative='./'+helper.relative_to(out).as_posix()
(out/'thread-pool.js').write_text("import * as wasm from './speechworks_web.js';\nexport {terminateWorkers, setWorkerOwner, setWorkerErrorHandler} from "+repr(relative)+";\nexport {wasm};\nexport const requiresRelaxedSimd="+str(requires_relaxed).lower()+";\n")
PY
cp "$root/LICENSE" "$out/LICENSE"
cp "$root/THIRD_PARTY_NOTICES.md" "$out/THIRD_PARTY_NOTICES.md"
cp "$root/docs/wasm-threads.md" "$out/README.md"
mkdir -p "$out/licenses"
cp "$root/licenses/wasm-bindgen-rayon.txt" "$out/licenses/wasm-bindgen-rayon.txt"
printf 'Optional shared-memory runtime: %s\n' "$out"
