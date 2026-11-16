# Optional shared-memory CPU runtime

The normal build has plain and guarded relaxed-SIMD **serial** runtimes. The
optional package adds a persistent CPU pool. The public client selects it only
when the package and browser capabilities are present; native kernels and
existing application server headers are unchanged.

## Install and select

```sh
# In addition to the normal Rust/wasm-bindgen prerequisites in the README:
rustup toolchain install nightly-2026-11-14 --profile minimal --component rust-src --target wasm32-unknown-unknown
SPEECHWORKS_WASM_THREADS=1 tools/install-web.sh \
  --dest /path/to/app/public/speechworks --model-dir /path/to/model-cache
```

The installer builds serial artifacts first, then `threaded/`, and records
availability in `runtime-config.js`. A normal rebuild removes stale threaded
artifacts and resets availability. Package the **entire installed directory**,
including `relaxed/`, `threaded/snippets/`, helper modules and notices. Do not
copy only the top-level WASM file or reuse helpers from a different build.

```js
import {createDictation} from '/speechworks/speechworks.js';
const speech = createDictation({backend: 'cpu', cpuThreads: 'auto'});
// 'auto' is the default; no separate caller-side feature detection is needed.
speech.onStatus(({device, fallback}) => console.log({device, fallback}));
await speech.preload();
// Use speech.transcribe(...) or speech.record(...).
speech.destroy();
```

Automatic sizing is `max(1, min(4, floor(hardwareConcurrency / 2)))`. An absent
or invalid processor count selects one. This is a conservative cap, **not** a
measurement of idle or performance cores. Two reported processors select one;
four select two; eight or more select four. `cpuThreads: 1` disables the pool;
explicit integers 2–32 override the automatic cap. A runtime's size is immutable:
create another engine to change it. CPU pressure is not measured and there is
no automatic mid-inference resizing or pressure backoff.

The pool is also prepared for `backend: 'auto'` so a GPU failure can use CPU.
The reported model device remains the actual GPU/CPU backend, rather than
claiming CPU was used merely because a pool exists.

## Hosting and fallback

Threading requires a secure, **cross-origin-isolated** context, SharedArrayBuffer,
and shared WebAssembly memory. The installer currently packages the relaxed-SIMD
threaded variant, so that instruction set is also probed before instantiation.
Common isolation headers are `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`, but adding them can break embeddings,
OAuth popup relationships and cross-origin assets. Hosting owners must review
those requirements. This library and its installer never change deployed headers.

When isolation, a capability or the optional artifact is absent, the client uses
guarded relaxed-SIMD serial, then plain SIMD if necessary. It does not fetch the
threaded binary in a known non-isolated context. A failed pool startup terminates
the attempted pool **before** loading a separate serial WASM instance. A failed
shared instance is never reused. Fallback reasons appear in backend status; the
device string identifies the actual CPU thread count and kernel path. Raw worker
protocol clients without independent child-worker ownership safely remain serial.

The runtime retains same-origin module workers and works with
`worker-src 'self'`; it does not require `blob:` or a relaxed CSP. Serve JavaScript
with its proper MIME type and WASM as `application/wasm`. Normal WASM CSP and
same-origin model requirements still apply.

## Lifetime and failure behavior

The UI owns compute Worker handles independently of the coordinator, because the
coordinator can block inside synchronous WASM. `destroy()`, coordinator failure,
child failure and an independent watchdog terminate children **before** the
coordinator and reject pending work. A replacement engine cannot execute canceled
requests from an earlier generation. Recording cancellation closes capture and
discards its result; `destroy()` is the immediate whole-engine stop.

Startup is bounded to 30 seconds. An active threaded inference has a 30-second
watchdog, excluding model download and loading. It is a failure guard, not a
performance goal or an estimate of free CPU. A fatal error aborts the request;
there is no silent replay or automatic recovery of a partially failed inference.
The application can start a new engine after handling the error. Background-tab
browser throttling and suspended pages can delay timers, so this is not a hard
real-time termination guarantee.

The lower-level `thread-pool.js` exposes `wasm`, `requiresRelaxedSimd`,
`setWorkerOwner`, `setWorkerErrorHandler` and async/idempotent `terminateWorkers`.
`wasm.initCpuThreadPool(1..32)` must complete once, before loading models; a second
initialization is rejected. Advanced consumers must provide equivalent independent
ownership and fatal-error handling. The owner contract is documented in
`web/pool-owner.js`; the supported application API is `createDictation`.

## Build internals

`tools/build-web-threaded.sh` can independently emit `dist-threaded/` without
changing the serial build. `SPEECHWORKS_THREAD_DEST` and
`SPEECHWORKS_THREAD_TARGET` choose output/build directories;
`SPEECHWORKS_THREAD_RELAXED=1` enables guarded relaxed SIMD. Its default standalone
output is plain SIMD. The helper uses pinned nightly Rust 1.102.0,
wasm-bindgen 0.2.129, wasm-bindgen-rayon 1.3.0 and locked Rayon dependencies. It
rebuilds `std` with atomics; a prebuilt stable WASM standard library is not
interchangeable. Shared imported memory has a 2 GiB maximum, not an up-front
allocation. All models within a module share its pool.

`tools/wasm-rayon-worker.js` adapts the Apache-2.0 helper from
[wasm-bindgen-rayon 1.3.0](https://crates.io/crates/wasm-bindgen-rayon/1.3.0).
Its license/header are retained. The generated snippet is replaced with a
same-origin implementation, with bounded startup and explicit disposal.

## Validation and benchmarks

- `node --test tests/*.test.mjs`: sizing, capability/startup fallbacks, helper
  lifecycle and canceled-generation regressions.
- `npm run test:wasm-kernels`: plain and guarded relaxed-SIMD numerical/bounds
  assertions in Node and a real browser.
- `tests/browser-lifecycle.mjs --root ... --fixture ... --output ...`: real public
  API, isolated/non-isolated selection, repeated use/recreation, partial startup,
  destroy during inference, late child failure, watchdog, UI handles and browser
  worker-target cleanup. Fault injection is test-local.
- `tests/threaded-smoke.mjs`: lower-level pool throughput/controls and startup
  guards, with `--threads 1 --force-threaded` for a same-binary one-worker control.

See [the measured CPU follow-up](wasm-cpu-phase2.md). Small fixture regressions
are not broad speech-accuracy, mobile, sustained-load or deployment guarantees.
