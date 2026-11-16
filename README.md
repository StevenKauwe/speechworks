# Speechworks

Local dictation in Rust: editable live previews, stable final phrases, optional
GPU acceleration, and the same engine in native applications and browser workers.
Built to be shared by Agency Platform and CrabScape.

**[Try the live playground →](https://stevenkauwe.github.io/speechworks/)**
Use your microphone or a sample clip, switch between base and tiny, and compare
automatic GPU acceleration with CPU inference. No account or installation;
recognition runs in your browser. The first use downloads the selected model.

The dictation behavior is independent of the model. Applications choose draft
and final models, backend, pause thresholds, preview cadence, phrase limits, and
memory budgets. The included inference backend supports Moonshine tiny and base
from GGUF metadata. Other architectures need an inference adapter; changing a
model name does not make an unsupported architecture work.

## What runs where

| Component | Responsibility |
|---|---|
| `speechworks` | Rust phrase boundaries, bounded queues, stop/flush, preview policy, resampling |
| `speechworks-inference` | Moonshine model, Rust CPU kernels, portable asynchronous WGSL/`wgpu` GPU kernels |
| `speechworks-vad` | Rust Silero voice detection, with streaming audio context preserved |
| `speechworks-web` | WASM bindings and Rust Silero VAD; no ONNX Runtime |
| `web/` | Small microphone, worker, caching, and application interface bridges |
| `speechworks-assets` | Pinned model download, checksum validation, same-origin bundle installation |

Native GPU backends are Metal, Vulkan, and DirectX 12; browser GPU execution uses
WebGPU. `auto` tries the GPU and replays the current phrase on CPU if GPU setup or
execution fails. `cpu` forces CPU; `gpu` makes GPU failures explicit. Detection
does not imply that every model is faster on the GPU; benchmark your workload.
Browsers without WebGPU use WASM SIMD CPU inference. Plain SIMD remains the
compatible fallback; relaxed SIMD and shared-memory workers are optional.
Cross-origin isolation is not required for the serial fallback.

Live previews currently re-decode the bounded phrase in progress. This provides
streaming dictation UX, not a cache-aware incremental Moonshine V2 encoder.
Final-only mode skips preview inference. Audio never goes to a transcription
service. Applications retain control of their own text fields.

The default profile uses base for both live and final text. Set
`models: {draft: 'tiny', final: 'base'}` for a smaller preview model. Preview
frequency adapts to measured inference time so CPU fallback leaves processing
time for capture and VAD; final phrases always have priority.

## Native Rust

```rust,no_run
use speechworks_inference::{Backend, Model};

async fn transcribe(weights: Vec<u8>, pcm: &[f32]) -> speechworks_inference::Result<String> {
    let model = Model::from_bytes(weights, Backend::Auto).await?;
    Ok(model.transcribe(pcm).await?.text)
}
```

PCM is mono 16 kHz, finite samples in `[-1, 1]`. The default inference call is
bounded to 30 seconds and 256 MiB of working memory, excluding weights. Use
`LoadOptions` to set explicit limits. For continuous audio, feed `speechworks::Session`
frames and speech probabilities, drain `take_final()` before `preview()`, and
discard a preview if its recording or phrase is no longer current. Session errors
must reach the application; do not turn recognition failures into empty text.
Session phrase limits include frame rounding and final silence padding within
the default 30-second inference budget.

## Browser applications

Prerequisites: Rust 1.99+, `wasm32-unknown-unknown`, `wasm-bindgen-cli` **0.2.129**,
and `curl`. Node is only used for development tests.

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked wasm-bindgen-cli --version 0.2.129
tools/install-web.sh --dest /path/to/app/public/speechworks --model-dir /path/to/model-cache
```

The installer builds the WASM module, fetches missing models, verifies each file,
and installs a complete directory. The default manifest downloads from exact
Hugging Face revisions. Supply `--manifest path/to/models.lock.json` to choose
other models or download hosts. SHA-256 and byte lengths are required. The
generated `models.json` points to same-origin, content-addressed model files.
The normal build includes both plain and relaxed-SIMD serial artifacts. The
worker probes support before loading the optional faster artifact and preserves
the compatible fallback. Direct imports of `speechworks_web.js` remain plain
SIMD; `runtime-loader.js` exposes the feature-selecting `loadRuntime()` helper.

For optional shared-memory CPU workers, see [the threaded build guide](docs/wasm-threads.md).
Install with `SPEECHWORKS_WASM_THREADS=1` and the pinned nightly prerequisite in
the guide. When that artifact is installed and hosting supports isolation,
`createDictation` defaults to a conservative `cpuThreads: 'auto'`: half the
reported logical processors, capped at four. Unsupported contexts retain the
fast serial path. Set `cpuThreads: 1` to opt out, or an explicit 2–32 to override.
The installer does not set server headers or relax CSP. See
[the follow-up benchmark notes](docs/wasm-cpu-phase2.md).

Serve `.wasm` as `application/wasm`, `.js` as JavaScript, and `.json` as JSON.
Use HTTPS (localhost is allowed for development), microphone permission, and
`worker-src 'self'` in your CSP; include `wasm-unsafe-eval` where your CSP requires it.

```js
import {createDictation} from '/speechworks/speechworks.js';

const speech = createDictation({
  backend: 'auto',
  models: {draft: 'base', final: 'base'},
  session: {previews: true, preview_interval_ms: 250},
});
speech.onStatus(({ready, progress, error}) => updateStatus({ready, progress, error}));
// Optional preload: downloads only models selected by this profile, without opening the microphone.
await speech.preload();
const recording = await speech.record({
  onText: ({committed, live}) => showTranscript({committed, live}),
  onError: error => showError(error.message),
});
const finalText = await recording.stop();
// recording.cancel() discards this recording; speech.destroy() frees its worker/models.
```

Keep tentative text separate from the user's editable value. Append only newly
committed final phrases to the *current* field value, preserving typing and edits.
Record supports `signal: AbortSignal`. A clip API is also available:
`await speech.transcribe(float32Pcm, {model: 'base'})`. Clip inputs are capped at
10 minutes; long clips are split into bounded speech phrases. Applications with
their own microphone capture can import `startWorker` from `worker.js` and use
the message protocol in `docs/protocol.md`.

Silero VAD v6.2 runs in Rust; microphone sample rates are converted to 16 kHz by
a stateful windowed-sinc resampler. Stop flushes the remaining microphone block
and final phrase. The audio queue is bounded; overload is reported explicitly.
Downloads and cached models are checked against the pinned checksum before use.
Offline transcription works once application assets and selected weights are cached.

## Demo page

The [playground](https://stevenkauwe.github.io/speechworks/) is built from `demo/`
and published from this repository with GitHub Pages. It uses the same browser
API and pinned assets as consumer applications. Models, WASM, and the public-domain
sample are served from the same site; no transcription service or analytics is used.
Live previews are optional, final text is editable, and the displayed backend and
sample transcription time come from the running engine.

To run it locally (Node 22+, plus the browser build prerequisites above):

```sh
npm run build:demo
python3 -m http.server 8080 --bind 127.0.0.1 --directory _site
# Open http://127.0.0.1:8080
```

Use `npm run build:demo -- --model-dir /path/to/model-cache` to reuse downloaded
weights. Only demo source is committed; the Pages workflow builds and verifies
the published runtime and models. GitHub Pages does not supply cross-origin
isolation headers, so this demo ships the compatible serial CPU variants and
uses WebGPU when available. Shared-memory CPU workers remain available to apps
with suitable hosting; see [the threaded build guide](docs/wasm-threads.md).

The Pages build runs `npm run test:demo` against real models before deployment,
including sample transcription, microphone stop/cancel, error recovery, and a
mobile layout check. Use `-- --root /path/to/demo` to test another built directory.

## Tests

```sh
cargo test --workspace
cargo check -p speechworks-web --target wasm32-unknown-unknown
node --test tests/*.test.mjs
```

Real-model and hardware tests are separate from unit tests: see
[`docs/validation.md`](docs/validation.md), `crates/inference/README.md`, and
`tests/browser-smoke.mjs`. No universal latency or
accuracy guarantee is implied by a smoke test. Check word accuracy, first useful
preview, pause-to-final delay, missed/duplicated boundary words, CPU fallback,
and memory use on the devices your application supports.

## Ownership and licenses

Speechworks is maintained here under MIT. Its inference implementation is derived
from Wavo's MIT-licensed code, with original notices preserved. The session layer,
browser portability, integration, and product behavior are owned here. See
`THIRD_PARTY_NOTICES.md` for the pinned upstream revision and separate model licenses.
New model adapters can be implemented without changing the recording or app APIs.
