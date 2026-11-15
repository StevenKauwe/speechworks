# Validation

Recorded 2026-11-15 on an Apple M2 Max, macOS/Darwin 27.2.0, Rust 1.99.1,
Chrome 157.0.8103.78. The browser reported an Apple `metal-3` adapter with
`isFallbackAdapter: false`. Its private device name was omitted; the recognizer
reported `GPU (WebGPU, other)`. Tests used an isolated Chrome profile.

The input was the public-domain 11-second JFK recording distributed with
[whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav),
16 kHz mono PCM. Its SHA-256 was
`59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
Tiny/base Q8_0 weights were the revisions and hashes in `models.lock.json`.

## Observed latency

The following are median milliseconds over three warm decodes of each growing
prefix. These are inference times, excluding microphone capture, VAD, fetching,
model loading and initial pipeline compilation.

| Model/backend | 1 second | 3 seconds | 5 seconds | 11 seconds |
| --- | ---: | ---: | ---: | ---: |
| Tiny, WASM CPU | 157 | 306 | 427 | 1133 |
| Base, WASM CPU | 263 | 659 | 865 | 2301 |
| Tiny, WebGPU | 10 | 16 | 21 | 52 |
| Base, WebGPU | 14 | 25 | 34 | 70 |

The first full-phrase decode on each GPU model instance took 264 ms for tiny and
114 ms for base. GPU model construction took 657 ms then 77 ms respectively;
the second model could benefit from shared driver/pipeline caches. These are
not independent cold-start measurements. Local weight reads took roughly
29–63 ms and WASM initialization 12–17 ms; internet download time was not measured.
A separate native CLI smoke processed the same 11-second clip with tiny on CPU
in approximately 83 ms, after approximately 19 ms of model loading.

With base used for both previews and finals, a real-time worker session emitted
24 previews on WebGPU and 9 on WASM CPU. First preview arrived after about
507 ms / 761 ms respectively, including speech onset and VAD confirmation.
Stop-to-final was about 42 ms / 833 ms. Browser fake-microphone capture through
the actual AudioWorklet, with previews disabled and `stopTailMs: 0`, finalized in about
20 ms / 638 ms after stop. All four final phrases retained the final words.
The worker measurements begin when it receives `stop`; the microphone API's
default 200 ms capture tail is additional. Worker comparisons used 15-second
soft/hard phrase limits; the microphone test used the default phrase limits.
These values describe this fixture and machine, not a latency or accuracy guarantee.

## Acceptance exercised

- Native tiny/base CPU and Metal produced reference transcripts. Growing and
  shrinking 5/11/5-second phrases retained exact CPU/GPU text parity.
- Actual browser WASM CPU and hardware WebGPU produced identical tiny/base text
  over repeated 1/3/5/11/3-second phrases. This was executed, not inferred from
  successful compilation.
- Real VAD, worker previews/finals, final-only sessions, cancellation followed by
  another clip, verified caching, corrupt model hash rejection, AudioWorklet
  capture and repeated `stop()` completed successfully on CPU and WebGPU.
- Auto selected CPU when WebGPU was unavailable. Destroying the actual GPUDevice
  caused CPU replay of the same phrase with identical text and a reported reason.
- The headless `--cpu-only` suite also passed without requesting a GPU device.
  All browser runs made no external requests and reported no unhandled errors,
  under the consuming application's policy:
  `script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; connect-src 'self'; worker-src 'self'`.

Real speech testing exposed a defect in upstream `silero-vad-pure` 0.1.1:
it omitted the previous 64 audio samples required by the official 16 kHz
streaming wrapper. The unmodified detector found zero voiced frames, with a
maximum probability of 0.00365. The corrected vendored detector found
241/344 voiced frames and a maximum probability of 0.999963, matching between
native and WASM runs. Both produced four final phrases. The 8 kHz context is
32 samples; reset clears audio history and recurrent state together.
See [`crates/vad/UPSTREAM.md`](../crates/vad/UPSTREAM.md) for source and model notices.
This is real-speech regression evidence, not complete ONNX parity validation.

## Reproduce

These POSIX shell commands require Rust, its WASM target, Node/npm and curl.
The installer builds WASM and verifies pinned model downloads. Browser tests
use local files and an isolated profile; they do not access a personal profile.

```sh
rustup target add wasm32-unknown-unknown
npm install
npx playwright install chromium
validation_dir=$(mktemp -d)
tools/install-web.sh --dest "$validation_dir/runtime" --model-dir "$validation_dir/models"
curl --fail --location --output "$validation_dir/models/jfk.wav" \
  https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav

cargo test --locked --workspace
cargo check --locked -p speechworks-web --target wasm32-unknown-unknown
npm test
SPEECHWORKS_TEST_AUDIO="$validation_dir/models/jfk.wav" \
  cargo test --locked -p speechworks-web --test real_vad -- --ignored --nocapture

# Headed mode for hardware GPU acceptance; use --cpu-only for headless CPU CI.
HEADLESS=0 node tests/browser-smoke.mjs \
  --root "$validation_dir/runtime" --fixture "$validation_dir/models/jfk.wav"
node tests/browser-smoke.mjs --cpu-only \
  --root "$validation_dir/runtime" --fixture "$validation_dir/models/jfk.wav"
```

The harness records fixture/model/WASM hashes, device details, transcript text
and timings in ignored `test-results/browser.json`. Verify the fixture hash above
when comparing measurements. `--direct-only`, `--runtime-only`, `--fallback-only`
and `--output PATH` select or save individual stages. `CHROME_EXECUTABLE` and
`PLAYWRIGHT_MODULE` optionally select an existing installation. Native real-model
commands and their named fixture requirements are documented in
[`crates/inference/README.md`](../crates/inference/README.md).

Windows/DX12, Linux/Vulkan, NVIDIA RTX, Intel/AMD GPUs, Firefox, Safari, mobile
devices, thermal limits and long recording endurance were not exercised here.
The portable kernels and CPU fallback support those deployment paths where the
required Rust/WASM/WebGPU capabilities exist, but compilation does not establish
device-specific performance or reliability. CPU selection can be faster for some
native workloads. Test representative voices, noise, boundary words and sustained
sessions on the devices an application intends to support.
