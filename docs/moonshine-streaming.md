# Streaming Medium: implementation and measurements

Recorded 2026-11-16 on an Apple M2 Max (12 logical processors), Rust 1.99.1,
Chrome 157.0.8103.78. [Machine-readable evidence](../benchmarks/2026-11-16-streaming-m2max.json)
includes model, fixture, inference-source, executable and WASM hashes.

Medium is the default for previews and final text. Small works through the same
metadata-driven implementation; it is also tested, rather than assumed compatible.
Both pinned Q8_0 models are MIT licensed. Downloads are approximately 295.8 MB
and 198.5 MB respectively. Original Tiny/Base remain selectable.

## Measured compute latency

The input is the same 11-second JFK recording and the same Medium Q8_0 bytes in
every Medium comparison. Updates supply 320 ms of new PCM, followed by a final
flush. Native figures are medians of two warm passes after one initial pass;
browser figures are the second pass after an initial pass. Runs were sequential,
without concurrent builds or other benchmark processes. These are descriptive
measurements on one machine, not statistical performance guarantees.

| Medium backend | Entire 11 s clip | Preview median | Preview p95 | Final flush |
| --- | ---: | ---: | ---: | ---: |
| Native CPU | 507 ms | 115 ms | 248 ms | 222 ms |
| Native Metal, portable kernels | 312 ms | 86 ms | 158 ms | 154 ms |
| Native Metal, probed matrix kernels | 190 ms | 61 ms | 131 ms | 124 ms |
| Browser WebGPU | 236 ms | 71 ms | 141 ms | 124 ms |
| Browser CPU, guarded relaxed SIMD, 4 threads | 848 ms | 131 ms | 262 ms | 237 ms |
| Browser CPU, guarded relaxed SIMD, 1 thread | 2,582 ms | 319 ms | 568 ms | 499 ms |
| Browser CPU, plain SIMD, 1 thread | 3,708 ms | 345 ms | 579 ms | 480 ms |

The plain-SIMD control was recorded earlier in the same implementation session;
its artifact hash is retained. Small's browser WebGPU result was 150 ms for the
clip, 43 ms median preview and 84 ms final flush. Timings exclude model download,
construction, microphone capture, speech detection and scheduling. The models
also have **240 ms of cumulative encoder look-ahead**. A 61 ms preview computation
therefore does not mean speech appears 61 ms after it is spoken. The worker adapts
preview frequency to compute time; slower CPU configurations deliver fewer updates.

For a reference, we built the MIT-licensed
[transcribe.cpp at adfc502](https://github.com/handy-computer/transcribe.cpp/tree/adfc50210a3f026a655a3e19937d90ef2bf0728c)
in Release with GGML CPU and Metal. Its last reported iteration of five took
414 ms on CPU (228 ms encoder, 186 ms decoder) and 217 ms on Metal (40 ms encoder,
177 ms decoder), using the same weights/audio and `--threads 12`. These are its
internal stage timers, whereas Speechworks uses call wall-clock time. Native
Speechworks CPU is slower in this comparison; the accelerated implementation is
in the same range as the native reference. This is not a PyTorch comparison.

## What was optimized

- The causal frontend and encoder retain left context and pending right context
  per layer. Already emitted rows and decoder cross-attention projections are
  reused. The autoregressive text decoder restarts for each revised hypothesis;
  its old self-attention cache cannot be reused when audio context changes.
- Native cooperative matrix multiplication is enabled only after adapter feature,
  subgroup-layout and numerical probes. Metal uses 8×8 matrices; eligible NVIDIA
  adapters use 16×16. Other adapters keep portable kernels. Browser WebGPU retains
  portable WGSL because the wgpu matrix extension is native-only.
- Four decoder steps share each readback. This alone made little difference
  (roughly 508 → 503 ms for the whole clip), so it is not credited with the larger
  improvement. Native matrix kernels lowered a typical encoder/cross-projection
  update from about 32 ms to 14 ms. Decoder work now dominates growing previews.
- Offline GPU encoding uses four-second blocks for better matrix utilization;
  live updates retain one-second maximum processing blocks. Together with matrix
  kernels this reduced the initial roughly 508 ms offline result to 190 ms.
  The controlled final comparison is portable 312 ms versus accelerated 190 ms.
- Activations and bindings are reused on GPU, decoder generation budgets are
  bounded, and CPU frontend buffers are released before encoder blocks run.
  Hash-verified model caching uses 64 MiB pieces: the tested Chrome context rejected
  a single Medium-sized cache entry despite sufficient origin quota. Reuse with
  model network requests blocked now passes.

The final native profile measured 79 ms encoder/cross projections plus 118 ms
decoder for the first offline call. A typical live update spent 14 ms encoding
and 48 ms decoding; finalization spent 14 ms encoding and 111 ms decoding.
Browser WASM linear memory grew to about 820–828 MB in Medium acceptance, including
weights, retained source bytes, allocator high-water marks and scratch. This is
not total browser/GPU memory. The optional shared-memory module has a 2 GiB maximum,
and concurrent resident Small/Medium models passed an actual browser test.

## Correctness and behavior checked

- Exact reference text for Medium and Small on JFK and a second FLEURS speaker,
  on native CPU and Metal. Incremental and one-shot encoder rows agree within
  0.01 maximum absolute difference over deliberately awkward chunk boundaries.
- Final text agrees for irregular chunks, interleaved independent phrases and a
  22-second concatenated recording. Retained encoder history lengths are bounded.
  Invalid PCM, excessive input and reuse after finalization are rejected.
- Browser plain/relaxed/threaded CPU and hardware WebGPU produce the expected text.
  Revised final boundaries, recording restart and actual GPU destruction with
  whole-phrase CPU recovery pass. Original Tiny/Base CPU–GPU regressions pass too.
- The actual demo passes microphone previews, final flush, edited-text preservation,
  cancellation, model switching, failed downloads, cache reuse and CPU fallback.
  Desktop and 320–390 px layouts were checked. No audio upload or external runtime
  request occurred.
- Public client tests cover pool creation/destruction, two resident models,
  non-isolated fallback, partial startup failure, a failed child and a stalled
  inference watchdog. Browser worker targets are confirmed destroyed.

These two recordings establish implementation regressions, not a general word
error rate. RTX 3090, Windows/DX12, Linux/Vulkan, mobile devices and sustained
thermal behavior still need device-specific runs. The NVIDIA fast path is capability
gated; no RTX performance measurement is claimed. Default session phrases are
bounded to eight seconds. Native audio and working-memory limits are independent;
larger whole-clip calls can hit the memory budget before the sample limit.

## Reproduce

```sh
tools/install-web.sh --dest /tmp/speechworks-runtime --model-dir /tmp/speechworks-models
node tools/prepare-streaming-fixtures.mjs /tmp/speechworks-models
SPEECHWORKS_TEST_STREAMING_MODELS=/tmp/speechworks-models cargo test --release -p speechworks-inference --test streaming_audio -- --ignored --nocapture --test-threads=1
SPEECHWORKS_TEST_STREAMING_MODELS=/tmp/speechworks-models cargo test --release -p speechworks-inference chunk_boundaries_preserve_encoder_values_and_bound_history -- --ignored --nocapture
cargo run --release -p speechworks-inference --example streaming_benchmark -- /tmp/speechworks-models/moonshine-streaming-medium-Q8_0.gguf /tmp/speechworks-models/jfk.wav gpu 320 3
node tests/browser-streaming.mjs --root /tmp/speechworks-runtime --fixture /tmp/speechworks-models/jfk.wav --backend gpu
```

Set `SPEECHWORKS_GPU_KERNELS=portable` for the native control or
`SPEECHWORKS_PROFILE=1` for stage timings. Select `cpu` for native CPU;
browser `--backend cpu --variant relaxed` selects the guarded serial path.
For `--variant threaded --threads 4`, first install with
`SPEECHWORKS_WASM_THREADS=1` and the [threaded compiler prerequisites](wasm-threads.md).
The reference command is `transcribe-cli -q -m MODEL --backend metal --threads 12 -r 5 AUDIO`.

The JFK fixture is the public-domain recording pinned from whisper.cpp. The second
fixture is **Google FLEURS**, `en_us` test id **1904**, **CC-BY-4.0**, unchanged from
[the pinned reference sample](https://github.com/handy-computer/transcribe.cpp/blob/adfc50210a3f026a655a3e19937d90ef2bf0728c/samples/README.md).
See the [dataset](https://huggingface.co/datasets/google/fleurs) and
[license](https://creativecommons.org/licenses/by/4.0/). The preparation tool pins
and verifies both audio hashes; audio and weights are not committed here.
