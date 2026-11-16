# speechworks-inference

Rust Moonshine inference over model bytes, with CPU and asynchronous GPU execution. Native Metal/Vulkan/DX12 and browser WebGPU use the same portable WGSL kernels. No Python, C++ inference library, ONNX Runtime, microphone, network client, or model download is included.

The model's GGUF metadata determines its dimensions and tokenizer. Moonshine tiny/base share one implementation; Moonshine Streaming Small/Medium share a separate implementation with independently sized encoder and decoder. Model weights are supplied by the application, which should verify their pinned hashes before calling this crate. See [UPSTREAM.md](UPSTREAM.md) and [LICENSE-WAVO](LICENSE-WAVO) for the MIT-licensed reference code and kernels.

```rust,no_run
# async fn example(bytes: Vec<u8>, pcm: &[f32]) -> Result<(), speechworks_inference::Error> {
use speechworks_inference::{Backend, Model};
let model = Model::from_bytes(bytes, Backend::Auto).await?;
let transcript = model.transcribe(pcm).await?;
println!("{}", transcript.text);
println!("{}", model.device()?);
# Ok(())
# }
```

Supply finite mono float PCM in `[-1, 1]` at 16 kHz. The default per-call limit is 30 seconds and the conservative scratch-memory budget is 256 MiB. `LoadOptions` configures both independently of model choice. Weights and retained GGUF source bytes are additional memory. Split longer audio into phrases before inference. CPU speech detection and session orchestration belong in the `speechworks` layer.

`Backend::Auto` attempts the GPU, then selects CPU if initialization fails. If GPU execution fails, it retries the complete phrase on CPU. `fallback_reason()` records that decision. `Backend::Gpu` requires GPU success and reports an error instead of hiding CPU use. Run one transcription at a time on each `Model`; a concurrent call returns `Error::Busy`. Native callers can use an async executor or `pollster::block_on` on a background thread. Browser callers must await the future in a worker.

GPU execution uses reusable activation buffers and cached uniform parameter buffers. There are no native-only push constants or blocking waits in the WASM path. Native optimized SIMD CPU code retains its upstream safety comments. WASM runs CPU inference on one worker without requiring shared memory or cross-origin isolation. With `simd128`, dot products and Q8 decoding use explicit plain-SIMD kernels; scalar builds retain the portable path. Plain SIMD uses separate multiply/add rounding rather than emulated fused arithmetic. See [CPU performance and validation](../../docs/wasm-cpu-performance.md) for measured gains and accuracy limits.

`Model::transcribe_chunk(&mut StreamState, new_pcm, final_chunk)` accepts incremental
audio. Retain one state per phrase; partial text remains tentative. Its causal
frontend and bounded per-layer histories emit only encoder rows whose right context
is complete. Cross-attention K/V is appended once per emitted row. The final call
zero-pads an incomplete 80-sample frame and flushes look-ahead. A state belongs to
one model and cannot be reused after finalization. Keep states separate for
interleaved phrases. CPU recovery replays the retained PCM, including all prior chunks.

On native adapters, capability checks and numerical probes can enable 8×8 Metal
or 16×16 NVIDIA cooperative matrix kernels. `Device::kernels` reports the selected
path. Set `SPEECHWORKS_GPU_KERNELS=portable` for a native comparison. Browser builds
retain portable WebGPU kernels: wgpu's cooperative-matrix extension is native-only.
No CUDA toolkit or C++ runtime is required by Speechworks.

## Validation

`cargo test -p speechworks-inference` runs numerical kernel, parser, tokenizer and invalid-model tests. `cargo check -p speechworks-inference --target wasm32-unknown-unknown` validates browser compilation.

Real-model tests are explicit and fail if their fixtures or hardware are unavailable. Put the pinned `moonshine-tiny-Q8_0.gguf`, `moonshine-base-Q8_0.gguf` and the 16 kHz mono public [JFK fixture](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav) in a directory, then run:

```sh
SPEECHWORKS_TEST_MODELS=/path/to/fixtures cargo test --release -p speechworks-inference --test real_audio -- --ignored --nocapture
SPEECHWORKS_TEST_MODELS=/path/to/fixtures cargo test --release -p speechworks-inference --lib device_loss_replays_current_phrase_on_cpu -- --ignored --nocapture
```

These check reference transcripts, tiny/base CPU–GPU equality while phrases grow and shrink, input and memory limits, and replay after the GPU device is actually destroyed. GPU tests require access to a hardware adapter. WebGPU execution must also be tested in the browser by the consumer's browser harness; compilation alone does not establish browser behavior.

For Streaming Medium/Small, put both pinned `moonshine-streaming-*-Q8_0.gguf`
files, `jfk.wav`, and `fleurs-en.wav` in `SPEECHWORKS_TEST_STREAMING_MODELS`:

```sh
SPEECHWORKS_TEST_STREAMING_MODELS=/path/to/fixtures cargo test --release -p speechworks-inference --test streaming_audio -- --ignored --nocapture
SPEECHWORKS_TEST_STREAMING_MODELS=/path/to/fixtures cargo test --release -p speechworks-inference chunk_boundaries_preserve_encoder_values_and_bound_history -- --ignored --nocapture
cargo run --release -p speechworks-inference --example streaming_benchmark -- MODEL.gguf AUDIO.wav gpu 320 3
```

Run `node tools/prepare-streaming-fixtures.mjs /path/to/fixtures` after the installer
to prepare verified models and audio. See [the streaming measurements](../../docs/moonshine-streaming.md) for fixture attribution and results.

The benchmark compares one-shot text with streaming final text and reports each
preview's compute time, p50/p95 latency and finalization. `SPEECHWORKS_PROFILE=1`
adds native frontend/encoder/decoder timing spans. `tests/browser-streaming.mjs`
checks the same model through real WASM/WebGPU, including a revised phrase boundary,
restart and GPU-device-loss replay. Use `--backend cpu` for the CPU fallback.
