# speechworks-inference

Rust Moonshine inference over model bytes, with CPU and asynchronous GPU execution. Native Metal/Vulkan/DX12 and browser WebGPU use the same portable WGSL kernels. No Python, C++ inference library, ONNX Runtime, microphone, network client, or model download is included.

The model's GGUF metadata determines its dimensions and tokenizer. Moonshine tiny and base share an implementation. Model weights are supplied by the application, which should verify their pinned hashes before calling this crate. See [UPSTREAM.md](UPSTREAM.md) and [LICENSE-WAVO](LICENSE-WAVO) for the MIT-licensed Wavo source this implementation derives from.

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

## Validation

`cargo test -p speechworks-inference` runs numerical kernel, parser, tokenizer and invalid-model tests. `cargo check -p speechworks-inference --target wasm32-unknown-unknown` validates browser compilation.

Real-model tests are explicit and fail if their fixtures or hardware are unavailable. Put the pinned `moonshine-tiny-Q8_0.gguf`, `moonshine-base-Q8_0.gguf` and the 16 kHz mono public [JFK fixture](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav) in a directory, then run:

```sh
SPEECHWORKS_TEST_MODELS=/path/to/fixtures cargo test --release -p speechworks-inference --test real_audio -- --ignored --nocapture
SPEECHWORKS_TEST_MODELS=/path/to/fixtures cargo test --release -p speechworks-inference --lib device_loss_replays_current_phrase_on_cpu -- --ignored --nocapture
```

These check reference transcripts, tiny/base CPU–GPU equality while phrases grow and shrink, input and memory limits, and replay after the GPU device is actually destroyed. GPU tests require access to a hardware adapter. WebGPU execution must also be tested in the browser by the consumer's browser harness; compilation alone does not establish browser behavior.
