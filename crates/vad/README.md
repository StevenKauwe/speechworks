# speechworks-vad

Pure Rust Silero voice activity detection with embedded MIT-licensed weights.
Feed normalized mono PCM in 512-sample frames at 16 kHz, or 256-sample frames
at 8 kHz. One instance owns the streaming audio history and LSTM state.

```rust
use speechworks_vad::{SampleRate, SileroVad};
let mut vad = SileroVad::new(SampleRate::Hz16000)?;
let probability = vad.process(&[0.0; 512])?;
vad.reset();
# Ok::<(), speechworks_vad::Error>(())
```

There is no ONNX, Python, C/C++, platform framework, network, or threading
requirement. See [UPSTREAM.md](UPSTREAM.md) for provenance and the streaming
context correction to the upstream implementation.

The workspace's external real-speech test runs with:

```sh
SPEECHWORKS_TEST_AUDIO=/path/to/jfk.wav cargo test -p speechworks-web --test real_vad -- --ignored
```
