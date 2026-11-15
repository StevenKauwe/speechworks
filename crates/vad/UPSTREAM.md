# Upstream provenance

The Rust kernels and embedded Silero v6.2 weights derive from MIT-licensed
[`silero-vad-pure` 0.1.1](https://crates.io/crates/silero-vad-pure/0.1.1),
repository commit `daa349430048ed07c35c1b62ccb4733c62306da2` from
[huanglizhuo/silero-vad-pure](https://github.com/huanglizhuo/silero-vad-pure).
The upstream copyright and full license are retained in `LICENSE-UPSTREAM`.
The model's MIT license is retained in `LICENSE-MODEL`.

The embedded `assets/silero_vad.bin` is copied unchanged from that published crate.
Its SHA-256 is
`ecae310f3cc59ce9a55fe6da7e87c778e86f1d887fdb3b6ba2e367ec181a7132`.
This asset contains both 8 kHz and 16 kHz model weights; no download is performed
at runtime. The upstream crate documents its conversion from
[snakers4/silero-vad](https://github.com/snakers4/silero-vad).

Speechworks adds the streaming input context required by the official
[Silero wrapper](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py):
64 previous samples at 16 kHz, or 32 at 8 kHz, prepended before the STFT.
Both audio history and LSTM state reset together. Without this context, 0.1.1
detected zero voiced frames in the public-domain JFK acceptance fixture.
The corrected detector identified 241 of its 344 frames with probability above
0.3. The regression checks actual recorded speech, independently of kernel
comparison tests. This observation is not a claim of complete ONNX parity.

The optional Accelerate dependency and framework calls were removed. The crate
uses Rust NEON kernels on AArch64 and pure Rust `matrixmultiply` elsewhere,
including WASM. Working buffers retain their allocation between frames.
