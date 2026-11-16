# Attribution

This crate derives its Moonshine model, GGUF reader, tokenizer, SIMD CPU code and portable WGSL kernels from [Wavo](https://github.com/vladkens/wavo), commit `06dfe6642db8a2f9159ad0f23719f23276431a84`, Copyright (c) vladkens, under the MIT license in `LICENSE-WAVO`.

Speechworks adds browser-safe asynchronous WebGPU operation, uniform parameters, in-memory model loading, bounded audio/work memory and CPU recovery. Model weights are separate artifacts under their own licenses; none are embedded in this crate.

Moonshine Streaming equations, tensor naming, strict local-attention windows,
adapter positions and generation budget follow handy-computer/transcribe.cpp
`adfc50210a3f026a655a3e19937d90ef2bf0728c` (MIT; full notice in
`../../licenses/transcribe-cpp.txt`). Its reference CLI is used for transcript
comparisons. Speechworks' CPU and WGSL implementations use bounded per-layer
history caches instead of re-encoding a sliding PCM window.

The native cooperative matrix GEMM is also derived from the pinned Wavo revision,
adapted from immediate parameters to uniform buffers. All six type/tile variants
must pass subgroup-layout and numerical probes before activation.
