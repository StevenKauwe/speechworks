# Attribution

This crate derives its Moonshine model, GGUF reader, tokenizer, SIMD CPU code and portable WGSL kernels from [Wavo](https://github.com/vladkens/wavo), commit `06dfe6642db8a2f9159ad0f23719f23276431a84`, Copyright (c) vladkens, under the MIT license in `LICENSE-WAVO`.

Speechworks adds browser-safe asynchronous WebGPU operation, uniform parameters, in-memory model loading, bounded audio/work memory and CPU recovery. Model weights are separate artifacts under their own licenses; none are embedded in this crate.
