# Further browser CPU optimization

This follow-up keeps the same Moonshine tiny/base Q8_0 models and 11-second JFK
fixture as [the original investigation](wasm-cpu-performance.md). It does not
substitute a smaller model, shorten the audio or include GPU work.

## Result on M2 Max

Chrome 157, macOS, reported hardware concurrency 12. Public `createDictation`
CPU API, one full warmup then 1/3/5/11/3-second prefixes with five repetitions
per prefix. Each setting uses a fresh browser context; model download, checksum
verification, module/pool initialization and construction precede warm timing.

| Final runtime setting | Tiny, warm 11s median | Base, warm 11s median |
|---|---:|---:|
| Auto (selected four workers) | 122.2 ms | **236.3 ms** |
| Explicit one (guarded relaxed SIMD serial) | 287.5 ms | 606.0 ms |
| Explicit two | 178.4 ms | 357.6 ms |
| Explicit four, later in the same run | 134.4 ms | **257.5 ms** |

Both four-worker runs beat the requested 300 ms base target. The difference
between 236 and 258 ms is retained, not hidden by reporting only the best run.
This is a short, local, headless-browser test, not a sustained-load/mobile or
arbitrary-hardware guarantee. Automatic sizing caps at four and half the reported
logical processors. It does not measure idle cores or dynamically resize under
pressure. The UI's 10 ms heartbeat had maxima of 12.0–19.8 ms across these runs;
that is evidence of responsiveness during these tests, not a real-time guarantee.

[Raw samples, hashes and timing caveats](../benchmarks/2026-11-15-m2max-cpu.json)
and `tests/browser-cpu-benchmark.mjs` make the final public-API run reproducible.
Preload was 253–273 ms from the local server. The tiny model's first call includes
lazy loading; do not compare that value with the already-preloaded base model's
first call as a pure first-inference ratio.

Separate direct-model serial controls measured the final plain-SIMD path at
368.4/810.4 ms (tiny/base) and guarded relaxed SIMD at 287.1/595.8 ms. An earlier
matched run of the new kernels versus `c6f1998` measured 289.6/598.1 versus
373.3/827.0 ms. Thus non-isolated applications still improve, but **do not get the
236–258 ms threaded result**. Historical pre-optimization Mac browser timings
were 1113.7/2281.9 ms; they are not simultaneous controls for the final run.

A separate lower-level prototype sweep used the identical shared binary with
one worker (287.4/594.8 ms) to distinguish threading from compiler differences.
Four workers measured 121.8/233.9 ms and eight 96.5/208.1 ms. The public API keeps
the more conservative four-worker automatic cap. These prototype measurements
are not substituted for the integrated-client results above.

## Omarchy retest

The **identical final Mac-built WASM files**, verified by SHA-256, were retested
on the i7-11700K Omarchy machine. Same pinned models, JFK PCM, local isolated
browser, public API, full warmup and five repeats per 1/3/5/11/3-second prefix:

| Setting | Tiny | Base |
|---|---:|---:|
| Auto (four workers) | 201.4 ms | 390.5 ms |
| One worker | 490.2 ms | 1122.0 ms |
| Two workers | 276.5 ms | 595.3 ms |
| Four workers, later run | 204.7 ms | 376.8 ms |

[Raw Omarchy samples](../benchmarks/2026-11-15-omarchy-cpu.json) retain the
host/browser details and caveats. All prefix text matched; the same nine lifecycle
cases passed, with worker destruction verified. Heartbeat maxima were 10.1–15.7 ms.
The historical stock browser baseline was 1463/3001 ms, but it was not measured
contemporaneously with this run. The 300 ms target was for the Mac: it is **not**
met by base on this Omarchy CPU. The old native one-thread 187/346 ms and
16-thread 105/173 ms are also historical controls, not rerun here.

## What changed, and why

Profiling identified GEMM as the dominant serial cost, not JavaScript glue.
The final serial base profile still spends about 52% of sampled self time in the
guarded fused GEMM, 13% in the ordinary GEMM tile and 16% in Q8 decoder rows.
Profiling is separate from timed runs. The original observation of scalar `fmaf`
is not evidence that it caused the whole browser/native gap.

- Plain SIMD128 remains available at `speechworks_web.js`.
- `relaxed/speechworks_web.js` uses a guarded fused linear GEMM kernel with four
  rows per tile. The loader validates relaxed-SIMD support before selection.
- Q8 decoding spells out four independent row updates so LLVM keeps sixteen
  vector accumulators in WASM locals rather than repeatedly spilling a scratch
  array to linear memory. Scaling, accumulation order, reduction and tail
  behavior are preserved; no persistent dequantized model copy is added.
- An optional separate shared-memory runtime uses a persistent Rayon worker
  pool for intra-inference CPU work. It is never substituted for the compatible
  serial binary without capability checks and independent worker ownership.

The fused kernel is limited to finite, bounded, half-rounded linear inputs.
Binary16 products are exactly representable in float32, so fused or unfused
relaxed madd produces the same result as separate multiply/add in this domain.
Runtime guards reject nonfinite/overrange operands and retain the old path.
Generic float32 attention products remain on plain SIMD. This proof does not
apply to arbitrary float32 values or to the earlier first-pass dot-product
change. Plain and relaxed numerical tests cover tails, cancellation, signed
zero, subnormals, extrema, guard boundaries and invalid slice bounds.

## Deployment and correctness

See [the threading guide](wasm-threads.md) for the pinned compiler, optional
installer flag, actual capability checks, same-origin workers and cleanup.
No deployed headers or CSP are changed. A non-isolated page stays on the fast
serial path and reports why; a failed pool startup destroys its workers before
initializing a separate serial module. Models and sessions wait for selection.

Validation includes 85 JavaScript unit tests, 996 plain/5425 relaxed numerical
assertions plus 5/8 bounds traps in Node and Chrome, real browser CPU/VAD/client
acceptance, and nine real-browser lifecycle cases. These include partial/mixed
startup, destroy during inference, late child failure, watchdog and recreation.
Every instrumented worker had a matching browser destruction event and no worker
target remained after cleanup. CI now includes both numerical variants and an
optional threaded build/lifecycle job; local passes do not assert remote CI status.

JFK prefixes and repeats match exactly across thread counts. Tiny/base transcripts
also match the previous serial build on five independently selected LibriSpeech
test-clean clips, five speakers, 4.0–14.4 seconds (ten model/clip cases). Existing
reference recognition errors remain. This is a small regression set, not broad
speech-accuracy validation. Native and non-SIMD WASM checks remain separate from
browser feature tests.

The older Omarchy RTX 3090 WebGPU result (~73 ms) is historical, not rerun here,
and cannot be used as a JavaScript-versus-Rust or CPU language-overhead ratio.
