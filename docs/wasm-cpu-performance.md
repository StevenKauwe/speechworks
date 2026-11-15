# Browser CPU performance handoff

Measured November 15, 2026. Local candidate branch: `perf/wasm-cpu-simd`.
Baseline: `e66a38c` (only a documentation change from `fe00377`). No push or merge.

## Result

The slow CPU fallback is **not specific to Omarchy**. On the M2 Max, the original
browser build was 4.79× / 4.42× slower than one-thread native for tiny / base.
Targeted plain-WASM-SIMD kernels materially reduce that gap without additional
inference threads, SharedArrayBuffer, cross-origin isolation, or relaxed SIMD.

**11-second inference medians, milliseconds:**

| Machine | Browser baseline tiny / base | Optimized tiny / base | Native one-thread tiny / base |
| --- | ---: | ---: | ---: |
| M2 Max, Chrome 157 | 1113.7 / 2281.9 | **382.0 / 841.5** | 232.398 / 516.832 |
| i7-11700K, Chromium 155, initial A/B | 1459.3 / 2943.5 | **562.9 / 1274.6** | 189.514 / 345.498 |

Mac speedups: **2.92× / 2.71×**. Omarchy initial speedups: **2.59× / 2.31×**.
Do not report only that best Omarchy pair: the reverse-order pair measured
baseline **1438.0 / 3004.2** and optimized **789.5 / 1821.9** (1.82× / 1.65×).
A third optimized run returned to **585.4 / 1234.5**. All three retained identical
text. The cause of this fresh-browser variability is **not established**;
JIT tiering, scheduling, and host conditions are hypotheses, not findings.
The native recheck remained near its earlier values. Load averages around 1–2
on 16 logical CPUs do not prove an otherwise idle machine.

Mac prefix medians before → after:

| Model | 1 second | 3 seconds | 5 seconds | 11 seconds |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 151.7 → 37.4 | 294.0 → 93.2 | 414.9 → 152.0 | 1113.7 → 382.0 |
| Base | 257.1 → 80.1 | 626.7 → 215.9 | 835.2 → 346.0 | 2281.9 → 841.5 |

## Method and controls

- Same public JFK WAV: 176,000 mono samples, 16 kHz, signed PCM16 converted to
  float by division by 32768. SHA-256:
  `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
  Same pinned tiny/base Q8_0 models from `models.lock.json` throughout.
- One full decode warmup, then **1 / 3 / 5 / 11 / 3 second** prefixes, **five
  repetitions each**. Fresh isolated headless browser per invocation; new model
  and context per model. Timings bracket `transcribe`, excluding file fetching,
  model loading, microphone capture, VAD, and browser startup. Direct API timings
  isolate inference; the actual worker/capture pipeline is tested separately.
- Rust 1.99.1, release optimization and thin LTO; WASM flags
  `+simd128,-relaxed-simd`, wasm-bindgen 0.2.129. Both machines executed the
  **same baseline and optimized WASM bytes**, not independently compiled variants.
- Both browser builds use one inference thread. Mac reports 12 hardware threads;
  Omarchy reports 16. `crossOriginIsolated` was false; Omarchy had no SAB.
- Mac native control used the same source with only `cpu::threads()` forced to 1
  in the isolated benchmark build, plus the archived benchmark example. That
  temporary change is **not in the candidate**. Mac had no CPU affinity control.
  Omarchy native recheck used the existing `fe00377` binary, `taskset` CPU 0,
  and three repeats per prefix. Native NEON / AVX2+FMA+AVX512-VNNI differ from the
  browser kernels: these ratios are not a measurement of generic “WASM overhead.”

## What the measurements justified

A separate Chrome sampling profile over three full decodes attributed roughly
30–31% of baseline self time to `dot`, 24–25% to the GEMV caller, 24–28% to
GEMM tiles, and 10–11% to software `fmaf`. Sampling was not enabled in latency runs.
Software FMA was a contributor, **not the sole cause**.

The retained changes are:

1. Explicit SIMD128 dot products with 16 accumulation lanes and pairwise reduction.
2. Four Q8 rows processed together, widening and scaling weights in SIMD registers,
   sharing activation loads, and avoiding a temporary dequantized vector per row.
3. Accurate `WASM SIMD128` device reporting and actual-WASM numerical/bounds tests
   wired into browser CI.

Mac ablations: SIMD dot alone gave 655.3 / 1359.0 ms (three repeats); adding Q8
register decoding supplied the remaining improvement. Explicit 6×16 GEMM gave
389.6 / 854.5 ms versus 387.7 / 852.4 with the original GEMM, while 6×8 was slower
at 449.1 / 1024.6. **Both GEMM changes were removed.** Native kernels, GPU kernels,
model selection, and worker scheduling are unchanged. Remaining final-profile
GEMM self time is about 66% / 74% on Mac and 71% / 76% on Omarchy.

## Correctness and limits

- Exact native / old-browser / optimized-browser transcript equality on every
  JFK prefix and full clip; exact repeat stability on both machines.
- **996 arithmetic assertions plus five bounds traps** passed in Node and actual
  Chrome/Chromium WASM, covering tails, Q8 short row groups, signed widening,
  checked-size overflow, and truncated inputs. The dot oracle also checks f64 error.
- Mac full CPU acceptance passed actual auto fallback, worker previews/finals,
  cancellation and reuse, cache/hash rejection, and fake-microphone AudioWorklet
  capture. No external requests or unhandled errors. Omarchy direct/fallback and
  kernel suites passed in every timed run.
- Workspace Rust tests: 37 passed, five opt-in hardware/fixture tests remained
  ignored. Scalar and SIMD WASM checks, 14 JavaScript tests, formatting, and diff
  checks passed. Clippy was unavailable on this Mac. Remote CI was not run.

**Numerical caveat:** plain SIMD uses separate float multiply/add operations.
Rounding differs from the previous browser's emulated fused arithmetic as well
as native FMA. These arithmetic checks and exact JFK text parity do **not** prove
unchanged recognition accuracy on a broader speech corpus. Validate representative
voices, noise, and near-tie decoding before claiming general accuracy parity.
Firefox, Safari, mobile, thermal endurance, and long-running sessions were not tested.

## TabICL and GPU comparison caveats

[TabICL at `6290de3`](https://github.com/StevenKauwe/tabicl-anywhere/blob/6290de3664520becce649c1ca8cf94f0c26ecbcf/wasm/src/lib.rs)
uses independent **single-threaded** WASM instances in separate workers, each
with weights, distributing independent ensemble members. It is not shared-memory
WASM threading. Its SIMD/activation-reuse pattern is applicable; copying its
worker pool does not parallelize one Moonshine utterance and would duplicate
weights. Intra-inference parallelism would be a separate design.

Earlier Omarchy measurements were browser CPU 1463.1 / 3001.1 ms, native one-thread
186.770 / 345.666, native 16-thread 105.150 / 172.729, and hardware RTX 3090 WebGPU
73.3 / 72.5, all three warm samples. GPU was **not rerun** here. GPU versus CPU
compares different compute hardware and kernels, not browser overhead.

## Reproduce and inspect

```sh
node tests/wasm-kernels.mjs --browser
tools/build-web.sh
# Install/link the pinned models and models.json in the runtime directory first.
node tests/browser-smoke.mjs --cpu-only --root dist --fixture /path/to/jfk.wav --repeats 5
node tests/profile-cpu.mjs dist test-results/cpu-profile /path/to/jfk.wav
```

`PLAYWRIGHT_MODULE` and `CHROME_EXECUTABLE` select existing installations.
The benchmark kit retains both runtime bundles, raw per-sample JSON, sampled
profiles, the native harness/control patch, model/artifact hashes, and all
Omarchy repetitions. Next investigation: explain Omarchy run-to-run variance
and profile the remaining GEMM cost. Do not infer those causes from this patch.
