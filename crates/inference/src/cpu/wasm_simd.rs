//! Ordinary WASM SIMD128 kernels. No relaxed SIMD, shared memory, or worker pool is required.
//!
//! WASM SIMD128 has no fused multiply-add. These kernels intentionally use separate f32
//! multiply/add operations (as the portable GEMM does), retaining the dot product's 16 lanes
//! and pairwise reduction. Results need not be bit-identical to native fused arithmetic.
//! Q8 rows are decoded in registers instead of allocating a dequantized vector per row.

use core::arch::wasm32::*;

#[inline]
fn sum16([a0, a1, a2, a3]: [v128; 4]) -> f32 {
    let v = f32x4_add(f32x4_add(a0, a2), f32x4_add(a1, a3));
    (f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<2>(v))
        + (f32x4_extract_lane::<1>(v) + f32x4_extract_lane::<3>(v))
}

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let aligned = n / 16 * 16;
    let mut acc = [f32x4_splat(0.0); 4];
    // SAFETY: every vector load is four floats within the complete 16-element chunks of
    // both slices. SIMD128 is a compile-time prerequisite for this module.
    unsafe {
        for i in (0..aligned).step_by(16) {
            for (c, acc) in acc.iter_mut().enumerate() {
                let x = v128_load(a.as_ptr().add(i + 4 * c).cast());
                let y = v128_load(b.as_ptr().add(i + 4 * c).cast());
                *acc = f32x4_add(*acc, f32x4_mul(x, y));
            }
        }
    }
    sum16(acc)
        + a[aligned..n]
            .iter()
            .zip(&b[aligned..n])
            .map(|(x, y)| x * y)
            .sum::<f32>()
}

pub(super) fn q8_rows(q: &[i8], d: &[f32], k: usize, x: &[f32], out: &mut [f32]) {
    assert_eq!(k % 32, 0);
    let total = out.len().checked_mul(k).expect("Q8 row size overflow");
    let (q, d, x) = (&q[..total], &d[..total / 32], &x[..k]);
    // SAFETY: k is a multiple of 32, so each 16-byte Q8 load and four-float activation
    // load is wholly inside its checked row. The scale index belongs to that row's block.
    unsafe {
        for (g, out) in out.chunks_mut(4).enumerate() {
            let mut acc = [[f32x4_splat(0.0); 4]; 4];
            for i in (0..k).step_by(16) {
                let xs: [_; 4] =
                    std::array::from_fn(|c| v128_load(x.as_ptr().add(i + 4 * c).cast()));
                for (j, acc) in acc.iter_mut().take(out.len()).enumerate() {
                    let at = (4 * g + j) * k + i;
                    let packed = v128_load(q.as_ptr().add(at).cast());
                    let lo = i16x8_extend_low_i8x16(packed);
                    let hi = i16x8_extend_high_i8x16(packed);
                    let ws = [
                        i32x4_extend_low_i16x8(lo),
                        i32x4_extend_high_i16x8(lo),
                        i32x4_extend_low_i16x8(hi),
                        i32x4_extend_high_i16x8(hi),
                    ];
                    let scale = f32x4_splat(d[at / 32]);
                    for (c, acc) in acc.iter_mut().enumerate() {
                        let w = f32x4_mul(f32x4_convert_i32x4(ws[c]), scale);
                        *acc = f32x4_add(*acc, f32x4_mul(w, xs[c]));
                    }
                }
            }
            for (o, acc) in out.iter_mut().zip(acc) {
                *o = sum16(acc);
            }
        }
    }
}

// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
