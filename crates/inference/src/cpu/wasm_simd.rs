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
            let rows: [_; 4] = std::array::from_fn(|j| (4 * g + j.min(out.len() - 1)) * k);
            let zero = [f32x4_splat(0.0); 4];
            let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
            for i in (0..k).step_by(16) {
                let xs: [_; 4] =
                    std::array::from_fn(|c| v128_load(x.as_ptr().add(i + 4 * c).cast()));
                // Spell out the four updates: a dynamic row loop leaves the accumulators
                // in linear-memory scratch instead of SIMD locals. A short group repeats
                // its last valid row, but only the requested outputs are written.
                macro_rules! row {
                    ($r:literal, $acc:ident) => {{
                        let at = rows[$r] + i;
                        let packed = v128_load(q.as_ptr().add(at).cast());
                        let scale = f32x4_splat(*d.get_unchecked(at / 32));
                        $acc = q8_step(packed, scale, xs, $acc);
                    }};
                }
                row!(0, a0);
                row!(1, a1);
                row!(2, a2);
                row!(3, a3);
            }
            for (o, acc) in out.iter_mut().zip([a0, a1, a2, a3]) {
                *o = sum16(acc);
            }
        }
    }
}

#[inline(always)]
fn q8_step(packed: v128, scale: v128, xs: [v128; 4], mut acc: [v128; 4]) -> [v128; 4] {
    let lo = i16x8_extend_low_i8x16(packed);
    let hi = i16x8_extend_high_i8x16(packed);
    let ws = [
        i32x4_extend_low_i16x8(lo),
        i32x4_extend_high_i16x8(lo),
        i32x4_extend_low_i16x8(hi),
        i32x4_extend_high_i16x8(hi),
    ];
    for (c, acc) in acc.iter_mut().enumerate() {
        let w = f32x4_mul(f32x4_convert_i32x4(ws[c]), scale);
        *acc = f32x4_add(*acc, f32x4_mul(w, xs[c]));
    }
    acc
}

/// Conservative guards for the exact finite-half product fast path.
#[cfg(target_feature = "relaxed-simd")]
pub(super) fn half_bounded(values: &[f32]) -> bool {
    values.iter().all(|x| x.is_finite() && x.abs() <= 65504.0)
}

#[cfg(target_feature = "relaxed-simd")]
pub(super) fn q8_half_bounded(scales: &[f32]) -> bool {
    scales.iter().all(|d| d.is_finite() && d.abs() <= 511.75)
}

/// Rounded linear GEMM only, not the unrounded attention product. The caller must ensure
/// both operands are finite binary16 values. Their products are exact in f32, so either
/// permitted relaxed-madd lowering has the same rounding as separate multiply/add.
#[cfg(target_feature = "relaxed-simd")]
pub(super) fn rounded_gemm_tile<const R: usize>(
    a: [&[f32]; R],
    b: &[f32],
    k: usize,
) -> [[f32; 16]; R] {
    assert!(a.iter().all(|row| row.len() >= k));
    let n = k.checked_mul(16).expect("GEMM panel size overflow");
    let b = &b[..n];
    let mut acc = [[f32x4_splat(0.0); 4]; R];
    // SAFETY: all input rows and the complete k-by-16 panel were checked above. SIMD and
    // relaxed SIMD are compile-time requirements; this kernel lives in a separate artifact.
    unsafe {
        for i in 0..k {
            let bv: [_; 4] =
                std::array::from_fn(|c| v128_load(b.as_ptr().add(i * 16 + c * 4).cast()));
            for (r, acc) in acc.iter_mut().enumerate() {
                let av = f32x4_splat(*a[r].get_unchecked(i));
                for (acc, bv) in acc.iter_mut().zip(bv) {
                    *acc = f32x4_relaxed_madd(av, bv, *acc);
                }
            }
        }
        let mut out = [[0.0; 16]; R];
        for (row, acc) in out.iter_mut().zip(acc) {
            for (c, acc) in acc.into_iter().enumerate() {
                v128_store(row.as_mut_ptr().add(c * 4).cast(), acc);
            }
        }
        out
    }
}

// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
