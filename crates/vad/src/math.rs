//! Pure Rust matrix multiplication; no platform frameworks or C runtime.

/// C = A @ B, row-major, beta = 0 (overwrite).
/// a: [m, k], b: [k, n], c: [m, n].
pub(crate) fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    backend::sgemm(m, n, k, a, b, c)
}

mod backend {
    pub(super) fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        // SAFETY: extents validated by the caller; strides describe row-major
        // layouts with unit column stride. beta = 0.0 overwrites c.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
    }

    #[test]
    fn sgemm_matches_naive() {
        // deterministic pseudo-random inputs
        let mut seed = 0x12345678u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        for (m, n, k) in [
            (1, 1, 1),
            (3, 5, 7),
            (128, 3, 387),
            (512, 1, 256),
            (64, 2, 384),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| rand()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rand()).collect();
            let mut c = vec![0.0f32; m * n];
            let mut c_ref = vec![0.0f32; m * n];
            sgemm(m, n, k, &a, &b, &mut c);
            naive(m, n, k, &a, &b, &mut c_ref);
            for (x, y) in c.iter().zip(&c_ref) {
                assert!((x - y).abs() <= 1e-4, "m={m} n={n} k={k}: {x} vs {y}");
            }
        }
    }
}
