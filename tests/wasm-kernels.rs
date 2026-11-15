// Standalone SIMD oracle tests: compile and execute the actual WASM kernels, not native paths.
#[path = "../crates/inference/src/cpu/wasm_simd.rs"]
mod simd;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as i32 as f32) / 134217728.0
        })
        .collect()
}

fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let end = n / 16 * 16;
    let mut acc = [0.0f32; 16];
    for i in (0..end).step_by(16) {
        for j in 0..16 {
            acc[j] += a[i + j] * b[i + j];
        }
    }
    for n in [8, 4, 2, 1] {
        for j in 0..n {
            acc[j] += acc[j + n];
        }
    }
    acc[0]
        + a[end..n]
            .iter()
            .zip(&b[end..n])
            .map(|(x, y)| x * y)
            .sum::<f32>()
}

#[unsafe(no_mangle)]
pub extern "C" fn run_tests() -> u32 {
    let mut assertions = 0;
    for seed in 0..8 {
        for n in [
            0, 1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 333, 1024,
        ] {
            let a = values(n, seed + 1);
            let b = values(n, seed + 81);
            let got = simd::dot(&a, &b);
            assert_eq!(got.to_bits(), reference_dot(&a, &b).to_bits());
            let want: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
            let scale: f64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| (x as f64 * y as f64).abs())
                .sum();
            let tolerance = (n / 16 + 5) as f64 * f32::EPSILON as f64 * scale;
            assert!((got as f64 - want).abs() <= tolerance.max(1e-30));
            assertions += 2;
        }
    }
    for k in [0, 32, 64, 96, 256, 1024] {
        for rows in [0, 1, 2, 3, 4, 5, 7, 8, 70] {
            let offset = 3;
            let q: Vec<i8> = (0..(rows + offset) * k).map(|i| (i * 7919) as i8).collect();
            let d: Vec<f32> = (0..(rows + offset) * k / 32)
                .map(|i| (i % 7 + 1) as f32 / 1024.0)
                .collect();
            let x = values(k, 41);
            let mut out = vec![12345.0; rows + 2];
            simd::q8_rows(
                &q[offset * k..],
                &d[offset * k / 32..],
                k,
                &x,
                &mut out[1..rows + 1],
            );
            assert_eq!(out[0], 12345.0);
            assert_eq!(out[rows + 1], 12345.0);
            assertions += 2;
            for r in 0..rows {
                let w: Vec<f32> = (0..k)
                    .map(|i| q[(offset + r) * k + i] as f32 * d[((offset + r) * k + i) / 32])
                    .collect();
                assert_eq!(out[r + 1].to_bits(), reference_dot(&w, &x).to_bits());
                assertions += 1;
            }
        }
    }
    #[cfg(target_feature = "relaxed-simd")]
    {
        for limit in [65504.0f32, 511.75] {
            let check = if limit == 65504.0 {
                simd::half_bounded
            } else {
                simd::q8_half_bounded
            };
            assert!(check(&[0.0, -0.0, limit, -limit]));
            let above = f32::from_bits(limit.to_bits() + 1);
            for invalid in [above, -above, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(!check(&[0.0, invalid]));
                assertions += 1;
            }
            assertions += 1;
        }
        assert!(!simd::q8_half_bounded(&[512.0, -512.0]));
        assertions += 1;
        // Every finite binary16 value has an exact f32 representation. Include both
        // signs, normal/subnormal values and large products, with no half dependency.
        let half = |bits: u16| {
            let e = ((bits >> 10) & 31) as i32;
            let frac = (bits & 1023) as f32;
            let value = if e == 0 {
                frac * (1.0 / 16_777_216.0)
            } else {
                (1.0 + frac / 1024.0) * f32::from_bits(((e - 15 + 127) as u32) << 23)
            };
            if bits & 32768 != 0 { -value } else { value }
        };
        for seed in 0..8 {
            for k in [0, 1, 3, 7, 32, 127, 416, 832] {
                let a: Vec<f32> = (0..4 * k)
                    .map(|i| {
                        half((((i * 7919 + seed * 313) % 31744) as u16) | (((i & 1) as u16) << 15))
                    })
                    .collect();
                let b: Vec<f32> = (0..16 * k)
                    .map(|i| {
                        half(
                            (((i * 3907 + seed * 727) % 31744) as u16)
                                | ((((i + 1) & 1) as u16) << 15),
                        )
                    })
                    .collect();
                let rows = std::array::from_fn(|r| &a[r * k..][..k]);
                let got = simd::rounded_gemm_tile::<4>(rows, &b, k);
                for r in 0..4 {
                    for c in 0..16 {
                        let want = (0..k).fold(0f32, |sum, i| sum + a[r * k + i] * b[i * 16 + c]);
                        assert_eq!(got[r][c].to_bits(), want.to_bits());
                        assertions += 1;
                    }
                }
            }
        }
    }
    #[cfg(target_feature = "relaxed-simd")]
    {
        let extremes = [
            0.0f32,
            -0.0,
            1.0 / 16_777_216.0,
            -1.0 / 16_777_216.0,
            1.0 / 16384.0,
            -1.0 / 16384.0,
            65504.0,
            -65504.0,
            1.0,
            -1.0,
            1.0009765625,
        ];
        for k in [1, 2, 11, 31, 128] {
            let a: Vec<_> = (0..4 * k).map(|i| extremes[i % extremes.len()]).collect();
            let b: Vec<_> = (0..16 * k)
                .map(|i| extremes[(i * 3 + 5) % extremes.len()])
                .collect();
            let got =
                simd::rounded_gemm_tile::<4>(std::array::from_fn(|r| &a[r * k..][..k]), &b, k);
            for r in 0..4 {
                for c in 0..16 {
                    let want = (0..k).fold(0f32, |sum, i| sum + a[r * k + i] * b[i * 16 + c]);
                    assert_eq!(got[r][c].to_bits(), want.to_bits());
                    assertions += 1;
                }
            }
        }
    }
    assertions
}

// Each case must trap before any unchecked vector load; the JS runner checks the traps.
#[unsafe(no_mangle)]
pub extern "C" fn reject_invalid(case: u32) {
    match case {
        0 => simd::q8_rows(&[], &[], 1 << 31, &[], &mut [0.0; 2]),
        1 => simd::q8_rows(&[0; 31], &[1.0], 32, &[0.0; 32], &mut [0.0]),
        2 => simd::q8_rows(&[0; 32], &[], 32, &[0.0; 32], &mut [0.0]),
        3 => simd::q8_rows(&[0; 32], &[1.0], 32, &[0.0; 31], &mut [0.0]),
        4 => simd::q8_rows(&[0; 33], &[1.0; 2], 33, &[0.0; 33], &mut [0.0]),
        #[cfg(target_feature = "relaxed-simd")]
        5 => {
            simd::rounded_gemm_tile::<4>([&[]; 4], &[0.0; 16], 1);
        }
        #[cfg(target_feature = "relaxed-simd")]
        6 => {
            simd::rounded_gemm_tile::<4>([&[0.0]; 4], &[0.0; 15], 1);
        }
        #[cfg(target_feature = "relaxed-simd")]
        7 => {
            simd::rounded_gemm_tile::<0>([], &[], 1 << 28);
        }
        _ => unreachable!(),
    }
}
