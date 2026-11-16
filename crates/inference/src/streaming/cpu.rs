use super::Shape;
use crate::cpu::gemm::{Linear, gemm};
use crate::cpu::ops::{exp, layer_norm};
use crate::cpu::transformer::{linear, norm};
use crate::cpu::{dot, par_each};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::Epilogue;

type Norm = (Vec<f32>, Vec<f32>);
fn weights(n: &Norm) -> (&[f32], &[f32]) {
    (&n.0, &n.1)
}

pub struct Encoder {
    log_k: f32,
    embed: Linear,
    conv1: Linear,
    conv2: Linear,
    blocks: Vec<Block>,
    norm: Norm,
    positions: Vec<f32>,
    proj: Option<Linear>,
}
struct Block {
    norm_attn: Norm,
    qkv: Linear,
    out: Linear,
    norm_ffn: Norm,
    fc1: Linear,
    fc2: Linear,
}

/// Every retained encoder row is immutable. A layer waits for its own right
/// context before emitting rows; finalization flushes the remaining tail.
#[derive(Default)]
pub struct State {
    pub raw_frames: usize,
    conv1: ConvState,
    conv2: ConvState,
    blocks: Vec<BlockState>,
    pub frames: usize,
}
#[derive(Default)]
struct ConvState {
    input: Vec<f32>,
    start: usize,
    emitted: usize,
}
#[derive(Default)]
struct BlockState {
    input: Vec<f32>,
    qkv: Vec<f32>,
    start: usize,
    emitted: usize,
}

impl Encoder {
    pub fn load(g: &Gguf, s: &Shape) -> Result<Self> {
        let (d, a) = (s.d, s.heads * s.hd);
        let blocks = (0..s.windows.len())
            .map(|i| {
                let p = |n: &str| format!("enc.blocks.{i}.{n}");
                let qkv = ["q", "k", "v"]
                    .iter()
                    .map(|name| g.tensor(&p(&format!("attn.{name}.weight")), &[d, a]))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Block {
                    norm_attn: norm(g, &p("norm_attn"), d, false)?,
                    qkv: Linear::new(&qkv, Vec::new())?,
                    out: linear(g, &p("attn.out"), &[a, d], false)?,
                    norm_ffn: norm(g, &p("norm_ffn"), d, false)?,
                    fc1: linear(g, &p("ffn.fc1"), &[d, s.f], true)?,
                    fc2: linear(g, &p("ffn.fc2"), &[s.f, d], true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let proj = if g.get::<bool>("stt.moonshine_streaming.adapter_has_proj")? {
            Some(linear(g, "adapter.proj", &[d, s.dec.d], false)?)
        } else {
            None
        };
        if proj.is_none() && d != s.dec.d {
            crate::error::bail!("missing streaming width adapter");
        }
        Ok(Self {
            log_k: g.tensor("enc.embedder.comp.log_k", &[1])?.to_f32()?[0],
            embed: linear(g, "enc.embedder.linear", &[s.frame, d], false)?,
            conv1: linear(g, "enc.embedder.conv1", &[5, d, 2 * d], true)?,
            conv2: linear(g, "enc.embedder.conv2", &[5, 2 * d, d], true)?,
            blocks,
            norm: norm(g, "enc.final_norm", d, false)?,
            positions: g
                .tensor("adapter.pos_emb.weight", &[d, s.positions])?
                .to_f32()?,
            proj,
        })
    }

    pub fn run(&self, pcm: &[f32], s: &Shape) -> Vec<f32> {
        let stem = crate::profile::span("cpu.frontend.offline");
        let raw = pcm.len().div_ceil(s.frame);
        let mut input = vec![0.0; raw * s.frame];
        input[..pcm.len()].copy_from_slice(pcm);
        let scale = self.log_k.exp();
        for row in input.chunks_exact_mut(s.frame) {
            let mean = row.iter().sum::<f32>() / s.frame as f32;
            let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / s.frame as f32;
            let gain = scale / (var + s.eps).sqrt();
            for x in row {
                *x = ((*x - mean) * gain).asinh();
            }
        }
        let mut embed = vec![0.0; raw * s.d];
        gemm(&input, &self.embed, &mut embed, raw, Epilogue::Silu);
        drop(input);
        let conv1 = conv(&embed, raw, s.d, &self.conv1, Epilogue::Silu);
        drop(embed);
        let mut x = conv(
            &conv1,
            raw.div_ceil(2),
            2 * s.d,
            &self.conv2,
            Epilogue::Bias,
        );
        drop(conv1);
        drop(stem);
        let _blocks = crate::profile::span("cpu.encoder.offline");
        let t = raw.div_ceil(4);
        let a = s.heads * s.hd;
        let mut y = vec![0.0; t * s.d];
        let mut qkv = vec![0.0; t * 3 * a];
        let mut att = vec![0.0; t * a];
        let mut h = vec![0.0; t * s.f];
        for (block, &window) in self.blocks.iter().zip(&s.windows) {
            layer_norm(&x, weights(&block.norm_attn), &mut y, None);
            gemm(&y, &block.qkv, &mut qkv, t, Epilogue::Bias);
            attention(&qkv, &mut att, 0, s.heads, s.hd, window);
            gemm(&att, &block.out, &mut x, t, Epilogue::Residual(1.0));
            layer_norm(&x, weights(&block.norm_ffn), &mut y, None);
            gemm(&y, &block.fc1, &mut h, t, Epilogue::Gelu);
            gemm(&h, &block.fc2, &mut x, t, Epilogue::Residual(1.0));
        }
        layer_norm(&x, weights(&self.norm), &mut y, None);
        for (x, pos) in y.iter_mut().zip(&self.positions) {
            *x += pos;
        }
        if let Some(proj) = &self.proj {
            let mut out = vec![0.0; t * s.dec.d];
            gemm(&y, proj, &mut out, t, Epilogue::Bias);
            out
        } else {
            y
        }
    }

    /// `pcm` is the complete phrase, held by the caller for CPU recovery.
    /// Only complete, previously unprocessed 5-ms frames are read. A final
    /// partial frame is zero-padded exactly as in the one-shot processor.
    pub fn update(&self, state: &mut State, pcm: &[f32], final_chunk: bool, s: &Shape) -> Vec<f32> {
        let stem = crate::profile::span("cpu.frontend.stream");
        let raw = if final_chunk {
            pcm.len().div_ceil(s.frame)
        } else {
            pcm.len() / s.frame
        };
        let count = raw - state.raw_frames;
        let mut input = vec![0.0; count * s.frame];
        let start = state.raw_frames * s.frame;
        let count_pcm = pcm.len().min(raw * s.frame) - start;
        input[..count_pcm].copy_from_slice(&pcm[start..start + count_pcm]);
        let scale = self.log_k.exp();
        for row in input.chunks_exact_mut(s.frame) {
            let mean = row.iter().sum::<f32>() / s.frame as f32;
            let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / s.frame as f32;
            let gain = scale / (var + s.eps).sqrt();
            for x in row {
                *x = ((*x - mean) * gain).asinh();
            }
        }
        let mut embed = vec![0.0; count * s.d];
        if count > 0 {
            gemm(&input, &self.embed, &mut embed, count, Epilogue::Silu);
        }
        drop(input);
        state.raw_frames = raw;
        let c1 = state.conv1.update(&embed, s.d, &self.conv1, Epilogue::Silu);
        drop(embed);
        let mut x = state
            .conv2
            .update(&c1, 2 * s.d, &self.conv2, Epilogue::Bias);
        drop(c1);
        drop(stem);
        let _blocks = crate::profile::span("cpu.encoder.stream");
        state
            .blocks
            .resize_with(self.blocks.len(), BlockState::default);
        let a = s.heads * s.hd;
        for ((block, cache), &window) in self.blocks.iter().zip(&mut state.blocks).zip(&s.windows) {
            let rows = x.len() / s.d;
            if rows > 0 {
                let mut y = vec![0.0; rows * s.d];
                layer_norm(&x, weights(&block.norm_attn), &mut y, None);
                let mut qkv = vec![0.0; rows * 3 * a];
                gemm(&y, &block.qkv, &mut qkv, rows, Epilogue::Bias);
                cache.input.extend_from_slice(&x);
                cache.qkv.extend(qkv);
            }
            let available = cache.start + cache.input.len() / s.d;
            let end = if final_chunk {
                available
            } else {
                available.saturating_sub(window[1].saturating_sub(1))
            };
            let rows = end.saturating_sub(cache.emitted);
            let at = cache.emitted - cache.start;
            x = cache.input[at * s.d..][..rows * s.d].to_vec();
            if rows > 0 {
                let mut att = vec![0.0; rows * a];
                attention(&cache.qkv, &mut att, at, s.heads, s.hd, window);
                gemm(&att, &block.out, &mut x, rows, Epilogue::Residual(1.0));
                let mut y = vec![0.0; rows * s.d];
                layer_norm(&x, weights(&block.norm_ffn), &mut y, None);
                let mut h = vec![0.0; rows * s.f];
                gemm(&y, &block.fc1, &mut h, rows, Epilogue::Gelu);
                gemm(&h, &block.fc2, &mut x, rows, Epilogue::Residual(1.0));
            }
            cache.emitted += rows;
            let keep = (cache.emitted + 1).saturating_sub(window[0]);
            let remove = keep.saturating_sub(cache.start);
            cache.input.drain(..remove * s.d);
            cache.qkv.drain(..remove * 3 * a);
            cache.start += remove;
        }
        let rows = x.len() / s.d;
        if rows == 0 {
            return Vec::new();
        }
        let mut y = vec![0.0; rows * s.d];
        layer_norm(&x, weights(&self.norm), &mut y, None);
        for (x, pos) in y.iter_mut().zip(&self.positions[state.frames * s.d..]) {
            *x += pos;
        }
        state.frames += rows;
        if let Some(proj) = &self.proj {
            let mut out = vec![0.0; rows * s.dec.d];
            gemm(&y, proj, &mut out, rows, Epilogue::Bias);
            out
        } else {
            y
        }
    }
}

impl ConvState {
    fn update(&mut self, x: &[f32], d: usize, w: &Linear, act: Epilogue) -> Vec<f32> {
        self.input.extend_from_slice(x);
        let end = (self.start + self.input.len() / d).div_ceil(2);
        let rows = end - self.emitted;
        if rows == 0 {
            return Vec::new();
        }
        let mut col = vec![0.0; rows * d * 5];
        for (i, col) in col.chunks_exact_mut(d * 5).enumerate() {
            for k in 0..5 {
                let index = 2 * (self.emitted + i) + k;
                if index < 4 {
                    continue;
                }
                let at = index - 4 - self.start;
                for c in 0..d {
                    col[c * 5 + k] = self.input[at * d + c];
                }
            }
        }
        let mut out = vec![0.0; rows * w.n];
        gemm(&col, w, &mut out, rows, act);
        self.emitted = end;
        let keep = (2 * end).saturating_sub(4);
        let remove = keep.saturating_sub(self.start);
        self.input.drain(..remove * d);
        self.start += remove;
        out
    }
}

fn conv(x: &[f32], t: usize, d: usize, w: &Linear, act: Epilogue) -> Vec<f32> {
    let rows = t.div_ceil(2);
    let mut col = vec![0.0; rows * d * 5];
    for (i, col) in col.chunks_exact_mut(d * 5).enumerate() {
        for k in 0..5 {
            if 2 * i + k < 4 {
                continue;
            }
            let at = 2 * i + k - 4;
            for c in 0..d {
                col[c * 5 + k] = x[at * d + c];
            }
        }
    }
    let mut out = vec![0.0; rows * w.n];
    gemm(&col, w, &mut out, rows, act);
    out
}

/// Strict (left, right) window: includes the current row, at most L-1
/// predecessors and R-1 successors. Work and scratch stay linear in audio length.
fn attention(
    qkv: &[f32],
    out: &mut [f32],
    query_start: usize,
    heads: usize,
    hd: usize,
    [left, right]: [usize; 2],
) {
    let a = heads * hd;
    let t = qkv.len() / (3 * a);
    let scale = 1.0 / (hd as f32).sqrt();
    par_each(
        out.chunks_mut(a).enumerate(),
        || vec![0.0; left + right],
        |scores, (q, row)| {
            let q = q + query_start;
            let start = (q + 1).saturating_sub(left);
            let end = (q + right.max(1)).min(t);
            for h in 0..heads {
                let query = &qkv[q * 3 * a + h * hd..][..hd];
                let scores = &mut scores[..end - start];
                for (k, score) in (start..end).zip(scores.iter_mut()) {
                    *score = dot(query, &qkv[k * 3 * a + a + h * hd..][..hd]) * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for score in scores.iter_mut() {
                    *score = exp(*score - max);
                    sum += *score;
                }
                for c in 0..hd {
                    row[h * hd + c] = (start..end)
                        .zip(scores.iter())
                        .map(|(k, p)| p * qkv[k * 3 * a + 2 * a + h * hd + c])
                        .sum::<f32>()
                        / sum;
                }
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_attention_window_and_retained_history() {
        // Zero Q/K makes the result the average of the allowed values. This
        // catches both off-by-one window errors and a query offset into history.
        let mut qkv = Vec::new();
        for value in [2.0, 4.0, 8.0, 16.0] {
            qkv.extend([0.0, 0.0, 0.0, 0.0, value, value]);
        }
        let mut out = vec![0.0; 8];
        attention(&qkv, &mut out, 0, 1, 2, [2, 2]);
        for (got, want) in out.chunks_exact(2).zip([3.0, 14.0 / 3.0, 28.0 / 3.0, 12.0]) {
            assert!((got[0] - want).abs() < 1e-5);
        }
        let mut tail = vec![0.0; 4];
        attention(&qkv[6..], &mut tail, 1, 1, 2, [2, 0]);
        assert_eq!(tail, [6.0, 6.0, 12.0, 12.0]);
    }

    #[test]
    #[ignore = "requires pinned streaming model weights and JFK fixture"]
    fn chunk_boundaries_preserve_encoder_values_and_bound_history() {
        use crate::runtime::{Runtime, Target};
        use crate::streaming::Streaming;
        let root = std::path::PathBuf::from(
            std::env::var("SPEECHWORKS_TEST_STREAMING_MODELS").expect("fixture directory"),
        );
        let mut wav = hound::WavReader::open(root.join("jfk.wav")).unwrap();
        let pcm: Vec<f32> = wav
            .samples::<i16>()
            .map(|x| x.unwrap() as f32 / 32768.0)
            .take(23451)
            .collect();
        for name in ["medium", "small"] {
            let g = Gguf::from_bytes(
                std::fs::read(root.join(format!("moonshine-streaming-{name}-Q8_0.gguf"))).unwrap(),
            )
            .unwrap();
            let model = pollster::block_on(Streaming::load(&g, Target::Cpu, 480000)).unwrap();
            let Runtime::Cpu(on) = &model.backend else {
                unreachable!()
            };
            let expected = on.encoder.run(&pcm, &model.shape);
            for step in [79, 321, 511, 10240] {
                let mut state = State::default();
                let mut actual = Vec::new();
                for end in (step..pcm.len()).step_by(step) {
                    actual.extend(
                        on.encoder
                            .update(&mut state, &pcm[..end], false, &model.shape),
                    );
                    for (cache, window) in state.blocks.iter().zip(&model.shape.windows) {
                        assert!(
                            cache.input.len() / model.shape.d
                                <= window[0] - 1 + window[1].saturating_sub(1)
                        );
                    }
                }
                actual.extend(on.encoder.update(&mut state, &pcm, true, &model.shape));
                assert_eq!(actual.len(), expected.len());
                let max = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max < 0.01,
                    "{name}, {step} samples, max absolute error {max}"
                );
            }
        }
    }
}
