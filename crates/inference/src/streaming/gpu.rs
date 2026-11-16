use super::Shape;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Channels, Epilogue, Gpu, Linear, Pass};
use crate::transformer;

pub const CHUNK_SAMPLES: usize = 16_000;

pub struct Encoder {
    gain: f32,
    embed: Linear,
    conv1: Linear,
    conv2: Linear,
    blocks: Vec<Block>,
    norm: Channels,
    positions: Buffer,
    proj: Option<Linear>,
}
struct Block {
    norm_attn: Channels,
    qkv: Linear,
    out: Linear,
    norm_ffn: Channels,
    fc1: Linear,
    fc2: Linear,
}

/// Histories contain at most a layer's left context and pending look-ahead.
/// Temporary activations are bounded by one second of new streaming audio (four
/// seconds for one-shot calls), regardless of
/// phrase length. Cross-attention caches are allocated separately by the decoder.
pub struct State {
    pub max_raw_frames: usize,
    pub raw_frames: usize,
    pub frames: usize,
    conv1: ConvState,
    conv2: ConvState,
    blocks: Vec<BlockState>,
    pub b: Buffers,
}
struct ConvState {
    history: Buffer,
    inputs: usize,
    emitted: usize,
}
struct BlockState {
    input: Buffer,
    qkv: Buffer,
    start: usize,
    inputs: usize,
    emitted: usize,
}
pub struct Buffers {
    pub pcm: Buffer,
    cmvn: Buffer,
    front: Buffer,
    join: Buffer,
    col: Buffer,
    x: Buffer,
    norm: Buffer,
    pad: Buffer,
    new_qkv: Buffer,
    qkv: Buffer,
    att: Buffer,
    h: Buffer,
    pub out: Buffer,
    pub cross: Buffer,
}

impl Encoder {
    pub fn load(gpu: &Gpu, g: &Gguf, s: &Shape) -> Result<Self> {
        let linear = |name: &str, dims: &[usize], bias: bool| {
            let b = if bias {
                g.tensor(&format!("{name}.bias"), &dims[dims.len() - 1..])?
                    .to_f32()?
            } else {
                Vec::new()
            };
            gpu.linear_compact(&[g.tensor(&format!("{name}.weight"), dims)?], &b)
        };
        let (d, a) = (s.d, s.heads * s.hd);
        let blocks = (0..s.windows.len())
            .map(|i| {
                let p = |n: &str| format!("enc.blocks.{i}.{n}");
                let qkv = ["q", "k", "v"]
                    .iter()
                    .map(|n| g.tensor(&p(&format!("attn.{n}.weight")), &[d, a]))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Block {
                    norm_attn: transformer::norm(gpu, g, &p("norm_attn"), d, false)?,
                    qkv: gpu.linear_compact(&qkv, &[])?,
                    out: linear(&p("attn.out"), &[a, d], false)?,
                    norm_ffn: transformer::norm(gpu, g, &p("norm_ffn"), d, false)?,
                    fc1: linear(&p("ffn.fc1"), &[d, s.f], true)?,
                    fc2: linear(&p("ffn.fc2"), &[s.f, d], true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gain: g.tensor("enc.embedder.comp.log_k", &[1])?.to_f32()?[0].exp(),
            embed: linear("enc.embedder.linear", &[s.frame, d], false)?,
            conv1: linear("enc.embedder.conv1", &[5, d, 2 * d], true)?,
            conv2: linear("enc.embedder.conv2", &[5, 2 * d, d], true)?,
            blocks,
            norm: transformer::norm(gpu, g, "enc.final_norm", d, false)?,
            positions: gpu.upload(
                &g.tensor("adapter.pos_emb.weight", &[d, s.positions])?
                    .to_f32()?,
            )?,
            proj: if g.get::<bool>("stt.moonshine_streaming.adapter_has_proj")? {
                Some(linear("adapter.proj", &[d, s.dec.d], false)?)
            } else {
                None
            },
        })
    }

    pub fn state(&self, gpu: &Gpu, s: &Shape, raw: usize) -> State {
        let history = s.windows.iter().map(|[l, r]| l + r).max().unwrap();
        let rows = raw.div_ceil(4)
            + s.windows
                .iter()
                .map(|w| w[1].saturating_sub(1))
                .sum::<usize>()
            + 1;
        let a = s.heads * s.hd;
        let conv = |d| ConvState {
            history: gpu.buffer(4 * d),
            inputs: 0,
            emitted: 0,
        };
        State {
            max_raw_frames: raw,
            raw_frames: 0,
            frames: 0,
            conv1: conv(s.d),
            conv2: conv(2 * s.d),
            blocks: s
                .windows
                .iter()
                .map(|[l, r]| BlockState {
                    input: gpu.buffer((l + r) * s.d),
                    qkv: gpu.buffer((l + r) * 3 * a),
                    start: 0,
                    inputs: 0,
                    emitted: 0,
                })
                .collect(),
            b: Buffers {
                pcm: gpu.buffer(raw * s.frame),
                cmvn: gpu.buffer(raw * s.frame.next_multiple_of(32)),
                front: gpu.buffer((raw * s.d).max(raw.div_ceil(2) * 2 * s.d)),
                join: gpu.buffer(
                    ((raw + 4) * s.d)
                        .max((raw.div_ceil(2) + 4) * 2 * s.d)
                        .max((rows + history) * s.d),
                ),
                col: gpu.buffer(
                    (raw.div_ceil(2) * (5 * s.d).next_multiple_of(32))
                        .max(raw.div_ceil(4) * (10 * s.d).next_multiple_of(32)),
                ),
                x: gpu.buffer(rows * s.d),
                norm: gpu.buffer(rows * s.d),
                pad: gpu.buffer(rows * s.d.max(s.f).next_multiple_of(32)),
                new_qkv: gpu.buffer(rows * 3 * a),
                qkv: gpu.buffer((rows + history) * 3 * a),
                att: gpu.buffer(rows * a),
                h: gpu.buffer(rows * s.f),
                out: gpu.buffer(rows * s.dec.d),
                cross: gpu.buffer(rows * 2 * s.dec.d),
            },
        }
    }

    /// `raw` newly completed 5-ms frames, already uploaded to `state.b.pcm`.
    /// Output is only newly stable encoder rows, in `state.b.out`.
    pub fn record(
        &self,
        p: &mut Pass,
        state: &mut State,
        raw: usize,
        final_chunk: bool,
        s: &Shape,
    ) -> usize {
        let b = &state.b;
        if raw > 0 {
            p.streaming_cmvn(&b.pcm, &b.cmvn, raw, s.frame, s.eps, self.gain);
            p.gemm(&b.cmvn, &self.embed, &b.front, raw, Epilogue::Silu);
        }
        state.raw_frames += raw;
        let c1 = conv(
            p,
            &mut state.conv1,
            &b.front,
            &b.front,
            &b.join,
            &b.col,
            raw,
            s.d,
            &self.conv1,
            Epilogue::Silu,
        );
        let mut rows = conv(
            p,
            &mut state.conv2,
            &b.front,
            &b.x,
            &b.join,
            &b.col,
            c1,
            2 * s.d,
            &self.conv2,
            Epilogue::Bias,
        );
        let a = s.heads * s.hd;
        for ((block, cache), &window) in self.blocks.iter().zip(&mut state.blocks).zip(&s.windows) {
            let old = cache.inputs - cache.start;
            p.join_rows(&cache.input, &b.x, &b.join, old * s.d, rows * s.d);
            if rows > 0 {
                p.layer_norm(&b.x, &block.norm_attn, &b.norm, rows, s.d, None);
                padded_gemm(
                    p,
                    &b.norm,
                    &b.pad,
                    &block.qkv,
                    &b.new_qkv,
                    rows,
                    s.d,
                    Epilogue::Bias,
                );
            }
            p.join_rows(&cache.qkv, &b.new_qkv, &b.qkv, old * 3 * a, rows * 3 * a);
            cache.inputs += rows;
            let end = if final_chunk {
                cache.inputs
            } else {
                cache.inputs.saturating_sub(window[1].saturating_sub(1))
            };
            rows = end.saturating_sub(cache.emitted);
            let at = cache.emitted - cache.start;
            if rows > 0 {
                p.copy_rows(&b.join, &b.x, rows, s.d, s.d, at);
                p.local_attention(
                    &b.qkv,
                    &b.att,
                    rows,
                    cache.inputs - cache.start,
                    s.heads,
                    s.hd,
                    at,
                    window,
                );
                padded_gemm(
                    p,
                    &b.att,
                    &b.pad,
                    &block.out,
                    &b.x,
                    rows,
                    a,
                    Epilogue::Residual(1.0),
                );
                p.layer_norm(&b.x, &block.norm_ffn, &b.norm, rows, s.d, None);
                padded_gemm(
                    p,
                    &b.norm,
                    &b.pad,
                    &block.fc1,
                    &b.h,
                    rows,
                    s.d,
                    Epilogue::Gelu,
                );
                padded_gemm(
                    p,
                    &b.h,
                    &b.pad,
                    &block.fc2,
                    &b.x,
                    rows,
                    s.f,
                    Epilogue::Residual(1.0),
                );
            }
            cache.emitted += rows;
            let keep = (cache.emitted + 1).saturating_sub(window[0]);
            let remove = keep.saturating_sub(cache.start);
            let remaining = cache.inputs - cache.start - remove;
            p.copy_rows(&b.join, &cache.input, remaining, s.d, s.d, remove);
            p.copy_rows(&b.qkv, &cache.qkv, remaining, 3 * a, 3 * a, remove);
            cache.start += remove;
        }
        if rows > 0 {
            p.layer_norm(&b.x, &self.norm, &b.norm, rows, s.d, None);
            p.audio_positions(&b.norm, &self.positions, &b.x, rows, s.d, state.frames);
            if let Some(proj) = &self.proj {
                padded_gemm(p, &b.x, &b.pad, proj, &b.out, rows, s.d, Epilogue::Bias);
            } else {
                p.copy_rows(&b.x, &b.out, rows, s.d, s.d, 0);
            }
            state.frames += rows;
        }
        rows
    }
}

#[allow(clippy::too_many_arguments)]
fn padded_gemm(
    p: &mut Pass,
    x: &Buffer,
    pad: &Buffer,
    w: &Linear,
    out: &Buffer,
    rows: usize,
    width: usize,
    act: Epilogue,
) {
    if width.is_multiple_of(32) {
        p.gemm(x, w, out, rows, act);
    } else {
        p.copy_rows(x, pad, rows, width, width.next_multiple_of(32), 0);
        p.gemm(pad, w, out, rows, act);
    }
}

#[allow(clippy::too_many_arguments)]
fn conv(
    p: &mut Pass,
    state: &mut ConvState,
    fresh: &Buffer,
    out: &Buffer,
    joined: &Buffer,
    columns: &Buffer,
    new: usize,
    d: usize,
    w: &Linear,
    act: Epilogue,
) -> usize {
    if new == 0 {
        return 0;
    }
    let old = state.inputs.min(4);
    let start = state.inputs - old;
    p.join_rows(&state.history, fresh, joined, old * d, new * d);
    state.inputs += new;
    let end = state.inputs.div_ceil(2);
    let rows = end - state.emitted;
    if rows > 0 {
        p.causal_columns(joined, columns, rows, d, start, state.emitted);
        p.gemm(columns, w, out, rows, act);
    }
    state.emitted = end;
    let remaining = state.inputs.min(4);
    p.copy_rows(
        joined,
        &state.history,
        remaining,
        d,
        d,
        old + new - remaining,
    );
    rows
}
