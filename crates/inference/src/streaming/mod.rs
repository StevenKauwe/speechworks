//! Moonshine Streaming: causal time-domain frontend, local encoder attention,
//! learned audio positions, and an untied autoregressive text decoder.
//!
//! The equations and GGUF contract are checked against transcribe.cpp's MIT
//! implementation (see THIRD_PARTY_NOTICES.md). Encoder and decoder widths are
//! deliberately separate: Small and Medium have non-square attention projections.

mod cpu;
mod gpu;

use crate::cpu::transformer::Decoder;
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Gpu, Groups};
use crate::moonshine::repeat;
use crate::runtime::{Runtime, Target};
use crate::sentencepiece::{self, Tokenizer};
use crate::transformer::Config;
use crate::{Token, Transcript};

#[derive(Clone)]
pub(super) struct Shape {
    pub d: usize,
    pub heads: usize,
    pub hd: usize,
    pub f: usize,
    pub windows: Vec<[usize; 2]>,
    pub frame: usize,
    pub eps: f32,
    pub positions: usize,
    pub dec: Config,
}

pub struct Streaming {
    backend: Runtime<OnGpu, OnCpu>,
    shape: Shape,
    tok: Tokenizer,
    bos: usize,
    eos: usize,
    table: Vec<f32>,
    context: usize,
}

struct OnCpu {
    encoder: cpu::Encoder,
    decoder: Decoder,
}
struct OnGpu {
    encoder: gpu::Encoder,
    decoder: crate::transformer::Decoder,
    rope: Buffer,
    capacity: usize,
}
struct GpuState {
    encoder: gpu::State,
    decoder: crate::transformer::Steps,
    next: Buffer,
    read: Buffer,
    groups: Groups,
    steps: Groups,
}

#[derive(Default)]
pub(crate) struct State {
    encoder: cpu::State,
    decoder: Option<crate::cpu::transformer::Cache>,
    gpu: Option<GpuState>,
    last: Transcript,
}

pub(super) fn frames(samples: usize, frame: usize) -> usize {
    samples.div_ceil(frame).div_ceil(4)
}

fn budget(frames: usize, frame: usize) -> usize {
    24 + frames * frame * 4 * 13 / 32_000
}

impl Streaming {
    pub async fn load(g: &Gguf, target: Target, max_samples: usize) -> Result<Self> {
        let u = |key: &str| {
            g.get::<u32>(&format!("stt.moonshine_streaming.{key}"))
                .map(|v| v as usize)
        };
        for (key, value) in [
            ("stt.frontend.type", "raw"),
            ("stt.moonshine_streaming.encoder.activation", "gelu"),
            ("stt.moonshine_streaming.decoder.activation", "silu"),
            ("tokenizer.ggml.model", "bpe"),
        ] {
            g.check(key, &[value])?;
        }
        g.check("stt.frontend.sample_rate", &[16_000u32])?;
        g.check("stt.moonshine_streaming.attention_bias", &[false])?;
        g.check(
            "stt.moonshine_streaming.decoder.tie_word_embeddings",
            &[false],
        )?;
        g.check(
            "stt.moonshine_streaming.encoder_layernorm_unit_offset",
            &[true],
        )?;
        let factor = g.get::<f32>("stt.moonshine_streaming.partial_rotary_factor")?;
        let theta = g.get::<f32>("stt.moonshine_streaming.rope_theta")?;
        let eps = g.get::<f32>("stt.moonshine_streaming.cmvn_eps")?;
        if !factor.is_finite()
            || !(0.0..=1.0).contains(&factor)
            || !theta.is_finite()
            || theta <= 0.0
            || !eps.is_finite()
            || eps <= 0.0
        {
            bail!("invalid streaming normalization/rotary metadata");
        }
        let (d, heads, hd, f, layers) = (
            u("encoder.d_model")?,
            u("encoder.n_heads")?,
            u("encoder.head_dim")?,
            u("encoder.ffn_dim")?,
            u("encoder.n_layers")?,
        );
        let dec = Config {
            d: u("decoder.d_model")?,
            heads: u("decoder.n_heads")?,
            layers: u("decoder.n_layers")?,
            d_ff: u("decoder.ffn_dim")?,
            bias: false,
            rot: (u("decoder.head_dim")? as f32 * factor) as usize & !1,
            gated: true,
        };
        if d == 0
            || d > 1024
            || heads == 0
            || heads > 32
            || hd == 0
            || hd > 128
            || hd % 2 != 0
            || layers == 0
            || layers > 32
            || f == 0
            || f > 8192
            || dec.d == 0
            || dec.d > 1024
            || dec.heads == 0
            || dec.heads > 32
            || dec.layers == 0
            || dec.layers > 32
            || dec.d_ff == 0
            || dec.d_ff > 8192
            || dec.heads * u("decoder.head_dim")? != dec.d
            || dec.head_dim() > 128
            || dec.head_dim() % 2 != 0
            || dec.rot == 0
            || u("encoder.n_kv_heads")? != heads
            || u("decoder.n_kv_heads")? != dec.heads
        {
            bail!("unsupported streaming transformer dimensions");
        }
        if !g.get::<bool>("stt.moonshine_streaming.adapter_has_proj")? && d != dec.d {
            bail!("missing streaming width adapter");
        }
        let windows = g.array::<u32>("stt.moonshine_streaming.encoder.sliding_windows")?;
        if windows.len() != 2 * layers
            || windows
                .chunks_exact(2)
                .any(|w| w[0] == 0 || w[0] > 1024 || w[1] > 1024)
        {
            bail!("invalid streaming attention windows");
        }
        let frame = u("encoder.frame_len")?;
        let positions = u("decoder.max_position_embeddings")?;
        let vocab = u("decoder.vocab_size")?;
        if frame != 80 || positions == 0 || positions > 4096 || vocab == 0 || vocab > 262144 {
            bail!("unsupported streaming frame/vocabulary/position dimensions");
        }
        if frames(max_samples, frame) > positions {
            bail!("audio budget exceeds learned audio positions");
        }
        let shape = Shape {
            d,
            heads,
            hd,
            f,
            frame,
            eps,
            positions,
            dec: dec.clone(),
            windows: windows
                .chunks_exact(2)
                .map(|w| [w[0] as usize, w[1] as usize])
                .collect(),
        };
        let tok = Tokenizer::new(&g.array("tokenizer.ggml.tokens")?);
        let (bos, eos) = (u("decoder_start_token_id")?, u("eos_token_id")?);
        if tok.len() != vocab || bos >= vocab || eos >= vocab {
            bail!("invalid streaming vocabulary or special tokens");
        }
        let context = positions.min(budget(frames(max_samples, frame), frame));
        let table: Vec<f32> = (0..context)
            .flat_map(|pos| {
                let angles = (0..dec.rot / 2)
                    .map(move |i| pos as f32 * theta.powf(-((2 * i) as f32) / dec.rot as f32));
                angles.clone().map(f32::cos).chain(angles.map(f32::sin))
            })
            .collect();
        let backend = Runtime::new(
            target,
            |gpu| {
                Ok(OnGpu {
                    encoder: gpu::Encoder::load(gpu, g, &shape)?,
                    decoder: crate::transformer::Decoder::load(
                        gpu,
                        g,
                        dec.clone(),
                        vocab,
                        context,
                    )?,
                    rope: gpu.upload(&table)?,
                    capacity: frames(max_samples, frame),
                })
            },
            || {
                Ok(OnCpu {
                    encoder: cpu::Encoder::load(g, &shape)?,
                    decoder: Decoder::load(g, dec.clone(), vocab, context)?,
                })
            },
        )
        .await?;
        Ok(Self {
            backend,
            shape,
            tok,
            bos,
            eos,
            table,
            context,
        })
    }

    pub fn device(&self) -> &crate::Device {
        self.backend.device()
    }

    pub async fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
        let t = frames(pcm.len(), self.shape.frame);
        if t == 0 {
            return Ok(Transcript::default());
        }
        match &self.backend {
            Runtime::Cpu(on) => {
                let enc = on.encoder.run(pcm, &self.shape);
                let mut cache = on.decoder.cache(&enc, t);
                Ok(self.decode(&on.decoder, &mut cache, t))
            }
            Runtime::Gpu((gpu, on)) => {
                let raw = pcm
                    .len()
                    .div_ceil(self.shape.frame)
                    .min(4 * gpu::CHUNK_SAMPLES / self.shape.frame);
                let mut state = State {
                    gpu: Some(self.gpu_state(gpu, on, raw)),
                    ..Default::default()
                };
                self.update_gpu(gpu, on, &mut state, pcm, true).await
            }
        }
    }

    pub async fn update(
        &self,
        state: &mut State,
        pcm: &[f32],
        final_chunk: bool,
    ) -> Result<Transcript> {
        if pcm.is_empty() {
            return Ok(Transcript::default());
        }
        match &self.backend {
            Runtime::Cpu(on) => {
                state.gpu = None;
                let before = state.encoder.frames;
                let enc = on
                    .encoder
                    .update(&mut state.encoder, pcm, final_chunk, &self.shape);
                if state.encoder.frames == before {
                    return Ok(state.last.clone());
                }
                let cache = state
                    .decoder
                    .get_or_insert_with(|| on.decoder.cache(&[], 0));
                on.decoder
                    .append_cross(cache, &enc, enc.len() / self.shape.dec.d);
                if state.encoder.frames == 0 {
                    return Ok(Transcript::default());
                }
                state.last = self.decode(&on.decoder, cache, state.encoder.frames);
                Ok(state.last.clone())
            }
            Runtime::Gpu((gpu, on)) => self.update_gpu(gpu, on, state, pcm, final_chunk).await,
        }
    }

    fn decode(
        &self,
        decoder: &Decoder,
        cache: &mut crate::cpu::transformer::Cache,
        t: usize,
    ) -> Transcript {
        let _decode = crate::profile::span("cpu.decoder");
        let mut token = self.bos;
        let mut ids = Vec::new();
        let limit = self.context.min(budget(t, self.shape.frame));
        for pos in 0..limit {
            let logits = decoder.step(cache, token, pos, Some(&self.table));
            token = (0..logits.len())
                .fold(0, |best, i| if logits[i] > logits[best] { i } else { best });
            if token == self.eos {
                break;
            }
            ids.push(token);
            if repeat::stop(&mut ids) {
                break;
            }
        }
        if ids.len() == limit {
            repeat::trim_at_budget(&mut ids);
        }
        self.transcript(&ids)
    }

    fn transcript(&self, ids: &[usize]) -> Transcript {
        let pieces = self.tok.pieces(ids);
        Transcript {
            text: sentencepiece::text(&pieces),
            tokens: ids
                .iter()
                .zip(pieces)
                .map(|(&id, piece)| Token {
                    id: id as u32,
                    piece,
                    start_ms: 0,
                })
                .collect(),
            language: None,
        }
    }

    fn gpu_state(&self, gpu: &Gpu, on: &OnGpu, raw: usize) -> GpuState {
        GpuState {
            encoder: on.encoder.state(gpu, &self.shape, raw),
            decoder: on.decoder.steps(gpu, on.capacity),
            next: gpu.buffer(5),
            read: gpu.readback(5),
            groups: Vec::new(),
            steps: Vec::new(),
        }
    }

    async fn update_gpu(
        &self,
        gpu: &Gpu,
        on: &OnGpu,
        state: &mut State,
        pcm: &[f32],
        final_chunk: bool,
    ) -> Result<Transcript> {
        gpu.check()?;
        let a = state
            .gpu
            .get_or_insert_with(|| self.gpu_state(gpu, on, gpu::CHUNK_SAMPLES / self.shape.frame));
        let before = a.encoder.frames;
        let frame = self.shape.frame;
        let target = if final_chunk {
            pcm.len().div_ceil(frame)
        } else {
            pcm.len() / frame
        };
        let mut flush = final_chunk;
        let encode = crate::profile::span("gpu.encoder_and_cross");
        while a.encoder.raw_frames < target || flush {
            let start = a.encoder.raw_frames * frame;
            let end = (a.encoder.raw_frames + a.encoder.max_raw_frames).min(target) * frame;
            let raw = (end - start) / frame;
            let last = end == target * frame;
            if raw > 0 {
                let mut samples = vec![0.0; raw * frame];
                let count = pcm.len().min(end) - start;
                samples[..count].copy_from_slice(&pcm[start..start + count]);
                gpu.write(&a.encoder.b.pcm, &samples);
            }
            let offset = a.encoder.frames;
            gpu.run(&mut a.groups, &a.next, &a.read, 1, |p| {
                let rows =
                    on.encoder
                        .record(p, &mut a.encoder, raw, last && final_chunk, &self.shape);
                if rows > 0 {
                    on.decoder.record_cross_append(
                        p,
                        &a.decoder,
                        &a.encoder.b.out,
                        &a.encoder.b.cross,
                        rows,
                        on.capacity,
                        offset,
                    );
                }
            })
            .await?;
            if last {
                flush = false;
            }
        }
        drop(encode);
        if a.encoder.frames == before {
            return Ok(state.last.clone());
        }
        let _decode = crate::profile::span("gpu.decoder");
        let mut ids = Vec::new();
        let limit = self.context.min(budget(a.encoder.frames, frame));
        let mut next = self.bos;
        'decode: for pos in (0..limit).step_by(4) {
            let count = (limit - pos).min(4);
            gpu.write(&a.next, &[f32::from_bits(next as u32)]);
            let result = gpu
                .run(&mut a.steps, &a.next, &a.read, count + 1, |p| {
                    for i in 0..count {
                        on.decoder.record_step_capacity(
                            p,
                            &a.decoder,
                            (&a.next, i),
                            pos + i,
                            a.encoder.frames,
                            on.capacity,
                            Some(&on.rope),
                        );
                        p.argmax_at(&a.decoder.logits, &a.next, on.decoder.vocab, i + 1);
                    }
                })
                .await?;
            for value in &result[1..] {
                next = value.to_bits() as usize;
                if next == self.eos {
                    break 'decode;
                }
                if next >= self.tok.len() {
                    return Err(crate::Error::Gpu("invalid streaming decoder token".into()));
                }
                ids.push(next);
                if repeat::stop(&mut ids) {
                    break 'decode;
                }
            }
        }
        if ids.len() == limit {
            repeat::trim_at_budget(&mut ids);
        }
        state.last = self.transcript(&ids);
        Ok(state.last.clone())
    }

    pub fn working_bytes(&self, n: usize) -> usize {
        let s = &self.shape;
        let t = frames(n, s.frame) as u64;
        let (d, f, dd, layers) = (s.d as u64, s.f as u64, s.dec.d as u64, s.dec.layers as u64);
        let context = self.context as u64;
        // Includes causal-convolution columns, simultaneous transformer scratch,
        // per-layer decoder caches, logits and the incoming PCM. No T×T scores.
        let history: u64 = s.windows.iter().map(|[l, r]| (l + r) as u64).sum();
        let a = (s.heads * s.hd) as u64;
        let cpu = (t * (32 * d + 2 * f + 4 * dd)
            + (t + context) * 2 * dd * layers
            + self.tok.len() as u64
            + n as u64
            + history * (d + 3 * a))
            * 4;
        let gpu = if let Runtime::Gpu((_, on)) = &self.backend {
            let raw = (n.min(4 * gpu::CHUNK_SAMPLES).max(gpu::CHUNK_SAMPLES) / s.frame) as u64;
            let rows = raw.div_ceil(4)
                + 1
                + s.windows
                    .iter()
                    .map(|w| w[1].saturating_sub(1) as u64)
                    .sum::<u64>();
            ((on.capacity as u64 + context) * dd * layers
                + raw * 16 * d
                + rows * (4 * d + 2 * f + 8 * a + 4 * dd)
                + 2 * history * (d + 3 * a)
                + n as u64)
                * 4
        } else {
            0
        };
        cpu.max(gpu).try_into().unwrap_or(usize::MAX)
    }

    #[cfg(test)]
    pub(crate) fn destroy_gpu_for_test(&self) {
        if let Runtime::Gpu((gpu, _)) = &self.backend {
            gpu.destroy_for_test();
        }
    }
}
