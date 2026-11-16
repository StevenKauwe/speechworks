// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Moonshine: an encoder over raw samples and an autoregressive decoder, both on the GPU or the
//! CPU, greedy
//! over the whole input in one pass as transcribe.cpp does (`src/arch/moonshine/model.cpp`): the
//! prompt is `<s>`, decoding stops at `</s>`, after the decoder's positions or on a loop. The text
//! has no timestamps: every token starts at 0, the one untimed segment.

mod cpu;
mod encoder;
pub(crate) mod repeat;

use std::sync::Mutex;

use encoder::{Encoder, STEM, frames};

use crate::cpu::transformer as on_cpu;
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Gpu, Groups};
use crate::runtime::{Runtime as Backend, Target};
use crate::sentencepiece::{self, Tokenizer};
use crate::transformer::{self, Decoder, Steps};
use crate::{Token, Transcript};

pub struct Moonshine {
    tok: Tokenizer,
    bos: usize,
    eos: usize,
    /// Rotary frequencies θ^(−2i/rot) for i < rot / 2, in f32 as ggml-metal's `rope.metal`
    /// computes them.
    freqs: Vec<f32>,
    /// The decoder's positions.
    context: usize,
    shape: (usize, usize, usize, usize),
    backend: Backend<OnGpu, OnCpu>,
}

struct OnGpu {
    encoder: Encoder,
    decoder: Decoder,
    arena: Mutex<Arena>,
}

struct OnCpu {
    encoder: cpu::Encoder,
    decoder: on_cpu::Decoder,
}

/// Activations, caches and the bind groups recorded over them, for the longest input so far.
struct Arena {
    enc: encoder::Buffers,
    dec: Steps,
    /// Per position (encoder frames or decoder positions) rot / 2 cosines, then as many sines.
    rope: Buffer,
    /// The next token's id, as the bits of a float.
    next: Buffer,
    read: Buffer,
    window: Groups,
    steps: Groups,
}

impl Moonshine {
    pub async fn load(g: &Gguf, target: Target) -> Result<Self> {
        let u = |key: &str| {
            g.get::<u32>(&format!("stt.moonshine.{key}"))
                .map(|v| v as usize)
        };
        for (key, want) in [
            ("stt.frontend.type", "raw"),
            ("stt.moonshine.encoder.activation", "gelu"),
            ("stt.moonshine.decoder.activation", "silu"),
        ] {
            g.check(key, &[want])?;
        }
        g.check("stt.frontend.sample_rate", &[16000u32])?;
        g.check("stt.moonshine.decoder.tie_word_embeddings", &[true])?;
        g.check("stt.moonshine.attention_bias", &[false])?;
        g.check("stt.moonshine.conv_stem.groupnorm_num_groups", &[1u32])?;
        let stem = |key: &str| g.array::<u32>(&format!("stt.moonshine.conv_stem.{key}"));
        if stem("kernel_sizes")? != STEM.map(|s| s.0 as u32)
            || stem("strides")? != STEM.map(|s| s.1 as u32)
        {
            bail!("conv stem kernels or strides other than {STEM:?}");
        }
        let factor = g.get::<f32>("stt.moonshine.partial_rotary_factor")?;
        let config = |part: &str| {
            let key = |name: &str| u(&format!("{part}.{name}"));
            let (d, heads) = (key("d_model")?, key("n_heads")?);
            let hd = d / heads.max(1);
            // The rotated dims of a head, rounded down to even (`weights.h`).
            let rot = (hd as f32 * factor) as usize & !1;
            // `attend` keeps a head of up to 128 as pairs.
            if d == 0
                || d > 1024
                || heads == 0
                || heads > 64
                || key("n_layers")? == 0
                || key("n_layers")? > 32
                || key("ffn_dim")? > 8192
                || hd * heads != d
                || key("n_kv_heads")? != heads
                || hd > 128
                || hd % 2 != 0
                || rot > hd
            {
                bail!("unsupported {part} shape: d_model {d}, {heads} heads, {rot} rotary dims");
            }
            Ok(transformer::Config {
                d,
                heads,
                layers: key("n_layers")?,
                d_ff: key("ffn_dim")?,
                bias: false,
                rot,
                gated: part == "decoder",
            })
        };
        let (enc, dec) = (config("encoder")?, config("decoder")?);
        if dec.d != enc.d || dec.rot != enc.rot {
            bail!("encoder and decoder of different widths or rotary dims");
        }
        let channels = stem("channels")?;
        if channels != [enc.d, 2 * enc.d, enc.d].map(|c| c as u32) {
            bail!(
                "conv stem channels {channels:?}, expected [d, 2·d, d] for d_model {}",
                enc.d
            );
        }
        let (vocab, context) = (
            u("decoder.vocab_size")?,
            u("decoder.max_position_embeddings")?,
        );
        if vocab == 0 || vocab > 262144 || context == 0 || context > 4096 {
            bail!("unsupported vocabulary/context size");
        }
        g.check("tokenizer.ggml.model", &["bpe"])?;
        let tok = Tokenizer::new(&g.array("tokenizer.ggml.tokens")?);
        let (bos, eos) = (u("decoder_start_token_id")?, u("eos_token_id")?);
        if tok.len() != vocab {
            bail!(
                "vocabulary of {} tokens, decoder.vocab_size {vocab}",
                tok.len()
            );
        }
        if bos >= vocab || eos >= vocab {
            bail!("special token ids {bos} and {eos} out of the vocabulary of {vocab}");
        }
        let (theta, rot) = (g.get::<f32>("stt.moonshine.rope_theta")?, enc.rot);
        let freqs = (0..rot / 2)
            .map(|i| theta.powf(-1.0 / rot as f32 * (2 * i) as f32))
            .collect();

        let eps = g.get::<f32>("stt.moonshine.conv_stem.groupnorm_eps")?;
        let gpu = |gpu: &Gpu| {
            let encoder = Encoder::load(gpu, g, config("encoder")?, eps)?;
            let decoder = Decoder::load(gpu, g, config("decoder")?, vocab, context)?;
            let arena = Arena {
                enc: encoder.buffers(gpu, 0),
                dec: decoder.steps(gpu, 0),
                rope: gpu.buffer(1),
                next: gpu.buffer(1),
                read: gpu.readback(1),
                window: Vec::new(),
                steps: Vec::new(),
            };
            Ok(OnGpu {
                encoder,
                decoder,
                arena: Mutex::new(arena),
            })
        };
        let cpu = || {
            let encoder = cpu::Encoder::load(g, config("encoder")?, eps)?;
            Ok(OnCpu {
                encoder,
                decoder: on_cpu::Decoder::load(g, config("decoder")?, vocab, context)?,
            })
        };
        let shape = (enc.d, enc.d_ff.max(dec.d_ff), enc.heads, dec.layers);
        let backend = Backend::new(target, gpu, cpu).await?;
        Ok(Self {
            tok,
            bos,
            eos,
            freqs,
            context,
            shape,
            backend,
        })
    }

    pub fn device(&self) -> &crate::Device {
        self.backend.device()
    }

    pub async fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
        let t = frames(pcm.len())[2];
        if t == 0 {
            return Ok(Transcript::default());
        }
        let ids = match &self.backend {
            Backend::Gpu(b) => self.on_gpu(&b.0, &b.1, pcm).await?,
            Backend::Cpu(on) => {
                let table = self.table(t.max(self.context));
                let enc = on.encoder.run(pcm, &table);
                let mut cache = on.decoder.cache(&enc, t);
                self.greedy(|token, pos| {
                    Ok(argmax(&on.decoder.step(
                        &mut cache,
                        token,
                        pos,
                        Some(&table),
                    )))
                })?
            }
        };
        let pieces = self.tok.pieces(&ids);
        let text = sentencepiece::text(&pieces);
        let token = |(&id, piece): (&usize, String)| Token {
            id: id as u32,
            piece,
            start_ms: 0,
        };
        Ok(Transcript {
            text,
            tokens: ids.iter().zip(pieces).map(token).collect(),
            language: None,
        })
    }

    /// As the reference's loop: `step(token, pos)` gives the token after `token` at `pos`, from `<s>`
    /// at 0, a step per position up to the context, the repetition guard after each token, and on a
    /// stop at the context a repeating tail trimmed.
    fn greedy(&self, mut step: impl FnMut(usize, usize) -> Result<usize>) -> Result<Vec<usize>> {
        let mut token = step(self.bos, 0)?;
        let mut ids = Vec::new();
        while token != self.eos {
            ids.push(token);
            if repeat::stop(&mut ids) {
                break;
            }

            if ids.len() == self.context {
                repeat::trim_at_budget(&mut ids);
                break;
            }

            token = step(token, ids.len())?;
        }

        Ok(ids)
    }

    async fn on_gpu(&self, gpu: &Gpu, on: &OnGpu, pcm: &[f32]) -> Result<Vec<usize>> {
        gpu.check()?;
        let (n, t) = (pcm.len(), frames(pcm.len())[2]);
        let mut a = on.arena.try_lock().map_err(|_| crate::Error::Busy)?;
        if n > a.enc.samples {
            self.grow(gpu, on, &mut a, n);
        }
        gpu.write(&a.enc.pcm, pcm);
        let Arena {
            enc,
            dec,
            rope,
            next,
            read,
            window,
            steps,
        } = &mut *a;
        let (enc, dec, rope, next) = (&*enc, &*dec, &*rope, &*next);
        let mut ids = Vec::new();
        gpu.write(next, &[f32::from_bits(self.bos as u32)]);
        for pos in 0..self.context {
            let groups = if pos == 0 { &mut *window } else { &mut *steps };
            let out = gpu
                .run(groups, next, read, 1, |p| {
                    if pos == 0 {
                        on.encoder.record(p, enc, n, rope);
                        on.decoder.record_cross(p, dec, &enc.t.y, &enc.t.h, t);
                    }
                    on.decoder
                        .record_step(p, dec, (next, 0), pos, t, Some(rope));
                    p.argmax(&dec.logits, next, on.decoder.vocab);
                })
                .await?;
            let token = out[0].to_bits() as usize;
            if token == self.eos {
                break;
            }
            if token >= self.tok.len() {
                return Err(crate::Error::Gpu("invalid decoder token".into()));
            }
            ids.push(token);
            if repeat::stop(&mut ids) {
                break;
            }
        }
        if ids.len() == self.context {
            repeat::trim_at_budget(&mut ids);
        }
        Ok(ids)
    }

    pub fn working_bytes(&self, n: usize) -> usize {
        let [t1, t2, t] = frames(n).map(|x| x as u64);
        let (d, f, heads, layers) = self.shape;
        let (d, f, heads, layers) = (d as u64, f as u64, heads as u64, layers as u64);
        let context = self.context as u64;
        // CPU attention has the largest score matrix. Include the old and new arenas during
        // growth; weights are separate. u64 arithmetic also bounds a 32-bit WASM address space.
        let cols = (t1 * 128).max(t2 * 7 * d).max(t * 6 * d);
        let stem = cols + (t1 * d).max(t2 * 2 * d);
        let blocks = t * (8 * d + 4 * f) + heads * t * t;
        let cache = (t + context) * 2 * d * layers;
        ((stem + blocks + cache + self.tok.len() as u64 + n as u64) * 8)
            .try_into()
            .unwrap_or(usize::MAX)
    }

    /// The rotary table for `positions`: per position rot / 2 cosines, then as many sines.
    fn table(&self, positions: usize) -> Vec<f32> {
        let mut table = Vec::new();
        for pos in 0..positions {
            let angles = self.freqs.iter().map(|f| pos as f32 * f);
            table.extend(angles.clone().map(f32::cos).chain(angles.map(f32::sin)));
        }

        table
    }

    /// Buffers for `n` samples, the rotary table for as many positions as they and the decoder need.
    fn grow(&self, gpu: &Gpu, on: &OnGpu, a: &mut Arena, n: usize) {
        let t = frames(n)[2];
        a.enc = on.encoder.buffers(gpu, n);
        a.dec = on.decoder.steps(gpu, t);
        let table = self.table(t.max(self.context));
        a.rope = gpu.buffer(table.len());
        gpu.write(&a.rope, &table);
        a.window.clear();
        a.steps.clear();
    }
}

/// The index of the largest logit, the lowest on a tie, as the GPU's `argmax`.
fn argmax(logits: &[f32]) -> usize {
    (0..logits.len()).fold(0, |best, i| if logits[i] > logits[best] { i } else { best })
}

#[cfg(test)]
impl Moonshine {
    pub(crate) fn destroy_gpu_for_test(&self) {
        if let Backend::Gpu((gpu, _)) = &self.backend {
            gpu.destroy_for_test();
        }
    }
}
