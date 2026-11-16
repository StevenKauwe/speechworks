//! Model inference for Speechworks. Models are supplied as GGUF bytes; the caller owns
//! downloading, persistence, model selection and microphone capture. Native and browser GPU
//! inference share the same portable WGSL kernels and asynchronous API.
//!
//! Moonshine dimensions, vocabulary and token limits come from the model metadata. Tiny and
//! base use the same implementation. The CPU backend uses only Rust. In `Auto`, GPU setup or
//! execution failure replays the entire supplied phrase on CPU, recording the reason.
#![allow(dead_code)] // Shared upstream math retains kernels useful to additional model families.
mod conformer;
mod cpu;
mod error;
mod gguf;
mod gpu;
mod moonshine;
mod profile;
mod runtime;
mod sentencepiece;
mod streaming;
mod transformer;
mod whisper {
    pub mod tokenizer;
}

/// Register the existing global Rayon pool before constructing a model.
/// The browser adapter calls this after its awaited thread-pool initialization.
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
#[doc(hidden)]
pub fn register_wasm_thread_pool() {
    cpu::register_wasm_thread_pool();
}

pub use error::{Error, Result};
pub use gpu::{Api, Device, DeviceKind, Kernels};
use runtime::Target;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

static MODEL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

/// Bounds for one phrase, independent of the model chosen by an application.
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    pub backend: Backend,
    /// Long recordings must be segmented by the caller.
    pub max_audio_samples: usize,
    /// Conservative bound for activations and decoder caches; model weights are separate.
    pub max_work_bytes: usize,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            max_audio_samples: 16_000 * 30,
            max_work_bytes: 256 << 20,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub text: String,
    pub tokens: Vec<Token>,
    pub language: Option<String>,
}
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u32,
    pub piece: String,
    pub start_ms: u32,
}

pub struct Model {
    id: u64,
    source: gguf::Gguf,
    model: Mutex<Family>,
    options: LoadOptions,
    fallback: Mutex<Option<String>>,
}

/// State of one phrase. Feed fresh 16-kHz PCM with `Model::transcribe_chunk`;
/// pass `final_chunk=true` once to flush look-ahead and the last partial frame.
/// Partial transcripts may change. A finalized stream cannot accept more audio.
/// Keep a separate state for every concurrent phrase; model weights are shared.
#[derive(Default)]
pub struct StreamState {
    model_id: Option<u64>,
    pcm: Vec<f32>,
    streaming: streaming::State,
    finished: bool,
}
impl StreamState {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn samples(&self) -> usize {
        self.pcm.len()
    }
    pub fn is_finished(&self) -> bool {
        self.finished
    }
    /// Whether a growing phrase still includes exactly the audio already fed.
    /// Useful when a phrase splitter can move a final boundary behind a preview.
    pub fn matches_prefix(&self, samples: &[f32]) -> bool {
        samples.starts_with(&self.pcm)
    }
}

enum Family {
    Moonshine(moonshine::Moonshine),
    Streaming(streaming::Streaming),
}
impl Family {
    async fn load(g: &gguf::Gguf, target: Target, max_samples: usize) -> Result<Self> {
        match g.get::<&str>("general.architecture")? {
            "moonshine" => Ok(Self::Moonshine(
                moonshine::Moonshine::load(g, target).await?,
            )),
            "moonshine_streaming" => Ok(Self::Streaming(
                streaming::Streaming::load(g, target, max_samples).await?,
            )),
            name => Err(Error::Model(format!(
                "unsupported architecture {name:?}; expected moonshine or moonshine_streaming"
            ))),
        }
    }
    fn device(&self) -> &Device {
        match self {
            Self::Moonshine(m) => m.device(),
            Self::Streaming(m) => m.device(),
        }
    }
    fn working_bytes(&self, n: usize) -> usize {
        match self {
            Self::Moonshine(m) => m.working_bytes(n),
            Self::Streaming(m) => m.working_bytes(n),
        }
    }
    async fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        match self {
            Self::Moonshine(m) => m.transcribe(samples).await,
            Self::Streaming(m) => m.transcribe(samples).await,
        }
    }
    #[cfg(test)]
    fn destroy_gpu_for_test(&self) {
        match self {
            Self::Moonshine(m) => m.destroy_gpu_for_test(),
            Self::Streaming(m) => m.destroy_gpu_for_test(),
        }
    }
}
impl Model {
    pub fn is_streaming(&self) -> bool {
        self.source
            .get::<&str>("general.architecture")
            .is_ok_and(|name| name == "moonshine_streaming")
    }
    pub async fn from_bytes(bytes: Vec<u8>, backend: Backend) -> Result<Self> {
        Self::from_bytes_with(
            bytes,
            LoadOptions {
                backend,
                ..Default::default()
            },
        )
        .await
    }
    pub async fn from_bytes_with(bytes: Vec<u8>, options: LoadOptions) -> Result<Self> {
        if options.max_audio_samples == 0
            || options.max_audio_samples > 16_000 * 60
            || options.max_work_bytes == 0
        {
            return Err(Error::Model(
                "audio/work budgets must be positive, with at most 60 seconds per phrase".into(),
            ));
        }
        let source = gguf::Gguf::from_bytes(bytes)?;
        let target = if options.backend == Backend::Cpu {
            Target::Cpu
        } else {
            Target::Gpu
        };
        let (model, fallback) = match Family::load(&source, target, options.max_audio_samples).await
        {
            Ok(model) => (model, None),
            Err(error) if options.backend == Backend::Auto && error.is_fatal() => (
                Family::load(&source, Target::Cpu, options.max_audio_samples).await?,
                Some(error.to_string()),
            ),
            Err(error) => return Err(error),
        };
        Ok(Self {
            id: MODEL_ID.fetch_add(1, Ordering::Relaxed),
            source,
            model: Mutex::new(model),
            options,
            fallback: Mutex::new(fallback),
        })
    }
    pub fn device(&self) -> Result<Device> {
        Ok(self
            .model
            .try_lock()
            .map_err(|_| Error::Busy)?
            .device()
            .clone())
    }
    pub fn fallback_reason(&self) -> Option<String> {
        self.fallback
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    pub async fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        if samples.len() > self.options.max_audio_samples {
            return Err(Error::Audio(format!(
                "{} samples exceed the configured {} sample limit",
                samples.len(),
                self.options.max_audio_samples
            )));
        }
        if samples.iter().any(|s| !s.is_finite() || s.abs() > 1.0) {
            return Err(Error::Audio(
                "expected finite mono PCM samples in [-1, 1] at 16 kHz".into(),
            ));
        }
        let mut model = self.model.try_lock().map_err(|_| Error::Busy)?;
        let needed = model.working_bytes(samples.len());
        if needed > self.options.max_work_bytes {
            return Err(Error::MemoryBudget {
                needed,
                budget: self.options.max_work_bytes,
            });
        }
        match model.transcribe(samples).await {
            Err(error) if self.options.backend == Backend::Auto && error.is_fatal() => {
                let reason = error.to_string();
                *model =
                    Family::load(&self.source, Target::Cpu, self.options.max_audio_samples).await?;
                *self
                    .fallback
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(reason);
                model.transcribe(samples).await
            }
            result => result,
        }
    }

    /// Append audio and return the current hypothesis. Streaming models cache
    /// every finalized encoder row and cross-attention projection. Original
    /// Moonshine models retain their bounded whole-phrase preview behavior.
    /// `Auto` replays the accumulated phrase on CPU after a GPU failure.
    pub async fn transcribe_chunk(
        &self,
        state: &mut StreamState,
        chunk: &[f32],
        final_chunk: bool,
    ) -> Result<Transcript> {
        if state.finished {
            return Err(Error::Audio("stream is already finalized".into()));
        }
        if state.model_id.is_some_and(|id| id != self.id) {
            return Err(Error::Audio("stream belongs to a different model".into()));
        }
        let samples = state
            .pcm
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| Error::Audio("audio size overflow".into()))?;
        if samples > self.options.max_audio_samples {
            return Err(Error::Audio(
                "stream exceeds the configured audio limit".into(),
            ));
        }
        if chunk.iter().any(|x| !x.is_finite() || x.abs() > 1.0) {
            return Err(Error::Audio(
                "expected finite mono PCM samples in [-1, 1] at 16 kHz".into(),
            ));
        }
        let mut model = self.model.try_lock().map_err(|_| Error::Busy)?;
        let needed = model.working_bytes(samples);
        if needed > self.options.max_work_bytes {
            return Err(Error::MemoryBudget {
                needed,
                budget: self.options.max_work_bytes,
            });
        }
        state.model_id = Some(self.id);
        state.pcm.extend_from_slice(chunk);
        let result = match &*model {
            Family::Moonshine(m) => m.transcribe(&state.pcm).await,
            Family::Streaming(m) => {
                m.update(&mut state.streaming, &state.pcm, final_chunk)
                    .await
            }
        };
        let result = match result {
            Err(error) if self.options.backend == Backend::Auto && error.is_fatal() => {
                match Family::load(&self.source, Target::Cpu, self.options.max_audio_samples).await
                {
                    Ok(cpu) => {
                        *model = cpu;
                        *self
                            .fallback
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            Some(error.to_string());
                        state.streaming = streaming::State::default();
                        match &*model {
                            Family::Moonshine(m) => m.transcribe(&state.pcm).await,
                            Family::Streaming(m) => {
                                m.update(&mut state.streaming, &state.pcm, final_chunk)
                                    .await
                            }
                        }
                    }
                    Err(load_error) => Err(load_error),
                }
            }
            result => result,
        };
        match result {
            Ok(text) => {
                state.finished = final_chunk;
                Ok(text)
            }
            Err(error) => {
                // A caller may retry the same chunk after a transient failure.
                // Invalidate mutated caches and roll back only this append.
                state.pcm.truncate(samples - chunk.len());
                state.streaming = streaming::State::default();
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_models_return_errors() {
        for bytes in [vec![], b"GGUF".to_vec(), vec![0; 1024]] {
            assert!(pollster::block_on(Model::from_bytes(bytes, Backend::Cpu)).is_err());
        }
    }
    #[test]
    #[ignore = "requires GPU access and SPEECHWORKS_TEST_MODELS fixtures"]
    fn device_loss_replays_current_phrase_on_cpu() {
        let dir = std::path::PathBuf::from(
            std::env::var("SPEECHWORKS_TEST_MODELS").expect("set fixture directory"),
        );
        let model = pollster::block_on(Model::from_bytes(
            std::fs::read(dir.join("moonshine-tiny-Q8_0.gguf")).unwrap(),
            Backend::Auto,
        ))
        .unwrap();
        assert_ne!(
            model.device().unwrap().api,
            Api::Cpu,
            "this acceptance test requires a hardware GPU"
        );
        let mut wav = hound::WavReader::open(dir.join("jfk.wav")).unwrap();
        let pcm: Vec<f32> = wav
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let before = pollster::block_on(model.transcribe(&pcm)).unwrap().text;
        model.model.lock().unwrap().destroy_gpu_for_test();
        let replayed = pollster::block_on(model.transcribe(&pcm)).unwrap().text;
        assert_eq!(replayed, before);
        assert_eq!(model.device().unwrap().api, Api::Cpu);
        assert!(model.fallback_reason().is_some());
    }
}
