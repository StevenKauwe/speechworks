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
mod runtime;
mod sentencepiece;
mod transformer;
mod whisper {
    pub mod tokenizer;
}

pub use error::{Error, Result};
pub use gpu::{Api, Device, DeviceKind, Kernels};
use runtime::Target;
use std::sync::Mutex;

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
    source: gguf::Gguf,
    model: Mutex<moonshine::Moonshine>,
    options: LoadOptions,
    fallback: Mutex<Option<String>>,
}
impl Model {
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
        let architecture = source.get::<&str>("general.architecture")?;
        if architecture != "moonshine" {
            return Err(Error::Model(format!(
                "unsupported architecture {architecture:?}; this build supports moonshine"
            )));
        }
        let target = if options.backend == Backend::Cpu {
            Target::Cpu
        } else {
            Target::Gpu
        };
        let (model, fallback) = match moonshine::Moonshine::load(&source, target).await {
            Ok(model) => (model, None),
            Err(error) if options.backend == Backend::Auto && error.is_fatal() => (
                moonshine::Moonshine::load(&source, Target::Cpu).await?,
                Some(error.to_string()),
            ),
            Err(error) => return Err(error),
        };
        Ok(Self {
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
                *model = moonshine::Moonshine::load(&self.source, Target::Cpu).await?;
                *self
                    .fallback
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(reason);
                model.transcribe(samples).await
            }
            result => result,
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
