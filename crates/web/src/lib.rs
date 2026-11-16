//! Thin browser bindings. Capture and UI remain with the application; the model,
//! speech detector and phrase boundaries execute in Rust in a worker.
use speechworks::{Config, DecodeRequest, Session};
use speechworks_inference::{Backend, Model, StreamState};
use speechworks_vad::{SampleRate, SileroVad};
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
static THREAD_POOL_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Initialize the optional persistent shared-memory CPU pool before loading any model.
/// A failed pool is not reusable; discard this module and fall back to the plain artifact.
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
#[wasm_bindgen(js_name = initCpuThreadPool)]
pub async fn init_cpu_thread_pool(threads: usize) -> Result<(), JsValue> {
    use std::sync::atomic::Ordering;
    if !(1..=32).contains(&threads) {
        return Err(js_error("CPU worker count must be between 1 and 32"));
    }
    if THREAD_POOL_STATE
        .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(js_error(
            "CPU thread pool initialization is already attempted",
        ));
    }
    match wasm_bindgen_futures::JsFuture::from(wasm_bindgen_rayon::init_thread_pool(threads)).await
    {
        Ok(_) => {
            speechworks_inference::register_wasm_thread_pool();
            THREAD_POOL_STATE.store(2, Ordering::Release);
            Ok(())
        }
        Err(error) => {
            THREAD_POOL_STATE.store(3, Ordering::Release);
            Err(error)
        }
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

#[wasm_bindgen]
pub struct Recognizer {
    model: Model,
    stream: Option<(String, StreamState)>,
}

#[wasm_bindgen(js_name = loadModel)]
pub async fn load_model(bytes: Vec<u8>, backend: String) -> Result<Recognizer, JsValue> {
    #[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
    if THREAD_POOL_STATE.load(std::sync::atomic::Ordering::Acquire) != 2 {
        return Err(js_error(
            "Call and await initCpuThreadPool before loading a threaded model",
        ));
    }
    let backend = match backend.as_str() {
        "auto" => Backend::Auto,
        "cpu" => Backend::Cpu,
        "gpu" => Backend::Gpu,
        _ => return Err(js_error("backend must be auto, cpu or gpu")),
    };
    Ok(Recognizer {
        model: Model::from_bytes(bytes, backend).await.map_err(js_error)?,
        stream: None,
    })
}

#[wasm_bindgen]
impl Recognizer {
    pub fn device(&self) -> Result<String, JsValue> {
        Ok(self.model.device().map_err(js_error)?.to_string())
    }
    #[wasm_bindgen(js_name = fallbackReason)]
    pub fn fallback_reason(&self) -> Option<String> {
        self.model.fallback_reason()
    }
    pub async fn transcribe(&self, samples: &[f32]) -> Result<String, JsValue> {
        Ok(self.model.transcribe(samples).await.map_err(js_error)?.text)
    }
    /// A growing, possibly revised phrase. Legacy models retain their existing
    /// padded-preview behavior; streaming models consume only newly captured
    /// audio, and append synthetic tail silence once at finalization.
    #[wasm_bindgen(js_name = transcribePhrase)]
    pub async fn transcribe_phrase(
        &mut self,
        samples: &[f32],
        key: String,
        is_final: bool,
        audio_samples: usize,
    ) -> Result<String, JsValue> {
        if audio_samples > samples.len() {
            return Err(js_error("captured audio length exceeds phrase length"));
        }
        if !self.model.is_streaming() {
            return self.transcribe(samples).await;
        }
        let samples = if is_final {
            samples
        } else {
            &samples[..audio_samples]
        };
        let reset = self.stream.as_ref().is_none_or(|(id, state)| {
            id != &key || state.is_finished() || !state.matches_prefix(samples)
        });
        if reset {
            self.stream = Some((key, StreamState::new()));
        }
        let (_, state) = self.stream.as_mut().unwrap();
        let offset = state.samples();
        let text = self
            .model
            .transcribe_chunk(state, &samples[offset..], is_final)
            .await
            .map_err(js_error)?
            .text;
        if is_final {
            self.stream = None;
        }
        Ok(text)
    }
    #[wasm_bindgen(js_name = resetStream)]
    pub fn reset_stream(&mut self) {
        self.stream = None;
    }
}

#[wasm_bindgen]
pub struct DecodeJob {
    request: DecodeRequest,
}

#[wasm_bindgen]
impl DecodeJob {
    #[wasm_bindgen(getter, js_name = isFinal)]
    pub fn is_final(&self) -> bool {
        self.request.is_final
    }
    #[wasm_bindgen(getter)]
    pub fn phrase(&self) -> u32 {
        self.request.phrase as u32
    }
    #[wasm_bindgen(getter)]
    pub fn samples(&self) -> Vec<f32> {
        self.request.samples.clone()
    }
    #[wasm_bindgen(getter, js_name = audioSamples)]
    pub fn audio_samples(&self) -> usize {
        self.request.audio_samples
    }
}

#[wasm_bindgen]
pub struct DictationSession {
    session: Session,
    vad: SileroVad,
    resampler: Option<(u32, speechworks::audio::Resampler)>,
    pending: Vec<f32>,
    probability: f32,
}

#[wasm_bindgen]
impl DictationSession {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<Self, JsValue> {
        let config: Config = if config.is_undefined() || config.is_null() {
            Config::default()
        } else {
            serde_wasm_bindgen::from_value(config).map_err(js_error)?
        };
        Ok(Self {
            session: Session::new(config).map_err(js_error)?,
            vad: SileroVad::new(SampleRate::Hz16000).map_err(js_error)?,
            resampler: None,
            pending: Vec::new(),
            probability: 0.0,
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<(), JsValue> {
        if samples.len() != speechworks::FRAME_SAMPLES || samples.iter().any(|x| !x.is_finite()) {
            return Err(js_error("expected 512 finite samples at 16 kHz"));
        }
        let probability = self.vad.process(samples).map_err(js_error)?;
        self.probability = probability;
        self.session.push(samples, probability).map_err(js_error)
    }

    /// Speech probability from the most recently processed 32 ms frame.
    #[wasm_bindgen(js_name = speechProbability)]
    pub fn speech_probability(&self) -> f32 {
        self.probability
    }

    #[wasm_bindgen(js_name = pushInput)]
    pub fn push_input(&mut self, samples: &[f32], rate: u32) -> Result<(), JsValue> {
        if samples.len() > 16384 {
            return Err(js_error("microphone block too large"));
        }
        if self.resampler.is_none() {
            self.resampler = Some((
                rate,
                speechworks::audio::Resampler::new(rate as usize).map_err(js_error)?,
            ));
        }
        let (previous, resampler) = self.resampler.as_mut().unwrap();
        if *previous != rate {
            return Err(js_error("microphone sample rate changed during recording"));
        }
        let output = resampler.push(samples).map_err(js_error)?;
        self.feed(output)
    }

    #[wasm_bindgen(js_name = nextFinal)]
    pub fn next_final(&mut self) -> Option<DecodeJob> {
        self.session
            .take_final()
            .map(|request| DecodeJob { request })
    }
    pub fn preview(&mut self) -> Option<DecodeJob> {
        self.session.preview().map(|request| DecodeJob { request })
    }
    #[wasm_bindgen(js_name = isCurrent)]
    pub fn is_current(&self, phrase: u32) -> bool {
        self.session.current_phrase() == phrase as u64
    }
    pub fn finish(&mut self) -> Result<(), JsValue> {
        if let Some((_, r)) = &mut self.resampler {
            let tail = r.finish();
            self.feed(tail)?;
        }
        if !self.pending.is_empty() {
            let mut tail = std::mem::take(&mut self.pending);
            tail.resize(512, 0.0);
            self.push(&tail)?;
        }
        self.session.finish().map_err(js_error)
    }
    pub fn cancel(&mut self) {
        self.session.cancel();
    }
}

impl DictationSession {
    fn feed(&mut self, samples: Vec<f32>) -> Result<(), JsValue> {
        self.pending.extend(samples);
        while self.pending.len() >= 512 {
            let frame: Vec<_> = self.pending.drain(..512).collect();
            self.push(&frame)?;
        }
        Ok(())
    }
}
