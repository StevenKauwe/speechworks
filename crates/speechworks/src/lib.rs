//! Dictation behavior, independent of the speech model and compute backend.
//!
//! Feed 16 kHz mono frames with their speech probabilities. Drain finished
//! phrases before decoding previews. A preview replaces tentative text; only
//! a successful final decode commits a phrase. Inference errors are errors,
//! never empty successful transcripts. The caller owns scheduling and models.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
pub mod audio;

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_SAMPLES: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub start_threshold: f32,
    pub keep_threshold: f32,
    pub min_speech_ms: usize,
    pub preroll_ms: usize,
    pub end_silence_ms: usize,
    pub soft_limit_ms: usize,
    pub hard_limit_ms: usize,
    pub split_lookback_ms: usize,
    pub tail_ms: usize,
    pub preview_interval_ms: usize,
    pub previews: bool,
    pub max_pending_audio_ms: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            start_threshold: 0.3,
            keep_threshold: 0.1,
            min_speech_ms: 256,
            preroll_ms: 512,
            end_silence_ms: 512,
            soft_limit_ms: 3008,
            hard_limit_ms: 8000,
            split_lookback_ms: 1504,
            tail_ms: 300,
            preview_interval_ms: 250,
            previews: true,
            max_pending_audio_ms: 60_000,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("invalid dictation configuration: {0}")]
    Configuration(&'static str),
    #[error("audio must contain exactly 512 finite samples and a probability between zero and one")]
    InvalidFrame,
    #[error(
        "transcription cannot keep up with the microphone; stop and retry with previews disabled"
    )]
    Backpressure,
    #[error("this dictation session has already finished")]
    Finished,
}

fn frames(ms: usize) -> usize {
    ms.div_ceil(32)
}

impl Config {
    pub fn validate(&self) -> Result<(), Error> {
        if !self.start_threshold.is_finite()
            || !self.keep_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.keep_threshold)
            || !(self.keep_threshold..=1.0).contains(&self.start_threshold)
        {
            return Err(Error::Configuration(
                "speech thresholds must be ordered probabilities",
            ));
        }
        if self.min_speech_ms == 0
            || self.end_silence_ms == 0
            || self.preview_interval_ms == 0
            || self.soft_limit_ms < self.min_speech_ms
            || self.hard_limit_ms < self.soft_limit_ms
            || self.hard_limit_ms > 30_000
            || self.preroll_ms > 2000
            || self.tail_ms > 2000
            || frames(self.hard_limit_ms) * 32 + self.tail_ms > 30_000
            || self.split_lookback_ms > self.hard_limit_ms
            || self.max_pending_audio_ms < self.hard_limit_ms + self.tail_ms
            || self.max_pending_audio_ms > 300_000
        {
            return Err(Error::Configuration(
                "invalid audio durations or memory budget",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DecodeRequest {
    pub phrase: u64,
    pub revision: u64,
    pub samples: Vec<f32>,
    /// Real captured audio before the optional synthetic decode tail. Native
    /// streaming previews must not cache the temporary silence appended here.
    pub audio_samples: usize,
    pub is_final: bool,
}

#[derive(Clone)]
struct Frame {
    audio: Vec<f32>,
    probability: f32,
}

/// A single recording. Session identifiers belong to the caller, so a cancelled
/// recording's asynchronous results can never enter a later recording.
pub struct Session {
    config: Config,
    pre: VecDeque<Frame>,
    audio: Vec<Frame>,
    finals: VecDeque<DecodeRequest>,
    pending_samples: usize,
    phrase: u64,
    revision: u64,
    voiced: usize,
    quiet: usize,
    low: usize,
    elapsed_samples: u64,
    preview_at: u64,
    finished: bool,
}

impl Session {
    pub fn new(config: Config) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            config,
            pre: VecDeque::new(),
            audio: Vec::new(),
            finals: VecDeque::new(),
            pending_samples: 0,
            phrase: 0,
            revision: 0,
            voiced: 0,
            quiet: 0,
            low: 0,
            elapsed_samples: 0,
            preview_at: 0,
            finished: false,
        })
    }

    pub fn push(&mut self, samples: &[f32], probability: f32) -> Result<(), Error> {
        if self.finished {
            return Err(Error::Finished);
        }
        if samples.len() != FRAME_SAMPLES
            || samples.iter().any(|v| !v.is_finite())
            || !probability.is_finite()
            || !(0.0..=1.0).contains(&probability)
        {
            return Err(Error::InvalidFrame);
        }
        self.elapsed_samples += FRAME_SAMPLES as u64;
        let speech = probability > self.config.start_threshold
            || (!self.audio.is_empty() && probability >= self.config.keep_threshold);
        let frame = Frame {
            audio: samples.to_vec(),
            probability,
        };
        if self.audio.is_empty() && !speech {
            self.pre.push_back(frame);
            while self.pre.len() > frames(self.config.preroll_ms) {
                self.pre.pop_front();
            }
            return Ok(());
        }
        if self.audio.is_empty() {
            self.audio.extend(self.pre.drain(..));
        }
        self.audio.push(frame);
        self.revision += 1;
        if speech {
            self.voiced += 1;
            self.quiet = 0;
        } else {
            self.quiet += 1;
        }
        self.low = if probability < 0.5 { self.low + 1 } else { 0 };
        if self.quiet >= frames(self.config.end_silence_ms) {
            if self.voiced >= frames(self.config.min_speech_ms) {
                self.commit(self.audio.len())?;
            } else {
                self.clear_phrase();
            }
        } else if self.audio.len() >= frames(self.config.soft_limit_ms)
            && self.low >= 2
            && self.voiced >= frames(self.config.min_speech_ms)
        {
            self.commit(self.audio.len())?;
        } else if self.audio.len() >= frames(self.config.hard_limit_ms) {
            let from = self
                .audio
                .len()
                .saturating_sub(frames(self.config.split_lookback_ms))
                .max(1);
            let split = (from..self.audio.len())
                .min_by(|&a, &b| {
                    self.audio[a]
                        .probability
                        .total_cmp(&self.audio[b].probability)
                })
                .map_or(self.audio.len(), |i| i + 1);
            self.commit(split)?;
        }
        Ok(())
    }

    fn clear_phrase(&mut self) {
        self.audio.clear();
        self.voiced = 0;
        self.quiet = 0;
        self.low = 0;
    }

    fn collect(&self, count: usize) -> Vec<f32> {
        let mut samples = Vec::with_capacity(count * FRAME_SAMPLES + self.config.tail_ms * 16);
        for frame in &self.audio[..count] {
            samples.extend_from_slice(&frame.audio);
        }
        samples.resize(samples.len() + self.config.tail_ms * 16, 0.0);
        samples
    }

    fn commit(&mut self, count: usize) -> Result<(), Error> {
        let sample_count = count * FRAME_SAMPLES + self.config.tail_ms * 16;
        if self.pending_samples + sample_count > self.config.max_pending_audio_ms * 16 {
            self.finished = true;
            return Err(Error::Backpressure);
        }
        let samples = self.collect(count);
        self.pending_samples += samples.len();
        self.finals.push_back(DecodeRequest {
            phrase: self.phrase,
            revision: self.revision,
            samples,
            audio_samples: count * FRAME_SAMPLES,
            is_final: true,
        });
        self.phrase += 1;
        self.audio.drain(..count);
        self.voiced = self
            .audio
            .iter()
            .filter(|f| f.probability > self.config.start_threshold)
            .count();
        self.quiet = 0;
        self.low = 0;
        Ok(())
    }

    /// Drain finals first. The caller must limit in-flight decodes to one.
    pub fn take_final(&mut self) -> Option<DecodeRequest> {
        let request = self.finals.pop_front()?;
        self.pending_samples -= request.samples.len();
        Some(request)
    }

    /// At most one preview per interval. Use `is_current` after inference to
    /// reject a result superseded by a phrase boundary, stop, or cancellation.
    pub fn preview(&mut self) -> Option<DecodeRequest> {
        if self.finished
            || !self.config.previews
            || !self.finals.is_empty()
            || self.voiced < frames(self.config.min_speech_ms)
            || self.elapsed_samples.saturating_sub(self.preview_at)
                < (self.config.preview_interval_ms * 16) as u64
        {
            return None;
        }
        self.preview_at = self.elapsed_samples;
        Some(DecodeRequest {
            phrase: self.phrase,
            revision: self.revision,
            samples: self.collect(self.audio.len()),
            audio_samples: self.audio.len() * FRAME_SAMPLES,
            is_final: false,
        })
    }

    pub fn is_current(&self, request: &DecodeRequest) -> bool {
        !self.finished && self.phrase == request.phrase
    }

    pub fn current_phrase(&self) -> u64 {
        self.phrase
    }

    /// Flush even a short voiced last word when the user explicitly stops.
    pub fn finish(&mut self) -> Result<(), Error> {
        if self.finished {
            return Ok(());
        }
        if self.voiced > 0 {
            self.commit(self.audio.len())?;
        }
        self.finished = true;
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.finished = true;
        self.clear_phrase();
        self.pre.clear();
        self.finals.clear();
        self.pending_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn push(s: &mut Session, n: usize, p: f32, value: f32) {
        for _ in 0..n {
            s.push(&[value; FRAME_SAMPLES], p).unwrap();
        }
    }
    #[test]
    fn silence_is_bounded_and_does_not_transcribe() {
        let mut s = Session::new(Config::default()).unwrap();
        push(&mut s, 20_000, 0.0, 0.0);
        assert_eq!(s.pre.len(), 16);
        assert!(s.preview().is_none());
        s.finish().unwrap();
        assert!(s.take_final().is_none());
    }
    #[test]
    fn phrase_limit_includes_frame_rounding_and_decode_padding() {
        let config = Config {
            soft_limit_ms: 29_696,
            hard_limit_ms: 29_696,
            split_lookback_ms: 0,
            ..Config::default()
        };
        let mut s = Session::new(config.clone()).unwrap();
        push(&mut s, frames(config.hard_limit_ms), 0.9, 0.25);
        assert!(s.take_final().unwrap().samples.len() <= 30 * SAMPLE_RATE);
        assert!(
            Config {
                hard_limit_ms: 29_697,
                ..config
            }
            .validate()
            .is_err()
        );
        assert!(
            Config {
                hard_limit_ms: 30_000,
                tail_ms: 0,
                ..Config::default()
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn pause_commits_once_and_stale_previews_are_rejected() {
        let mut s = Session::new(Config::default()).unwrap();
        push(&mut s, 8, 0.9, 0.25);
        let preview = s.preview().unwrap();
        assert_eq!(preview.audio_samples, 8 * 512);
        assert!(
            preview.samples[preview.audio_samples..]
                .iter()
                .all(|x| *x == 0.0)
        );
        assert!(s.is_current(&preview));
        push(&mut s, 16, 0.0, 0.0);
        assert!(!s.is_current(&preview));
        let final_ = s.take_final().unwrap();
        assert_eq!(final_.phrase, 0);
        assert_eq!(final_.samples.len(), 24 * 512 + 4800);
        assert_eq!(final_.audio_samples, 24 * 512);
        s.finish().unwrap();
        assert!(s.take_final().is_none());
    }
    #[test]
    fn explicit_stop_keeps_short_last_word_and_finish_is_idempotent() {
        let mut s = Session::new(Config::default()).unwrap();
        push(&mut s, 2, 0.8, 0.3);
        s.finish().unwrap();
        s.finish().unwrap();
        assert_eq!(s.take_final().unwrap().samples[0], 0.3);
        assert!(s.take_final().is_none());
        assert_eq!(s.push(&[0.0; 512], 0.0), Err(Error::Finished));
    }
    #[test]
    fn hard_split_preserves_every_input_sample_in_order() {
        let mut s = Session::new(Config {
            tail_ms: 0,
            ..Config::default()
        })
        .unwrap();
        for i in 0..650 {
            s.push(&[i as f32; 512], 0.9).unwrap();
        }
        s.finish().unwrap();
        let mut samples = Vec::new();
        while let Some(r) = s.take_final() {
            samples.extend(r.samples);
        }
        assert_eq!(samples.len(), 650 * 512);
        for (i, c) in samples.chunks_exact(512).enumerate() {
            assert!(c.iter().all(|x| *x == i as f32));
        }
    }
    #[test]
    fn slow_inference_fails_explicitly_instead_of_dropping_words() {
        let mut s = Session::new(Config {
            max_pending_audio_ms: 9000,
            ..Config::default()
        })
        .unwrap();
        let mut error = None;
        for _ in 0..1000 {
            if let Err(e) = s.push(&[0.2; 512], 0.9) {
                error = Some(e);
                break;
            }
        }
        assert_eq!(error, Some(Error::Backpressure));
        assert!(s.pending_samples <= 9000 * 16);
    }
    #[test]
    fn cancel_releases_audio_and_final_only_never_previews() {
        let mut s = Session::new(Config {
            previews: false,
            ..Config::default()
        })
        .unwrap();
        push(&mut s, 20, 0.9, 0.5);
        assert!(s.preview().is_none());
        s.cancel();
        assert!(s.take_final().is_none());
        assert!(s.audio.is_empty());
    }
}
