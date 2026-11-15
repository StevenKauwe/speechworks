//! Pure-Rust inference engine for silero-vad (v6.2 ONNX model).
//!
//! No inference frameworks: STFT, Conv1d, LSTM and activations are
//! implemented in Rust, including CPU matrix multiplication. Each chunk includes
//! the previous 64 (16 kHz) or 32 (8 kHz) samples of context before the STFT.
//! Based on MIT-licensed silero-vad-pure 0.1.1; see UPSTREAM.md.
//!
//! ```
//! use speechworks_vad::{SampleRate, SileroVad};
//! let mut vad = SileroVad::new(SampleRate::Hz16000).unwrap();
//! let chunk = vec![0.0f32; 512];
//! let speech_prob = vad.process(&chunk).unwrap();
//! assert!((0.0..=1.0).contains(&speech_prob));
//! ```

mod conv;
mod lstm;
mod math;
#[cfg(target_arch = "aarch64")]
mod neon;
mod stft;
mod vad;
mod wav;
mod weights;

#[cfg(feature = "profile")]
pub use vad::StageProfile;
pub use vad::{SileroVad, Trace};
pub use wav::read_wav;

use std::fmt;

/// Supported sample rates. Chunk sizes: 512 samples at 16kHz, 256 at 8kHz
/// (both 32 ms of audio per chunk).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleRate {
    Hz8000,
    Hz16000,
}

/// Errors returned by this crate.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// Malformed or incompatible weight blob.
    Weights(String),
    /// `process` called with a chunk of the wrong length.
    BadChunkSize {
        expected: usize,
        got: usize,
    },
    /// Malformed or unsupported wav file.
    Wav(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Weights(msg) => write!(f, "weights error: {msg}"),
            Error::BadChunkSize { expected, got } => {
                write!(f, "bad chunk size: expected {expected} samples, got {got}")
            }
            Error::Wav(msg) => write!(f, "wav error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
