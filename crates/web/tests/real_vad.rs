//! External real-speech acceptance fixture, never replaced with synthetic VAD probabilities.
//! SPEECHWORKS_TEST_AUDIO points to whisper.cpp's public-domain samples/jfk.wav.
use speechworks::{Config, Session};
use speechworks_vad::{SampleRate, SileroVad, read_wav};
use std::path::Path;

#[test]
#[ignore = "requires SPEECHWORKS_TEST_AUDIO=.../jfk.wav"]
fn real_speech_is_detected_and_preserved_for_final_inference() {
    let path = std::env::var("SPEECHWORKS_TEST_AUDIO").expect("real JFK WAV fixture");
    let (rate, samples) = read_wav(Path::new(&path)).unwrap();
    assert_eq!(rate, 16000);
    let mut vad = SileroVad::new(SampleRate::Hz16000).unwrap();
    let mut session = Session::new(Config::default()).unwrap();
    let mut probabilities = Vec::new();
    for chunk in samples.chunks(512) {
        let mut frame = [0.0; 512];
        frame[..chunk.len()].copy_from_slice(chunk);
        let probability = vad.process(&frame).unwrap();
        probabilities.push(probability);
        session.push(&frame, probability).unwrap();
    }
    session.finish().unwrap();
    let max = probabilities.iter().copied().fold(0.0, f32::max);
    let voiced = probabilities.iter().filter(|&&p| p > 0.3).count();
    let mut final_samples = 0;
    let mut finals = 0;
    while let Some(job) = session.take_final() {
        finals += 1;
        final_samples += job.samples.len();
    }
    eprintln!(
        "real Silero: max={max} voiced={voiced}/{} finals={finals} final_samples={final_samples}",
        probabilities.len()
    );
    assert!(
        voiced > probabilities.len() / 4,
        "real speech was not detected"
    );
    assert!(finals > 0, "real speech must reach final inference");
    assert!(
        final_samples > samples.len() / 2,
        "speech frames were dropped"
    );
}
