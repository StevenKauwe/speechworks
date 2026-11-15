//! Opt-in real-model acceptance. Fixtures remain external to avoid bundling model weights.
//! SPEECHWORKS_TEST_MODELS=/path cargo test --release -p speechworks-inference --test real_audio -- --ignored --nocapture
use speechworks_inference::{Backend, Error, LoadOptions, Model};
use std::path::PathBuf;
fn fixtures(name: &str) -> (Vec<u8>, Vec<f32>) {
    let root = PathBuf::from(
        std::env::var("SPEECHWORKS_TEST_MODELS").expect("set SPEECHWORKS_TEST_MODELS"),
    );
    let weights =
        std::fs::read(root.join(format!("moonshine-{name}-Q8_0.gguf"))).expect("model weights");
    let mut wav = hound::WavReader::open(root.join("jfk.wav")).expect("JFK 16 kHz fixture");
    assert_eq!(
        (
            wav.spec().sample_rate,
            wav.spec().channels,
            wav.spec().bits_per_sample
        ),
        (16_000, 1, 16)
    );
    let pcm = wav
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    (weights, pcm)
}
#[test]
#[ignore = "requires downloaded Moonshine tiny/base and JFK audio"]
fn cpu_recognizes_real_speech_and_enforces_limits() {
    for name in ["tiny", "base"] {
        let (weights, pcm) = fixtures(name);
        let model = pollster::block_on(Model::from_bytes(weights, Backend::Cpu)).unwrap();
        let text = pollster::block_on(model.transcribe(&pcm)).unwrap().text;
        eprintln!("{name} CPU: {text}");
        let expected = if name == "tiny" {
            "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
        } else {
            "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
        };
        // Wavo's reference fixture at the upstream commit recorded in UPSTREAM.md.
        assert_eq!(text, expected);
        assert!(
            text.to_lowercase()
                .contains("what you can do for your country")
        );
        assert!(matches!(
            pollster::block_on(model.transcribe(&[f32::NAN])),
            Err(Error::Audio(_))
        ));
        assert!(matches!(
            pollster::block_on(model.transcribe(&vec![0.0; 16_000 * 31])),
            Err(Error::Audio(_))
        ));
    }
    let (weights, pcm) = fixtures("tiny");
    let options = LoadOptions {
        backend: Backend::Cpu,
        max_work_bytes: 1,
        ..Default::default()
    };
    let model = pollster::block_on(Model::from_bytes_with(weights, options)).unwrap();
    assert!(matches!(
        pollster::block_on(model.transcribe(&pcm)),
        Err(Error::MemoryBudget { .. })
    ));
}
#[test]
#[ignore = "requires hardware GPU, downloaded Moonshine tiny/base and JFK audio"]
fn gpu_matches_cpu_when_phrase_grows_and_shrinks() {
    for name in ["tiny", "base"] {
        let (weights, pcm) = fixtures(name);
        let cpu = pollster::block_on(Model::from_bytes(weights.clone(), Backend::Cpu)).unwrap();
        let gpu = pollster::block_on(Model::from_bytes(weights, Backend::Gpu)).unwrap();
        eprintln!("{name}: {}", gpu.device().unwrap());
        for len in [16_000 * 5, pcm.len(), 16_000 * 5] {
            let cpu = pollster::block_on(cpu.transcribe(&pcm[..len])).unwrap();
            let on_gpu = pollster::block_on(gpu.transcribe(&pcm[..len])).unwrap();
            assert_eq!(on_gpu.text, cpu.text, "{name}, {len} samples");
        }
    }
}
