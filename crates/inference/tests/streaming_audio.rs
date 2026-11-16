//! Real-model acceptance; no model weights or audio are embedded in the crate.
use speechworks_inference::{Backend, Error, Model, StreamState};

fn fixtures(name: &str) -> (Vec<u8>, Vec<f32>) {
    let root = std::path::PathBuf::from(
        std::env::var("SPEECHWORKS_TEST_STREAMING_MODELS").expect("fixture directory"),
    );
    let weights =
        std::fs::read(root.join(format!("moonshine-streaming-{name}-Q8_0.gguf"))).unwrap();
    (weights, audio("jfk.wav"))
}

fn audio(file: &str) -> Vec<f32> {
    let root = std::path::PathBuf::from(
        std::env::var("SPEECHWORKS_TEST_STREAMING_MODELS").expect("fixture directory"),
    );
    let mut wav = hound::WavReader::open(root.join(file)).unwrap();
    wav.samples::<i16>()
        .map(|v| v.unwrap() as f32 / 32768.0)
        .collect()
}

fn check(backend: Backend) {
    for name in ["medium", "small"] {
        let (weights, pcm) = fixtures(name);
        let model = pollster::block_on(Model::from_bytes(weights, backend)).unwrap();
        assert!(model.is_streaming());
        let expected = pollster::block_on(model.transcribe(&pcm)).unwrap();
        // Reference: transcribe.cpp adfc502, pinned Q8_0 weights, CPU/Metal.
        assert_eq!(
            expected.text,
            "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
        );
        for step in [5130, 8000] {
            let mut state = StreamState::new();
            for chunk in pcm.chunks(step) {
                pollster::block_on(model.transcribe_chunk(&mut state, chunk, false)).unwrap();
            }
            assert!(state.matches_prefix(&pcm));
            assert!(!state.matches_prefix(&pcm[1..]));
            let actual = pollster::block_on(model.transcribe_chunk(&mut state, &[], true)).unwrap();
            assert_eq!(
                actual.text, expected.text,
                "{name}, {backend:?}, step {step}"
            );
            assert!(matches!(
                pollster::block_on(model.transcribe_chunk(&mut state, &[], false)),
                Err(Error::Audio(_))
            ));
        }
        // A second speaker and content, pinned to the upstream reference output.
        let fleurs = audio("fleurs-en.wav");
        let reference = if name == "medium" {
            "However, due to the slow communication channels, styles in the West could lag behind by twenty-five to thirty years."
        } else {
            "However, due to the slow communication channels, styles in the West could lag behind by 25 to 30 years."
        };
        assert_eq!(
            pollster::block_on(model.transcribe(&fleurs)).unwrap().text,
            reference
        );
        // Interleaving unrelated phrases must not share encoder/decoder history.
        let mut a = StreamState::new();
        let mut b = StreamState::new();
        pollster::block_on(model.transcribe_chunk(&mut a, &pcm[..32037], false)).unwrap();
        for chunk in fleurs.chunks(7997) {
            pollster::block_on(model.transcribe_chunk(&mut b, chunk, false)).unwrap();
        }
        assert_eq!(
            pollster::block_on(model.transcribe_chunk(&mut a, &pcm[32037..], true))
                .unwrap()
                .text,
            expected.text
        );
        assert_eq!(
            pollster::block_on(model.transcribe_chunk(&mut b, &[], true))
                .unwrap()
                .text,
            reference
        );

        // Longer than the UI's 8-second phrase limit; odd frame boundaries and a
        // second sentence exercise cache growth and learned-position offsets.
        let long: Vec<_> = pcm
            .iter()
            .copied()
            .chain(std::iter::repeat_n(0.0, 8037))
            .chain(fleurs)
            .collect();
        let wanted = pollster::block_on(model.transcribe(&long)).unwrap();
        let mut state = StreamState::new();
        for chunk in long.chunks(16013) {
            pollster::block_on(model.transcribe_chunk(&mut state, chunk, false)).unwrap();
        }
        let actual = pollster::block_on(model.transcribe_chunk(&mut state, &[], true)).unwrap();
        assert_eq!(actual.text, wanted.text, "long {name} {backend:?}");
        let mut state = StreamState::new();
        assert!(matches!(
            pollster::block_on(model.transcribe_chunk(&mut state, &[f32::NAN], false)),
            Err(Error::Audio(_))
        ));
        assert_eq!(state.samples(), 0);
        assert!(matches!(
            pollster::block_on(model.transcribe_chunk(&mut state, &vec![0.0; 480001], false)),
            Err(Error::Audio(_))
        ));
        assert_eq!(state.samples(), 0);
        assert_eq!(
            pollster::block_on(model.transcribe_chunk(&mut state, &[], true))
                .unwrap()
                .text,
            ""
        );
    }
}

#[test]
#[ignore = "requires pinned Streaming Small/Medium, JFK and FLEURS fixtures"]
fn cpu_streaming_matches_reference() {
    check(Backend::Cpu);
}

#[test]
#[ignore = "requires hardware GPU and pinned streaming fixtures"]
fn gpu_streaming_matches_reference() {
    check(Backend::Gpu);
}
