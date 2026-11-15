//! cargo run --release -p speechworks-inference --example transcribe -- MODEL.gguf AUDIO.wav cpu|gpu|auto
use speechworks_inference::{Backend, Model};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("expected MODEL.gguf AUDIO.wav cpu|gpu|auto".into());
    }
    let backend = match args[3].as_str() {
        "cpu" => Backend::Cpu,
        "gpu" => Backend::Gpu,
        "auto" => Backend::Auto,
        _ => return Err("unknown backend".into()),
    };
    let mut wav = hound::WavReader::open(&args[2])?;
    if wav.spec().sample_rate != 16_000
        || wav.spec().channels != 1
        || wav.spec().bits_per_sample != 16
    {
        return Err("fixture must be 16 kHz mono signed 16-bit WAV".into());
    }
    let pcm: Vec<f32> = wav
        .samples::<i16>()
        .map(|s| s.map(|s| s as f32 / 32768.0))
        .collect::<Result<_, _>>()?;
    let started = std::time::Instant::now();
    let model = pollster::block_on(Model::from_bytes(std::fs::read(&args[1])?, backend))?;
    eprintln!(
        "device={} load_ms={}",
        model.device()?,
        started.elapsed().as_millis()
    );
    let started = std::time::Instant::now();
    let result = pollster::block_on(model.transcribe(&pcm))?;
    eprintln!(
        "audio_ms={} inference_ms={} fallback={:?}",
        pcm.len() / 16,
        started.elapsed().as_millis(),
        model.fallback_reason()
    );
    println!("{}", result.text);
    Ok(())
}
