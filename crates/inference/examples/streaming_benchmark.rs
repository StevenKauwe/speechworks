//! Reproducible offline/streaming comparison on the same loaded model.
//! MODEL.gguf AUDIO.wav cpu|gpu|auto [chunk_ms=320] [repeat=3]
use speechworks_inference::{Backend, Model, StreamState};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err("expected MODEL.gguf AUDIO.wav cpu|gpu|auto [chunk_ms] [repeat]".into());
    }
    let backend = match args[3].as_str() {
        "cpu" => Backend::Cpu,
        "gpu" => Backend::Gpu,
        "auto" => Backend::Auto,
        _ => return Err("unknown backend".into()),
    };
    let chunk_ms: usize = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(320);
    let repeat: usize = args.get(5).map(|v| v.parse()).transpose()?.unwrap_or(3);
    if chunk_ms == 0 || repeat == 0 {
        return Err("chunk and repeat must be positive".into());
    }
    let mut wav = hound::WavReader::open(&args[2])?;
    if wav.spec().sample_rate != 16_000
        || wav.spec().channels != 1
        || wav.spec().bits_per_sample != 16
    {
        return Err("expected mono 16-kHz PCM16 WAV".into());
    }
    let pcm = wav
        .samples::<i16>()
        .map(|v| v.map(|v| v as f32 / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;
    let started = Instant::now();
    let model = pollster::block_on(Model::from_bytes(std::fs::read(&args[1])?, backend))?;
    println!(
        "device={} kernels={:?} load_ms={:.3} audio_ms={:.3}",
        model.device()?,
        model.device()?.kernels,
        started.elapsed().as_secs_f64() * 1000.0,
        pcm.len() as f64 / 16.0
    );
    for run in 0..repeat {
        let start = Instant::now();
        let offline = pollster::block_on(model.transcribe(&pcm))?;
        println!(
            "run={run} offline_ms={:.3} text={:?}",
            start.elapsed().as_secs_f64() * 1000.0,
            offline.text
        );
        let mut state = StreamState::new();
        let start = Instant::now();
        let mut latencies = Vec::new();
        for chunk in pcm.chunks(chunk_ms * 16) {
            let call = Instant::now();
            let preview = pollster::block_on(model.transcribe_chunk(&mut state, chunk, false))?;
            let ms = call.elapsed().as_secs_f64() * 1000.0;
            latencies.push(ms);
            if run == 0 {
                println!(
                    "preview audio_ms={} compute_ms={ms:.3} text={:?}",
                    state.samples() / 16,
                    preview.text
                );
            }
        }
        let call = Instant::now();
        let final_text = pollster::block_on(model.transcribe_chunk(&mut state, &[], true))?;
        let final_ms = call.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        latencies.sort_by(f64::total_cmp);
        println!(
            "run={run} stream_compute_ms={total_ms:.3} update_p50_ms={:.3} update_p95_ms={:.3} final_ms={final_ms:.3} parity={} text={:?}",
            latencies[latencies.len() / 2],
            latencies[((latencies.len() * 95) / 100).min(latencies.len() - 1)],
            final_text.text == offline.text,
            final_text.text
        );
        if final_text.text != offline.text {
            return Err("streaming final differs from one-shot".into());
        }
    }
    Ok(())
}
