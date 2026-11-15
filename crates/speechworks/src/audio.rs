//! Stateful windowed-sinc conversion to the recognizer's 16 kHz mono input.
//! Chunk boundaries do not reset filter state or round away samples.
use std::collections::VecDeque;

pub struct Resampler {
    input_rate: usize,
    samples: VecDeque<f32>,
    offset: i64,
    received: usize,
    next: f64,
}

impl Resampler {
    pub fn new(input_rate: usize) -> Result<Self, &'static str> {
        if !(8000..=192000).contains(&input_rate) {
            return Err("unsupported microphone sample rate");
        }
        Ok(Self {
            input_rate,
            samples: VecDeque::new(),
            offset: 0,
            received: 0,
            next: 0.0,
        })
    }
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>, &'static str> {
        if input.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite microphone sample");
        }
        if self.input_rate == 16000 {
            return Ok(input.iter().map(|x| x.clamp(-1.0, 1.0)).collect());
        }
        self.samples.extend(input);
        self.received += input.len();
        Ok(self.drain(false))
    }
    pub fn finish(&mut self) -> Vec<f32> {
        self.drain(true)
    }
    fn drain(&mut self, final_: bool) -> Vec<f32> {
        let radius = 24i64;
        let ratio = self.input_rate as f64 / 16000.0;
        let cutoff = (1.0 / ratio).min(1.0) * 0.94;
        let mut output = Vec::new();
        while self.next < self.received as f64
            && (final_ || self.next + (radius as f64) < self.received as f64)
        {
            let center = self.next.floor() as i64;
            let mut total = 0.0;
            let mut norm = 0.0;
            for index in center - radius + 1..=center + radius {
                let x = self.next - index as f64;
                let angle = std::f64::consts::PI * x * cutoff;
                let sinc = if angle.abs() < 1e-10 {
                    1.0
                } else {
                    angle.sin() / angle
                };
                let weight =
                    cutoff * sinc * 0.5 * (1.0 + (std::f64::consts::PI * x / radius as f64).cos());
                let value = if index < 0 {
                    0.0
                } else {
                    self.samples
                        .get((index - self.offset) as usize)
                        .copied()
                        .unwrap_or(0.0)
                };
                total += value as f64 * weight;
                norm += weight;
            }
            output.push(((total / norm) as f32).clamp(-1.0, 1.0));
            self.next += ratio;
        }
        let keep = (self.next.floor() as i64 - radius).max(0);
        while self.offset < keep && !self.samples.is_empty() {
            self.samples.pop_front();
            self.offset += 1;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_boundaries_preserve_output_at_44100_and_48000() {
        for rate in [44100, 48000] {
            let input: Vec<_> = (0..rate).map(|i| (i as f32 * 0.071).sin()).collect();
            let mut one = Resampler::new(rate).unwrap();
            let mut a = one.push(&input).unwrap();
            a.extend(one.finish());
            let mut many = Resampler::new(rate).unwrap();
            let mut b = Vec::new();
            for chunk in input.chunks(137) {
                b.extend(many.push(chunk).unwrap());
            }
            b.extend(many.finish());
            assert_eq!(a, b);
            assert!((16000..=16001).contains(&a.len()));
        }
    }
    #[test]
    fn downsample_rejects_above_nyquist_noise() {
        let mut r = Resampler::new(48000).unwrap();
        let input: Vec<_> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 12000.0 * i as f32 / 48000.0).sin())
            .collect();
        let output = r.push(&input).unwrap();
        let rms = (output[100..output.len() - 100]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / (output.len() - 200) as f32)
            .sqrt();
        assert!(rms < 0.02, "aliased RMS {rms}");
    }
}
