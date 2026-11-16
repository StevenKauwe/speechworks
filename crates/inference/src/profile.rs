//! Opt-in native wall-clock stage measurements. Kept out of browser builds.
//! Set SPEECHWORKS_PROFILE=1 when running the benchmark example.
#[cfg(not(target_arch = "wasm32"))]
pub struct Span {
    name: &'static str,
    start: Option<std::time::Instant>,
}
#[cfg(target_arch = "wasm32")]
pub struct Span;
pub fn span(name: &'static str) -> Span {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| std::env::var_os("SPEECHWORKS_PROFILE").is_some());
        Span {
            name,
            start: enabled.then(std::time::Instant::now),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = name;
        Span
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            eprintln!(
                "speechworks_profile stage={} ms={:.3}",
                self.name,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}
