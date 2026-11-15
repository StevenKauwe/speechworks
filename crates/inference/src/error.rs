//! Errors reported without silently losing an audio phrase.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid model: {0}")]
    Model(String),
    #[error("no GPU adapter: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error("GPU device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("GPU readback: {0}")]
    Readback(#[from] wgpu::BufferAsyncError),
    #[error("GPU mapping: {0}")]
    Map(#[from] wgpu::MapRangeError),
    #[error("GPU polling: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("GPU: {0}")]
    Gpu(String),
    #[error("another transcription is in progress on this model")]
    Busy,
    #[error("invalid audio: {0}")]
    Audio(String),
    #[error("working memory estimate {needed} bytes exceeds configured budget {budget} bytes")]
    MemoryBudget { needed: usize, budget: usize },
}
impl Error {
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::Adapter(_)
                | Self::Device(_)
                | Self::Readback(_)
                | Self::Map(_)
                | Self::Poll(_)
                | Self::Gpu(_)
        )
    }
}
pub type Result<T> = std::result::Result<T, Error>;
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::Error::Model(format!($($arg)*))) };
}
pub(crate) use bail;
