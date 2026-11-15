use crate::{Device, Result, gpu::Gpu};
pub enum Target {
    Gpu,
    Cpu,
}
pub enum Runtime<G, C> {
    Gpu((Gpu, G)),
    Cpu(C),
}
impl<G, C> Runtime<G, C> {
    pub async fn new(
        target: Target,
        gpu: impl FnOnce(&Gpu) -> Result<G>,
        cpu: impl FnOnce() -> Result<C>,
    ) -> Result<Self> {
        match target {
            Target::Cpu => cpu().map(Self::Cpu),
            Target::Gpu => {
                let device = Gpu::new().await?;
                let model = gpu(&device)?;
                Ok(Self::Gpu((device, model)))
            }
        }
    }
    pub fn device(&self) -> &Device {
        match self {
            Self::Gpu((gpu, _)) => gpu.info(),
            Self::Cpu(_) => crate::cpu::device(),
        }
    }
}
