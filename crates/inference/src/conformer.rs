// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
use crate::{
    error::Result,
    gguf::Gguf,
    gpu::{Gpu, Linear},
};
pub fn linear(gpu: &Gpu, g: &Gguf, names: &[String], dims: &[usize], bias: bool) -> Result<Linear> {
    let w = names.iter().map(|p| g.tensor(&format!("{p}.weight"), dims));
    let mut b = Vec::new();
    for p in names.iter().filter(|_| bias) {
        b.extend(
            g.tensor(&format!("{p}.bias"), &dims[dims.len() - 1..])?
                .to_f32()?,
        );
    }
    gpu.linear(&w.collect::<Result<Vec<_>>>()?, &b)
}
