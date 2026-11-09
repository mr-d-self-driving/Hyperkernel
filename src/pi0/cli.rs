use std::path::{Path, PathBuf};

use crate::bundle::Bundle;
use crate::gpu::{cuda, load, ranks};
use crate::pi0::chunk::{self, Observation};
use crate::pi0::{Config, Device, Pi0};
use crate::weights::Checkpoint;

pub struct Flags {
    pub images: PathBuf,
    pub ids: PathBuf,
    pub lens: Option<i32>,
    pub lens_file: Option<PathBuf>,
    pub count: usize,
    pub fixed60: bool,
    pub out: Option<PathBuf>,
}

fn read(p: &Path, what: &str) -> Result<Vec<u8>, String> {
    std::fs::read(p).map_err(|e| format!("--{what} {}: {e}", p.display()))
}

pub fn run(c: &Config, kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let images: Vec<f32> =
        read(&flags.images, "images")?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let ids: Vec<i32> =
        read(&flags.ids, "ids")?.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
    let lens = match (flags.lens, &flags.lens_file) {
        (Some(n), _) => n,
        (None, Some(p)) => read(p, "lens-file")?
            .first_chunk::<4>()
            .map(|b| i32::from_le_bytes(*b))
            .ok_or("--lens-file: under 4 bytes")?,
        (None, None) => return Err("--lens N or --lens-file FILE".into()),
    };
    let bundle = Bundle::<Pi0>::read(kernels, 0)?;
    cuda::visible(c.gpus)?;
    let ranks = ranks::Ranks::join(0, bundle.reference.kernel.symmetric, &c.ranks)?;
    let loaded = load::load(&bundle, &ranks, Some(&Checkpoint::open(weights)?), &[], &c.loader)?;
    let mut dev = Device::boot(c, &bundle, loaded)?;
    let o = Observation { images: &images, ids: &ids, lens };
    let c = chunk::chunks(c, &o, flags.count, flags.fixed60.then_some(c.fixed), &mut dev)?;
    dev.close()?;
    print!("{}", c.lines);
    if let (Some(p), Some(last)) = (flags.out, c.tokens.last()) {
        let bytes: Vec<u8> = last.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&p, bytes).map_err(|e| format!("--out {}: {e}", p.display()))?;
    }
    Ok(())
}
