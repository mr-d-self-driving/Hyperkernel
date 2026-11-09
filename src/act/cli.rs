use std::path::{Path, PathBuf};

use crate::act::tick;
use crate::act::{Act, Config, Device};
use crate::bundle::Bundle;
use crate::gpu::{Pinned, cuda, load, ranks};
use crate::weights::Checkpoint;

pub struct Flags {
    pub images: PathBuf,
    pub states: Option<PathBuf>,
    pub count: Option<usize>,
    pub out: Option<PathBuf>,
}

pub fn run(c: &Config, kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let images =
        std::fs::read(&flags.images).map_err(|e| format!("--images {}: {e}", flags.images.display()))?;
    let states = match &flags.states {
        Some(p) => std::fs::read(p).map_err(|e| format!("--states {}: {e}", p.display()))?,
        None => Vec::new(),
    };
    let n = flags.count.unwrap_or(images.len() / c.image_bytes());
    let bundle = Bundle::<Act>::read(kernels, 0)?;
    cuda::visible(c.gpus)?;
    let ranks = ranks::Ranks::join(0, bundle.reference.kernel.symmetric, &c.ranks)?;
    let loaded = load::load(&bundle, &ranks, Some(&Checkpoint::open(weights)?), &[], &c.loader)?;
    let stage = Pinned::new(c.image_bytes() + c.state_bytes())?;
    let t = tick::ticks(c, &images, &states, n, &mut Device { bundle: &bundle, loaded, stage })?;
    print!("{}", t.lines);
    if let Some(dir) = flags.out {
        std::fs::create_dir_all(&dir).map_err(|e| format!("--out {}: {e}", dir.display()))?;
        std::fs::write(dir.join("actions.f32"), &t.actions)
            .map_err(|e| format!("--out {}: {e}", dir.display()))?;
    }
    Ok(())
}
