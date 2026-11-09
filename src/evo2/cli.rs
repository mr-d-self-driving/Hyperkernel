use std::path::{Path, PathBuf};

use clap::Args;

use crate::bundle::Bundle;
use crate::evo2::{self, Config, Evo2, Program, Sampling};
use crate::gpu::cuda;
use crate::weights::Checkpoint;

#[derive(Args)]
pub struct Flags {
    #[arg(long, help = "The prompt: a FASTA or plain file of A, C, G and T")]
    sequence: PathBuf,
    #[arg(long, default_value_t = 1500, help = "Bases to generate")]
    steps: usize,
    #[arg(long, default_value_t = 1, help = "Sample from the K likeliest bases (1 is greedy)")]
    top_k: usize,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Sample from the likeliest bases holding this much probability (0 is off)"
    )]
    top_p: f32,
    #[arg(long, default_value_t = 1.0, help = "The sampling temperature")]
    temperature: f32,
    #[arg(long, default_value_t = 0, help = "The sampling seed")]
    seed: u64,
    #[arg(
        long,
        help = "A directory to write the prompt's logits (logits.f32) and layer states (x*.bf16) into"
    )]
    out: Option<PathBuf>,
}

pub fn bundle(_: &Flags) -> &'static str {
    "evo2"
}

pub fn run(kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let c = Config::default();
    if !(0.0..=1.0).contains(&flags.top_p) || flags.temperature.is_nan() || flags.temperature <= 0.0 {
        return Err(format!(
            "--top-p {} (0 to 1), --temperature {} (past 0)",
            flags.top_p, flags.temperature
        ));
    }
    let sampling =
        Sampling { top_k: flags.top_k, top_p: flags.top_p, temperature: flags.temperature, seed: flags.seed };
    let read = std::fs::read(&flags.sequence).map_err(|e| e.to_string());
    let bases = read
        .and_then(|text| evo2::bases(c.alphabet, &text))
        .map_err(|e| format!("--sequence {}: {e}", flags.sequence.display()))?;
    let (prefill, step) =
        (Bundle::<Evo2>::read(&kernels.join(c.prefill), 0)?, Bundle::<Evo2>::read(&kernels.join(c.step), 0)?);
    evo2::room(&prefill.reference.served.evo2, bases.len(), flags.steps)
        .map_err(|e| format!("--sequence and --steps: {e}"))?;
    if let Some(dir) = &flags.out {
        std::fs::create_dir_all(dir).map_err(|e| format!("--out {}: {e}", dir.display()))?;
    }
    cuda::visible(c.gpus)?;
    let weights = Checkpoint::open(weights)?;
    let (mut prefill, mut step) =
        (Program::boot(&c, &prefill, &weights)?, Program::boot(&c, &step, &weights)?);
    print!(
        "{}",
        evo2::generate(&mut prefill, &mut step, &bases, flags.steps, flags.out.as_deref(), sampling)?
    );
    Ok(())
}
