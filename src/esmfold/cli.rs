use std::path::{Path, PathBuf};

use clap::Args;

use crate::bundle::Bundle;
use crate::esmfold::fold;
use crate::esmfold::{Config, Device, Esmfold, Program, check};
use crate::gpu::{cuda, load, ranks};
use crate::weights::Checkpoint;

#[derive(Args)]
pub struct Flags {
    #[arg(
        long,
        help = "The protein: one-letter amino acids",
        conflicts_with = "fasta",
        required_unless_present = "fasta"
    )]
    sequence: Option<String>,
    #[arg(long, help = "A FASTA (or plain) file whose first record is the protein")]
    fasta: Option<PathBuf>,
    #[arg(long, default_value_t = 3, help = "Timed folds, after one warm-up fold")]
    folds: usize,
    #[arg(
        long,
        help = "A file to write the last fold's backbone (N, CA, C) to, as PDB with CA pLDDT in the B-factor"
    )]
    out: Option<PathBuf>,
    #[arg(long, help = "Print each program's non-finite output counts before the folds")]
    probe: bool,
    #[arg(long, help = "Wait for each launch and copy to finish, naming the one that fails")]
    sync: bool,
}

pub fn bundle(_: &Flags) -> &'static str {
    "esmfold"
}

pub fn lengths(cfg: &Config, root: &Path) -> Vec<usize> {
    let mut ls: Vec<usize> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .filter(|l: &usize| cfg.programs.iter().all(|(_, d)| root.join(l.to_string()).join(d).is_dir()))
        .collect();
    ls.sort();
    ls
}

pub fn run(kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let cfg = Config { probe: flags.probe, sync: flags.sync, ..Config::default() };
    let text = match (&flags.sequence, &flags.fasta) {
        (Some(s), _) => s.clone(),
        (None, Some(p)) => std::fs::read_to_string(p).map_err(|e| format!("--fasta {}: {e}", p.display()))?,
        (None, None) => return Err("--sequence AA or --fasta FILE".into()),
    };
    let aa = fold::sequence(&text)?;
    let l = aa.len();
    let dir = kernels.join(l.to_string());
    if !dir.is_dir() {
        return Err(format!(
            "no esmfold bundle for {l} residues under {}: the kernels are compiled per length; lengths present: {:?}",
            kernels.display(),
            lengths(&cfg, kernels)
        ));
    }
    cuda::visible(cfg.gpus)?;
    let ck = Checkpoint::open(weights)?;
    let mut programs = Vec::with_capacity(cfg.programs.len());
    for (prog, sub) in cfg.programs {
        let bundle = Bundle::<Esmfold>::read(&dir.join(sub), 0)?;
        check(&cfg, prog, &bundle.reference.served.ports, l)?;
        let ranks = ranks::Ranks::join(0, bundle.reference.kernel.symmetric, &cfg.ranks)?;
        let loaded = load::load(&bundle, &ranks, Some(&ck), &[], &cfg.loader)?;
        programs.push(Program { bundle, loaded, ranks });
    }
    let [lm, trunk, sm]: [Program; 3] = programs.try_into().map_err(|_| "three esmfold programs")?;
    let mut dev = Device { lm, trunk, sm, sync: cfg.sync };
    if cfg.probe {
        print!("{}", fold::probe(&cfg, &mut dev, &aa)?);
    }
    let r = fold::run(&cfg, &mut dev, &aa, flags.folds)?;
    print!("{}", r.lines);
    if let Some(out) = flags.out {
        std::fs::write(&out, &r.pdb).map_err(|e| format!("--out {}: {e}", out.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_are_the_folders_holding_all_three_programs() {
        let root = std::env::temp_dir().join(format!("hyperserve-esmfold-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (l, subs) in [
            ("164", &["lm", "trunk", "sm"][..]),
            ("76", &["lm", "trunk", "sm"]),
            ("200", &["lm", "trunk"]),
            ("x", &["lm", "trunk", "sm"]),
        ] {
            subs.iter().for_each(|s| std::fs::create_dir_all(root.join(l).join(s)).unwrap());
        }
        assert_eq!(lengths(&Config::default(), &root), [76, 164]);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
