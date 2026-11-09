use std::path::{Path, PathBuf};

use clap::Args;

use crate::{act, pi0};

#[derive(Args)]
pub struct Flags {
    #[arg(
        long,
        help = "act: raw u8 RGB images, 480 x 640 x 3 each, back to back. pi0-fast: the 3 cameras' images, f32 [3, 3, 224, 224], each pixel x / 127.5 - 1"
    )]
    images: PathBuf,
    #[arg(long, help = "act: joint states, 14 f32 per tick, back to back, cycled [default: zeros]")]
    states: Option<PathBuf>,
    #[arg(long, help = "pi0-fast: the prompt's 200 ids, i32, padded with 0")]
    ids: Option<PathBuf>,
    #[arg(long, help = "pi0-fast: the prefix's real rows (the images' rows, then the prompt's real tokens)")]
    lens: Option<i32>,
    #[arg(long, help = "pi0-fast: a file holding lens as one i32, in place of --lens")]
    lens_file: Option<PathBuf>,
    #[arg(
        long,
        help = "Ticks (act) or action chunks (pi0-fast) to run [default: act one per image, pi0-fast 13]"
    )]
    count: Option<usize>,
    #[arg(
        long,
        help = "pi0-fast: the measured run's kernel, exactly 60 tokens with EOS off (the pi0fast-fixed60 bundle)"
    )]
    fixed60: bool,
    #[arg(
        long,
        help = "act: a directory to write actions.f32 into. pi0-fast: a file to write the last chunk's token ids to"
    )]
    out: Option<PathBuf>,
}

#[derive(Clone, Copy)]
pub enum Model {
    Act,
    Pi0Fast,
}

impl Flags {
    pub fn bundle(&self, model: &Model) -> &'static str {
        match model {
            Model::Act => "act",
            Model::Pi0Fast if self.fixed60 => "pi0fast-fixed60",
            Model::Pi0Fast => "pi0fast",
        }
    }

    fn refuse(&self, model: &str, flags: &[(&str, bool)]) -> Result<(), String> {
        match flags.iter().find(|(_, set)| *set) {
            Some((f, _)) => Err(format!("--{f} is not {model}'s")),
            None => Ok(()),
        }
    }

    pub fn act(self) -> Result<act::cli::Flags, String> {
        self.refuse(
            "act",
            &[
                ("ids", self.ids.is_some()),
                ("lens", self.lens.is_some()),
                ("lens-file", self.lens_file.is_some()),
                ("fixed60", self.fixed60),
            ],
        )?;
        Ok(act::cli::Flags { images: self.images, states: self.states, count: self.count, out: self.out })
    }

    pub fn pi0(self, count: usize) -> Result<pi0::cli::Flags, String> {
        self.refuse("pi0-fast", &[("states", self.states.is_some())])?;
        let ids = self.ids.ok_or("pi0-fast: --ids FILE (the prompt's 200 ids)")?;
        Ok(pi0::cli::Flags {
            images: self.images,
            ids,
            lens: self.lens,
            lens_file: self.lens_file,
            count: self.count.unwrap_or(count),
            fixed60: self.fixed60,
            out: self.out,
        })
    }
}

pub fn run(model: Model, kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    match model {
        Model::Act => act::cli::run(&act::Config::default(), kernels, weights, flags.act()?),
        Model::Pi0Fast => {
            let c = pi0::Config::default();
            let flags = flags.pi0(c.count)?;
            pi0::cli::run(&c, kernels, weights, flags)
        }
    }
}
