use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use hyperserve::cli::At;
use hyperserve::{esmfold, evo2, mg3, model, openpilot, robot};

#[derive(Parser)]
#[command(
    name = "hyperserve",
    version,
    about = "Runs models from HyperKernels megakernel bundles on NVIDIA B300 GPUs, the first ones CUDA_VISIBLE_DEVICES lists"
)]
enum Hyperserve {
    #[command(about = "Serve an LLM over an OpenAI-compatible HTTP API (4 GPUs)")]
    Llm {
        #[arg(long, help = "The model")]
        model: Llm,
        #[command(flatten)]
        at: At,
        #[command(flatten)]
        flags: model::cli::Flags,
    },
    #[command(about = "Run a driving model over camera frames (1 GPU)")]
    Drive {
        #[arg(long, help = "The model")]
        model: Drive,
        #[command(flatten)]
        at: At,
        #[command(flatten)]
        flags: openpilot::cli::Flags,
    },
    #[command(about = "Generate DNA from a prompt sequence (1 GPU)")]
    Dna {
        #[arg(long, help = "The model")]
        model: Dna,
        #[command(flatten)]
        at: At,
        #[command(flatten)]
        flags: evo2::cli::Flags,
    },
    #[command(about = "Generate video with a world model (2 GPUs)")]
    World {
        #[arg(long, help = "The model")]
        model: World,
        #[command(flatten)]
        at: At,
        #[command(flatten)]
        flags: mg3::cli::Flags,
    },
    #[command(about = "Run a robot policy over camera images and joint states (1 GPU)")]
    Robot {
        #[arg(long, help = "The model")]
        model: Robot,
        #[command(flatten)]
        at: At,
        #[command(flatten)]
        flags: robot::Flags,
    },
    #[command(about = "Fold a protein sequence into a structure (1 GPU)")]
    Fold {
        #[arg(long, help = "The model")]
        model: Fold,
        #[command(flatten)]
        at: At,
        #[command(flatten)]
        flags: esmfold::cli::Flags,
    },
}

#[derive(Clone, ValueEnum)]
enum Llm {
    #[value(name = "deepseek-v4.1-flash")]
    DeepseekV41Flash,
}

#[derive(Clone, ValueEnum)]
enum Drive {
    Openpilot,
}

#[derive(Clone, ValueEnum)]
enum Dna {
    #[value(name = "evo2-7b")]
    Evo2,
}

#[derive(Clone, ValueEnum)]
enum World {
    Mg3,
}

#[derive(Clone, ValueEnum)]
enum Robot {
    Act,
    #[value(name = "pi0-fast")]
    Pi0Fast,
}

#[derive(Clone, ValueEnum)]
enum Fold {
    Esmfold,
}

fn main() -> ExitCode {
    match run(Hyperserve::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hyperserve: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(task: Hyperserve) -> Result<(), String> {
    match task {
        Hyperserve::Llm { model: Llm::DeepseekV41Flash, at, flags } => {
            model::cli::run(&at.ready(model::cli::bundle(&flags))?, &at.weights, flags)
        }
        Hyperserve::Drive { model: Drive::Openpilot, at, flags } => {
            openpilot::cli::run(&at.ready(openpilot::cli::bundle(&flags))?, &at.weights, flags)
        }
        Hyperserve::Dna { model: Dna::Evo2, at, flags } => {
            evo2::cli::run(&at.ready(evo2::cli::bundle(&flags))?, &at.weights, flags)
        }
        Hyperserve::World { model: World::Mg3, at, flags } => {
            mg3::cli::run(&at.ready(mg3::cli::bundle(&flags))?, &at.weights, flags)
        }
        Hyperserve::Robot { model, at, flags } => {
            let model = match model {
                Robot::Act => robot::Model::Act,
                Robot::Pi0Fast => robot::Model::Pi0Fast,
            };
            robot::run(model, &at.ready(flags.bundle(&model))?, &at.weights, flags)
        }
        Hyperserve::Fold { model: Fold::Esmfold, at, flags } => {
            esmfold::cli::run(&at.ready(esmfold::cli::bundle(&flags))?, &at.weights, flags)
        }
    }
}
