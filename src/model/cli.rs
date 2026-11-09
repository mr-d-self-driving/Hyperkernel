use std::net::{IpAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use crate::api::{Server, ServerConfig};
use crate::bundle::Bundle;
use crate::engine::{self, EngineConfig, Metrics, Plan};
use crate::gpu::feed::{Feed, Leader};
use crate::gpu::{cuda, load, ranks};
use crate::model::tokenizer::{Tokenizer, TokenizerConfig};
use crate::weights::Checkpoint;
use clap::builder::RangedU64ValueParser;
use clap::{Args, ValueEnum};

pub struct LlmConfig {
    pub gpus: usize,
    pub tokenizer: TokenizerConfig,
    pub server: ServerConfig,
    pub engine: EngineConfig,
    pub ranks: ranks::RanksConfig,
    pub loader: load::LoaderConfig,
    pub ring: crate::gpu::ring::RingConfig,
}

impl Default for LlmConfig {
    fn default() -> LlmConfig {
        LlmConfig {
            gpus: 4,
            tokenizer: TokenizerConfig::default(),
            server: ServerConfig::default(),
            engine: EngineConfig::default(),
            ranks: Default::default(),
            loader: Default::default(),
            ring: Default::default(),
        }
    }
}

#[derive(Args)]
pub struct Flags {
    #[arg(long, value_enum, default_value_t = Preset::C1, help = "The maximum concurrency: picks the kernel bundle and its tuned prefill plan")]
    preset: Preset,
    #[arg(long, help = "The tokenizer.json [default: WEIGHTS/tokenizer.json]")]
    tokenizer: Option<PathBuf>,
    #[arg(
        long,
        default_value = "127.0.0.1",
        help = "The address to listen on (0.0.0.0 for every interface)"
    )]
    host: IpAddr,
    #[arg(long, default_value_t = 8000, help = "The port to listen on")]
    port: u16,
    #[arg(
        long,
        env = "HYPERSERVE_API_KEY",
        hide_env_values = true,
        help = "Answer only requests sending Authorization: Bearer KEY (all but /health)"
    )]
    api_key: Option<String>,
    #[arg(long, value_parser = RangedU64ValueParser::<usize>::new().range(1..), help = "Prompt tokens per step across the prefilling lanes, less the decoding lanes' [default: the preset's]")]
    budget: Option<usize>,
    #[arg(long, value_parser = RangedU64ValueParser::<usize>::new().range(1..), help = "Prefill once every N steps while lanes decode [default: the preset's]")]
    pf_every: Option<usize>,
    #[arg(long, default_value_t = Plan::default().headroom, help = "Output tokens of cache a request reserves when it is seated")]
    headroom: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Preset {
    C1,
    C2,
    C4,
    C5,
    C6,
    C7,
    C8,
}

impl Preset {
    pub fn bundle(self) -> &'static str {
        match self {
            Preset::C1 => "c1",
            Preset::C2 => "c2",
            Preset::C4 => "c4",
            Preset::C5 | Preset::C6 | Preset::C7 => "c5",
            Preset::C8 => "c8",
        }
    }

    pub fn plan(self) -> Plan {
        let tuned = |every| Plan { budget: Some(8192), every, ..Plan::default() };
        match self {
            Preset::C1 => Plan::default(),
            Preset::C2 => tuned(224),
            Preset::C4 | Preset::C5 => tuned(448),
            Preset::C6 | Preset::C7 | Preset::C8 => tuned(640),
        }
    }
}

pub fn bundle(flags: &Flags) -> &'static str {
    flags.preset.bundle()
}

pub fn run(kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let config = LlmConfig::default();
    cuda::visible(config.gpus)?;
    let tok = Tokenizer::open(
        &flags.tokenizer.unwrap_or_else(|| weights.join("tokenizer.json")),
        &config.tokenizer,
    )?;
    let stated = ranks::rank()?;
    let rank = stated.unwrap_or(0);
    let bundle: Bundle = Bundle::read(kernels, rank)?;
    let world = bundle.reference.served.model.ranks;
    if world > 1 && stated.is_none() {
        return ranks::spawn(world, &std::env::temp_dir().join(format!("hyperserve-{}", std::process::id())));
    }
    let ranks = ranks::Ranks::join(rank, bundle.reference.kernel.symmetric, &config.ranks)?;
    let running = load::boot(&bundle, &ranks, &Checkpoint::open(weights)?, &config.loader, &config.ring)?;
    let mut feed = Feed::new(&bundle.reference, running)?;
    if rank > 0 {
        return feed.follow(&ranks);
    }
    let (queue, jobs) = mpsc::channel();
    let metrics = Arc::new(Metrics::new(config.server.metrics.iteration_tokens.clone()));
    let server = Arc::new(Server::new(
        tok,
        bundle.reference.served.model.context,
        queue,
        metrics.clone(),
        flags.api_key,
        config.server,
    )?);
    let listener = TcpListener::bind((flags.host, flags.port))
        .map_err(|e| format!("--host {} --port {}: {e}", flags.host, flags.port))?;
    let bound =
        listener.local_addr().map_err(|e| format!("--host {} --port {}: {e}", flags.host, flags.port))?;
    thread::spawn(move || server.serve(listener));
    println!("ready: http://{bound}/v1");
    let preset = flags.preset.plan();
    let plan = Plan {
        budget: flags.budget.or(preset.budget),
        every: flags.pf_every.unwrap_or(preset.every),
        headroom: flags.headroom,
    };
    engine::run(&bundle.reference, &mut Leader { feed, ranks: &ranks }, jobs, &metrics, plan, &config.engine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_c6_and_c7_run_the_c5_bundle() {
        let all = [Preset::C1, Preset::C2, Preset::C4, Preset::C5, Preset::C6, Preset::C7, Preset::C8];
        assert_eq!(all.map(Preset::bundle), ["c1", "c2", "c4", "c5", "c5", "c5", "c8"]);
    }

    #[test]
    fn each_preset_carries_its_tuned_plan() {
        let plan = |p: Preset| (p.plan().budget, p.plan().every, p.plan().headroom);
        assert_eq!(plan(Preset::C1), (None, 1, 2048));
        assert_eq!(plan(Preset::C2), (Some(8192), 224, 2048));
        assert_eq!([Preset::C4, Preset::C5].map(plan), [(Some(8192), 448, 2048); 2]);
        assert_eq!([Preset::C6, Preset::C7, Preset::C8].map(plan), [(Some(8192), 640, 2048); 3]);
    }
}
