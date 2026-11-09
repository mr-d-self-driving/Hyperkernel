pub mod cli;
pub mod fold;

use serde::Deserialize;

pub use crate::bundle::{Port, Ports};

use crate::bundle::{Bundle, Served};
use crate::gpu::load::Loaded;
use crate::gpu::ranks::Ranks;
use fold::Prog;

#[derive(Deserialize)]
pub struct Esmfold {
    pub model: Model,
    pub ports: Ports,
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
}

impl Served for Esmfold {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

pub struct Config {
    pub gpus: usize,
    pub ranks: crate::gpu::ranks::RanksConfig,
    pub loader: crate::gpu::load::LoaderConfig,
    pub programs: [(Prog, &'static str); 3],
    pub single: usize,
    pub pair: usize,
    pub sm_single: usize,
    pub sm_pair: usize,
    pub recycles: usize,
    pub recycle_bins: usize,
    pub lddt_atoms: usize,
    pub lddt_bins: usize,
    pub warmup: usize,
    pub backbone: [[[f32; 3]; 3]; 21],
    pub probe: bool,
    pub sync: bool,
}

impl Default for Config {
    fn default() -> Config {
        let r = |nx: f32, ny: f32, cx: f32| [[nx, ny, 0.0], [0.0, 0.0, 0.0], [cx, 0.0, 0.0]];
        Config {
            gpus: 1,
            ranks: Default::default(),
            loader: Default::default(),
            programs: [(Prog::Lm, "lm"), (Prog::Trunk, "trunk"), (Prog::Sm, "sm")],
            single: 1024,
            pair: 128,
            sm_single: 384,
            sm_pair: 128,
            recycles: 4,
            recycle_bins: 15,
            lddt_atoms: 37,
            lddt_bins: 50,
            warmup: 1,
            backbone: [
                r(-0.525, 1.363, 1.526),
                r(-0.524, 1.362, 1.525),
                r(-0.536, 1.357, 1.526),
                r(-0.525, 1.362, 1.527),
                r(-0.522, 1.362, 1.524),
                r(-0.526, 1.361, 1.526),
                r(-0.528, 1.361, 1.526),
                r(-0.572, 1.337, 1.517),
                r(-0.527, 1.360, 1.525),
                r(-0.493, 1.373, 1.527),
                r(-0.520, 1.363, 1.525),
                r(-0.526, 1.362, 1.526),
                r(-0.521, 1.364, 1.525),
                r(-0.518, 1.363, 1.524),
                r(-0.566, 1.351, 1.527),
                r(-0.529, 1.360, 1.525),
                r(-0.517, 1.364, 1.526),
                r(-0.521, 1.363, 1.525),
                r(-0.522, 1.362, 1.524),
                r(-0.494, 1.373, 1.527),
                r(0.0, 0.0, 0.0),
            ],
            probe: false,
            sync: false,
        }
    }
}

pub struct Program {
    pub bundle: Bundle<Esmfold>,
    pub loaded: Loaded,
    pub ranks: Ranks,
}

pub struct Device {
    pub lm: Program,
    pub trunk: Program,
    pub sm: Program,
    pub sync: bool,
}

impl Device {
    fn at(&self, prog: Prog) -> &Program {
        match prog {
            Prog::Lm => &self.lm,
            Prog::Trunk => &self.trunk,
            Prog::Sm => &self.sm,
        }
    }
}

impl Device {
    fn serial(&self, prog: Prog) -> Result<(), String> {
        let p = self.at(prog);
        [&self.lm, &self.trunk, &self.sm].iter().try_for_each(|o| p.loaded.after(&o.loaded))
    }
}

impl fold::Step for Device {
    fn write(&mut self, prog: Prog, name: &str, bytes: &[u8]) -> Result<(), String> {
        let p = self.at(prog);
        let port = p.bundle.reference.served.ports.input(name, bytes.len())?;
        p.loaded.write(port.home, port.at.start, bytes)
    }

    fn zero(&mut self, prog: Prog, name: &str, len: usize) -> Result<(), String> {
        self.serial(prog)?;
        let p = self.at(prog);
        let port = p.bundle.reference.served.ports.input(name, len)?;
        p.loaded.zero(port.home, port.at.start, len)
    }

    fn copy(&mut self, from: (Prog, &str), to: (Prog, &str), len: usize) -> Result<(), String> {
        self.serial(to.0)?;
        let (src, dst) = (self.at(from.0), self.at(to.0));
        let out = src.bundle.reference.served.ports.output(from.1, len)?;
        let into = dst.bundle.reference.served.ports.input(to.1, len)?;
        dst.loaded.queued((into.home, into.at.start), &src.loaded, (out.home, out.at.start), len)?;
        synced(self.sync, &dst.loaded, || {
            format!("copy {:?}.{} -> {:?}.{} ({len} B)", from.0, from.1, to.0, to.1)
        })
    }

    fn launch(&mut self, prog: Prog) -> Result<(), String> {
        self.serial(prog)?;
        let p = self.at(prog);
        p.loaded.launch(&p.bundle.reference.kernel)?;
        synced(self.sync, &p.loaded, || format!("launch {prog:?}"))
    }

    fn read(&mut self, prog: Prog, name: &str, len: usize) -> Result<Vec<u8>, String> {
        let p = self.at(prog);
        let port = p.bundle.reference.served.ports.output(name, len)?;
        p.loaded.finish()?;
        p.loaded.read(port.home, port.at.start, len)
    }
}

fn synced(sync: bool, l: &Loaded, what: impl Fn() -> String) -> Result<(), String> {
    match sync {
        true => l.finish().map_err(|e| format!("{}: {e}", what())),
        false => Ok(()),
    }
}

pub fn wanted(cfg: &Config, prog: Prog, l: usize) -> Vec<(&'static str, bool, usize)> {
    let pairs = l * l;
    match prog {
        Prog::Lm => vec![("ids", true, 4 * (l + 2)), ("aa", true, 4 * l), ("s0", false, 4 * l * cfg.single)],
        Prog::Trunk => vec![
            ("s0", true, 4 * l * cfg.single),
            ("s_prev", true, 2 * l * cfg.single),
            ("z_prev", true, 2 * pairs * cfg.pair),
            ("bins", true, 4 * pairs),
            ("s", false, 2 * l * cfg.single),
            ("z", false, 2 * pairs * cfg.pair),
            ("sm_s", false, 2 * l * cfg.sm_single),
            ("sm_z", false, 2 * pairs * cfg.sm_pair),
        ],
        Prog::Sm => vec![
            ("sm_s", true, 2 * l * cfg.sm_single),
            ("sm_z", true, 2 * pairs * cfg.sm_pair),
            ("backbone", true, 4 * l * 9),
            ("frame", false, 4 * l * 7),
            ("lddt", false, 4 * l * cfg.lddt_atoms * cfg.lddt_bins),
            ("bins", false, 4 * pairs),
        ],
    }
}

pub fn check(cfg: &Config, prog: Prog, ports: &Ports, l: usize) -> Result<(), String> {
    for (name, input, bytes) in wanted(cfg, prog, l) {
        let p = if input { ports.input(name, bytes)? } else { ports.output(name, bytes)? };
        if p.at.end - p.at.start != bytes as u64 {
            return Err(format!(
                "esmfold {prog:?}: port {name} holds {} B, not {bytes} ({l} residues)",
                p.at.end - p.at.start
            ));
        }
    }
    Ok(())
}
