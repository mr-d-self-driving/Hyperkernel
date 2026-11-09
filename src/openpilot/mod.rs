pub mod cli;
pub mod drive;

use serde::Deserialize;

pub use crate::bundle::{Port, Ports};

use crate::bundle::{Bundle, Served};
use crate::gpu::load::Loaded;

#[derive(Deserialize)]
pub struct Openpilot {
    pub model: Model,
    pub ports: Ports,
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
}

impl Served for Openpilot {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

pub struct Config {
    pub gpus: usize,
    pub ranks: crate::gpu::ranks::RanksConfig,
    pub loader: crate::gpu::load::LoaderConfig,
    pub side: [usize; 2],
    pub width: usize,
    pub desire: usize,
    pub traffic_laid: usize,
    pub cameras: usize,
    pub planes: usize,
    pub img_back: usize,
    pub warmup: usize,
    pub hidden: &'static str,
    pub pad: &'static str,
    pub slices: [(&'static str, usize); 14],
}

impl Default for Config {
    fn default() -> Config {
        let (hidden, pad) = ("hidden_state", "pad");
        Config {
            gpus: 1,
            ranks: Default::default(),
            loader: Default::default(),
            side: [128, 256],
            width: 512,
            desire: 200,
            traffic_laid: 8,
            cameras: 2,
            planes: 6,
            img_back: 4,
            warmup: 20,
            hidden,
            pad,
            slices: [
                ("meta", 55),
                ("desire_pred", 32),
                ("pose", 12),
                ("wide_from_device_euler", 6),
                ("road_transform", 12),
                ("lane_lines", 528),
                ("lane_lines_prob", 8),
                ("road_edges", 264),
                ("lead", 144),
                ("lead_prob", 3),
                (hidden, 512),
                ("plan", 990),
                ("desire_state", 8),
                (pad, 2),
            ],
        }
    }
}

impl Config {
    pub fn channels(&self) -> usize {
        self.cameras * 2 * self.planes
    }

    pub fn frame_bytes(&self) -> usize {
        self.cameras * self.planes * self.side[0] * self.side[1]
    }

    pub fn outputs(&self) -> usize {
        self.slices.iter().map(|s| s.1).sum()
    }
}

pub struct Device<'a> {
    pub bundle: &'a Bundle<Openpilot>,
    pub loaded: Loaded,
}

impl drive::Step for Device<'_> {
    fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String> {
        let ports = &self.bundle.reference.served.ports;
        items.iter().try_for_each(|&(name, bytes)| {
            ports.input(name, bytes.len()).and_then(|p| self.loaded.write(p.home, p.at.start, bytes))
        })
    }

    fn launch(&mut self) -> Result<(), String> {
        self.loaded.launch(&self.bundle.reference.kernel)?;
        self.loaded.finish()
    }

    fn read_all(&self, items: &[(&str, usize)]) -> Result<Vec<Vec<u8>>, String> {
        let ports = &self.bundle.reference.served.ports;
        items
            .iter()
            .map(|&(name, bytes)| {
                ports.output(name, bytes).and_then(|p| self.loaded.read(p.home, p.at.start, bytes))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_facts_read_beside_the_kernel_and_ports_check_their_bytes() {
        let json = serde_json::json!({
            "format": "hyperserve-bundle-2", "point": "program=openpilot",
            "model": { "name": "openpilot", "ranks": 1 },
            "kernel": { "entries": [["k", 148]], "threads": 256, "cluster": 1, "smem": 0, "static_smem": 0, "cooperative": true, "stack": 0, "workspace": 0, "symmetric": 0,
                "l2": null, "homew": null, "params": ["arena"], "next": null, "loaders": [],
                "globals": { "fault": "f", "fabric": "g", "peers": "p", "peer_bytes": 32, "multicast": false } },
            "homes": [], "image": [],
            "ports": {
                "inputs": [{ "name": "img", "home": 3, "at": { "start": 0, "end": 64 }, "dtype": "bf16", "shape": [4, 8] }],
                "outputs": [{ "name": "plan", "home": 5, "at": { "start": 16, "end": 80 }, "dtype": "f32", "shape": [1, 16] }]
            }
        });
        let r: crate::bundle::Reference<Openpilot> = serde_json::from_value(json).expect("a reference");
        let ports = &r.served.ports;
        assert_eq!(r.served.ranks(), 1);
        assert_eq!(r.kernel.globals.rings, "");
        assert_eq!(ports.input("img", 64).unwrap().home, 3);
        assert!(ports.input("img", 65).is_err());
        assert_eq!(ports.output("plan", 40).unwrap().at.start, 16);
        assert!(ports.output("img", 4).is_err());
    }
}
