pub mod cli;
pub mod tick;

use serde::Deserialize;

pub use crate::bundle::{Port, Ports};

use crate::bundle::{Bundle, Served};
use crate::gpu::Pinned;
use crate::gpu::load::Loaded;

#[derive(Deserialize)]
pub struct Act {
    pub model: Model,
    pub ports: Ports,
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
}

impl Served for Act {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

pub struct Config {
    pub gpus: usize,
    pub ranks: crate::gpu::ranks::RanksConfig,
    pub loader: crate::gpu::load::LoaderConfig,
    pub image: [usize; 2],
    pub colours: usize,
    pub joints: usize,
    pub state_cols: usize,
    pub chunk: usize,
    pub warmup: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            gpus: 1,
            ranks: Default::default(),
            loader: Default::default(),
            image: [480, 640],
            colours: 3,
            joints: 14,
            state_cols: 16,
            chunk: 100,
            warmup: 20,
        }
    }
}

impl Config {
    pub fn image_bytes(&self) -> usize {
        self.image[0] * self.image[1] * self.colours
    }

    pub fn state_bytes(&self) -> usize {
        self.state_cols * 4
    }

    pub fn actions_bytes(&self) -> usize {
        self.chunk * self.joints * 4
    }
}

pub struct Device<'a> {
    pub bundle: &'a Bundle<Act>,
    pub loaded: Loaded,
    pub stage: Pinned,
}

impl tick::Step for Device<'_> {
    fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String> {
        let ports = &self.bundle.reference.served.ports;
        let mut from = 0;
        items.iter().try_for_each(|&(name, bytes)| {
            let p = ports.input(name, bytes.len())?;
            self.loaded.write_via(p.home, p.at.start, bytes, &self.stage, from)?;
            from += bytes.len();
            Ok(())
        })
    }

    fn launch(&mut self) -> Result<(), String> {
        self.loaded.launch(&self.bundle.reference.kernel)?;
        self.loaded.finish()
    }

    fn read(&self, name: &str, bytes: usize) -> Result<Vec<u8>, String> {
        let p = self.bundle.reference.served.ports.output(name, bytes)?;
        self.loaded.read(p.home, p.at.start, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_facts_read_beside_the_kernel_and_ports_check_their_bytes() {
        let c = Config::default();
        let json = serde_json::json!({
            "format": "hyperserve-bundle-2", "point": "program=act",
            "model": { "name": "act", "ranks": 1 },
            "kernel": { "entries": [["k", 148]], "threads": 256, "cluster": 1, "smem": 0, "static_smem": 0, "cooperative": true, "stack": 0, "workspace": 0, "symmetric": 0,
                "l2": null, "homew": null, "params": ["arena"], "next": null, "loaders": [],
                "globals": { "fault": "f", "fabric": "g", "peers": "p", "peer_bytes": 32, "multicast": false } },
            "homes": [], "image": [],
            "ports": {
                "inputs": [
                    { "name": "image", "home": 2, "at": { "start": 0, "end": c.image_bytes() }, "dtype": "u8", "shape": [307200, 3] },
                    { "name": "state", "home": 4, "at": { "start": 0, "end": c.state_bytes() }, "dtype": "f32", "shape": [1, 16] }
                ],
                "outputs": [{ "name": "actions", "home": 7, "at": { "start": 0, "end": c.actions_bytes() }, "dtype": "f32", "shape": [100, 14] }]
            }
        });
        let r: crate::bundle::Reference<Act> = serde_json::from_value(json).expect("a reference");
        let ports = &r.served.ports;
        assert_eq!(r.served.ranks(), 1);
        assert_eq!(ports.input("image", c.image_bytes()).unwrap().home, 2);
        assert!(ports.input("state", c.state_bytes() + 1).is_err());
        assert_eq!(ports.output("actions", c.actions_bytes()).unwrap().home, 7);
    }
}
