pub mod camera;
pub mod cli;
pub mod demo;
pub mod draw;
pub mod gpu;
pub mod link;
pub mod rope;
pub mod run;
pub mod spec;

use std::ops::Range;
use std::time::Duration;

use serde::Deserialize;

use crate::bundle::{Dtype, Served};
use camera::Action;
use spec::Spec;

pub struct Config {
    pub gpus: usize,
    pub ranks: crate::gpu::ranks::RanksConfig,
    pub loader: crate::gpu::load::LoaderConfig,
    pub link: &'static str,
    pub visible: &'static str,
    pub dial: Duration,
    pub primed: usize,
    pub history: &'static str,
    pub past: &'static str,
    pub scripted: Action,
    pub camera: camera::Config,
    pub demo: demo::Config,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            gpus: 2,
            ranks: Default::default(),
            loader: Default::default(),
            link: "MG3_LINK",
            visible: "CUDA_VISIBLE_DEVICES",
            dial: Duration::from_secs(1800),
            primed: 10,
            history: "l0.history",
            past: "l0.past",
            scripted: Action { keys: [1.0, 0.0, 0.0, 0.0], mouse: [0.0, 0.05] },
            camera: camera::Config::default(),
            demo: demo::Config::default(),
        }
    }
}

impl Config {
    pub fn chunk_bytes(&self) -> usize {
        self.camera.new_frames * self.demo.frame_bytes()
    }
}

#[derive(Deserialize)]
pub struct Mg3 {
    pub model: Spec,
    pub ports: Vec<Port>,
    pub codes: Codes,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Port {
    pub name: String,
    pub code: Option<usize>,
    pub home: usize,
    pub at: Range<u64>,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Codes {
    pub at: u64,
    pub n: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prog {
    Prompt,
    Chunk,
    FirstPrompt,
    First,
    Vae,
}

#[derive(Clone, Copy, Debug)]
pub struct At<'a> {
    pub prog: Prog,
    pub name: &'a str,
    pub at: usize,
}

pub trait Dev {
    fn write(&mut self, p: Prog, name: &str, at: usize, bytes: &[u8]) -> Result<(), String>;
    fn read(&mut self, p: Prog, name: &str, range: Range<usize>) -> Result<Vec<u8>, String>;
    fn launch(&mut self, p: Prog) -> Result<(), String>;
    fn finish(&mut self) -> Result<(), String>;
    fn copy(&mut self, from: At<'_>, to: At<'_>, n: usize) -> Result<(), String>;
    fn randn(&mut self, p: Prog, name: &str, stream: usize) -> Result<(), String>;
}

impl Served for Mg3 {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

impl Mg3 {
    pub fn port(&self, code: usize, name: &str) -> Result<&Port, String> {
        let named = |c: Option<usize>| self.ports.iter().find(|p| p.name == name && p.code == c);
        named(Some(code))
            .or_else(|| named(None))
            .ok_or_else(|| format!("`{name}` is no port of code {code} of {}", self.model.name))
    }
}

impl Port {
    pub fn span(&self, range: Range<usize>) -> Result<Range<u64>, String> {
        let (start, end) = (self.at.start + range.start as u64, self.at.start + range.end as u64);
        match range.start <= range.end && end <= self.at.end {
            true => Ok(start..end),
            false => Err(format!("bytes {range:?} of `{}`, its {}", self.name, self.at.end - self.at.start)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Codes, Mg3, Port, spec};
    use crate::bundle::Dtype;

    fn port(name: &str, code: Option<usize>, at: std::ops::Range<u64>) -> Port {
        Port { name: name.into(), code, home: 0, at, dtype: Dtype::Bf16, shape: Vec::new() }
    }

    #[test]
    fn a_name_is_its_codes_own_else_the_shared() {
        let f = Mg3 {
            model: spec::chunk(),
            ports: vec![port("noise", Some(1), 0..8), port("noise", None, 8..16), port("rope", None, 16..32)],
            codes: Codes { at: 0, n: 2 },
        };
        assert_eq!(f.port(1, "noise").unwrap().at, 0..8);
        assert_eq!(f.port(0, "noise").unwrap().at, 8..16);
        assert!(f.port(0, "frames").is_err());
        let rope = f.port(0, "rope").unwrap();
        assert_eq!(rope.span(4..8), Ok(20..24));
        assert!(rope.span(8..17).is_err());
    }

    #[test]
    fn its_facts_read_beside_the_kernel() {
        let json = serde_json::json!({
            "format": "hyperserve-bundle-2", "point": "program=mg3chunk",
            "kernel": { "entries": [["k", 148]], "threads": 256, "cluster": 1, "smem": 0, "static_smem": 0, "cooperative": true, "stack": 0, "workspace": 0, "symmetric": 0,
                "l2": null, "homew": null, "params": ["arena", "replay"], "next": "advance", "loaders": [],
                "globals": { "fault": "f", "fabric": "g", "peers": "p", "peer_bytes": 32, "multicast": false } },
            "homes": [], "image": [],
            "model": { "name": "mg3chunk", "ranks": 1, "tokens": 880, "grid": [22, 40], "patch_cols": 192, "heads": 24, "head": 128, "turns": 2048, "sigma": 0.8,
                "text": 512, "text_dim": 4096, "frames": 4, "windows": 3, "key_in": 6, "mouse_in": 2, "key_cols": 8, "mouse_cols": 24,
                "clip": { "memory": 5, "past": 4, "latents": 14 } },
            "ports": [{ "name": "latents", "code": 1, "home": 3, "at": { "start": 0, "end": 64 }, "dtype": "bf16", "shape": [4, 8] }],
            "codes": { "at": 4096, "n": 2 }
        });
        let r: crate::bundle::Reference<Mg3> = serde_json::from_value(json).expect("a reference");
        assert_eq!(r.served.model, spec::chunk());
        assert_eq!(crate::bundle::Served::ranks(&r.served), 1);
        assert_eq!(
            (r.served.port(1, "latents").unwrap().home, r.served.codes),
            (3, Codes { at: 4096, n: 2 })
        );
        assert!(r.served.port(0, "latents").is_err());
    }
}
