pub mod chunk;
pub mod cli;

use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::bundle::{Bundle, Rings, Served};
use crate::gpu::Pinned;
use crate::gpu::load::Loaded;
use crate::gpu::ring::{Ring, Value, encode};
use chunk::Observation;

pub struct Config {
    pub gpus: usize,
    pub ranks: crate::gpu::ranks::RanksConfig,
    pub loader: crate::gpu::load::LoaderConfig,
    pub ring: crate::gpu::ring::RingConfig,
    pub fixed: usize,
    pub count: usize,
    pub cameras: usize,
    pub side: usize,
    pub prompt: usize,
    pub tokens: usize,
    pub eos: i32,
    pub warmup: usize,
    pub request: &'static str,
    pub reply: &'static str,
    pub deadline: Duration,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            gpus: 1,
            ranks: Default::default(),
            loader: Default::default(),
            ring: Default::default(),
            fixed: 60,
            count: 13,
            cameras: 3,
            side: 224,
            prompt: 200,
            tokens: 257,
            eos: 1,
            warmup: 3,
            request: "observation",
            reply: "token",
            deadline: Duration::from_secs(60),
        }
    }
}

impl Config {
    pub fn image_floats(&self) -> usize {
        self.cameras * 3 * self.side * self.side
    }
}

#[derive(Deserialize)]
pub struct Pi0 {
    pub model: Model,
    pub rings: Rings,
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
}

impl Served for Pi0 {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

pub struct Device<'a> {
    config: &'a Config,
    pub bundle: &'a Bundle<Pi0>,
    pub loaded: Loaded,
    requests: Ring,
    replies: Ring,
    ctl: Pinned,
    closed: bool,
    images: Pinned,
    ids: Pinned,
    tokens: Pinned,
    n_tokens: usize,
}

impl<'a> Device<'a> {
    pub fn boot(config: &'a Config, bundle: &'a Bundle<Pi0>, loaded: Loaded) -> Result<Device<'a>, String> {
        let r = &bundle.reference;
        let rings = &r.served.rings;
        for (name, ring) in [(config.request, 0), (config.reply, 1)] {
            rings
                .messages
                .iter()
                .find(|m| m.name == name && m.ring == ring)
                .ok_or_else(|| format!("the bundle's rings have no {name}"))?;
        }
        let n_tokens = tokens_of(config, &r.served)?;
        let (requests, replies, ctl) =
            (Ring::requests(rings, &config.ring)?, Ring::replies(rings, &config.ring)?, Pinned::new(16)?);
        let words: Vec<u8> =
            [requests.device(), replies.device()].iter().flat_map(|a| a.to_le_bytes()).collect();
        for (name, bytes) in
            [(&r.kernel.globals.rings, words), (&r.kernel.globals.ctl, ctl.device().to_le_bytes().to_vec())]
        {
            if !loaded.put(name, &bytes)? {
                return Err(format!("the kernel has no global {name}"));
            }
        }
        loaded.launch(&r.kernel)?;
        let (images, ids, tokens) = (
            Pinned::new(config.image_floats() * 4)?,
            Pinned::new(config.prompt * 4)?,
            Pinned::new(n_tokens * 4)?,
        );
        Ok(Device {
            config,
            bundle,
            loaded,
            requests,
            replies,
            ctl,
            closed: false,
            images,
            ids,
            tokens,
            n_tokens,
        })
    }
}

pub fn tokens_of(c: &Config, p: &Pi0) -> Result<usize, String> {
    let m = p
        .rings
        .messages
        .iter()
        .find(|m| m.name == c.request && m.ring == 0)
        .ok_or("no observation message")?;
    match m.fields.iter().find(|f| f.name == "tokens").map(|f| f.n) {
        Some(n) if (1..=c.tokens).contains(&n) => Ok(n),
        other => Err(format!("the observation's tokens span of {other:?} elements, not 1 ..= {}", c.tokens)),
    }
}

impl Device<'_> {
    pub fn close(&mut self) -> Result<(), String> {
        if !self.closed {
            self.closed = true;
            self.ctl.word(0).store(1, std::sync::atomic::Ordering::Release);
            self.loaded.finish()?;
        }
        Ok(())
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl chunk::Step for Device<'_> {
    fn chunk(&mut self, o: &Observation<'_>) -> Result<Vec<i32>, String> {
        let (c, rings) = (self.config, &self.bundle.reference.served.rings);
        let request =
            rings.messages.iter().find(|m| m.name == c.request && m.ring == 0).expect("checked at boot");
        let reply =
            rings.messages.iter().find(|m| m.name == c.reply && m.ring == 1).expect("checked at boot");
        self.images.write(0, &floats(o.images));
        self.ids.write(0, &ints(o.ids));
        self.tokens.write(0, &vec![0u8; self.n_tokens * 4]);
        let span = |p: &Pinned, n: usize| Value::Span { addr: p.device(), elements: n as u64 };
        let body = encode(
            request,
            &[
                ("images", span(&self.images, c.image_floats())),
                ("ids", span(&self.ids, c.prompt)),
                ("tokens", span(&self.tokens, self.n_tokens)),
                ("lens", Value::Int(i64::from(o.lens))),
            ],
        );
        let t0 = Instant::now();
        while !self.requests.push(request.id, &body) {
            if t0.elapsed() > c.deadline {
                return Err("pi0-fast: the request ring stayed full".into());
            }
            std::hint::spin_loop();
        }
        loop {
            if let Some(id) = self.replies.pop(|_, id, _| id) {
                if id == reply.id {
                    break;
                }
                continue;
            }
            if t0.elapsed() > c.deadline {
                return Err(format!("pi0-fast: no {} reply within {:?}", c.reply, c.deadline));
            }
            std::hint::spin_loop();
        }
        Ok(self
            .tokens
            .read(0, self.n_tokens * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_rings_read_from_the_reference_and_the_observation_encodes_its_spans() {
        let c = Config::default();
        let field = |name: &str, at: usize, kind: &str, dtype: &str, n: usize| serde_json::json!({"name": name, "at": at, "kind": kind, "dtype": dtype, "n": n});
        let json = serde_json::json!({
            "format": "hyperserve-bundle-2", "point": "program=pi0fast",
            "model": { "name": "pi0fast", "ranks": 1 },
            "kernel": { "entries": [["k", 148]], "threads": 256, "cluster": 1, "smem": 0, "static_smem": 0, "cooperative": true, "stack": 0, "workspace": 0, "symmetric": 0,
                "l2": null, "homew": null, "params": ["arena"], "next": null, "loaders": [],
                "globals": { "fault": "f", "fabric": "g", "peers": "p", "peer_bytes": 32, "multicast": false, "rings": "ht_rings", "ctl": "ht_ctl" } },
            "homes": [], "image": [],
            "rings": { "slots": [2, 2], "commits_head": 0, "logprobs": 0, "messages": [
                { "name": "observation", "id": 0, "ring": 0, "bytes": 56, "fields": [
                    field("images", 0, "span", "f32", c.image_floats()), field("ids", 16, "span", "i32", c.prompt), field("tokens", 32, "span", "i32", c.tokens), field("lens", 48, "one", "i32", 0)] },
                { "name": "token", "id": 0, "ring": 1, "bytes": 4, "fields": [field("top", 0, "one", "i32", 0)] }
            ] }
        });
        let r: crate::bundle::Reference<Pi0> = serde_json::from_value(json).expect("a reference");
        assert_eq!(
            (r.served.ranks(), r.kernel.globals.rings.as_str(), r.kernel.globals.ctl.as_str()),
            (1, "ht_rings", "ht_ctl")
        );
        let m = r.served.rings.messages.iter().find(|m| m.name == c.request).unwrap();
        let b =
            encode(m, &[("images", Value::Span { addr: 0x1000, elements: 1 }), ("lens", Value::Int(902))]);
        assert_eq!(b[0..16], [0x1000u64.to_le_bytes(), (c.image_floats() as u64).to_le_bytes()].concat());
        assert_eq!(b[48..52], 902i32.to_le_bytes());
        assert_eq!(tokens_of(&c, &r.served).unwrap(), c.tokens);
    }
}
