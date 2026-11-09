pub mod cli;
pub mod draw;

use std::ops::Range;
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;

pub use draw::Sampling;

use crate::bundle::{Bundle, Ports, Served};
use crate::gpu::cuda;
use crate::gpu::load::{self, Loaded};
use crate::gpu::ranks::Ranks;
use crate::weights::Checkpoint;

pub struct Config {
    pub gpus: usize,
    pub ranks: crate::gpu::ranks::RanksConfig,
    pub loader: crate::gpu::load::LoaderConfig,
    pub prefill: &'static str,
    pub step: &'static str,
    pub alphabet: &'static [u8],
    pub copy_piece: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            gpus: 1,
            ranks: Default::default(),
            loader: Default::default(),
            prefill: "prefill",
            step: "step",
            alphabet: b"ACGT",
            copy_piece: 16 << 20,
        }
    }
}

#[derive(Deserialize)]
pub struct Evo2 {
    pub model: Model,
    pub ports: Ports,
    pub evo2: Facts,
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
}

impl Served for Evo2 {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Facts {
    pub vocab: usize,
    pub rows: usize,
    pub cache: usize,
    pub layers: usize,
    pub width: usize,
    pub head: usize,
    pub theta: f32,
    pub scale: f32,
    pub states: Vec<String>,
}

impl Facts {
    pub fn rope(&self, at: usize, rows: usize) -> Vec<f32> {
        let inv: Vec<f32> = (0..self.head / 2)
            .map(|i| bf16(1.0 / self.theta.powf((2 * i) as f32 / self.head as f32)))
            .collect();
        let mut out = Vec::with_capacity(rows * self.head);
        for p in at..at + rows {
            let angles: Vec<f32> = inv.iter().map(|&w| p as f32 / self.scale * w).collect();
            out.extend(angles.iter().map(|a| bf16(a.cos())));
            out.extend(angles.iter().map(|a| bf16(a.sin())));
        }
        out
    }
}

pub fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

pub trait Side {
    fn facts(&self) -> &Facts;
    fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), String>;
    fn read(&self, name: &str, range: Range<usize>) -> Result<Vec<u8>, String>;
    fn launch(&mut self) -> Result<(), String>;
    fn copy(&self, name: &str, to: &Self) -> Result<(), String>;
}

pub struct Program<'a> {
    bundle: &'a Bundle<Evo2>,
    loaded: Loaded,
    _ranks: Ranks,
    copy_piece: usize,
}

impl<'a> Program<'a> {
    pub fn boot(c: &Config, bundle: &'a Bundle<Evo2>, weights: &Checkpoint) -> Result<Program<'a>, String> {
        let ranks = Ranks::join(0, bundle.reference.kernel.symmetric, &c.ranks)?;
        let loaded = load::load(bundle, &ranks, Some(weights), &[], &c.loader)?;
        Ok(Program { bundle, loaded, _ranks: ranks, copy_piece: c.copy_piece })
    }

    fn ports(&self) -> &Ports {
        &self.bundle.reference.served.ports
    }
}

impl Side for Program<'_> {
    fn facts(&self) -> &Facts {
        &self.bundle.reference.served.evo2
    }

    fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), String> {
        let p = self.ports().input(name, bytes.len())?;
        self.loaded.finish()?;
        self.loaded.write(p.home, p.at.start, bytes)
    }

    fn read(&self, name: &str, range: Range<usize>) -> Result<Vec<u8>, String> {
        let p = self.ports().output(name, range.end)?;
        self.loaded.finish()?;
        self.loaded.read(p.home, p.at.start + range.start as u64, range.len())
    }

    fn launch(&mut self) -> Result<(), String> {
        self.loaded.launch(&self.bundle.reference.kernel)
    }

    fn copy(&self, name: &str, to: &Self) -> Result<(), String> {
        let (a, b) = (self.ports().input(name, 0)?, to.ports().input(name, 0)?);
        let n = (a.at.end - a.at.start) as usize;
        if b.at.end - b.at.start != n as u64 {
            return Err(format!("state {name}: {n} B into {}", b.at.end - b.at.start));
        }
        let (from, into) = (self.loaded.device(a.home) + a.at.start, to.loaded.device(b.home) + b.at.start);
        self.loaded.finish()?;
        to.loaded.finish()?;
        (0..n)
            .step_by(self.copy_piece)
            .try_for_each(|at| cuda::dtod(into + at as u64, from + at as u64, self.copy_piece.min(n - at)))
    }
}

pub fn bases(alphabet: &[u8], text: &[u8]) -> Result<Vec<u8>, String> {
    let lines = text.split(|&b| b == b'\n').filter(|line| line.trim_ascii_start().first() != Some(&b'>'));
    let bases: Vec<u8> =
        lines.flatten().filter(|b| !b.is_ascii_whitespace()).map(u8::to_ascii_uppercase).collect();
    match bases.iter().position(|b| !alphabet.contains(b)) {
        Some(i) => Err(format!(
            "base {i} is '{}', not one of {}",
            bases[i].escape_ascii(),
            String::from_utf8_lossy(alphabet)
        )),
        None if bases.is_empty() => Err("no bases".into()),
        None => Ok(bases),
    }
}

pub fn room(f: &Facts, bases: usize, steps: usize) -> Result<(), String> {
    match bases + steps {
        _ if bases == 0 || steps == 0 => {
            Err(format!("a prompt of {bases} bases and {steps} steps, not at least one of each"))
        }
        n if n - 1 > f.cache => Err(format!(
            "a prompt of {bases} bases and {steps} steps need {} cache positions, past the {} the bundle's caches hold",
            n - 1,
            f.cache
        )),
        _ => Ok(()),
    }
}

pub fn generate<S: Side>(
    prefill: &mut S,
    step: &mut S,
    prompt: &[u8],
    steps: usize,
    out: Option<&Path>,
    sampling: Sampling,
) -> Result<String, String> {
    let (f, g) = (prefill.facts().clone(), step.facts().clone());
    if g.rows != 1 || (f.vocab, f.cache, &f.states) != (g.vocab, g.cache, &g.states) {
        return Err(format!("a prefill of {f:?} and a step of {g:?}"));
    }
    room(&f, prompt.len(), steps)?;
    let (n, whole) = (prompt.len(), prompt.len() - prompt.len() % f.rows);
    let i32s = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let feed = |s: &mut S, bases: &[u8], at: usize| -> Result<(), String> {
        s.write("ids", &i32s(&bases.iter().map(|&b| i32::from(b)).collect::<Vec<_>>()))?;
        s.write("rope", &f32s(&f.rope(at, bases.len())))?;
        s.write("at", &i32s(&[at as i32]))?;
        s.launch()
    };
    let mut kept: Vec<Vec<u8>> = vec![Vec::new(); f.layers + 2];
    let mut keep = |s: &S, rows: usize| -> Result<(), String> {
        if out.is_some() {
            kept[0].extend(s.read("logits", 0..rows * f.vocab * 4)?);
            for j in 0..=f.layers {
                kept[j + 1].extend(s.read(&format!("x{j}"), 0..rows * f.width * 2)?);
            }
        }
        Ok(())
    };
    let t0 = Instant::now();
    for at in (0..whole).step_by(f.rows) {
        feed(prefill, &prompt[at..at + f.rows], at)?;
        keep(prefill, f.rows)?;
    }
    let chunked = t0.elapsed();
    if whole > 0 {
        for name in &f.states {
            prefill.copy(name, step)?;
        }
    }
    let copied = t0.elapsed() - chunked;
    for at in whole..n {
        feed(step, &prompt[at..=at], at)?;
        keep(step, 1)?;
    }
    let mut state = sampling.seed;
    let (last, row) = if whole == n { (&*prefill, f.rows - 1) } else { (&*step, 0) };
    let mut drawn = vec![base(last, row, &sampling, &mut state)?];
    let prefilled = t0.elapsed() - copied;
    let t1 = Instant::now();
    for i in 1..steps {
        feed(step, &drawn[i - 1..i], n + i - 1)?;
        drawn.push(base(step, 0, &sampling, &mut state)?);
    }
    let per = t1.elapsed().as_secs_f64() * 1e3 / (steps.max(2) - 1) as f64;
    if let Some(dir) = out {
        let write = |name: String, bytes: &[u8]| {
            std::fs::write(dir.join(&name), bytes).map_err(|e| format!("{}: {e}", dir.join(name).display()))
        };
        write("logits.f32".into(), &kept[0])?;
        for j in 0..=f.layers {
            write(format!("x{j}.bf16"), &kept[j + 1])?;
        }
    }
    let tokens = drawn.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
    Ok(format!(
        "EVO2GEN bases={} prefill_ms={:.1} copy_ms={:.2} step_ms={per:.2} tokens={tokens}\n",
        String::from_utf8_lossy(&drawn),
        prefilled.as_secs_f64() * 1e3,
        copied.as_secs_f64() * 1e3
    ))
}

fn base<S: Side>(s: &S, row: usize, sampling: &Sampling, state: &mut u64) -> Result<u8, String> {
    let vocab = s.facts().vocab;
    let bytes = s.read("logits", row * vocab * 4..(row + 1) * vocab * 4)?;
    let v: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let token = sampling.draw(&v, state);
    u8::try_from(token).map_err(|_| format!("a token drawn at {token}, past a byte"))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::BTreeMap;

    use super::*;

    struct Fake {
        facts: Facts,
        named: BTreeMap<String, Vec<u8>>,
        ats: Vec<i32>,
        copies: Cell<usize>,
    }

    impl Fake {
        fn new(rows: usize) -> Fake {
            let facts = Facts {
                vocab: 256,
                rows,
                cache: 16,
                layers: 1,
                width: 2,
                head: 4,
                theta: 10000.0,
                scale: 1.0,
                states: vec!["l0.k".into()],
            };
            let sizes = [
                ("ids", 4 * rows),
                ("rope", 16 * rows),
                ("at", 4),
                ("logits", 4 * 256 * rows),
                ("x0", 4 * rows),
                ("x1", 4 * rows),
                ("l0.k", 8),
            ];
            Fake {
                facts,
                named: sizes.iter().map(|&(n, b)| (n.to_string(), vec![0; b])).collect(),
                ats: Vec::new(),
                copies: Cell::new(0),
            }
        }

        fn i32s(&self, name: &str) -> Vec<i32> {
            self.named[name].as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect()
        }
    }

    impl Side for Fake {
        fn facts(&self) -> &Facts {
            &self.facts
        }

        fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), String> {
            let b = self.named.get_mut(name).ok_or("no place")?;
            b[..bytes.len()].copy_from_slice(bytes);
            Ok(())
        }

        fn read(&self, name: &str, range: Range<usize>) -> Result<Vec<u8>, String> {
            self.named
                .get(name)
                .and_then(|b| b.get(range).map(<[u8]>::to_vec))
                .ok_or_else(|| "no bytes".into())
        }

        fn launch(&mut self) -> Result<(), String> {
            let next_of = *b"ACGT";
            let (ids, at) = (self.i32s("ids"), self.i32s("at")[0]);
            self.ats.push(at);
            self.named.get_mut("l0.k").unwrap()[0] += 1;
            let logits = self.named.get_mut("logits").unwrap();
            logits.fill(0);
            for (r, &id) in ids.iter().enumerate() {
                let next =
                    next_of[(next_of.iter().position(|&b| i32::from(b) == id).unwrap() + 1) % 4] as usize;
                logits[(r * 256 + next) * 4..(r * 256 + next + 1) * 4].copy_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(())
        }

        fn copy(&self, _: &str, _: &Self) -> Result<(), String> {
            self.copies.set(self.copies.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn greedy_generation_feeds_each_chunk_and_step_at_its_position() {
        let (mut p, mut s) = (Fake::new(4), Fake::new(1));
        let line = generate(&mut p, &mut s, b"GTACGTAC", 5, None, Sampling::greedy()).unwrap();
        assert!(line.starts_with("EVO2GEN bases=GTACG prefill_ms="), "{line}");
        assert!(line.trim_end().ends_with("tokens=71,84,65,67,71"), "{line}");
        assert_eq!(p.ats, [0, 4]);
        assert_eq!(s.ats, [8, 9, 10, 11]);
        assert_eq!(s.i32s("ids"), [i32::from(b'C')]);
        assert_eq!(p.named["l0.k"][0], 2);
    }

    #[test]
    fn a_tail_past_the_whole_chunks_steps_a_base_at_a_time_from_their_states() {
        let (mut p, mut s) = (Fake::new(4), Fake::new(1));
        let line = generate(&mut p, &mut s, b"GTACGTACGT", 3, None, Sampling::greedy()).unwrap();
        assert!(line.starts_with("EVO2GEN bases=ACG prefill_ms="), "{line}");
        assert_eq!((p.ats, s.ats), (vec![0, 4], vec![8, 9, 10, 11]));
        assert_eq!((p.copies.get(), s.named["l0.k"][0]), (1, 4));
    }

    #[test]
    fn a_prompt_short_of_a_chunk_runs_on_the_step_alone() {
        let (mut p, mut s) = (Fake::new(4), Fake::new(1));
        let line = generate(&mut p, &mut s, b"TAC", 2, None, Sampling::greedy()).unwrap();
        assert!(line.starts_with("EVO2GEN bases=GT prefill_ms="), "{line}");
        assert_eq!((p.ats, s.ats), (vec![], vec![0, 1, 2, 3]));
        assert_eq!(p.copies.get(), 0);
    }

    #[test]
    fn a_prompt_past_the_caches_or_a_mismatched_step_is_refused() {
        let (mut p, mut s) = (Fake::new(4), Fake::new(1));
        assert!(generate(&mut p, &mut s, b"", 2, None, Sampling::greedy()).is_err());
        assert!(generate(&mut p, &mut s, b"ACGTACGT", 0, None, Sampling::greedy()).is_err());
        assert!(generate(&mut p, &mut s, b"ACGTACGT", 10, None, Sampling::greedy()).is_err());
        assert!(generate(&mut p, &mut s, b"ACGTACGT", 9, None, Sampling::greedy()).is_ok());
        assert!(generate(&mut p, &mut Fake::new(2), b"ACGT", 2, None, Sampling::greedy()).is_err());
    }

    #[test]
    fn room_names_the_positions_needed_and_the_caches_limit() {
        let f = Fake::new(4).facts;
        assert!(room(&f, 16, 1).is_ok() && room(&f, 1, 16).is_ok());
        let e = room(&f, 10, 8).unwrap_err();
        assert!(e.contains("17 cache positions") && e.contains("past the 16"), "{e}");
        assert!(room(&f, 0, 1).is_err() && room(&f, 1, 0).is_err());
    }

    #[test]
    fn bases_skip_fasta_headers_and_whitespace_and_upper_their_case() {
        let a = Config::default().alphabet;
        assert_eq!(bases(a, b">chr1 a header\nacgt\r\n  ACgt \n>two\n").unwrap(), b"ACGTACGT");
        assert_eq!(bases(a, b"gattaca\n").unwrap(), b"GATTACA");
        assert_eq!(bases(a, b" >x\nT").unwrap(), b"T");
    }

    #[test]
    fn bases_refuse_a_byte_past_the_alphabet_or_none_at_all() {
        let a = Config::default().alphabet;
        let e = bases(a, b"ACGTN").unwrap_err();
        assert!(e.contains("base 4 is 'N'") && e.contains("ACGT"), "{e}");
        assert!(bases(a, b"AC-GT").unwrap_err().contains("base 2 is '-'"));
        assert!(bases(a, "AC\u{e9}".as_bytes()).unwrap_err().contains("'\\xc3'"));
        assert_eq!(bases(a, b">only a header\n \n").unwrap_err(), "no bases");
    }

    #[test]
    fn out_keeps_every_prompt_rows_logits_and_streams() {
        let dir = std::env::temp_dir().join(format!("evo2-out-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (mut p, mut s) = (Fake::new(4), Fake::new(1));
        generate(&mut p, &mut s, b"ACGTACGTAC", 2, Some(&dir), Sampling::greedy()).unwrap();
        let (logits, x1) =
            (std::fs::read(dir.join("logits.f32")).unwrap(), std::fs::read(dir.join("x1.bf16")).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!((logits.len(), x1.len()), (10 * 256 * 4, 10 * 2 * 2));
        let one = |row: usize, base: u8| {
            logits[(row * 256 + usize::from(base)) * 4..(row * 256 + usize::from(base) + 1) * 4]
                == 1.0f32.to_le_bytes()
        };
        assert!(one(1, b'G') && one(8, b'C') && one(9, b'G'));
    }

    #[test]
    fn rope_rows_are_cosines_then_sines_rounded_to_bf16() {
        let f = Fake::new(1).facts;
        let t = f.rope(0, 2);
        assert_eq!(t[..4], [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            t[4..],
            [bf16(1f32.cos()), bf16(bf16(0.01).cos()), bf16(1f32.sin()), bf16(bf16(0.01).sin())]
        );
        assert_eq!(bf16(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(bf16(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
    }

    #[test]
    fn its_facts_and_ports_read_beside_the_kernel() {
        let json = serde_json::json!({
            "format": "hyperserve-bundle-2", "point": "evo2 prefill", "model": { "name": "evo2", "ranks": 1 },
            "kernel": { "entries": [["k", 148]], "threads": 256, "cluster": 1, "smem": 0, "static_smem": 0, "cooperative": true, "stack": 0, "workspace": 0, "symmetric": 0,
                "l2": null, "homew": null, "params": ["arena"], "next": null, "loaders": [],
                "globals": { "fault": "f", "fabric": "g", "peers": "p", "peer_bytes": 32, "multicast": false } },
            "homes": [], "image": [{ "after": { "from": [0, { "start": 0, "end": 8 }], "to": [1, { "start": 8, "end": 16 }] } }],
            "ports": {
                "inputs": [{ "name": "ids", "home": 3, "at": { "start": 0, "end": 4096 }, "dtype": "i32", "shape": [1024] }],
                "outputs": [{ "name": "logits", "home": 5, "at": { "start": 0, "end": 2097152 }, "dtype": "f32", "shape": [1024, 512] }]
            },
            "evo2": { "vocab": 512, "rows": 1024, "cache": 8256, "layers": 32, "width": 4096, "head": 128, "theta": 10000.0, "scale": 128.0, "states": ["l0.zpast", "l0.upast"] }
        });
        let r: crate::bundle::Reference<Evo2> = serde_json::from_value(json).expect("a reference");
        let f = &r.served.evo2;
        assert_eq!(
            (r.served.ranks(), f.vocab, f.rows, f.cache, f.layers, f.head, f.states.len()),
            (1, 512, 1024, 8256, 32, 128, 2)
        );
        assert!(
            r.served.ports.input("ids", 4096).is_ok()
                && r.served.ports.output("logits", 4096 * 512 + 1).is_err()
        );
        assert!(
            matches!(&r.image[..], [crate::bundle::Record::After { from: (0, f), to: (1, t) }] if *f == (0..8) && *t == (8..16))
        );
    }
}
