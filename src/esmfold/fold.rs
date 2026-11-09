use std::time::Instant;

use super::Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prog {
    Lm,
    Trunk,
    Sm,
}

pub trait Step {
    fn write(&mut self, prog: Prog, name: &str, bytes: &[u8]) -> Result<(), String>;
    fn zero(&mut self, prog: Prog, name: &str, len: usize) -> Result<(), String>;
    fn copy(&mut self, from: (Prog, &str), to: (Prog, &str), len: usize) -> Result<(), String>;
    fn launch(&mut self, prog: Prog) -> Result<(), String>;
    fn read(&mut self, prog: Prog, name: &str, len: usize) -> Result<Vec<u8>, String>;
}

pub fn aa(c: char) -> u32 {
    let restypes = "ARNDCQEGHILKMFPSTWYV";
    restypes.find(c.to_ascii_uppercase()).map_or(20, |i| i as u32)
}

pub fn sequence(text: &str) -> Result<Vec<u32>, String> {
    let mut body = String::new();
    for (i, l) in text.lines().map(str::trim).filter(|l| !l.is_empty()).enumerate() {
        match (i, l.starts_with('>')) {
            (0, true) => {}
            (_, true) => break,
            _ => body += l,
        }
    }
    let aas: Vec<u32> = body
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| if c.is_ascii_alphabetic() { Ok(aa(c)) } else { Err(c) })
        .collect::<Result<_, _>>()
        .map_err(|c| format!("a sequence holds {c:?}: one-letter amino acids only"))?;
    match aas.is_empty() {
        true => Err("an empty sequence".into()),
        false => Ok(aas),
    }
}

pub fn ids(aa: &[u32]) -> Vec<u8> {
    let esm: [u32; 21] = [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7, 24];
    let (cls, eos) = (0, 2);
    std::iter::once(cls)
        .chain(aa.iter().map(|&a| esm[a.min(20) as usize]))
        .chain([eos])
        .flat_map(u32::to_le_bytes)
        .collect()
}

pub fn words(aa: &[u32]) -> Vec<u8> {
    aa.iter().flat_map(|a| a.to_le_bytes()).collect()
}

pub fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

pub fn literature(cfg: &Config, aa: &[u32]) -> Vec<u8> {
    aa.iter()
        .flat_map(|&a| cfg.backbone[(a as usize).min(20)].into_iter().flatten().flat_map(f32::to_le_bytes))
        .collect()
}

fn rigid(f: &[f32]) -> ([[f32; 3]; 3], [f32; 3]) {
    let n = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3]).sqrt();
    let (w, x, y, z) = (f[0] / n, f[1] / n, f[2] / n, f[3] / n);
    let r = [
        [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
    ];
    (r, [10.0 * f[4], 10.0 * f[5], 10.0 * f[6]])
}

pub fn backbone(cfg: &Config, frame: &[f32], aa: &[u32]) -> Vec<[[f32; 3]; 3]> {
    frame
        .as_chunks::<7>()
        .0
        .iter()
        .zip(aa)
        .map(|(f, &a)| {
            let (r, t) = rigid(f);
            cfg.backbone[(a as usize).min(20)]
                .map(|v| std::array::from_fn(|i| r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] + t[i]))
        })
        .collect()
}

fn bounds(cfg: &Config) -> Vec<f32> {
    let (lo, hi, n) = (3.375f32, 21.375f32, cfg.recycle_bins - 1);
    let step = (hi - lo) / (n - 1) as f32;
    (0..n)
        .map(|i| {
            let b = lo + step * i as f32;
            b * b
        })
        .collect()
}

pub fn bins(cfg: &Config, frame: &[f32], aa: &[u32]) -> Vec<u8> {
    let sub = |p: [f32; 3], q: [f32; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let cb: Vec<[f32; 3]> = backbone(cfg, frame, aa)
        .into_iter()
        .map(|[n, ca, c]| {
            let (b, c) = (sub(ca, n), sub(c, ca));
            let a = [b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]];
            std::array::from_fn(|i| -0.582_734_3 * a[i] + 0.568_028_3 * b[i] - 0.540_674_7 * c[i] + ca[i])
        })
        .collect();
    let bounds = bounds(cfg);
    let mut out = vec![0u8; 4 * cb.len() * cb.len()];
    for (p, row) in cb.iter().zip(out.chunks_exact_mut(4 * cb.len())) {
        for (q, w) in cb.iter().zip(row.as_chunks_mut::<4>().0) {
            let d = sub(*q, *p);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let bin: i32 = bounds.iter().map(|&b| i32::from(d2 > b)).sum();
            w.copy_from_slice(&bin.to_le_bytes());
        }
    }
    out
}

pub fn plddt(cfg: &Config, logits: &[f32], l: usize) -> Vec<f64> {
    let (atoms, bins) = (cfg.lddt_atoms, cfg.lddt_bins);
    (0..l)
        .map(|i| {
            let row = &logits[(i * atoms + 1) * bins..(i * atoms + 2) * bins];
            let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let (mut z, mut e) = (0.0f64, 0.0f64);
            for (k, &v) in row.iter().enumerate() {
                let w = f64::from(v - max).exp();
                z += w;
                e += w * (k as f64 + 0.5) / bins as f64;
            }
            100.0 * e / z
        })
        .collect()
}

pub fn pdb(cfg: &Config, frame: &[f32], aa: &[u32], plddt: &[f64]) -> String {
    let resnames = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
    ];
    let mut s = String::from(
        "REMARK   1 HYPERSERVE ESMFOLD: BACKBONE N, CA, C FROM THE STRUCTURE MODULE'S FRAMES; B-FACTOR = CA PLDDT\n",
    );
    let mut serial = 1;
    for (i, ((atoms, &a), &p)) in backbone(cfg, frame, aa).iter().zip(aa).zip(plddt).enumerate() {
        for (name, x, el) in [(" N  ", atoms[0], "N"), (" CA ", atoms[1], "C"), (" C  ", atoms[2], "C")] {
            s += &format!(
                "ATOM  {serial:>5} {name} {} A{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {el:>2}\n",
                resnames[(a as usize).min(20)],
                i + 1,
                x[0],
                x[1],
                x[2],
                1.0,
                p
            );
            serial += 1;
        }
    }
    s += &format!(
        "TER   {serial:>5}      {} A{:>4}\nEND\n",
        resnames[(*aa.last().unwrap_or(&20) as usize).min(20)],
        aa.len()
    );
    s
}

pub struct Folded {
    pub wall_ms: f64,
    pub frame: Vec<f32>,
    pub plddt: Vec<f64>,
}

pub fn fold(cfg: &Config, step: &mut impl Step, aa: &[u32]) -> Result<Folded, String> {
    let (l, pairs) = (aa.len(), aa.len() * aa.len());
    let (ids, words, lit) = (ids(aa), words(aa), literature(cfg, aa));
    let t0 = Instant::now();
    step.write(Prog::Lm, "ids", &ids)?;
    step.write(Prog::Lm, "aa", &words)?;
    step.write(Prog::Sm, "backbone", &lit)?;
    step.launch(Prog::Lm)?;
    step.zero(Prog::Trunk, "s_prev", 2 * l * cfg.single)?;
    step.zero(Prog::Trunk, "z_prev", 2 * pairs * cfg.pair)?;
    step.zero(Prog::Trunk, "bins", 4 * pairs)?;
    step.copy((Prog::Lm, "s0"), (Prog::Trunk, "s0"), 4 * l * cfg.single)?;
    for r in 0..cfg.recycles {
        let more = r + 1 < cfg.recycles;
        step.launch(Prog::Trunk)?;
        step.copy((Prog::Trunk, "sm_s"), (Prog::Sm, "sm_s"), 2 * l * cfg.sm_single)?;
        step.copy((Prog::Trunk, "sm_z"), (Prog::Sm, "sm_z"), 2 * pairs * cfg.sm_pair)?;
        if more {
            step.copy((Prog::Trunk, "s"), (Prog::Trunk, "s_prev"), 2 * l * cfg.single)?;
            step.copy((Prog::Trunk, "z"), (Prog::Trunk, "z_prev"), 2 * pairs * cfg.pair)?;
        }
        step.launch(Prog::Sm)?;
        if more {
            step.copy((Prog::Sm, "bins"), (Prog::Trunk, "bins"), 4 * pairs)?;
        }
    }
    let frame = floats(&step.read(Prog::Sm, "frame", 4 * l * 7)?);
    let lddt = floats(&step.read(Prog::Sm, "lddt", 4 * l * cfg.lddt_atoms * cfg.lddt_bins)?);
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(Folded { wall_ms, frame, plddt: plddt(cfg, &lddt, l) })
}

fn bad_f32(b: &[u8]) -> usize {
    b.as_chunks::<4>().0.iter().filter(|w| !f32::from_le_bytes(**w).is_finite()).count()
}

fn bad_bf16(b: &[u8]) -> usize {
    b.as_chunks::<2>().0.iter().filter(|w| u16::from_le_bytes(**w) & 0x7f80 == 0x7f80).count()
}

pub fn probe(cfg: &Config, step: &mut impl Step, aa: &[u32]) -> Result<String, String> {
    let (l, pairs) = (aa.len(), aa.len() * aa.len());
    let mut lines = String::new();
    step.write(Prog::Lm, "ids", &ids(aa))?;
    step.write(Prog::Lm, "aa", &words(aa))?;
    step.write(Prog::Sm, "backbone", &literature(cfg, aa))?;
    step.launch(Prog::Lm)?;
    let s0 = step.read(Prog::Lm, "s0", 4 * l * cfg.single)?;
    lines += &format!("PROBE lm s0 bad={} of {}\n", bad_f32(&s0), s0.len() / 4);
    step.zero(Prog::Trunk, "s_prev", 2 * l * cfg.single)?;
    step.zero(Prog::Trunk, "z_prev", 2 * pairs * cfg.pair)?;
    step.zero(Prog::Trunk, "bins", 4 * pairs)?;
    step.copy((Prog::Lm, "s0"), (Prog::Trunk, "s0"), 4 * l * cfg.single)?;
    step.launch(Prog::Trunk)?;
    for (name, n) in [
        ("s", 2 * l * cfg.single),
        ("z", 2 * pairs * cfg.pair),
        ("sm_s", 2 * l * cfg.sm_single),
        ("sm_z", 2 * pairs * cfg.sm_pair),
    ] {
        let b = step.read(Prog::Trunk, name, n)?;
        lines += &format!("PROBE trunk {name} bad={} of {}\n", bad_bf16(&b), b.len() / 2);
    }
    step.copy((Prog::Trunk, "sm_s"), (Prog::Sm, "sm_s"), 2 * l * cfg.sm_single)?;
    step.copy((Prog::Trunk, "sm_z"), (Prog::Sm, "sm_z"), 2 * pairs * cfg.sm_pair)?;
    step.launch(Prog::Sm)?;
    for (name, n) in [("frame", 4 * l * 7), ("lddt", 4 * l * cfg.lddt_atoms * cfg.lddt_bins)] {
        let b = step.read(Prog::Sm, name, n)?;
        lines += &format!("PROBE sm {name} bad={} of {}\n", bad_f32(&b), b.len() / 4);
    }
    Ok(lines)
}

pub struct Run {
    pub lines: String,
    pub pdb: String,
}

pub fn run(cfg: &Config, step: &mut impl Step, aa: &[u32], folds: usize) -> Result<Run, String> {
    let (l, warmup) = (aa.len(), cfg.warmup);
    let mut lines = format!("FOLDS residues={l} warmup={warmup} folds={folds} recycles={}\n", cfg.recycles);
    let mut walls = Vec::new();
    let mut last = None;
    for k in 0..warmup + folds.max(1) {
        let f = fold(cfg, step, aa)?;
        let mean = f.plddt.iter().sum::<f64>() / l as f64;
        lines += &format!("FOLD fold={k} warmup={} wall_ms={:.2} plddt={mean:.2}\n", k < warmup, f.wall_ms);
        if k >= warmup {
            walls.push(f.wall_ms);
        }
        last = Some(f);
    }
    walls.sort_by(f64::total_cmp);
    let f = last.expect("one fold at least");
    let mean = f.plddt.iter().sum::<f64>() / l as f64;
    lines += &format!(
        "ESMFOLD residues={l} folds={} median_ms={:.2} min_ms={:.2} max_ms={:.2} plddt={mean:.2}\n",
        walls.len(),
        walls[walls.len() / 2],
        walls[0],
        walls[walls.len() - 1]
    );
    Ok(Run { lines, pdb: pdb(cfg, &f.frame, aa, &f.plddt) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fasta_or_plain_sequence_reads_as_af2_indices() {
        assert_eq!(sequence(">p1 lysozyme\nMKV\nfg\n>p2\nAAA\n").unwrap(), [12, 11, 19, 13, 7]);
        assert_eq!(sequence("ARX\n").unwrap(), [0, 1, 20]);
        assert!(sequence(">only a header\n").is_err());
        assert!(sequence("MK1").is_err());
        assert_eq!(
            ids(&[0, 20]).as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect::<Vec<_>>(),
            [0, 5, 24, 2]
        );
    }

    #[test]
    fn backbone_and_bins() {
        let frame = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let cfg = Config::default();
        let bb = backbone(&cfg, &frame, &[0, 0]);
        assert_eq!(bb[0][0], [-0.525, 1.363, 0.0]);
        assert_eq!(bb[1][1], [30.0, 0.0, 0.0]);
        let b: Vec<i32> =
            bins(&cfg, &frame, &[0, 0]).as_chunks::<4>().0.iter().map(|w| i32::from_le_bytes(*w)).collect();
        assert_eq!(b, vec![0, 14, 14, 0]);
    }

    #[test]
    fn plddt_of_one_hot_logits_is_its_bins_center_and_the_pdb_holds_three_atoms_a_residue() {
        let cfg = Config::default();
        let (atoms, bins) = (cfg.lddt_atoms, cfg.lddt_bins);
        let mut logits = vec![0.0f32; 2 * atoms * bins];
        logits[bins + 49] = 1e4;
        logits[(atoms + 1) * bins] = 1e4;
        let p = plddt(&cfg, &logits, 2);
        assert!((p[0] - 99.0).abs() < 1e-9 && (p[1] - 1.0).abs() < 1e-9, "{p:?}");
        let frame = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.38, 0.0, 0.0];
        let s = pdb(&cfg, &frame, &[12, 20], &p);
        let atoms: Vec<&str> = s.lines().filter(|l| l.starts_with("ATOM")).collect();
        assert_eq!(atoms.len(), 6);
        assert_eq!(&atoms[1][12..26], " CA  MET A   1");
        assert_eq!(atoms[4][30..54].split_whitespace().collect::<Vec<_>>(), ["3.800", "0.000", "0.000"]);
        assert_eq!(atoms[4][60..66].trim(), "1.00");
        assert!(s.ends_with("END\n"));
    }

    struct Log {
        ops: Vec<String>,
    }

    impl Step for Log {
        fn write(&mut self, prog: Prog, name: &str, _: &[u8]) -> Result<(), String> {
            self.ops.push(format!("w {prog:?}.{name}"));
            Ok(())
        }
        fn zero(&mut self, prog: Prog, name: &str, _: usize) -> Result<(), String> {
            self.ops.push(format!("z {prog:?}.{name}"));
            Ok(())
        }
        fn copy(&mut self, from: (Prog, &str), to: (Prog, &str), _: usize) -> Result<(), String> {
            self.ops.push(format!("c {:?}.{}>{:?}.{}", from.0, from.1, to.0, to.1));
            Ok(())
        }
        fn launch(&mut self, prog: Prog) -> Result<(), String> {
            self.ops.push(format!("l {prog:?}"));
            Ok(())
        }
        fn read(&mut self, prog: Prog, name: &str, len: usize) -> Result<Vec<u8>, String> {
            self.ops.push(format!("r {prog:?}.{name}"));
            let mut b = vec![0u8; len];
            if name == "frame" {
                b.as_chunks_mut::<28>()
                    .0
                    .iter_mut()
                    .for_each(|f| f[..4].copy_from_slice(&1.0f32.to_le_bytes()));
            }
            Ok(b)
        }
    }

    #[test]
    fn a_fold_runs_the_lm_then_four_recycles_feeding_each_ones_bins_and_states_to_the_next() {
        let cfg = Config::default();
        let mut log = Log { ops: Vec::new() };
        let r = run(&cfg, &mut log, &[0, 1, 2], 2).unwrap();
        let per_fold = log.ops.len() / 3;
        let ops = &log.ops[..per_fold];
        assert_eq!(
            &ops[..8],
            [
                "w Lm.ids",
                "w Lm.aa",
                "w Sm.backbone",
                "l Lm",
                "z Trunk.s_prev",
                "z Trunk.z_prev",
                "z Trunk.bins",
                "c Lm.s0>Trunk.s0"
            ]
        );
        assert_eq!(ops.iter().filter(|o| *o == "l Trunk").count(), cfg.recycles);
        assert_eq!(ops.iter().filter(|o| *o == "l Sm").count(), cfg.recycles);
        assert_eq!(ops.iter().filter(|o| *o == "c Sm.bins>Trunk.bins").count(), cfg.recycles - 1);
        assert_eq!(ops.iter().filter(|o| *o == "c Trunk.z>Trunk.z_prev").count(), cfg.recycles - 1);
        let last_trunk = ops.iter().rposition(|o| o == "l Trunk").unwrap();
        assert_eq!(
            &ops[last_trunk..],
            ["l Trunk", "c Trunk.sm_s>Sm.sm_s", "c Trunk.sm_z>Sm.sm_z", "l Sm", "r Sm.frame", "r Sm.lddt"]
        );
        assert_eq!(r.lines.lines().filter(|l| l.starts_with("FOLD ")).count(), 3);
        assert!(r.lines.lines().last().unwrap().starts_with("ESMFOLD residues=3 folds=2 "));
        assert_eq!(r.pdb.lines().filter(|l| l.starts_with("ATOM")).count(), 9);
    }
}
