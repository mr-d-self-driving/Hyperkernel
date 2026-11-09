use std::time::Instant;

use crate::openpilot::Config;

pub trait Step {
    fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String>;
    fn launch(&mut self) -> Result<(), String>;
    fn read_all(&self, items: &[(&str, usize)]) -> Result<Vec<Vec<u8>>, String>;
}

pub struct Drive {
    pub outputs: Vec<u8>,
    pub lines: String,
}

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn widened(b: &[u8]) -> Vec<u8> {
    b.chunks_exact(2)
        .flat_map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).to_le_bytes())
        .collect()
}

pub fn img(c: &Config, frames: &[u8], t: usize) -> Vec<u8> {
    let ([h, w], frame_bytes) = (c.side, c.frame_bytes());
    let at = |f: Option<usize>, cam: usize, plane: usize, y: usize, x: usize| {
        f.map_or(0, |f| frames[f * frame_bytes + ((cam * c.planes + plane) * h + y) * w + x])
    };
    let mut b = Vec::with_capacity(h * w * c.channels() * 2);
    for y in 0..h {
        for x in 0..w {
            for cam in 0..c.cameras {
                for f in [t.checked_sub(c.img_back), Some(t)] {
                    for plane in 0..c.planes {
                        b.extend_from_slice(&bf16(f32::from(at(f, cam, plane, y, x))).to_le_bytes());
                    }
                }
            }
        }
    }
    b
}

pub fn feats(c: &Config, hidden: &[Vec<u8>], t: usize) -> Vec<u8> {
    let feats_back = [32, 28, 24, 20, 16, 12, 8, 4];
    let mut b = Vec::with_capacity(feats_back.len() * c.width * 2);
    for back in feats_back {
        match t.checked_sub(back) {
            Some(s) => b.extend_from_slice(&hidden[s]),
            None => b.resize(b.len() + c.width * 2, 0),
        }
    }
    b
}

pub fn traffic(c: &Config) -> Vec<u8> {
    let mut b = vec![0; c.traffic_laid * 2];
    b[..2].copy_from_slice(&bf16(1.0).to_le_bytes());
    b
}

pub fn reads(c: &Config) -> Vec<(&'static str, usize)> {
    c.slices
        .iter()
        .filter(|(name, _)| *name != c.pad)
        .map(|&(name, n)| (name, if name == c.hidden { n * 2 } else { n * 4 }))
        .collect()
}

pub fn row(c: &Config, got: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    let mut got = got.iter();
    let mut row = Vec::with_capacity(c.outputs() * 4);
    for (name, n) in c.slices {
        match name {
            _ if name == c.pad => row.resize(row.len() + n * 4, 0),
            _ if name == c.hidden => row.extend(widened(got.next().ok_or("openpilot: an output short")?)),
            _ => row.extend_from_slice(got.next().ok_or("openpilot: an output short")?),
        }
    }
    Ok(row)
}

pub fn drive(c: &Config, frames: &[u8], n: usize, step: &mut impl Step) -> Result<Drive, String> {
    if n == 0 {
        return Err("no frames to drive".into());
    }
    if frames.len() < n * c.frame_bytes() {
        return Err(format!("{} bytes of frames, under {n} frames of {}", frames.len(), c.frame_bytes()));
    }
    let (desire, road, reads) = (vec![0u8; c.desire * 2], traffic(c), reads(c));
    let hidden_at = reads.iter().position(|&(name, _)| name == c.hidden).expect("a hidden state read");
    let (mut hidden, mut outputs, mut walls, mut lines) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n * c.outputs() * 4),
        Vec::with_capacity(n),
        String::new(),
    );
    for t in 0..n {
        let (i, f) = (img(c, frames, t), feats(c, &hidden, t));
        let t0 = Instant::now();
        step.write_all(&[("img", &i), ("feats", &f), ("desire", &desire), ("traffic", &road)])?;
        let t1 = Instant::now();
        step.launch()?;
        let t2 = Instant::now();
        let got = step.read_all(&reads)?;
        let t3 = Instant::now();
        outputs.extend(row(c, &got)?);
        hidden.push(got[hidden_at].clone());
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        lines += &format!(
            "FRAME f={t} wall_ms={:.3} write_ms={:.3} launch_ms={:.3} read_ms={:.3}\n",
            ms(t0, t3),
            ms(t0, t1),
            ms(t1, t2),
            ms(t2, t3)
        );
        walls.push(ms(t0, t3));
    }
    lines += &summary(c.warmup, n, &walls);
    Ok(Drive { outputs, lines })
}

fn summary(warmup: usize, n: usize, walls: &[f64]) -> String {
    let warmup = warmup.min(n - 1);
    let mut past: Vec<f64> = walls.iter().skip(warmup).copied().collect();
    past.sort_by(f64::total_cmp);
    let q = |p: f64| past[((past.len() as f64 * p) as usize).min(past.len() - 1)];
    format!(
        "DRIVE frames={n} warmup={warmup} median_ms={:.3} p99_ms={:.3} max_ms={:.3} frames_per_s={:.1}\n",
        q(0.5),
        q(0.99),
        q(1.0),
        1e3 / q(0.5)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn img_orders_as_model_cat() {
        let c = Config::default();
        let mut d = vec![0u8; 5 * c.frame_bytes()];
        let plane = c.side[0] * c.side[1];
        let at = |f: usize, cam: usize, p: usize, y: usize, x: usize| {
            f * c.frame_bytes() + (cam * 6 + p) * plane + y * c.side[1] + x
        };
        d[at(4, 1, 5, 3, 7)] = 200;
        d[at(0, 0, 2, 3, 7)] = 9;
        let b = img(&c, &d, 4);
        let word = |ch: usize| {
            let o = (3 * c.side[1] + 7) * 24 * 2 + ch * 2;
            f32::from_bits(u32::from(u16::from_le_bytes([b[o], b[o + 1]])) << 16)
        };
        assert_eq!(word(23), 200.0);
        assert_eq!(word(2), 9.0);
        assert_eq!(img(&c, &d, 3).len(), c.side[0] * c.side[1] * c.channels() * 2);
        let first = img(&c, &d, 0);
        let o = (3 * c.side[1] + 7) * 24 * 2 + 2 * 2;
        assert_eq!((first[o], first[o + 1]), (0, 0));
    }

    #[test]
    fn feats_rows() {
        let c = Config::default();
        let hidden: Vec<Vec<u8>> = (0..40).map(|s| vec![s as u8; c.width * 2]).collect();
        let b = feats(&c, &hidden, 30);
        assert_eq!(b.len(), 8 * c.width * 2);
        assert_eq!(b[0], 0);
        assert_eq!(b[c.width * 2], 2);
        assert_eq!(b[7 * c.width * 2], 26);
    }

    struct Echo {
        fed: Vec<Vec<u8>>,
        launches: usize,
    }

    impl Step for Echo {
        fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String> {
            assert_eq!(items.iter().map(|i| i.0).collect::<Vec<_>>(), ["img", "feats", "desire", "traffic"]);
            self.fed.push(items[1].1.to_vec());
            Ok(())
        }

        fn launch(&mut self) -> Result<(), String> {
            self.launches += 1;
            Ok(())
        }

        fn read_all(&self, items: &[(&str, usize)]) -> Result<Vec<Vec<u8>>, String> {
            let (step, hidden) = (self.launches as f32, Config::default().hidden);
            Ok(items
                .iter()
                .map(|&(name, n)| match name {
                    _ if name == hidden => (0..n / 2).flat_map(|_| bf16(step).to_le_bytes()).collect(),
                    _ => (0..n / 4).flat_map(|_| (-step).to_le_bytes()).collect(),
                })
                .collect())
        }
    }

    #[test]
    fn a_drive_feeds_each_steps_hidden_state_back_and_writes_the_graphs_rows() {
        let (c, n) = (Config::default(), 40);
        let (frame_bytes, outputs) = (c.frame_bytes(), c.outputs());
        let mut echo = Echo { fed: Vec::new(), launches: 0 };
        let d = drive(&c, &vec![7u8; n * frame_bytes], n, &mut echo).unwrap();
        assert_eq!(d.outputs.len(), n * outputs * 4);
        assert_eq!(d.lines.lines().filter(|l| l.starts_with("FRAME ")).count(), n);
        assert!(d.lines.lines().last().unwrap().starts_with("DRIVE frames=40"));
        let word = |f: usize, i: usize| {
            f32::from_le_bytes(d.outputs[(f * outputs + i) * 4..][..4].try_into().unwrap())
        };
        let hidden = c.slices.iter().take_while(|s| s.0 != c.hidden).map(|s| s.1).sum::<usize>();
        assert_eq!(word(5, 0), -6.0);
        assert_eq!(word(5, hidden), 6.0);
        assert_eq!(word(5, outputs - 1), 0.0);
        assert!(echo.fed[3].iter().all(|&b| b == 0));
        assert_eq!(
            f32::from_bits(u32::from(u16::from_le_bytes([echo.fed[32][0], echo.fed[32][1]])) << 16),
            1.0
        );
        assert!(drive(&c, &[0u8; 10], 1, &mut echo).is_err());
        assert!(drive(&c, &[], 0, &mut echo).is_err());
        let short = drive(&c, &vec![7u8; 3 * frame_bytes], 3, &mut echo).unwrap();
        let last = short.lines.lines().last().unwrap();
        assert!(last.starts_with("DRIVE frames=3 warmup=2 ") && !last.contains("NaN"), "{last}");
    }
}
