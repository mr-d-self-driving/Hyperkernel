use std::time::Instant;

use crate::pi0::Config;

pub struct Observation<'a> {
    pub images: &'a [f32],
    pub ids: &'a [i32],
    pub lens: i32,
}

pub trait Step {
    fn chunk(&mut self, o: &Observation<'_>) -> Result<Vec<i32>, String>;
}

pub fn ended(tokens: &[i32], eos: i32, fixed: Option<usize>) -> &[i32] {
    match fixed {
        Some(n) => &tokens[..n.min(tokens.len())],
        None => &tokens[..tokens.iter().position(|&t| t == eos).map_or(tokens.len(), |k| k + 1)],
    }
}

pub struct Chunks {
    pub tokens: Vec<Vec<i32>>,
    pub lines: String,
}

pub fn chunks(
    c: &Config,
    o: &Observation<'_>,
    n: usize,
    fixed: Option<usize>,
    step: &mut impl Step,
) -> Result<Chunks, String> {
    if n == 0 {
        return Err("no chunks to run".into());
    }
    if o.images.len() != c.image_floats() || o.ids.len() != c.prompt {
        return Err(format!(
            "an observation of {} image floats and {} ids, not {} ({} cameras of 3 x {} x {}) and {}",
            o.images.len(),
            o.ids.len(),
            c.image_floats(),
            c.cameras,
            c.side,
            c.side,
            c.prompt
        ));
    }
    let (mut tokens, mut walls, mut lines) = (Vec::with_capacity(n), Vec::with_capacity(n), String::new());
    for k in 0..n {
        let t0 = Instant::now();
        let all = step.chunk(o)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let got = ended(&all, c.eos, fixed).to_vec();
        lines += &format!("CHUNK run={k} ms={ms:.3} tokens={} ids={got:?}\n", got.len());
        walls.push(ms);
        tokens.push(got);
    }
    let warmup = c.warmup.min(n - 1);
    let mut past: Vec<f64> = walls[warmup..].to_vec();
    past.sort_by(f64::total_cmp);
    let q = |p: f64| past[((past.len() as f64 * p) as usize).min(past.len() - 1)];
    lines += &format!(
        "PI0FAST chunks={n} warmup={warmup} median_ms={:.3} p99_ms={:.3} max_ms={:.3}\n",
        q(0.5),
        q(0.99),
        q(1.0)
    );
    Ok(Chunks { tokens, lines })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        calls: usize,
    }

    impl Step for Fake {
        fn chunk(&mut self, o: &Observation<'_>) -> Result<Vec<i32>, String> {
            let c = Config::default();
            assert_eq!((o.images.len(), o.ids.len(), o.lens), (c.image_floats(), c.prompt, 7));
            self.calls += 1;
            let mut t: Vec<i32> = (10..10 + c.tokens as i32).collect();
            t[5] = c.eos;
            Ok(t)
        }
    }

    #[test]
    fn a_chunk_ends_at_eos_or_its_fixed_count_and_warmup_is_left_out() {
        let config = Config::default();
        let (images, ids) = (vec![0.0f32; config.image_floats()], vec![0i32; config.prompt]);
        let o = Observation { images: &images, ids: &ids, lens: 7 };
        let mut f = Fake { calls: 0 };
        let c = chunks(&config, &o, 5, None, &mut f).unwrap();
        assert_eq!(f.calls, 5);
        assert_eq!(c.tokens[0], [10, 11, 12, 13, 14, config.eos]);
        assert!(c.lines.lines().last().unwrap().starts_with("PI0FAST chunks=5 warmup=3 "));
        let fixed = chunks(&config, &o, 1, Some(60), &mut f).unwrap();
        assert_eq!(fixed.tokens[0].len(), 60);
        assert!(fixed.lines.lines().last().unwrap().starts_with("PI0FAST chunks=1 warmup=0 "));
        assert!(
            chunks(&config, &Observation { images: &images[1..], ids: &ids, lens: 7 }, 1, None, &mut f)
                .is_err()
        );
        assert!(chunks(&config, &o, 0, None, &mut f).is_err());
        assert_eq!(ended(&[3, 4], config.eos, None), [3, 4]);
    }
}
