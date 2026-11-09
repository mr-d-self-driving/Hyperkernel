use std::time::Instant;

use crate::act::Config;

pub trait Step {
    fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String>;
    fn launch(&mut self) -> Result<(), String>;
    fn read(&self, name: &str, bytes: usize) -> Result<Vec<u8>, String>;
}

pub struct Ticks {
    pub actions: Vec<u8>,
    pub lines: String,
}

pub fn state(c: &Config, states: &[u8], t: usize) -> Vec<u8> {
    let mut b = vec![0u8; c.state_bytes()];
    if !states.is_empty() {
        let rows = states.len() / (c.joints * 4);
        let at = (t % rows) * c.joints * 4;
        b[..c.joints * 4].copy_from_slice(&states[at..at + c.joints * 4]);
    }
    b
}

pub fn ticks(
    c: &Config,
    images: &[u8],
    states: &[u8],
    n: usize,
    step: &mut impl Step,
) -> Result<Ticks, String> {
    if n == 0 {
        return Err("no ticks to run".into());
    }
    let image_bytes = c.image_bytes();
    let frames = images.len() / image_bytes;
    if frames == 0 || !images.len().is_multiple_of(image_bytes) {
        return Err(format!(
            "{} bytes of images, not whole {}x{}x{} u8 frames",
            images.len(),
            c.image[0],
            c.image[1],
            c.colours
        ));
    }
    if !states.len().is_multiple_of(c.joints * 4) {
        return Err(format!("{} bytes of states, not whole rows of {} f32", states.len(), c.joints));
    }
    let (mut actions, mut walls, mut lines) =
        (Vec::with_capacity(n * c.actions_bytes()), Vec::with_capacity(n), String::new());
    for t in 0..n {
        let f = t % frames;
        let (image, s) = (&images[f * image_bytes..(f + 1) * image_bytes], state(c, states, t));
        let t0 = Instant::now();
        step.write_all(&[("image", image), ("state", &s)])?;
        let t1 = Instant::now();
        step.launch()?;
        let t2 = Instant::now();
        let a = step.read("actions", c.actions_bytes())?;
        let t3 = Instant::now();
        actions.extend(a);
        let us = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e6;
        lines += &format!(
            "TICK t={t} frame={f} wall_us={:.1} write_us={:.1} launch_us={:.1} read_us={:.1}\n",
            us(t0, t3),
            us(t0, t1),
            us(t1, t2),
            us(t2, t3)
        );
        walls.push(us(t0, t3));
    }
    lines += &summary(c.warmup, n, &walls);
    Ok(Ticks { actions, lines })
}

fn summary(warmup: usize, n: usize, walls: &[f64]) -> String {
    let warmup = warmup.min(n - 1);
    let mut past: Vec<f64> = walls.iter().skip(warmup).copied().collect();
    past.sort_by(f64::total_cmp);
    let q = |p: f64| past[((past.len() as f64 * p) as usize).min(past.len() - 1)];
    format!(
        "ACT ticks={n} warmup={warmup} median_us={:.1} p99_us={:.1} max_us={:.1} ticks_per_s={:.1}\n",
        q(0.5),
        q(0.99),
        q(1.0),
        1e6 / q(0.5)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo {
        fed: Vec<(u8, Vec<u8>)>,
        launches: usize,
    }

    impl Step for Echo {
        fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String> {
            let c = Config::default();
            assert_eq!(
                items.iter().map(|i| (i.0, i.1.len())).collect::<Vec<_>>(),
                [("image", c.image_bytes()), ("state", c.state_bytes())]
            );
            self.fed.push((items[0].1[0], items[1].1.to_vec()));
            Ok(())
        }

        fn launch(&mut self) -> Result<(), String> {
            self.launches += 1;
            Ok(())
        }

        fn read(&self, name: &str, bytes: usize) -> Result<Vec<u8>, String> {
            assert_eq!((name, bytes), ("actions", Config::default().actions_bytes()));
            Ok((0..bytes / 4).flat_map(|_| (self.launches as f32).to_le_bytes()).collect())
        }
    }

    fn word(b: &[u8], i: usize) -> f32 {
        f32::from_le_bytes(b[i * 4..][..4].try_into().unwrap())
    }

    #[test]
    fn ticks_cycle_the_frames_pad_the_state_and_keep_each_chunk() {
        let c = Config::default();
        let images: Vec<u8> = (0..3u8).flat_map(|f| vec![f + 1; c.image_bytes()]).collect();
        let states: Vec<u8> = (0..2)
            .flat_map(|r| (0..c.joints).flat_map(move |j| (r as f32 * 100.0 + j as f32).to_le_bytes()))
            .collect();
        let mut echo = Echo { fed: Vec::new(), launches: 0 };
        let t = ticks(&c, &images, &states, 25, &mut echo).unwrap();
        assert_eq!(echo.fed.iter().map(|f| f.0).take(4).collect::<Vec<_>>(), [1, 2, 3, 1]);
        let s = &echo.fed[3].1;
        assert_eq!((word(s, 0), word(s, 13), word(s, 14), word(s, 15)), (100.0, 113.0, 0.0, 0.0));
        assert_eq!(t.actions.len(), 25 * c.actions_bytes());
        assert_eq!(word(&t.actions, 24 * c.chunk * c.joints), 25.0);
        assert_eq!(t.lines.lines().filter(|l| l.starts_with("TICK ")).count(), 25);
        assert!(t.lines.lines().last().unwrap().starts_with("ACT ticks=25 warmup=20 "));
    }

    #[test]
    fn no_states_feed_zeros_and_bad_inputs_are_refused() {
        let c = Config::default();
        let b = c.image_bytes();
        let mut echo = Echo { fed: Vec::new(), launches: 0 };
        ticks(&c, &vec![9u8; b], &[], 2, &mut echo).unwrap();
        assert!(echo.fed[1].1.iter().all(|&b| b == 0));
        assert!(ticks(&c, &vec![0u8; b - 1], &[], 1, &mut echo).is_err());
        assert!(ticks(&c, &vec![0u8; b], &[0u8; 10], 1, &mut echo).is_err());
        assert!(ticks(&c, &vec![0u8; b], &[], 0, &mut echo).is_err());
        let one = ticks(&c, &vec![0u8; b], &[], 1, &mut echo).unwrap();
        assert!(one.lines.lines().last().unwrap().starts_with("ACT ticks=1 warmup=0 "));
    }
}
