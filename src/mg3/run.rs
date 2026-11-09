use std::time::{Duration, Instant};

use super::camera::{self, Action, Camera};
use super::demo::{self, Demo, Frames, Times};
use super::spec::Spec;
use super::{At, Config, Dev, Prog, draw, rope};

pub struct Specs<'a> {
    pub chunk: &'a Spec,
    pub first: &'a Spec,
}

pub fn run(
    dev: &mut dyn Dev,
    cfg: &Config,
    s: Specs<'_>,
    image: Option<&[u8]>,
    chunks: usize,
    demo: &Demo,
) -> Result<String, String> {
    let route = route(cfg, demo, chunks)?;
    let sink = demo.frames.as_deref().map(|p| Frames::open(cfg.demo, p)).transpose()?;
    text(dev, s.chunk, demo)?;
    let table = rope::dit(s.chunk);
    dev.write(Prog::Chunk, "rope", 0, &table)?;
    dev.write(Prog::First, "rope", 0, &table)?;
    let (mut cam, mut actions) = (Camera::default(), vec![route(0)]);
    act(cfg, &mut cam, &mut actions, cfg.camera.first_frames - 1, &route);
    first_clip(dev, cfg, &s, image, &cam, &actions, demo)?;
    let latents = s.chunk.rows() * s.chunk.row_bytes();
    let (mut starts, mut leads) = (Vec::with_capacity(chunks), Vec::with_capacity(chunks));
    let mut held: Vec<Instant> = Vec::with_capacity(chunks + 1);
    let mut read: Vec<Duration> = Vec::with_capacity(chunks + 1);
    let mut read_frames = |dev: &mut dyn Dev| -> Result<(), String> {
        if let Some(sink) = &sink {
            let t = Instant::now();
            sink.push(dev.read(Prog::Vae, "frames", 0..cfg.chunk_bytes())?);
            read.push(t.elapsed());
            held.push(Instant::now());
        }
        Ok(())
    };
    if sink.is_some() {
        prime(dev, cfg, &s, demo)?;
    }
    for c in 1..=chunks {
        let t0 = Instant::now();
        starts.push(t0);
        act(cfg, &mut cam, &mut actions, cfg.camera.new_frames, &route);
        inputs(dev, cfg, Prog::Chunk, s.chunk, &cam, &actions, c)?;
        denoise(dev, Prog::Chunk, c, demo)?;
        read_frames(dev)?;
        let rows = dev.read(Prog::Chunk, "latents", 0..latents)?;
        if let Some(path) = &demo.frames {
            let new = s.chunk.split() * s.chunk.row_bytes();
            std::fs::write(path.with_extension(format!("latent-{c}.bin")), &rows[new..])
                .map_err(|e| format!("--frames-out latents: {e}"))?;
        }
        keep(dev, cfg, s.chunk, c)?;
        leads.push(t0.elapsed());
        dev.write(Prog::Vae, "latents", 0, &rows)?;
        dev.launch(Prog::Vae)?;
    }
    let drained = Instant::now();
    dev.finish()?;
    read_frames(dev)?;
    let done = Instant::now();
    if let Some(sink) = sink {
        sink.close()?;
    }
    if let (Some(path), Some(&first)) = (&demo.timing, starts.first()) {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let times: Vec<Times> = (0..starts.len())
            .map(|i| Times {
                chunk: i + 1,
                start_ms: ms(starts[i] - first),
                lead_ms: ms(leads[i]),
                frames_ms: held.get(i + 1).map(|&t| ms(t - first)),
                read_ms: read.get(i + 1).map(|&d| ms(d)),
            })
            .collect();
        demo::write_times(path, &times, cfg.camera.new_frames)?;
    }
    Ok(lines(cfg.camera.new_frames, &starts, &leads, drained, done))
}

fn route<'a>(cfg: &Config, demo: &'a Demo, chunks: usize) -> Result<impl Fn(usize) -> Action + 'a, String> {
    let (need, scripted) = (cfg.camera.first_frames + chunks * cfg.camera.new_frames, cfg.scripted);
    if let Some(a) = demo.actions.as_ref().filter(|a| a.len() < need) {
        return Err(format!("--actions: {} frames, {chunks} chunks need {need}", a.len()));
    }
    Ok(move |t: usize| demo.actions.as_ref().map_or(scripted, |a| a[t]))
}

fn prime(dev: &mut dyn Dev, cfg: &Config, s: &Specs<'_>, demo: &Demo) -> Result<(), String> {
    let (latent, clip) = (s.chunk.latent_bytes(), s.first.latents());
    let rows = dev.read(Prog::First, "latents", (clip - cfg.primed) * latent..clip * latent)?;
    if let Some(path) = &demo.frames {
        std::fs::write(path.with_extension("latent-0.bin"), &rows)
            .map_err(|e| format!("--frames-out latents: {e}"))?;
    }
    dev.write(Prog::Vae, "latents", s.chunk.split() * s.chunk.row_bytes(), &rows)?;
    dev.launch(Prog::Vae)
}

fn act(
    cfg: &Config,
    cam: &mut Camera,
    actions: &mut Vec<Action>,
    n: usize,
    route: &impl Fn(usize) -> Action,
) {
    for _ in 0..n {
        cam.act(&cfg.camera, actions[actions.len() - 1]);
        actions.push(route(actions.len()));
    }
}

fn text(dev: &mut dyn Dev, s: &Spec, demo: &Demo) -> Result<(), String> {
    if let Some(path) = &demo.context {
        let bytes = std::fs::read(path).map_err(|e| format!("--context {}: {e}", path.display()))?;
        let want = s.text * s.text_dim * 2;
        if bytes.len() != want {
            return Err(format!(
                "--context {}: {} bytes, not {want} (BF16 text rows)",
                path.display(),
                bytes.len()
            ));
        }
        dev.write(Prog::Prompt, "context", 0, &bytes)?;
        dev.write(Prog::FirstPrompt, "context", 0, &bytes)?;
    }
    dev.launch(Prog::Prompt)?;
    dev.launch(Prog::FirstPrompt)?;
    dev.finish()
}

fn inputs(
    dev: &mut dyn Dev,
    cfg: &Config,
    p: Prog,
    s: &Spec,
    cam: &Camera,
    actions: &[Action],
    c: usize,
) -> Result<(), String> {
    let k = cam.chunk(&cfg.camera, s, c);
    dev.write(p, "poses", 0, &k.poses)?;
    dev.write(p, "positions", 0, &k.positions)?;
    let n = s.latent_bytes();
    for (slot, &l) in k.picks.iter().enumerate() {
        dev.copy(
            At { prog: p, name: cfg.history, at: l * n },
            At { prog: p, name: "memory", at: slot * n },
            n,
        )?;
    }
    let (keyboard, mouse) = windows(&cfg.camera, s, actions, c);
    dev.write(p, "keyboard_windows", 0, &keyboard)?;
    dev.write(p, "mouse_windows", 0, &mouse)
}

fn denoise(dev: &mut dyn Dev, p: Prog, c: usize, demo: &Demo) -> Result<(), String> {
    match &demo.noise {
        Some(dir) => {
            let path = dir.join(format!("noise-{c}.bin"));
            let bytes = std::fs::read(&path).map_err(|e| format!("--noise {}: {e}", path.display()))?;
            dev.write(p, "noise", 0, &bytes)?;
        }
        None => dev.randn(p, "noise", c)?,
    }
    dev.launch(p)
}

fn keep(dev: &mut dyn Dev, cfg: &Config, s: &Spec, c: usize) -> Result<(), String> {
    let n = s.latent_bytes();
    let (first, fresh) = camera::kept(&cfg.camera, s, c);
    let from = At { prog: Prog::Chunk, name: "latents", at: s.split() / s.tokens * n };
    dev.copy(from, At { prog: Prog::Chunk, name: cfg.history, at: first * n }, fresh * n)
}

fn first_clip(
    dev: &mut dyn Dev,
    cfg: &Config,
    s: &Specs<'_>,
    image: Option<&[u8]>,
    cam: &Camera,
    actions: &[Action],
    demo: &Demo,
) -> Result<(), String> {
    let (latent, clip) = (s.chunk.latent_bytes(), s.first.latents());
    let past = (s.chunk.split() - s.chunk.memory() * s.chunk.tokens) / s.chunk.tokens;
    if let Some(bytes) = image {
        if bytes.len() != latent {
            return Err(format!(
                "an image latent of {} bytes, not a latent's {latent} (BF16 token rows)",
                bytes.len()
            ));
        }
        dev.write(Prog::First, "image_latent", 0, bytes)?;
    }
    inputs(dev, cfg, Prog::First, s.first, cam, actions, 0)?;
    denoise(dev, Prog::First, 0, demo)?;
    dev.copy(
        At { prog: Prog::First, name: "latents", at: 0 },
        At { prog: Prog::Chunk, name: cfg.history, at: 0 },
        clip * latent,
    )?;
    dev.copy(
        At { prog: Prog::First, name: "latents", at: (clip - past) * latent },
        At { prog: Prog::Chunk, name: cfg.past, at: 0 },
        past * latent,
    )
}

pub fn windows(cfg: &camera::Config, s: &Spec, actions: &[Action], c: usize) -> (Vec<u8>, Vec<u8>) {
    let (f, w, keys, tokens) = (s.frames, s.window(), s.key_in, s.tokens);
    let (key_cols, mouse_cols, memory) = (s.key_cols, s.mouse_cols, s.memory());
    let span = camera::frames(cfg, c);
    let (start, pad) = (span.start, if span.len() % f == 0 { w - f } else { w });
    let frames = move |i: usize| (0..w).map(move |j| start + (f * i + j).saturating_sub(pad));
    let laid = |mut row: Vec<f32>, cols: usize| {
        row.resize(cols, 0.0);
        row
    };
    let key = |a: &Action| (0..keys).map(|k| a.keys.get(k).copied().unwrap_or(0.0)).collect::<Vec<f32>>();
    let clip = 0..s.latents() - memory;
    let mut keyboard: Vec<f32> = (0..memory * w).flat_map(|_| laid(vec![-1.0; keys], key_cols)).collect();
    keyboard.extend(clip.clone().flat_map(|i| frames(i).flat_map(|t| laid(key(&actions[t]), key_cols))));
    let mut mouse: Vec<f32> =
        (0..memory * tokens).flat_map(|_| laid(vec![1.0; w * s.mouse_in], mouse_cols)).collect();
    for i in clip {
        let row = laid(frames(i).flat_map(|t| actions[t].mouse).collect(), mouse_cols);
        (0..tokens).for_each(|_| mouse.extend(&row));
    }
    let bf16 = |v: f32| draw::bf16(v).to_le_bytes();
    (keyboard.into_iter().flat_map(bf16).collect(), mouse.into_iter().flat_map(bf16).collect())
}

fn lines(frames: usize, starts: &[Instant], leads: &[Duration], drained: Instant, done: Instant) -> String {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let ends = starts.iter().skip(1).copied().chain([done]);
    let walls: Vec<f64> = starts.iter().zip(ends).map(|(s, e)| ms(e - *s)).collect();
    let mut out: String = walls
        .iter()
        .zip(leads)
        .enumerate()
        .map(|(i, (w, l))| format!("STEP chunk={} wall_ms={w:.2} lead_ms={:.2}\n", i + 1, ms(*l)))
        .collect();
    let mut sorted = walls.clone();
    sorted.sort_by(f64::total_cmp);
    let chunk = sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN);
    let total = starts.first().map_or(0.0, |s| ms(done - *s));
    out += &format!(
        "PIPE chunks={} frames={frames} chunk_ms={chunk:.2} fps={:.2} drain_ms={:.2} total_ms={total:.1}\n",
        walls.len(),
        frames as f64 * 1e3 / chunk,
        ms(done - drained)
    );
    out
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::ops::Range;

    use super::{Specs, run, windows};
    use crate::mg3::camera::Action;
    use crate::mg3::demo::Demo;
    use crate::mg3::{At, Config, Dev, Prog, spec};

    #[derive(Default)]
    struct Calls {
        said: RefCell<Vec<String>>,
    }

    impl Calls {
        fn say(&self, what: String) {
            self.said.borrow_mut().push(what);
        }
    }

    impl Dev for Calls {
        fn write(&mut self, p: Prog, name: &str, at: usize, bytes: &[u8]) -> Result<(), String> {
            self.say(format!("write {} {name} {at} {}", p as usize, bytes.len()));
            Ok(())
        }
        fn read(&mut self, p: Prog, name: &str, range: Range<usize>) -> Result<Vec<u8>, String> {
            self.say(format!("read {} {name}", p as usize));
            Ok(vec![0; range.len()])
        }
        fn launch(&mut self, p: Prog) -> Result<(), String> {
            self.say(format!("launch {}", p as usize));
            Ok(())
        }
        fn finish(&mut self) -> Result<(), String> {
            self.say("finish".into());
            Ok(())
        }
        fn copy(&mut self, from: At<'_>, to: At<'_>, n: usize) -> Result<(), String> {
            self.say(format!(
                "copy {} {} {} -> {} {} {} {n}",
                from.prog as usize, from.name, from.at, to.prog as usize, to.name, to.at
            ));
            Ok(())
        }
        fn randn(&mut self, p: Prog, name: &str, stream: usize) -> Result<(), String> {
            self.say(format!("randn {} {name} {stream}", p as usize));
            Ok(())
        }
    }

    fn specs<'a>(chunk: &'a spec::Spec, first: &'a spec::Spec) -> Specs<'a> {
        Specs { chunk, first }
    }

    #[test]
    fn no_demo_reads_no_frames() {
        let (cfg, c, f) = (Config::default(), spec::chunk(), spec::first());
        let mut dev = Calls::default();
        let out = run(&mut dev, &cfg, specs(&c, &f), None, 2, &Demo::default()).unwrap();
        assert!(out.contains("PIPE chunks=2"), "{out}");
        let said = dev.said.into_inner();
        assert!(!said.iter().any(|c| c.contains("frames")));
        assert!(
            said.contains(&"randn 1 noise 2".to_string()) && said.contains(&"randn 3 noise 0".to_string())
        );
        assert!(said.contains(&format!("copy 3 latents 0 -> 1 l0.history 0 {}", 15 * 880 * 384)));
        assert!(said.contains(&format!(
            "copy 3 latents {} -> 1 l0.past 0 {}",
            11 * 880 * 384,
            4 * 880 * 384
        )));
        assert!(said.contains(&format!(
            "copy 1 latents {} -> 1 l0.history {} {}",
            9 * 880 * 384,
            15 * 880 * 384,
            10 * 880 * 384
        )));
    }

    #[test]
    fn a_short_route_is_refused() {
        let (cfg, c, f) = (Config::default(), spec::chunk(), spec::first());
        let mut dev = Calls::default();
        let demo = Demo {
            actions: Some(vec![cfg.scripted; cfg.camera.first_frames + cfg.camera.new_frames - 1]),
            ..Demo::default()
        };
        let e = run(&mut dev, &cfg, specs(&c, &f), None, 1, &demo).err();
        assert_eq!(e.as_deref(), Some("--actions: 96 frames, 1 chunks need 97"));
        assert!(dev.said.borrow().is_empty());
    }

    #[test]
    fn frames_read_once_the_next_dit_is_launched() {
        let (cfg, c, f) = (Config::default(), spec::chunk(), spec::first());
        let dir = std::env::temp_dir().join(format!("hs-mg3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (rgb, jsonl) = (dir.join("frames.rgb"), dir.join("timing.jsonl"));
        let walk = Action { keys: [1.0, 0.0, 0.0, 0.0], mouse: [0.0, 0.0] };
        let demo = Demo {
            actions: Some(vec![walk; cfg.camera.first_frames + 2 * cfg.camera.new_frames]),
            frames: Some(rgb.clone()),
            timing: Some(jsonl.clone()),
            context: None,
            noise: None,
        };
        let mut dev = Calls::default();
        run(&mut dev, &cfg, specs(&c, &f), None, 2, &demo).unwrap();
        let said = dev.said.into_inner();
        let (d, v, r) = (
            format!("launch {}", Prog::Chunk as usize),
            format!("launch {}", Prog::Vae as usize),
            format!("read {} frames", Prog::Vae as usize),
        );
        let order: Vec<&str> = said
            .iter()
            .map(String::as_str)
            .filter(|c| [d.as_str(), v.as_str(), r.as_str(), "finish"].contains(c))
            .collect();
        assert_eq!(
            order[order.len() - 6..],
            [v.as_str(), d.as_str(), r.as_str(), v.as_str(), "finish", r.as_str()]
        );
        assert!(said.contains(&format!(
            "write {} latents {} {}",
            Prog::Vae as usize,
            7920 * 384,
            10 * 880 * 384
        )));
        assert!(said.contains(&format!("read {} latents", Prog::First as usize)));
        let bytes = std::fs::metadata(&rgb).unwrap().len() as usize;
        assert_eq!(bytes, 3 * cfg.camera.new_frames * cfg.demo.height * cfg.demo.width * 3);
        let latent = std::fs::metadata(dir.join("frames.latent-2.bin")).unwrap().len() as usize;
        assert_eq!(latent, 10 * 880 * 384);
        let times = std::fs::read_to_string(&jsonl).unwrap();
        assert_eq!(times.lines().count(), 2);
        assert!(!times.contains("null"), "{times}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn windows_of_a_chunk() {
        let (cfg, s) = (Config::default(), spec::chunk());
        let actions = vec![cfg.scripted; cfg.camera.first_frames + 2 * cfg.camera.new_frames];
        let (keyboard, mouse) = windows(&cfg.camera, &s, &actions, 2);
        let w = s.window();
        assert_eq!(keyboard.len(), s.latents() * w * s.key_cols * 2);
        assert_eq!(mouse.len(), s.rows() * s.mouse_cols * 2);
        let at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
        assert_eq!(at(&keyboard, 0), 0xbf80);
        assert_eq!(at(&keyboard, s.memory() * w * s.key_cols), 0x3f80);
        assert_eq!(at(&keyboard, s.key_in), 0);
    }
}
