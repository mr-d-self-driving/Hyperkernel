use super::ranks::Ranks;
use super::ring::{self, Value};
use super::{Command, Event, Kernel, Pinned, Running, Sampled, Share};
use crate::bundle::{Name, Reference};

pub struct Feed<'a> {
    reference: &'a Reference,
    running: Running,
    rows: Blocks,
    tokens: Blocks,
    reading: Reading,
    events: Vec<Event>,
}

struct Reading {
    lps: Blocks,
    commits: Blocks,
    prompt: Option<u32>,
}

impl<'a> Feed<'a> {
    pub fn new(reference: &'a Reference, running: Running) -> Result<Feed<'a>, String> {
        let (m, r, pool) = (&reference.served.model, &reference.served.rings, &reference.served.pool);
        let lanes = *m.batches.last().ok_or("a bundle of no decode width")?;
        let chunk = *m.chunks.first().ok_or("a bundle of no prompt chunk")?;
        Ok(Feed {
            rows: Blocks::new(4 * (pool.entries + pool.window), r.slots[0] + 1)?,
            tokens: Blocks::new(4 * chunk, 1)?,
            reading: Reading {
                lps: Blocks::new(4 * r.logprobs * lanes * m.block, r.slots[1])?,
                commits: Blocks::new(4 * (r.commits_head + lanes * m.block), r.slots[1])?,
                prompt: None,
            },
            events: Vec::new(),
            reference,
            running,
        })
    }

    fn command(&mut self, at: u32, c: &Command) -> (Name, Vec<(&'static str, Value)>) {
        let (model, rings) = (&self.reference.served.model, &self.reference.served.rings);
        let int = |x: u32| Value::Int(x as i64);
        let (at, lane) = (("at", int(at)), |l: u32| ("lane", int(l)));
        match c {
            Command::End { lane: l } => (Name::End, vec![at, lane(*l)]),
            Command::Pages { lane: l, from, upto, row } => {
                let n = self.running.requests.count();
                self.rows.write(n, &row.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                let row = Value::Span { addr: self.rows.device(n), elements: row.len() as u64 };
                (Name::Pages, vec![at, lane(*l), ("from", int(*from)), ("upto", int(*upto)), ("row", row)])
            }
            Command::Seat { lane: l, pos, back, sampling: s } => {
                let back = Value::Ints(back.iter().map(|&b| b as i64).collect());
                let (seed_lo, seed_hi) = (int(s.seed as u32), int((s.seed >> 32) as u32));
                (
                    Name::Seat,
                    vec![
                        at,
                        lane(*l),
                        ("pos", int(*pos)),
                        ("back", back),
                        ("slot", int(l + 1)),
                        ("temp", Value::Float(s.temperature)),
                        ("seed_lo", seed_lo),
                        ("seed_hi", seed_hi),
                        ("top_p", Value::Float(s.top_p)),
                        ("top_k", int(s.top_k)),
                        ("logprobs", int(s.logprobs)),
                    ],
                )
            }
            Command::Prompt { lane: l, tokens } => {
                let (rows, lens) = (model.chunks[0], int(tokens.len() as u32));
                let mut bytes: Vec<u8> = tokens.iter().flat_map(|x| x.to_le_bytes()).collect();
                bytes.resize(4 * rows, 0);
                self.tokens.write(0, &bytes);
                self.reading.prompt = Some(*l);
                let tokens = Value::Span { addr: self.tokens.device(0), elements: rows as u64 };
                let lps = Value::Span { addr: self.reading.lps.device(0), elements: rings.logprobs as u64 };
                (Name::Prompt, vec![lane(*l), ("lens", lens), ("tokens", tokens), ("lps", lps)])
            }
            Command::Start { lane: l } => {
                let (lps, commits) = (&self.reading.lps, &self.reading.commits);
                let lps = Value::Span { addr: lps.device(0), elements: (lps.each / 4) as u64 };
                let commits = Value::Span {
                    addr: commits.device(0),
                    elements: (commits.each * commits.count / 4) as u64,
                };
                (
                    Name::Start,
                    vec![
                        at,
                        lane(*l),
                        ("noise", int(model.noise)),
                        ("idle", int(model.context as u32 - 1)),
                        ("scratch", int(0)),
                        ("lps", lps),
                        ("commits", commits),
                    ],
                )
            }
        }
    }

    pub fn follow(&mut self, ranks: &Ranks) -> Result<(), String> {
        while let Ok(share) = ranks.recv_share() {
            self.push(&share)?;
            self.drain()?;
        }
        Ok(())
    }

    fn wait(&mut self) -> Result<(), String> {
        if !self.take()? {
            std::thread::yield_now();
        }
        Ok(())
    }

    fn take(&mut self) -> Result<bool, String> {
        let Feed { reference, running, reading, events, .. } = self;
        match running.replies.pop(|n, id, bytes| reading.event(reference, n, id, bytes)) {
            None => Ok(false),
            Some(e) => {
                events.push(e?);
                Ok(true)
            }
        }
    }
}

impl Kernel for Feed<'_> {
    fn pools(&self) -> (usize, usize) {
        (self.running.pages, self.running.window_pages)
    }

    fn push(&mut self, share: &Share) -> Result<(), String> {
        for c in &share.commands {
            if matches!(c, Command::Prompt { .. }) {
                while self.reading.prompt.is_some() {
                    self.wait()?;
                }
            }
            let (name, fields) = self.command(share.at, c);
            let layout = self.reference.served.rings.message(name)?;
            let bytes = ring::encode(layout, &fields);
            while !self.running.requests.push(layout.id, &bytes) {
                self.wait()?;
            }
        }
        let grant =
            (self.running.requests.count() & 0xffff_ffff) << 32 | (share.horizon as u64 + 1) & 0xffff_ffff;
        self.running.ctl.word(8).store(grant, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn drain(&mut self) -> Result<Vec<Event>, String> {
        while self.take()? {}
        Ok(std::mem::take(&mut self.events))
    }
}

impl Reading {
    fn event(&mut self, reference: &Reference, n: u64, id: u32, bytes: &[u8]) -> Result<Event, String> {
        let (model, rings) = (&reference.served.model, &reference.served.rings);
        let (layout, none) = (rings.reply(id)?, -2);
        let lps: Vec<f32> =
            self.lps.read(n).as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
        let mut rows = lps.chunks_exact(rings.logprobs);
        let mut blocks = lps.chunks_exact(rings.logprobs * model.block);
        match Name::of(layout) {
            Some(Name::Idle) => Ok(Event::Idle { step: ring::decode(layout, bytes, "step")[0] as u32 }),
            Some(Name::Last) => {
                let lane = self.prompt.take().ok_or("a prompt's reply with no prompt pushed")?;
                let token =
                    sampled(ring::decode(layout, bytes, "id")[0], rows.next().expect("a logprobs row"));
                Ok(Event::Prompted { lane, token, clipped: ring::decode(layout, bytes, "clip")[0] != 0 })
            }
            _ if model.batches.contains(&(id as usize)) => {
                let words: Vec<i32> =
                    self.commits.read(n).as_chunks::<4>().0.iter().map(|&c| i32::from_le_bytes(c)).collect();
                let lanes = words[rings.commits_head..]
                    .chunks_exact(model.block)
                    .map(|ids| {
                        let ids: Vec<i32> = ids.iter().copied().take_while(|&x| x != none).collect();
                        let block = if ids.is_empty() {
                            &[][..]
                        } else {
                            blocks.next().expect("a logprobs block each lane that ran")
                        };
                        let mut rows = block.chunks_exact(rings.logprobs);
                        ids.iter().map(|&x| sampled(x, rows.next().unwrap_or(&[]))).collect()
                    })
                    .collect();
                Ok(Event::Step { step: words[0] as u32, lanes })
            }
            _ => Err(format!("a reply {} of id {id:#x}, no decode width", layout.name)),
        }
    }
}

fn sampled(id: i32, row: &[f32]) -> Sampled {
    let Some((own, pairs)) = row.split_last().filter(|_| id >= 0) else {
        return Sampled { id, logprobs: Vec::new() };
    };
    let top = pairs.as_chunks::<2>().0.iter().take_while(|p| p[0] >= 0.0).map(|&[id, lp]| (id as u32, lp));
    Sampled { id, logprobs: top.chain([(id as u32, *own)]).collect() }
}

struct Blocks {
    mem: Pinned,
    each: usize,
    count: usize,
}

impl Blocks {
    fn new(each: usize, count: usize) -> Result<Blocks, String> {
        Ok(Blocks { mem: Pinned::new(each * count)?, each, count })
    }

    fn at(&self, n: u64) -> usize {
        (n % self.count as u64) as usize * self.each
    }

    fn device(&self, n: u64) -> u64 {
        self.mem.device() + self.at(n) as u64
    }

    fn write(&mut self, n: u64, bytes: &[u8]) {
        assert!(bytes.len() <= self.each, "{} bytes into a block of {}", bytes.len(), self.each);
        self.mem.write(self.at(n), bytes);
    }

    fn read(&self, n: u64) -> &[u8] {
        self.mem.read(self.at(n), self.each)
    }
}

pub struct Leader<'a> {
    pub feed: Feed<'a>,
    pub ranks: &'a Ranks,
}

impl Kernel for Leader<'_> {
    fn pools(&self) -> (usize, usize) {
        self.feed.pools()
    }

    fn push(&mut self, share: &Share) -> Result<(), String> {
        self.ranks.send_share(share)?;
        self.feed.push(share)
    }

    fn drain(&mut self) -> Result<Vec<Event>, String> {
        self.feed.drain()
    }
}
