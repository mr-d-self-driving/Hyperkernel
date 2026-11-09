use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use hyperserve::api::metrics::MetricsConfig;
use hyperserve::bundle::Reference;
use hyperserve::engine::{self, End, EngineConfig, Job, Metrics, Out, Plan, Request, Sampling};
use hyperserve::gpu::{Command, Event, Kernel, Sampled, Share};

fn page_rows() -> usize {
    4
}

fn entries() -> usize {
    16
}

fn context() -> usize {
    64
}

fn width() -> usize {
    4
}

fn chunk() -> usize {
    8
}

fn greedy() -> Sampling {
    Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 0, logprobs: 0 }
}

fn nucleus() -> Sampling {
    Sampling { temperature: 0.7, top_p: 0.9, top_k: 0, seed: 1, logprobs: 0 }
}

fn reference(keep: usize) -> Reference {
    drafting(keep, 1)
}

fn drafting(keep: usize, block: usize) -> Reference {
    let json = serde_json::json!({
        "format": "hyperserve-bundle-2", "point": "test",
        "model": { "name": "fake", "ranks": 1, "vocab": 1000, "context": context(), "batches": [1, 2, width()], "chunks": [chunk()], "block": block, "noise": 0, "pad": 7 },
        "kernel": { "entries": [], "threads": 0, "cluster": 0, "smem": 0, "static_smem": 0, "cooperative": false, "stack": 0, "workspace": 0, "symmetric": 0,
            "l2": null, "homew": null, "params": [], "next": null, "loaders": [],
            "globals": { "fault": "", "fabric": "", "peers": "", "peer_bytes": 0, "multicast": false, "rings": "", "ctl": "" } },
        "homes": [], "image": [],
        "rings": { "slots": [16, 8], "messages": [], "commits_head": 5, "logprobs": 33 },
        "pool": { "page_rows": page_rows(), "pages": 0, "window_pages": 0, "slots": 0, "window_keep": keep, "entries": entries(), "window": entries(), "states": [] }
    });
    serde_json::from_value(json).expect("a reference")
}

fn next(seq: &[u32]) -> u32 {
    let h = seq
        .iter()
        .enumerate()
        .fold(1469598103u64, |h, (i, &t)| (h ^ (t as u64 + 1000 * i as u64)).wrapping_mul(1099511628211));
    (h % 400) as u32 + 10
}

#[derive(Clone, Default)]
struct Lane {
    row: Vec<i32>,
    upto: u32,
    pos: u32,
    live: bool,
    tok: u32,
}

struct Fake {
    lanes: Vec<Lane>,
    queue: VecDeque<(u32, Command)>,
    s: u32,
    horizon: i64,
    applied: bool,
    pages: Vec<Vec<Option<u32>>>,
    windows: Vec<Vec<Option<u32>>>,
    events: Vec<Event>,
    drafts: usize,
    clip: bool,
    told: Option<Sender<Share>>,
}

impl Fake {
    fn new(pages: usize, windows: usize) -> Fake {
        let lane = Lane { row: vec![0; 2 * entries()], ..Lane::default() };
        Fake {
            lanes: vec![lane; width()],
            queue: VecDeque::new(),
            s: 0,
            horizon: -1,
            applied: false,
            pages: vec![vec![None; page_rows()]; pages],
            windows: vec![vec![None; page_rows()]; windows],
            events: Vec::new(),
            drafts: 0,
            clip: false,
            told: None,
        }
    }

    fn write(&mut self, l: usize, p: usize, t: u32) {
        let row = &self.lanes[l].row;
        let (page, window) = (row[p / page_rows()] as usize, row[entries() + p / page_rows()] as usize);
        assert!(page != 0, "lane {l} writes position {p} to the scratch page");
        self.pages[page][p % page_rows()] = Some(t);
        if window != 0 {
            self.windows[window][p % page_rows()] = Some(t);
        }
    }

    fn token(&self, l: usize, last: usize) -> u32 {
        let row = &self.lanes[l].row;
        let read = |p: usize| {
            self.pages[row[p / page_rows()] as usize][p % page_rows()]
                .unwrap_or_else(|| panic!("lane {l} reads position {p} unwritten"))
        };
        let seq: Vec<u32> = (0..=last).map(read).collect();
        for q in (last + 1).saturating_sub(page_rows())..=last {
            let w = row[entries() + q / page_rows()] as usize;
            assert!(w != 0, "lane {l} reads position {q}'s window from the scratch window page");
            assert_eq!(self.windows[w][q % page_rows()], Some(seq[q]), "lane {l}'s window at {q}");
        }
        next(&seq)
    }

    fn run(&mut self) {
        loop {
            while let Some((at, c)) = self.queue.front().cloned() {
                let delta = !matches!(c, Command::Prompt { .. });
                if delta && at > self.s {
                    break;
                }
                self.queue.pop_front();
                self.applied |= delta;
                match c {
                    Command::End { lane } => {
                        (self.lanes[lane as usize].live, self.lanes[lane as usize].upto) = (false, 0)
                    }
                    Command::Pages { lane, from, upto, row } => {
                        let l = &mut self.lanes[lane as usize];
                        l.row[from as usize..from as usize + row.len()].copy_from_slice(&row);
                        l.upto = upto;
                    }
                    Command::Seat { lane, pos, .. } => self.lanes[lane as usize].pos = pos,
                    Command::Start { lane } => self.lanes[lane as usize].live = true,
                    Command::Prompt { lane, tokens } => {
                        let (l, pos) = (lane as usize, self.lanes[lane as usize].pos as usize);
                        assert!(tokens.len() <= chunk());
                        for (j, &t) in tokens.iter().enumerate() {
                            let w = self.lanes[l].row[entries() + (pos + j) / page_rows()];
                            assert!(
                                w != 0,
                                "lane {l}'s chunk writes position {}'s window to the scratch window page",
                                pos + j
                            );
                            self.write(l, pos + j, t);
                        }
                        let end = pos + tokens.len();
                        let tok = self.token(l, end - 1);
                        (self.lanes[l].pos, self.lanes[l].tok) = (end as u32, tok);
                        self.events.push(Event::Prompted {
                            lane,
                            token: Sampled { id: tok as i32, logprobs: Vec::new() },
                            clipped: self.clip,
                        });
                    }
                }
            }
            let runnable: Vec<usize> = (0..width())
                .filter(|&l| self.lanes[l].live && self.lanes[l].pos < self.lanes[l].upto)
                .collect();
            if !runnable.is_empty() && (self.s as i64) <= self.horizon {
                let mut lanes = vec![Vec::new(); width()];
                for l in runnable {
                    for k in 0..=(self.s as usize + l) % (self.drafts + 1) {
                        let (pos, tok) = (self.lanes[l].pos as usize, self.lanes[l].tok);
                        if k > 0 && (pos >= self.lanes[l].upto as usize || pos >= context() - 1) {
                            break;
                        }
                        assert!(pos < context() - 1, "lane {l} decodes at the idle position");
                        self.write(l, pos, tok);
                        let t = self.token(l, pos);
                        (self.lanes[l].pos, self.lanes[l].tok) = (pos as u32 + 1, t);
                        lanes[l].push(Sampled { id: t as i32, logprobs: Vec::new() });
                    }
                }
                self.events.push(Event::Step { step: self.s, lanes });
                self.s += 1;
                continue;
            }
            if runnable.is_empty() && self.applied {
                self.events.push(Event::Idle { step: self.s });
                self.applied = false;
            }
            match self.queue.front() {
                Some(&(at, ref c))
                    if runnable.is_empty()
                        && !matches!(c, Command::Prompt { .. })
                        && at > self.s
                        && at as i64 <= self.horizon + 1 =>
                {
                    self.s = at
                }
                _ => return,
            }
        }
    }
}

impl Kernel for Fake {
    fn pools(&self) -> (usize, usize) {
        (self.pages.len(), self.windows.len())
    }

    fn push(&mut self, share: &Share) -> Result<(), String> {
        assert!(
            share.at as i64 == self.horizon + 1,
            "a share at {} past a horizon of {}",
            share.at,
            self.horizon
        );
        assert!(share.horizon >= share.at);
        self.queue.extend(share.commands.iter().map(|c| (share.at, c.clone())));
        self.horizon = share.horizon as i64;
        if let Some(told) = &self.told {
            told.send(share.clone()).expect("a listener to the shares");
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<Vec<Event>, String> {
        self.run();
        Ok(std::mem::take(&mut self.events))
    }
}

struct Ask {
    prompt: Vec<u32>,
    limit: usize,
    stop: Vec<u32>,
    sampling: Sampling,
}

fn ask(prompt: Vec<u32>, limit: usize) -> Ask {
    Ask { prompt, limit, stop: Vec::new(), sampling: greedy() }
}

fn expected(a: &Ask) -> (Vec<u32>, End) {
    if a.prompt.len() >= context() {
        return (Vec::new(), End::Refused(String::new()));
    }
    let (mut seq, mut out) = (a.prompt.clone(), Vec::new());
    loop {
        if out.len() == a.limit || seq.len() == context() {
            return (out, End::Length);
        }
        let t = next(&seq);
        out.push(t);
        if a.stop.contains(&t) {
            return (out, End::Stop);
        }
        seq.push(t);
    }
}

fn heard(rx: &Receiver<Out>) -> (Vec<u32>, End) {
    let mut tokens = Vec::new();
    for o in rx.iter() {
        match o {
            Out::Seated { .. } => {}
            Out::Token(t) => tokens.push(t.id),
            Out::End(End::Refused(_)) => return (tokens, End::Refused(String::new())),
            Out::End(e) => return (tokens, e),
        }
    }
    panic!("no end")
}

fn job(a: &Ask) -> (Job, Receiver<Out>) {
    let (out, rx) = channel();
    (
        Job::Run(Request {
            prompt: a.prompt.clone(),
            sampling: a.sampling,
            limit: a.limit,
            stop: a.stop.clone(),
            out,
        }),
        rx,
    )
}

fn serve(
    plan: Plan,
    pages: usize,
    windows: usize,
    keep: usize,
    asks: &[Ask],
) -> (Vec<(Vec<u32>, End)>, Vec<Share>) {
    let (heard, shares, _) = serve_told(plan, pages, windows, keep, asks);
    (heard, shares)
}

fn serve_told(
    plan: Plan,
    pages: usize,
    windows: usize,
    keep: usize,
    asks: &[Ask],
) -> (Vec<(Vec<u32>, End)>, Vec<Share>, Metrics) {
    serve_on(&reference(keep), Fake::new(pages, windows), plan, asks)
}

fn serve_on(
    reference: &Reference,
    mut fake: Fake,
    plan: Plan,
    asks: &[Ask],
) -> (Vec<(Vec<u32>, End)>, Vec<Share>, Metrics) {
    let (tx, queue) = channel();
    let mut outs = Vec::new();
    for a in asks {
        let (j, rx) = job(a);
        tx.send(j).unwrap();
        outs.push(rx);
    }
    drop(tx);
    let (told, shares) = channel();
    fake.told = Some(told);
    let metrics = Metrics::new(MetricsConfig::default().iteration_tokens);
    engine::run(reference, &mut fake, queue, &metrics, plan, &EngineConfig::default()).expect("served");
    drop(fake);
    (outs.iter().map(heard).collect(), shares.iter().collect(), metrics)
}

fn check(plan: Plan, pages: usize, windows: usize, keep: usize, asks: &[Ask]) -> Vec<Share> {
    let (got, shares) = serve(plan, pages, windows, keep, asks);
    for (i, (a, g)) in asks.iter().zip(&got).enumerate() {
        assert_eq!(g, &expected(a), "request {i}");
    }
    shares
}

fn conversations() -> Vec<Ask> {
    let system: Vec<u32> = (100..111).collect();
    let mut asks = Vec::new();
    for c in 0..3u32 {
        let mut prompt = system.clone();
        prompt.extend([200 + c, 201, 202]);
        for turn in 0..3 {
            let a = ask(prompt.clone(), 6);
            let (reply, _) = expected(&a);
            asks.push(a);
            prompt.extend(reply);
            prompt.extend([300 + turn, 301]);
        }
    }
    asks
}

fn resumed(shares: &[Share]) -> usize {
    let firsts = shares
        .iter()
        .flat_map(|s| &s.commands)
        .filter(|c| matches!(c, Command::Seat { pos, .. } if *pos > 0 && *pos as usize % page_rows() == 0));
    firsts.count()
}

#[test]
fn one_request_hears_the_models_tokens() {
    check(Plan::default(), 40, 20, 0, &[ask((1..20).collect(), 30)]);
}

#[test]
fn a_request_ends_at_its_stop_its_limit_or_the_context() {
    let prompt: Vec<u32> = (1..20).collect();
    let (tokens, _) = expected(&ask(prompt.clone(), 10));
    let asks = [
        Ask { stop: vec![tokens[4]], ..ask(prompt.clone(), 10) },
        ask((1..60).collect(), 100),
        ask(prompt.clone(), 0),
        ask((0..context() as u32).collect(), 5),
        ask((0..context() as u32 - 1).collect(), 5),
    ];
    check(Plan::default(), 80, 20, 0, &asks);
}

#[test]
fn more_requests_than_lanes_share_their_prefixes() {
    let asks = conversations();
    for plan in [
        Plan::default(),
        Plan { budget: Some(12), ..Plan::default() },
        Plan { every: 3, ..Plan::default() },
        Plan { budget: Some(20), every: 2, ..Plan::default() },
    ] {
        let shares = check(plan, 80, 24, 0, &asks);
        assert!(resumed(&shares) > 0, "{plan:?}: no prompt resumed a prefix");
    }
}

#[test]
fn a_small_pool_evicts_and_waits() {
    let asks = conversations();
    check(Plan::default(), 16, 10, 0, &asks);
    check(Plan { budget: Some(16), headroom: 4, ..Plan::default() }, 16, 10, 1, &asks);
}

#[test]
fn a_listener_gone_ends_its_request_and_the_rest_run() {
    let reference = reference(0);
    let mut fake = Fake::new(80, 24);
    let (tx, queue) = channel();
    let asks: Vec<Ask> = (0..6u32).map(|i| ask(vec![i + 1; 9], 20)).collect();
    let mut outs = Vec::new();
    for (i, a) in asks.iter().enumerate() {
        let (j, rx) = job(a);
        tx.send(j).unwrap();
        outs.push((i % 2 == 0).then_some(rx));
    }
    drop(tx);
    engine::run(
        &reference,
        &mut fake,
        queue,
        &Metrics::new(MetricsConfig::default().iteration_tokens),
        Plan::default(),
        &EngineConfig::default(),
    )
    .expect("served");
    for (a, rx) in asks.iter().zip(&outs) {
        if let Some(rx) = rx {
            assert_eq!(heard(rx), expected(a));
        }
    }
}

#[test]
fn the_cadence_sends_pieces_every_nth_step_while_lanes_decode() {
    let asks: Vec<Ask> = (0..4u32).map(|i| ask(vec![i + 1; 30], 12)).collect();
    let shares = check(Plan { every: 4, ..Plan::default() }, 80, 24, 0, &asks);
    let prompts: Vec<&Share> =
        shares.iter().filter(|s| s.commands.iter().any(|c| matches!(c, Command::Prompt { .. }))).collect();
    assert!(prompts.windows(2).skip(4).any(|w| w[1].at >= w[0].at + 3), "the pieces ran back to back");
}

#[test]
fn a_request_seats_as_a_lane_frees_without_waiting_for_the_cadence() {
    let asks: Vec<Ask> = [4usize, 20, 20, 20, 3]
        .iter()
        .enumerate()
        .map(|(i, &limit)| ask(vec![i as u32 + 1; 6], limit))
        .collect();
    let shares = check(Plan { every: 1000, ..Plan::default() }, 80, 24, 0, &asks);
    let commands: Vec<&Command> = shares.iter().flat_map(|s| &s.commands).collect();
    let fifth = commands
        .iter()
        .position(|c| matches!(c, Command::Prompt { tokens, .. } if tokens[0] == 5))
        .expect("the fifth prefilled");
    let ends: Vec<usize> = commands
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c, Command::End { .. }))
        .map(|(i, _)| i)
        .collect();
    assert!(fifth < ends[3], "the fifth prefilled before four requests had ended: {fifth} vs {ends:?}");
}

#[test]
fn requests_of_no_limit_run_at_once_on_a_pool_short_of_their_extents() {
    let asks: Vec<Ask> = (0..4u32).map(|i| ask(vec![i + 1; 9], usize::MAX)).collect();
    let shares = check(Plan { headroom: 8, ..Plan::default() }, 31, 24, 0, &asks);
    let commands: Vec<&Command> = shares.iter().flat_map(|s| &s.commands).collect();
    let first_end = commands.iter().position(|c| matches!(c, Command::End { .. })).expect("an end");
    let starts = commands[..first_end].iter().filter(|c| matches!(c, Command::Start { .. })).count();
    assert_eq!(starts, width(), "every lane started before any ended");
}

#[test]
fn the_metrics_count_what_ran() {
    let asks = conversations();
    let (got, _, m) = serve_told(Plan::default(), 80, 24, 0, &asks);
    let m = m.total();
    assert_eq!(m.prompt_tokens as usize, asks.iter().map(|a| a.prompt.len()).sum::<usize>());
    assert_eq!(m.cached_tokens as usize % page_rows(), 0, "cached by whole pages");
    assert_eq!(m.generation_tokens as usize, got.iter().map(|(t, _)| t.len()).sum::<usize>());
    assert_eq!((m.running, m.waiting, m.held_pages, m.pages, m.preemptions), (0, 0, 0, 79, 0));
    assert_eq!((m.spec_tokens, m.drafts, m.top_p_clipped), (0, 0, 0), "no drafts, no clips");
}

#[test]
fn drafts_clips_and_rounds_are_counted() {
    let mixed = |(i, a): (usize, Ask)| if i % 2 == 0 { Ask { sampling: nucleus(), ..a } } else { a };
    let asks: Vec<Ask> = conversations().into_iter().enumerate().map(mixed).collect();
    let fake = Fake { drafts: 2, clip: true, ..Fake::new(80, 24) };
    let (got, shares, m) = serve_on(&drafting(0, 3), fake, Plan::default(), &asks);
    for (i, (a, g)) in asks.iter().zip(&got).enumerate() {
        assert_eq!(g, &expected(a), "request {i}");
    }
    let (rounds, m) = (&m.iteration_tokens, m.total());
    let (drafts, accepted) = (m.drafts as usize, m.accepted_tokens() as usize);
    assert_eq!(m.spec_tokens, 2);
    assert!(drafts > 0 && accepted > 0, "{drafts} drafts, {accepted} accepted");
    assert_eq!(m.draft_tokens() as usize, 2 * drafts);
    let per_pos: Vec<usize> = m.accepted_per_pos.iter().map(|&c| c as usize).collect();
    assert!(
        per_pos[0] >= per_pos[1] && per_pos[1] > 0 && per_pos[2..].iter().all(|&c| c == 0),
        "{per_pos:?}"
    );
    assert_eq!(per_pos.iter().sum::<usize>(), accepted);
    assert!(per_pos.len() <= EngineConfig::default().max_draft_positions);
    assert!(m.generation_tokens as usize <= accepted + drafts + asks.len());
    let prompts: Vec<usize> = shares
        .iter()
        .flat_map(|s| &s.commands)
        .filter_map(|c| if let Command::Prompt { tokens, .. } = c { Some(tokens.len()) } else { None })
        .collect();
    let seats = shares
        .iter()
        .flat_map(|s| &s.commands)
        .filter(|c| matches!(c, Command::Seat { sampling, .. } if sampling.nucleated()));
    assert_eq!(m.top_p_clipped as usize, seats.count(), "every nucleus chunk's clip");
    assert!(m.top_p_clipped > 0 && (m.top_p_clipped as usize) < prompts.len());
    let sum = f64::from_bits(rounds.sum.load(std::sync::atomic::Ordering::Relaxed)) as usize;
    assert_eq!(
        sum,
        prompts.iter().sum::<usize>() + m.generation_tokens as usize,
        "a round's pieces and tokens heard"
    );
    assert!(rounds.counts.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).sum::<u64>() > 0);
}

#[test]
fn a_reset_empties_the_prefix_cache() {
    let (tx, queue) = channel::<Job>();
    let (told, shares) = channel::<Share>();
    let metrics = std::sync::Arc::new(Metrics::new(MetricsConfig::default().iteration_tokens));
    let m = metrics.clone();
    let loop_ = thread::spawn(move || {
        engine::run(
            &reference(0),
            &mut Fake { told: Some(told), ..Fake::new(80, 24) },
            queue,
            &m,
            Plan::default(),
            &EngineConfig::default(),
        )
    });
    let a = ask((1..20).collect(), 6);
    let run = |tx: &Sender<Job>| {
        let (j, rx) = job(&a);
        tx.send(j).unwrap();
        assert_eq!(heard(&rx), expected(&a));
        let shares: Vec<Share> = shares.try_iter().collect();
        shares.into_iter().flat_map(|s| s.commands).find_map(|c| match c {
            Command::Seat { pos, .. } => Some(pos as usize),
            _ => None,
        })
    };
    let reset = |tx: &Sender<Job>| {
        let (answer, freed) = channel();
        tx.send(Job::Reset(answer)).unwrap();
        freed.recv().unwrap()
    };
    assert_eq!(run(&tx), Some(0));
    assert_eq!(run(&tx), Some(16), "the second resumed the first's prefix");
    let cached = metrics.total().cached_tokens;
    assert_eq!(cached, 16, "the metrics tell the prefix cached");
    assert_eq!(reset(&tx), (a.prompt.len() + a.limit - 1) / page_rows(), "the tree's whole pages freed");
    assert_eq!(run(&tx), Some(0), "nothing resumed after the reset");
    assert_eq!(metrics.total().cached_tokens, cached, "nothing more cached");
    assert_eq!(reset(&tx), (a.prompt.len() + a.limit - 1) / page_rows());
    assert_eq!(reset(&tx), 0, "an empty tree frees nothing");
    drop(tx);
    loop_.join().unwrap().expect("served");
}
