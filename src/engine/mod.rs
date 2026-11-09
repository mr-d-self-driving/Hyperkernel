pub mod cache;
pub mod clock;
pub mod lanes;
pub mod metrics;
pub mod pacing;
pub mod request;

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::thread;
use std::time::Instant;

use crate::bundle::{Model, Reference};
use crate::gpu::{Command, Event, Kernel, Sampled, Share};
use cache::{Cache, Wait};
use clock::Clock;
use lanes::Lanes;
use pacing::Pacing;

pub use metrics::{Hist, Metrics, StepMetrics};
pub use pacing::Plan;
pub use request::{End, Job, Out, Request, Sampling, Token};

pub struct EngineConfig {
    pub gap: usize,
    pub lead: u32,
    pub ahead: usize,
    pub max_draft_positions: usize,
}

impl Default for EngineConfig {
    fn default() -> EngineConfig {
        EngineConfig { gap: 16, lead: 2, ahead: 16, max_draft_positions: 16 }
    }
}

pub fn run(
    reference: &Reference,
    kernel: &mut impl Kernel,
    jobs: Receiver<Job>,
    metrics: &Metrics,
    plan: Plan,
    config: &EngineConfig,
) -> Result<(), String> {
    let drafts = reference.served.model.block - 1;
    if drafts > config.max_draft_positions {
        return Err(format!(
            "{drafts} drafts a step, past the {} the metrics count",
            config.max_draft_positions
        ));
    }
    let mut engine = Engine::new(reference, kernel.pools(), plan, config);
    metrics.record(engine.step_metrics());
    let failure = loop {
        if !engine.take(&jobs) {
            return Ok(());
        }
        if let Some(share) = engine.schedule()
            && let Err(e) = kernel.push(&share)
        {
            break e;
        }
        match kernel.drain() {
            Ok(events) if events.is_empty() => thread::yield_now(),
            Ok(events) => events.into_iter().for_each(|e| engine.hear(e)),
            Err(e) => break e,
        }
        metrics.record(engine.step_metrics());
    };
    engine.refuse(&failure, &jobs);
    metrics.record(engine.step_metrics());
    Err(failure)
}

struct Engine<'a> {
    model: &'a Model,
    config: &'a EngineConfig,
    waiting: VecDeque<Request>,
    lanes: Lanes,
    cache: Cache,
    asked: VecDeque<Asked>,
    clock: Clock,
    pacing: Pacing,
    step: StepMetrics,
}

struct Asked {
    lane: usize,
    seat: u64,
    last: bool,
    nucleated: bool,
}

impl<'a> Engine<'a> {
    fn new(
        reference: &'a Reference,
        (pages, window_pages): (usize, usize),
        plan: Plan,
        config: &'a EngineConfig,
    ) -> Engine<'a> {
        let model = &reference.served.model;
        let width = *model.batches.last().expect("a decode width");
        Engine {
            model,
            config,
            waiting: VecDeque::new(),
            lanes: Lanes::new(width),
            cache: Cache::new(&reference.served.pool, pages, window_pages, width),
            asked: VecDeque::new(),
            clock: Clock::new(config.lead),
            pacing: Pacing::new(plan),
            step: StepMetrics::default(),
        }
    }

    fn take(&mut self, jobs: &Receiver<Job>) -> bool {
        loop {
            match jobs.try_recv() {
                Ok(j) => self.job(j),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return !self.idle(),
            }
        }
        if self.idle() {
            match jobs.recv() {
                Ok(j) => self.job(j),
                Err(_) => return false,
            }
        }
        true
    }

    fn job(&mut self, j: Job) {
        match j {
            Job::Run(r) => self.waiting.push_back(r),
            Job::Reset(answer) => {
                let _ = answer.send(self.cache.reset());
            }
        }
    }

    fn idle(&self) -> bool {
        self.lanes.is_empty() && self.waiting.is_empty() && self.asked.is_empty()
    }

    fn schedule(&mut self) -> Option<Share> {
        self.extend_decodes();
        let prompts = self.prefill();
        let mut commands: Vec<Command> =
            self.lanes.ended().into_iter().map(|lane| Command::End { lane }).collect();
        for lane in self.lanes.held() {
            commands.extend(self.cache.pages(lane, self.config.gap));
        }
        commands.extend(prompts);
        self.clock.grant(commands, !self.lanes.decoding().is_empty())
    }

    fn extend_decodes(&mut self) {
        for lane in self.lanes.decoding() {
            let Some(l) = self.lanes.get(lane) else { continue };
            let Some(rows) =
                l.ahead(self.cache.upto(lane), self.model.context, self.model.block, self.config.ahead)
            else {
                continue;
            };
            if self.cover(lane, rows.clone()) {
                self.cache.run_to(lane, rows.end);
            }
        }
    }

    fn prefill(&mut self) -> Vec<Command> {
        let seated = self.seat_waiting();
        let due = self.pacing.due(self.lanes.decoding().len());
        let chunks = self.chunks(if due { self.lanes.prompting() } else { seated });
        if due && !chunks.is_empty() {
            self.pacing.prefilled();
        }
        chunks
    }

    fn seat_waiting(&mut self) -> Vec<usize> {
        let (context, mut seated) = (self.model.context, Vec::new());
        while let (Some(lane), Some(r)) = (self.lanes.free(), self.waiting.front()) {
            let n = r.prompt.len();
            let end = match n {
                0 => End::Refused("an empty prompt".into()),
                n if n >= context => End::Refused(format!(
                    "a prompt of {n} tokens: the context holds {context}, a token past it"
                )),
                _ if r.limit == 0 => End::Length,
                _ => match self.cache.admit(
                    lane,
                    &r.prompt,
                    r.reach(self.pacing.headroom(), context, self.model.block),
                ) {
                    Ok(cached) => {
                        self.seat(lane, cached);
                        seated.push(lane);
                        continue;
                    }
                    Err(Wait::Later) => return seated,
                    Err(Wait::Never(why)) => End::Refused(format!("the cache: {why}")),
                },
            };
            let r = self.waiting.pop_front().expect("a waiting request");
            let _ = r.out.send(Out::End(end));
        }
        seated
    }

    fn chunks(&mut self, lanes: Vec<usize>) -> Vec<Command> {
        let (at, rows, pad) = (self.clock.next(), self.model.chunks[0], self.model.pad as i32);
        let mut allowance = self.pacing.allowance(self.lanes.decoding().len(), self.model.block);
        let mut out = Vec::new();
        for lane in lanes {
            if !allowance.left() {
                break;
            }
            let Some(l) = self.lanes.get(lane) else { continue };
            let piece = l.next_chunk(rows, allowance.tokens);
            let last = piece.end == l.request.prompt.len();
            let ahead = l
                .ahead(self.cache.upto(lane), self.model.context, self.model.block, self.config.ahead)
                .filter(|_| last);
            if !self.cover(lane, piece.clone())
                || ahead.as_ref().is_some_and(|rows| !self.cover(lane, piece.start..rows.end))
            {
                continue;
            }
            if let Some(rows) = ahead {
                self.cache.run_to(lane, rows.end);
            }
            let l = self.lanes.get_mut(lane).expect("a lane its pages held");
            out.extend(l.chunk(piece.clone(), pad));
            l.sent = piece.end;
            if last {
                l.from = Some(at);
            }
            self.asked.push_back(Asked {
                lane,
                seat: l.seat,
                last,
                nucleated: l.request.sampling.nucleated(),
            });
            self.step.prefill_tokens += piece.len() as u64;
            allowance.spend(piece.len());
        }
        out
    }

    fn cover(&mut self, lane: usize, rows: Range<usize>) -> bool {
        loop {
            let Err(e) = self.cache.hold(lane, rows.clone()) else { return true };
            let newest = self.lanes.newest().expect("a held lane");
            if self.lanes.held().len() == 1 {
                self.end(lane, End::Refused(format!("the cache: {e}")));
                return false;
            }
            self.preempt(newest);
            if newest == lane {
                return false;
            }
        }
    }

    fn hear(&mut self, e: Event) {
        let at = Instant::now();
        match e {
            Event::Idle { step } => self.clock.saw(step),
            Event::Step { step, lanes } => {
                self.clock.saw(step);
                self.pacing.stepped();
                for (lane, tokens) in lanes.iter().enumerate() {
                    self.decoded(lane, step, tokens, at);
                }
            }
            Event::Prompted { lane, token, clipped } => self.prompted(lane as usize, &token, clipped, at),
        }
    }

    fn prompted(&mut self, lane: usize, token: &Sampled, clipped: bool, at: Instant) {
        let asked = self.asked.pop_front().expect("a prompt chunk the reply answers");
        assert_eq!(asked.lane, lane, "a prompt chunk's reply for another lane");
        self.step.top_p_clipped += u64::from(clipped && asked.nucleated);
        if asked.last && self.lanes.get(lane).is_some_and(|l| l.seat == asked.seat) {
            self.heard(lane, token, at);
        }
    }

    fn decoded(&mut self, lane: usize, step: u32, tokens: &[Sampled], at: Instant) {
        if !self.lanes.get(lane).is_some_and(|l| l.decodes_at(step)) {
            return;
        }
        if self.model.block > 1 && !tokens.is_empty() && tokens.iter().all(|t| t.id >= 0) {
            self.step.verified(tokens.len() - 1);
        }
        for t in tokens {
            if !self.heard(lane, t, at) {
                return;
            }
        }
    }

    fn heard(&mut self, lane: usize, t: &Sampled, at: Instant) -> bool {
        self.step.generation_tokens += 1;
        let end = self.lanes.get_mut(lane).expect("a held lane").hear(t, self.model.context, at);
        end.map(|why| self.end(lane, why)).is_none()
    }

    fn seat(&mut self, lane: usize, cached: usize) {
        let r = self.waiting.pop_front().expect("a waiting request");
        self.step.prompt_tokens += r.prompt.len() as u64;
        self.step.cached_tokens += cached as u64;
        let _ = r.out.send(Out::Seated { at: Instant::now(), cached });
        self.lanes.seat(lane, r, cached);
    }

    fn preempt(&mut self, lane: usize) {
        self.step.preemptions += 1;
        let r = self.free(lane);
        self.waiting.push_front(r);
    }

    fn end(&mut self, lane: usize, why: End) {
        let _ = self.free(lane).out.send(Out::End(why));
    }

    fn free(&mut self, lane: usize) -> Request {
        let l = self.lanes.take(lane);
        self.cache.finish(lane, &l.request.prompt[..l.written()]);
        l.request
    }

    fn step_metrics(&mut self) -> StepMetrics {
        StepMetrics {
            running: self.lanes.held().len() as u64,
            waiting: self.waiting.len() as u64,
            held_pages: self.cache.held() as u64,
            pages: self.cache.pool() as u64,
            spec_tokens: self.model.block as u64 - 1,
            ..std::mem::take(&mut self.step)
        }
    }

    fn refuse(&mut self, why: &str, jobs: &Receiver<Job>) {
        for lane in self.lanes.held() {
            self.end(lane, End::Refused(why.to_string()));
        }
        let queued = jobs.try_iter().filter_map(|j| match j {
            Job::Run(r) => Some(r),
            Job::Reset(_) => None,
        });
        for r in self.waiting.drain(..).chain(queued) {
            let _ = r.out.send(Out::End(End::Refused(why.to_string())));
        }
    }
}
