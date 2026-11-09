use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepMetrics {
    pub running: u64,
    pub waiting: u64,
    pub held_pages: u64,
    pub pages: u64,
    pub spec_tokens: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub prefill_tokens: u64,
    pub generation_tokens: u64,
    pub preemptions: u64,
    pub top_p_clipped: u64,
    pub drafts: u64,
    pub accepted_per_pos: Vec<u64>,
}

impl StepMetrics {
    pub fn then(&self, next: StepMetrics) -> StepMetrics {
        StepMetrics {
            running: next.running,
            waiting: next.waiting,
            held_pages: next.held_pages,
            pages: next.pages,
            spec_tokens: next.spec_tokens,
            prompt_tokens: self.prompt_tokens + next.prompt_tokens,
            cached_tokens: self.cached_tokens + next.cached_tokens,
            prefill_tokens: self.prefill_tokens + next.prefill_tokens,
            generation_tokens: self.generation_tokens + next.generation_tokens,
            preemptions: self.preemptions + next.preemptions,
            top_p_clipped: self.top_p_clipped + next.top_p_clipped,
            drafts: self.drafts + next.drafts,
            accepted_per_pos: (0..self.accepted_per_pos.len().max(next.accepted_per_pos.len()))
                .map(|i| {
                    self.accepted_per_pos.get(i).unwrap_or(&0) + next.accepted_per_pos.get(i).unwrap_or(&0)
                })
                .collect(),
        }
    }

    pub fn verified(&mut self, accepted: usize) {
        self.drafts += 1;
        if self.accepted_per_pos.len() < accepted {
            self.accepted_per_pos.resize(accepted, 0);
        }
        self.accepted_per_pos.iter_mut().take(accepted).for_each(|c| *c += 1);
    }

    pub fn tokens(&self) -> u64 {
        self.prefill_tokens + self.generation_tokens
    }

    pub fn draft_tokens(&self) -> u64 {
        self.drafts * self.spec_tokens
    }

    pub fn accepted_tokens(&self) -> u64 {
        self.accepted_per_pos.iter().sum()
    }
}

#[derive(Debug)]
pub struct Metrics {
    total: Mutex<StepMetrics>,
    pub iteration_tokens: Hist,
}

impl Metrics {
    pub fn new(iteration_tokens: Vec<f64>) -> Metrics {
        Metrics { total: Mutex::default(), iteration_tokens: Hist::new(iteration_tokens) }
    }

    pub fn record(&self, step: StepMetrics) {
        if step.tokens() > 0 {
            self.iteration_tokens.observe(step.tokens() as f64);
        }
        let mut total = self.total.lock().unwrap_or_else(PoisonError::into_inner);
        *total = total.then(step);
    }

    pub fn total(&self) -> StepMetrics {
        self.total.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }
}

#[derive(Debug)]
pub struct Hist {
    pub bounds: Vec<f64>,
    pub counts: Vec<AtomicU64>,
    pub sum: AtomicU64,
}

impl Hist {
    pub fn new(bounds: Vec<f64>) -> Hist {
        debug_assert!(bounds.windows(2).all(|w| w[0] < w[1]), "{bounds:?}");
        Hist {
            counts: (0..=bounds.len()).map(|_| AtomicU64::default()).collect(),
            bounds,
            sum: AtomicU64::default(),
        }
    }

    pub fn observe(&self, x: f64) {
        self.counts[self.bounds.partition_point(|&b| b < x)].fetch_add(1, Ordering::Relaxed);
        let _ = self
            .sum
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |s| Some((f64::from_bits(s) + x).to_bits()));
    }
}
