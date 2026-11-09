#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub budget: Option<usize>,
    pub every: usize,
    pub headroom: usize,
}

impl Default for Plan {
    fn default() -> Plan {
        Plan { budget: None, every: 1, headroom: 2048 }
    }
}

pub struct Pacing {
    plan: Plan,
    steps: usize,
}

pub struct Allowance {
    pub chunks: usize,
    pub tokens: usize,
}

impl Pacing {
    pub fn new(plan: Plan) -> Pacing {
        Pacing { plan, steps: 0 }
    }

    pub fn due(&self, decoding: usize) -> bool {
        decoding == 0 || self.steps + 1 >= self.plan.every
    }

    pub fn allowance(&self, decoding: usize, block: usize) -> Allowance {
        match self.plan.budget {
            None => Allowance { chunks: 1, tokens: usize::MAX },
            Some(b) => Allowance { chunks: usize::MAX, tokens: b.saturating_sub(decoding * block) },
        }
    }

    pub fn headroom(&self) -> usize {
        self.plan.headroom
    }

    pub fn stepped(&mut self) {
        self.steps += 1;
    }

    pub fn prefilled(&mut self) {
        self.steps = 0;
    }
}

impl Allowance {
    pub fn left(&self) -> bool {
        self.chunks > 0 && self.tokens > 0
    }

    pub fn spend(&mut self, tokens: usize) {
        (self.chunks, self.tokens) = (self.chunks - 1, self.tokens - tokens);
    }
}
