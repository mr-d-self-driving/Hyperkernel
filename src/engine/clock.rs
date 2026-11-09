use crate::gpu::{Command, Share};

pub struct Clock {
    lead: u32,
    seen: u32,
    granted: u32,
}

impl Clock {
    pub fn new(lead: u32) -> Clock {
        Clock { lead, seen: 0, granted: 0 }
    }

    pub fn saw(&mut self, step: u32) {
        self.seen = step;
    }

    pub fn next(&self) -> u32 {
        self.granted
    }

    pub fn grant(&mut self, commands: Vec<Command>, decoding: bool) -> Option<Share> {
        let running_out = self.seen + self.lead >= self.granted;
        if commands.is_empty() && !(decoding && running_out) {
            return None;
        }
        let at = self.granted;
        let from = if commands.is_empty() { self.seen } else { self.seen.max(at.saturating_sub(1)) };
        self.granted = from + self.lead + 1;
        Some(Share { at, horizon: from + self.lead, commands })
    }
}
