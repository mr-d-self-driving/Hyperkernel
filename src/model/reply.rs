use std::mem;

use crate::model::grammar::{GrammarConfig, Item, Miss};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    Call { name: String, arguments: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opening {
    Reasoning,
    Content,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Calls {
    Read,
    Ignored,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Reasoning(Calls),
    Content,
    Block,
    Calls,
    Text,
}

pub struct Reader<'g> {
    grammar: &'g GrammarConfig,
    phase: Phase,
    pending: String,
}

impl<'g> Reader<'g> {
    pub fn new(grammar: &'g GrammarConfig, opening: Opening, calls: Calls) -> Reader<'g> {
        let phase = match opening {
            Opening::Reasoning => Phase::Reasoning(calls),
            Opening::Content => Phase::content(calls),
        };
        Reader { grammar, phase, pending: String::new() }
    }

    pub fn push(&mut self, text: &str) -> Vec<Delta> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        while let Some(phase) = self.step(&mut out) {
            self.phase = phase;
        }
        out
    }

    pub fn finish(&mut self) -> Vec<Delta> {
        let rest = mem::take(&mut self.pending);
        let delta =
            if let Phase::Reasoning(_) = self.phase { Delta::Reasoning(rest) } else { Delta::Content(rest) };
        unless_empty(delta).into_iter().collect()
    }

    fn step(&mut self, out: &mut Vec<Delta>) -> Option<Phase> {
        let g = self.grammar;
        match self.phase {
            Phase::Reasoning(calls) => {
                self.stream_until(&g.end_think, Delta::Reasoning, out)?;
                self.pending.drain(..g.end_think.len());
                Some(Phase::content(calls))
            }
            Phase::Content => self.stream_until(&g.block_start, Delta::Content, out).map(|()| Phase::Block),
            Phase::Block => self.read(g.read_first(&self.pending), out),
            Phase::Calls => self.read(g.read_item(&self.pending), out),
            Phase::Text => {
                out.extend(unless_empty(Delta::Content(mem::take(&mut self.pending))));
                None
            }
        }
    }

    fn read(&mut self, read: Result<(Item, usize), Miss>, out: &mut Vec<Delta>) -> Option<Phase> {
        let (item, n) = match read {
            Ok(read) => read,
            Err(Miss::Broken) => return Some(Phase::Text),
            Err(Miss::Wait) => return None,
        };
        self.pending.drain(..n);
        match item {
            Item::Call(call) => {
                out.push(Delta::Call { arguments: call.arguments(), name: call.name });
                Some(Phase::Calls)
            }
            Item::End => Some(Phase::Text),
        }
    }

    fn stream_until(&mut self, marker: &str, delta: fn(String) -> Delta, out: &mut Vec<Delta>) -> Option<()> {
        let found = self.pending.find(marker);
        let end = found.unwrap_or_else(|| {
            self.pending
                .char_indices()
                .map(|(i, _)| i)
                .find(|&i| marker.starts_with(&self.pending[i..]))
                .unwrap_or(self.pending.len())
        });
        out.extend(unless_empty(delta(self.pending.drain(..end).collect())));
        found.map(|_| ())
    }
}

impl Phase {
    fn content(calls: Calls) -> Phase {
        match calls {
            Calls::Read => Phase::Content,
            Calls::Ignored => Phase::Text,
        }
    }
}

fn unless_empty(delta: Delta) -> Option<Delta> {
    match &delta {
        Delta::Reasoning(t) | Delta::Content(t) if t.is_empty() => None,
        Delta::Reasoning(_) | Delta::Content(_) | Delta::Call { .. } => Some(delta),
    }
}
