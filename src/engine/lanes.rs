use std::ops::Range;
use std::time::Instant;

use crate::engine::{End, Out, Request, Token};
use crate::gpu::{Command, Sampled};

pub struct Lane {
    pub lane: usize,
    pub seat: u64,
    pub request: Request,
    pub sent: usize,
    pub from: Option<u32>,
}

impl Lane {
    pub fn decoding(&self) -> bool {
        self.from.is_some()
    }

    pub fn decodes_at(&self, step: u32) -> bool {
        self.from.is_some_and(|from| step >= from)
    }

    pub fn written(&self) -> usize {
        if self.decoding() { self.request.prompt.len() - 1 } else { self.sent }
    }

    pub fn next_chunk(&self, rows: usize, tokens: usize) -> Range<usize> {
        self.sent..self.sent + (self.request.prompt.len() - self.sent).min(rows).min(tokens)
    }

    pub fn chunk(&self, piece: Range<usize>, pad: i32) -> Vec<Command> {
        let (lane, tokens) = (self.lane as u32, &self.request.prompt);
        let back =
            std::array::from_fn(|j| (piece.start + j).checked_sub(3).map_or(pad, |i| tokens[i] as i32));
        let mut out = vec![
            Command::Seat { lane, pos: piece.start as u32, back, sampling: self.request.sampling },
            Command::Prompt { lane, tokens: tokens[piece.clone()].to_vec() },
        ];
        if piece.end == tokens.len() {
            out.push(Command::Start { lane });
        }
        out
    }

    pub fn ahead(&self, upto: usize, context: usize, block: usize, ahead: usize) -> Option<Range<usize>> {
        let next = self.request.prompt.len() - 1;
        let last = self.request.prompt.len().saturating_add(self.request.limit).min(context) - 2;
        let end = (next + ahead).min(last + 1) + block - 1;
        (end > next + block - 1 && upto < end && upto < next + ahead / 2 + block).then_some(next..end)
    }

    pub fn hear(&mut self, t: &Sampled, context: usize, at: Instant) -> Option<End> {
        let Ok(id) = u32::try_from(t.id) else {
            return Some(End::Refused("the kernel's logits were not finite".into()));
        };
        let r = &mut self.request;
        let ask = r.sampling.logprobs as usize;
        let logprobs = match t.logprobs.split_last() {
            Some((own, top)) if ask > 0 => {
                std::iter::once(own).chain(top.iter().take(ask)).copied().collect()
            }
            _ => Vec::new(),
        };
        if r.out.send(Out::Token(Token { id, logprobs, at })).is_err() {
            return Some(End::Stop);
        }
        if r.stop.contains(&id) {
            return Some(End::Stop);
        }
        r.took(id);
        (r.limit == 0 || r.prompt.len() >= context).then_some(End::Length)
    }
}

enum Slot {
    Free,
    Held(Lane),
    Ending,
}

pub struct Lanes {
    slots: Vec<Slot>,
    seats: u64,
}

impl Lanes {
    pub fn new(width: usize) -> Lanes {
        Lanes { slots: (0..width).map(|_| Slot::Free).collect(), seats: 0 }
    }

    pub fn free(&self) -> Option<usize> {
        self.slots.iter().position(|s| matches!(s, Slot::Free))
    }

    pub fn seat(&mut self, lane: usize, request: Request, cached: usize) {
        self.seats += 1;
        self.slots[lane] = Slot::Held(Lane { lane, seat: self.seats, request, sent: cached, from: None });
    }

    pub fn take(&mut self, lane: usize) -> Lane {
        match std::mem::replace(&mut self.slots[lane], Slot::Ending) {
            Slot::Held(l) => l,
            _ => panic!("lane {lane} taken unheld"),
        }
    }

    pub fn ended(&mut self) -> Vec<u32> {
        let ending =
            (0..self.slots.len()).filter(|&l| matches!(self.slots[l], Slot::Ending)).collect::<Vec<usize>>();
        ending.iter().for_each(|&l| self.slots[l] = Slot::Free);
        ending.into_iter().map(|l| l as u32).collect()
    }

    pub fn get(&self, lane: usize) -> Option<&Lane> {
        match &self.slots[lane] {
            Slot::Held(l) => Some(l),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, lane: usize) -> Option<&mut Lane> {
        match &mut self.slots[lane] {
            Slot::Held(l) => Some(l),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| matches!(s, Slot::Free))
    }

    pub fn held(&self) -> Vec<usize> {
        self.by_seat(|_| true)
    }

    pub fn prompting(&self) -> Vec<usize> {
        self.by_seat(|l| !l.decoding())
    }

    pub fn decoding(&self) -> Vec<usize> {
        self.by_seat(Lane::decoding)
    }

    pub fn newest(&self) -> Option<usize> {
        self.held().last().copied()
    }

    fn by_seat(&self, pick: impl Fn(&Lane) -> bool) -> Vec<usize> {
        let mut lanes: Vec<&Lane> = self
            .slots
            .iter()
            .filter_map(|s| if let Slot::Held(l) = s { Some(l) } else { None })
            .filter(|l| pick(l))
            .collect();
        lanes.sort_by_key(|l| l.seat);
        lanes.into_iter().map(|l| l.lane).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::channel;
    use std::time::Instant;

    use super::Lane;
    use crate::engine::{Out, Request, Sampling};
    use crate::gpu::Sampled;

    #[test]
    fn a_token_goes_with_the_logprobs_asked_and_is_taken() {
        let (out, heard) = channel();
        let sampling = Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 0, logprobs: 2 };
        let mut l = Lane {
            lane: 0,
            seat: 1,
            request: Request { prompt: vec![1], sampling, limit: 5, stop: Vec::new(), out },
            sent: 1,
            from: Some(0),
        };
        let row = vec![(7, -0.1), (8, -2.0), (9, -3.0), (7, -0.1)];
        assert_eq!(l.hear(&Sampled { id: 7, logprobs: row }, 64, Instant::now()), None);
        let Ok(Out::Token(t)) = heard.try_recv() else { panic!("no token") };
        assert_eq!(t.logprobs, [(7, -0.1), (7, -0.1), (8, -2.0)], "its own, then the top two");
        assert_eq!((l.request.prompt, l.request.limit), (vec![1, 7], 4));
    }
}
