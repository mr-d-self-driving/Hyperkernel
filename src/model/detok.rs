use std::str;

use crate::model::tokenizer::{Special, Tokenizer};

pub struct Detok<'t> {
    tok: &'t Tokenizer,
    special: Special,
    ends: Vec<u32>,
    stops: Vec<String>,
    include: bool,
    text: String,
    partial: Vec<u8>,
    sent: usize,
    cut: Option<usize>,
}

impl<'t> Detok<'t> {
    pub fn new(
        tok: &'t Tokenizer,
        special: Special,
        ends: Vec<u32>,
        stops: Vec<String>,
        include: bool,
    ) -> Detok<'t> {
        Detok {
            tok,
            special,
            ends,
            stops,
            include,
            text: String::new(),
            partial: Vec::new(),
            sent: 0,
            cut: None,
        }
    }

    pub fn push(&mut self, id: u32) -> (String, Option<String>) {
        if self.ends.contains(&id) {
            return (String::new(), None);
        }
        let before = self.text.len();
        self.partial.extend_from_slice(self.tok.bytes(id, self.special));
        self.take_whole_chars();
        let from = floor_char(&self.text, before.saturating_sub(self.hold()));
        let found = self
            .stops
            .iter()
            .enumerate()
            .filter_map(|(i, s)| self.text[from..].find(s.as_str()).map(|at| (at, i)))
            .min();
        let stop = found.map(|(at, i)| {
            let stop = self.stops[i].clone();
            self.cut = Some(from + at + if self.include { stop.len() } else { 0 });
            stop
        });
        let held = floor_char(&self.text, self.text.len().saturating_sub(self.hold()).max(self.sent));
        (self.send(self.cut.unwrap_or(held)), stop)
    }

    pub fn finish(&mut self) -> String {
        self.text.push_str(&String::from_utf8_lossy(&std::mem::take(&mut self.partial)));
        self.send(self.cut.unwrap_or(self.text.len()))
    }

    fn send(&mut self, end: usize) -> String {
        let text = self.text[self.sent..end].to_owned();
        self.sent = end;
        text
    }

    fn hold(&self) -> usize {
        self.stops.iter().map(String::len).max().unwrap_or(1).saturating_sub(1)
    }

    fn take_whole_chars(&mut self) {
        let mut rest = &self.partial[..];
        loop {
            match str::from_utf8(rest) {
                Ok(whole) => {
                    self.text.push_str(whole);
                    rest = &[];
                    break;
                }
                Err(e) => {
                    self.text.push_str(str::from_utf8(&rest[..e.valid_up_to()]).expect("valid up to"));
                    let Some(bad) = e.error_len() else {
                        rest = &rest[e.valid_up_to()..];
                        break;
                    };
                    self.text.push(char::REPLACEMENT_CHARACTER);
                    rest = &rest[e.valid_up_to() + bad..];
                }
            }
        }
        self.partial = rest.to_vec();
    }
}

fn floor_char(s: &str, mut at: usize) -> usize {
    while !s.is_char_boundary(at) {
        at -= 1;
    }
    at
}
