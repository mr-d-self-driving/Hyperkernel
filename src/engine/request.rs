use std::sync::mpsc::Sender;
use std::time::Instant;

use serde::{Deserialize, Serialize};

pub enum Job {
    Run(Request),
    Reset(Sender<usize>),
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub sampling: Sampling,
    pub limit: usize,
    pub stop: Vec<u32>,
    pub out: Sender<Out>,
}

impl Request {
    pub fn took(&mut self, id: u32) {
        self.prompt.push(id);
        self.limit -= 1;
    }

    pub fn reach(&self, headroom: usize, context: usize, block: usize) -> usize {
        (self.prompt.len() + self.limit.min(headroom)).min(context) + block - 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: u64,
    pub logprobs: u32,
}

impl Sampling {
    pub fn nucleated(&self) -> bool {
        self.temperature > 0.0 && (self.top_p < 1.0 || self.top_k > 0)
    }
}

#[derive(Debug, PartialEq)]
pub enum Out {
    Seated { at: Instant, cached: usize },
    Token(Token),
    End(End),
}

#[derive(Debug, PartialEq)]
pub struct Token {
    pub id: u32,
    pub logprobs: Vec<(u32, f32)>,
    pub at: Instant,
}

#[derive(Debug, PartialEq)]
pub enum End {
    Stop,
    Length,
    Refused(String),
}
