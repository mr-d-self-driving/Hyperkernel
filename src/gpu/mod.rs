pub mod cuda;
pub mod feed;
pub mod load;
pub mod ranks;
pub mod ring;

use std::sync::atomic::AtomicU64;

use serde::{Deserialize, Serialize};

use crate::engine::Sampling;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Share {
    pub at: u32,
    pub horizon: u32,
    pub commands: Vec<Command>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Command {
    End { lane: u32 },
    Pages { lane: u32, from: u32, upto: u32, row: Vec<i32> },
    Seat { lane: u32, pos: u32, back: [i32; 3], sampling: Sampling },
    Prompt { lane: u32, tokens: Vec<u32> },
    Start { lane: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Idle { step: u32 },
    Step { step: u32, lanes: Vec<Vec<Sampled>> },
    Prompted { lane: u32, token: Sampled, clipped: bool },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sampled {
    pub id: i32,
    pub logprobs: Vec<(u32, f32)>,
}

pub trait Kernel {
    fn pools(&self) -> (usize, usize);
    fn push(&mut self, share: &Share) -> Result<(), String>;
    fn drain(&mut self) -> Result<Vec<Event>, String>;
}

pub struct Running {
    pub requests: ring::Ring,
    pub replies: ring::Ring,
    pub ctl: Pinned,
    pub pages: usize,
    pub window_pages: usize,
}

pub struct Pinned {
    host: *mut u8,
    device: u64,
    len: usize,
}

unsafe impl Send for Pinned {}

impl Pinned {
    pub fn new(len: usize) -> Result<Pinned, String> {
        let (host, device) = cuda::host_alloc(len)?;
        Ok(Pinned { host, device, len })
    }

    pub fn host(&self) -> *mut u8 {
        self.host
    }

    pub fn device(&self) -> u64 {
        self.device
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn word(&self, at: usize) -> &AtomicU64 {
        assert!(at.is_multiple_of(8) && at + 8 <= self.len, "word {at} of {} pinned bytes", self.len);
        unsafe { &*(self.host.add(at) as *const AtomicU64) }
    }

    pub fn write(&mut self, at: usize, bytes: &[u8]) {
        assert!(at + bytes.len() <= self.len, "a pinned write past its {} bytes", self.len);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.host.add(at), bytes.len()) }
    }

    pub fn read(&self, at: usize, len: usize) -> &[u8] {
        assert!(at + len <= self.len, "a pinned read past its {} bytes", self.len);
        unsafe { std::slice::from_raw_parts(self.host.add(at), len) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe { cuda::host_free(self.host) }
    }
}
