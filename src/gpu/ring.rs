use std::sync::atomic::{Ordering, fence};

use super::Pinned;
use crate::bundle::{Dtype, Field, Kind, Message, Rings};

#[derive(Clone, Copy, Debug)]
pub struct RingConfig {
    pub head: usize,
    pub tail: usize,
    pub slots: usize,
    pub body: usize,
}

impl Default for RingConfig {
    fn default() -> RingConfig {
        RingConfig { head: 0, tail: 64, slots: 128, body: 16 }
    }
}

pub struct Ring {
    mem: Pinned,
    layout: RingConfig,
    slots: usize,
    slot: usize,
    n: u64,
    mine: usize,
    theirs: usize,
}

impl Ring {
    pub fn requests(rings: &Rings, layout: &RingConfig) -> Result<Ring, String> {
        Ring::new(rings, *layout, 0, layout.head, layout.tail)
    }

    pub fn replies(rings: &Rings, layout: &RingConfig) -> Result<Ring, String> {
        Ring::new(rings, *layout, 1, layout.tail, layout.head)
    }

    fn new(
        rings: &Rings,
        layout: RingConfig,
        ring: usize,
        mine: usize,
        theirs: usize,
    ) -> Result<Ring, String> {
        let widest = rings.messages.iter().filter(|m| m.ring == ring).map(|m| m.bytes).max().unwrap_or(0);
        let (slots, slot) = (rings.slots[ring], (layout.body + widest).next_multiple_of(16));
        Ok(Ring { mem: Pinned::new(layout.slots + slots * slot)?, layout, slots, slot, n: 0, mine, theirs })
    }

    pub fn device(&self) -> u64 {
        self.mem.device()
    }

    pub fn count(&self) -> u64 {
        self.n
    }

    pub fn push(&mut self, id: u32, body: &[u8]) -> bool {
        let at = self.layout.body;
        self.next(|_, slot| {
            slot[..4].copy_from_slice(&id.to_le_bytes());
            slot[at..at + body.len()].copy_from_slice(body);
        })
        .is_some()
    }

    pub fn pop<T>(&mut self, read: impl FnOnce(u64, u32, &[u8]) -> T) -> Option<T> {
        let body = self.layout.body;
        self.next(|n, slot| {
            read(n, u32::from_le_bytes(slot[..4].try_into().expect("4 bytes")), &slot[body..])
        })
    }

    fn next<T>(&mut self, f: impl FnOnce(u64, &mut [u8]) -> T) -> Option<T> {
        let (theirs, slots) = (self.mem.word(self.theirs).load(Ordering::Acquire), self.slots as u64);
        let ready = if self.mine == self.layout.head { self.n - theirs < slots } else { theirs > self.n };
        if !ready {
            return None;
        }
        let at = self.layout.slots + (self.n % slots) as usize * self.slot;
        let slot = unsafe { std::slice::from_raw_parts_mut(self.mem.host().add(at), self.slot) };
        let out = f(self.n, slot);
        self.n += 1;
        fence(Ordering::Release);
        self.mem.word(self.mine).store(self.n, Ordering::Release);
        Some(out)
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Span { addr: u64, elements: u64 },
}

pub fn encode(layout: &Message, fields: &[(&str, Value)]) -> Vec<u8> {
    let mut bytes = vec![0; layout.bytes];
    for (name, v) in fields {
        let f = field(layout, name);
        let (at, e) = (f.at, f.dtype.bytes());
        match (f.kind, v) {
            (Kind::One, &Value::Int(x)) => int(f.dtype, x, &mut bytes[at..at + e]),
            (Kind::One, Value::Float(x)) if f.dtype == Dtype::F32 => {
                bytes[at..at + 4].copy_from_slice(&x.to_le_bytes())
            }
            (Kind::Array, Value::Ints(xs)) if xs.len() == f.n => {
                xs.iter()
                    .enumerate()
                    .for_each(|(i, &x)| int(f.dtype, x, &mut bytes[at + i * e..at + (i + 1) * e]));
            }
            (Kind::Span, &Value::Span { addr, elements }) => {
                let elements = if f.n > 0 { f.n as u64 } else { elements };
                bytes[at..at + 8].copy_from_slice(&addr.to_le_bytes());
                bytes[at + 8..at + 16].copy_from_slice(&elements.to_le_bytes());
            }
            (kind, v) => {
                panic!("{}.{name} ({kind:?} of {} {:?}) written as {v:?}", layout.name, f.n, f.dtype)
            }
        }
    }
    bytes
}

pub fn decode(layout: &Message, bytes: &[u8], name: &str) -> Vec<i32> {
    let f = field(layout, name);
    let n = if f.kind == Kind::One { 1 } else { f.n };
    bytes[f.at..f.at + 4 * n].as_chunks::<4>().0.iter().map(|&c| i32::from_le_bytes(c)).collect()
}

fn int(dtype: Dtype, x: i64, out: &mut [u8]) {
    match dtype {
        Dtype::I32 => out.copy_from_slice(&(x as i32).to_le_bytes()),
        Dtype::U32 => out.copy_from_slice(&(x as u32).to_le_bytes()),
        Dtype::U64 => out.copy_from_slice(&(x as u64).to_le_bytes()),
        d => panic!("{x} written as {d:?}"),
    }
}

fn field<'a>(m: &'a Message, name: &str) -> &'a Field {
    m.fields.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("{} has no field {name}", m.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_encode_by_their_layout_and_decode_back() {
        let f = |name: &str, at, kind, dtype, n| Field { name: name.into(), at, kind, dtype, n };
        let m = Message {
            name: "seat".into(),
            id: 0x104,
            ring: 0,
            bytes: 56,
            fields: vec![
                f("lane", 0, Kind::One, Dtype::I32, 0),
                f("back", 4, Kind::Array, Dtype::I32, 3),
                f("temp", 16, Kind::One, Dtype::F32, 0),
                f("row", 24, Kind::Span, Dtype::I32, 0),
                f("lps", 40, Kind::Span, Dtype::F32, 33),
            ],
        };
        let span = |addr, elements| Value::Span { addr, elements };
        let b = encode(
            &m,
            &[
                ("lane", Value::Int(-1)),
                ("back", Value::Ints(vec![1, 2, 3])),
                ("temp", Value::Float(0.5)),
                ("row", span(0x1000, 7)),
                ("lps", span(0x2000, 99)),
            ],
        );
        assert_eq!(b.len(), 56);
        assert_eq!(decode(&m, &b, "lane"), [-1]);
        assert_eq!(decode(&m, &b, "back"), [1, 2, 3]);
        assert_eq!(b[16..20], 0.5f32.to_le_bytes());
        assert_eq!(b[24..40], [0x1000u64.to_le_bytes(), 7u64.to_le_bytes()].concat());
        assert_eq!(b[40..56], [0x2000u64.to_le_bytes(), 33u64.to_le_bytes()].concat(), "the layout's count");
    }
}
