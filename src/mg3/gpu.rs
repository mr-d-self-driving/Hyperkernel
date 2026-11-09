use std::ops::Range;
use std::path::Path;

use super::link::{Ask, Link};
use super::{At, Dev, Mg3, Port, Prog, draw};
use crate::bundle::{Bundle, Dtype};
use crate::gpu::load::{self, Loaded};
use crate::gpu::ranks::Ranks;
use crate::weights::Checkpoint;

pub struct Kernel {
    bundle: Bundle<Mg3>,
    loaded: Loaded,
}

impl Kernel {
    pub fn open(dir: &Path) -> Result<Bundle<Mg3>, String> {
        Bundle::read(dir, 0)
    }

    pub fn boot(
        bundle: Bundle<Mg3>,
        ranks: &Ranks,
        weights: &Checkpoint,
        config: &load::LoaderConfig,
    ) -> Result<Kernel, String> {
        let loaded = load::load(&bundle, ranks, Some(weights), &[], config)?;
        Ok(Kernel { bundle, loaded })
    }

    pub fn facts(&self) -> &Mg3 {
        &self.bundle.reference.served
    }

    fn span(&self, code: usize, name: &str, range: Range<usize>) -> Result<(&Port, Range<u64>), String> {
        let p = self.facts().port(code, name)?;
        Ok((p, p.span(range)?))
    }

    pub fn write(&self, code: usize, name: &str, at: usize, bytes: &[u8]) -> Result<(), String> {
        let (p, span) = self.span(code, name, at..at + bytes.len())?;
        self.loaded.finish()?;
        self.loaded.write(p.home, span.start, bytes)
    }

    pub fn read(&self, code: usize, name: &str, range: Range<usize>) -> Result<Vec<u8>, String> {
        let (p, span) = self.span(code, name, range.clone())?;
        self.loaded.finish()?;
        self.loaded.read(p.home, span.start, range.len())
    }

    pub fn launch(&self, code: usize) -> Result<(), String> {
        let codes = self.facts().codes;
        if code >= codes.n {
            return Err(format!("code {code} of {}'s {}", self.facts().model.name, codes.n));
        }
        if codes.n > 1 {
            self.loaded.word(codes.at, code as u64)?;
        }
        self.loaded.launch(&self.bundle.reference.kernel)
    }

    pub fn finish(&self) -> Result<(), String> {
        self.loaded.finish()
    }

    pub fn copy(
        &self,
        (code, name, at): (usize, &str, usize),
        from: &Kernel,
        (other, source, start): (usize, &str, usize),
        n: usize,
    ) -> Result<(), String> {
        let (p, span) = self.span(code, name, at..at + n)?;
        let (q, from_span) = from.span(other, source, start..start + n)?;
        self.loaded.copy((p.home, span.start), &from.loaded, (q.home, from_span.start), n)
    }

    pub fn randn(&self, code: usize, name: &str, stream: usize) -> Result<(), String> {
        let p = self.facts().port(code, name)?;
        if p.dtype != Dtype::Bf16 {
            return Err(format!("a draw of `{name}` as {:?}: bf16 alone", p.dtype));
        }
        self.write(code, name, 0, &draw::normal_bf16(name, stream, (p.at.end - p.at.start) as usize))
    }

    pub fn answer(&self, ask: Ask, payload: Vec<u8>) -> Result<Vec<u8>, String> {
        match ask {
            Ask::Write { code, name, at } => self.write(code, &name, at, &payload).map(|()| Vec::new()),
            Ask::Read { code, name, start, end } => self.read(code, &name, start..end),
            Ask::Launch { code } => self.launch(code).map(|()| Vec::new()),
            Ask::Finish => self.finish().map(|()| Vec::new()),
            Ask::Randn { code, name, stream } => self.randn(code, &name, stream).map(|()| Vec::new()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seat {
    Here { kernel: usize, code: usize },
    There { code: usize },
}

pub struct Pipe {
    pub kernels: Vec<Kernel>,
    pub there: Link,
    pub seats: Vec<Seat>,
}

impl Pipe {
    fn seat(&self, p: Prog) -> Result<Seat, String> {
        self.seats
            .get(p as usize)
            .copied()
            .ok_or_else(|| format!("program {} of {}", p as usize, self.seats.len()))
    }
}

impl Dev for Pipe {
    fn write(&mut self, p: Prog, name: &str, at: usize, bytes: &[u8]) -> Result<(), String> {
        match self.seat(p)? {
            Seat::Here { kernel, code } => self.kernels[kernel].write(code, name, at, bytes),
            Seat::There { code } => {
                self.there.ask(&Ask::Write { code, name: name.into(), at }, bytes).map(drop)
            }
        }
    }

    fn read(&mut self, p: Prog, name: &str, range: Range<usize>) -> Result<Vec<u8>, String> {
        match self.seat(p)? {
            Seat::Here { kernel, code } => self.kernels[kernel].read(code, name, range),
            Seat::There { code } => self
                .there
                .ask(&Ask::Read { code, name: name.into(), start: range.start, end: range.end }, &[]),
        }
    }

    fn launch(&mut self, p: Prog) -> Result<(), String> {
        match self.seat(p)? {
            Seat::Here { kernel, code } => self.kernels[kernel].launch(code),
            Seat::There { code } => self.there.ask(&Ask::Launch { code }, &[]).map(drop),
        }
    }

    fn finish(&mut self) -> Result<(), String> {
        self.kernels.iter().try_for_each(Kernel::finish)?;
        self.there.ask(&Ask::Finish, &[]).map(drop)
    }

    fn copy(&mut self, from: At<'_>, to: At<'_>, n: usize) -> Result<(), String> {
        let (Seat::Here { kernel: a, code: ca }, Seat::Here { kernel: b, code: cb }) =
            (self.seat(from.prog)?, self.seat(to.prog)?)
        else {
            return Err(format!("a copy of `{}` into `{}` across GPUs", from.name, to.name));
        };
        self.kernels[b].copy((cb, to.name, to.at), &self.kernels[a], (ca, from.name, from.at), n)
    }

    fn randn(&mut self, p: Prog, name: &str, stream: usize) -> Result<(), String> {
        match self.seat(p)? {
            Seat::Here { kernel, code } => self.kernels[kernel].randn(code, name, stream),
            Seat::There { code } => {
                self.there.ask(&Ask::Randn { code, name: name.into(), stream }, &[]).map(drop)
            }
        }
    }
}
