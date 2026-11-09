use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::cuda::{self, Attribute, Function, Launch, Limit, Module, Stream};
use super::ranks::Ranks;
use super::ring::{Ring, RingConfig};
use super::{Pinned, Running};
use crate::bundle::{
    Addr, Arg, Bundle, Fill, Home, Input, Kernel, Level, Param, Pool, PoolState, Record, Reference, Rope,
    Served, Unit,
};
use crate::weights::Checkpoint;

#[derive(Clone, Copy, Debug)]
pub struct LoaderConfig {
    pub constant_records: &'static str,
    pub block: u64,
    pub split_clusters: u32,
    pub slack: u64,
    pub read_chunk: usize,
    pub read_threads: usize,
}

impl Default for LoaderConfig {
    fn default() -> LoaderConfig {
        LoaderConfig {
            constant_records: "mega_homew",
            block: 256,
            split_clusters: 7,
            slack: 4 << 30,
            read_chunk: 256 << 20,
            read_threads: 16,
        }
    }
}

pub fn boot(
    bundle: &Bundle,
    ranks: &Ranks,
    weights: &Checkpoint,
    config: &LoaderConfig,
    ring: &RingConfig,
) -> Result<Running, String> {
    let r = &bundle.reference;
    let granule = cuda::granularity(false)? as u64;
    let Plan { pages, window_pages, holes } =
        plan(r, ranks.rank(), ranks.min(cuda::free_memory()?)?, granule, config)?;
    let loaded = load(bundle, ranks, Some(weights), &holes, config)?;
    let (requests, replies, ctl) =
        (Ring::requests(&r.served.rings, ring)?, Ring::replies(&r.served.rings, ring)?, Pinned::new(16)?);
    let rings: Vec<u8> = [requests.device(), replies.device()].iter().flat_map(|a| a.to_le_bytes()).collect();
    for (name, bytes) in
        [(&r.kernel.globals.rings, rings), (&r.kernel.globals.ctl, ctl.device().to_le_bytes().to_vec())]
    {
        if !loaded.put(name, &bytes)? {
            return Err(format!("the kernel has no global {name}"));
        }
    }
    ranks.barrier()?;
    loaded.launch(&r.kernel)?;
    println!("BOOT rank={} phase=launch", ranks.rank());
    Ok(Running { requests, replies, ctl, pages, window_pages })
}

pub struct Loaded {
    code: Code,
    arena: u64,
    homes: Vec<Place>,
    params: Vec<u64>,
    window: Option<Launch>,
    replay: u64,
    afters: Vec<(u64, u64, usize)>,
    done: cuda::Event,
}

impl Loaded {
    pub fn put(&self, name: &str, bytes: &[u8]) -> Result<bool, String> {
        put(self.code.entry, name, bytes)
    }

    pub fn launch(&self, k: &Kernel) -> Result<(), String> {
        self.code.launch(k, &self.params, self.window, self.replay)?;
        self.afters.iter().try_for_each(|&(to, from, n)| cuda::dtod_async(to, from, n, self.code.stream))
    }

    pub fn device(&self, home: usize) -> u64 {
        self.homes[home].device
    }

    pub fn finish(&self) -> Result<(), String> {
        cuda::sync(self.code.stream)
    }

    pub fn word(&self, at: u64, word: u64) -> Result<(), String> {
        cuda::memset32_async(self.arena + at, word as u32, 1, self.code.stream)?;
        cuda::memset32_async(self.arena + at + 4, (word >> 32) as u32, 1, self.code.stream)
    }

    pub fn zero(&self, home: usize, at: u64, len: usize) -> Result<(), String> {
        if !len.is_multiple_of(4) {
            return Err(format!("{len} B to zero in home {home}: not whole words"));
        }
        cuda::memset32_async(self.target(home)?.device + at, 0, len / 4, self.code.stream)
    }

    pub fn copy(
        &self,
        (home, at): (usize, u64),
        from: &Loaded,
        (other, start): (usize, u64),
        len: usize,
    ) -> Result<(), String> {
        from.finish()?;
        cuda::dtod_async(
            self.target(home)?.device + at,
            from.homes[other].device + start,
            len,
            self.code.stream,
        )
    }

    pub fn queued(
        &self,
        (home, at): (usize, u64),
        from: &Loaded,
        (other, start): (usize, u64),
        len: usize,
    ) -> Result<(), String> {
        if !std::ptr::eq(self, from) {
            cuda::after(self.code.stream, from.code.stream, &from.done)?;
        }
        cuda::dtod_async(
            self.target(home)?.device + at,
            from.homes[other].device + start,
            len,
            self.code.stream,
        )
    }

    pub fn after(&self, other: &Loaded) -> Result<(), String> {
        if std::ptr::eq(self, other) {
            return Ok(());
        }
        cuda::after(self.code.stream, other.code.stream, &other.done)
    }

    pub fn write_via(
        &self,
        home: usize,
        at: u64,
        bytes: &[u8],
        stage: &super::Pinned,
        from: usize,
    ) -> Result<(), String> {
        let to = self.target(home)?;
        if to.host.is_some() || from + bytes.len() > stage.len() {
            return self.write(home, at, bytes);
        }
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), stage.host().add(from), bytes.len()) };
        cuda::dtod_async(to.device + at, stage.device() + from as u64, bytes.len(), self.code.stream)
    }

    pub fn write(&self, home: usize, at: u64, bytes: &[u8]) -> Result<(), String> {
        let to = self.target(home)?;
        match to.host {
            Some(p) => {
                Ok(unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(at as usize), bytes.len()) })
            }
            None => cuda::htod(to.device + at, bytes),
        }
    }

    fn target(&self, home: usize) -> Result<Place, String> {
        match self.homes[home] {
            p if p.shared => Err(format!("a write into home {home}, shared by every rank")),
            p => Ok(p),
        }
    }

    pub fn read(&self, home: usize, at: u64, len: usize) -> Result<Vec<u8>, String> {
        let mut out = vec![0; len];
        match self.homes[home].host {
            Some(p) => unsafe { std::ptr::copy_nonoverlapping(p.add(at as usize), out.as_mut_ptr(), len) },
            None => cuda::dtoh(&mut out, self.homes[home].device + at)?,
        }
        Ok(out)
    }
}

pub fn load<S: Served>(
    bundle: &Bundle<S>,
    ranks: &Ranks,
    weights: Option<&Checkpoint>,
    holes: &[Range<u64>],
    config: &LoaderConfig,
) -> Result<Loaded, String> {
    let (r, rank) = (&bundle.reference, ranks.rank());
    let k = &r.kernel;
    if bundle.inputs.rank != rank || r.served.ranks() != ranks.world() {
        return Err(format!(
            "rank {rank} of {} booting rank {}'s inputs of a {}-rank bundle",
            ranks.world(),
            bundle.inputs.rank,
            r.served.ranks()
        ));
    }
    let mut phase = Phase { rank, at: Instant::now() };

    let code = Code::load(bundle, rank, config.split_clusters)?;
    phase.done("modules");

    let granule = cuda::granularity(false)? as u64;
    let arena = arena(k.workspace.next_multiple_of(granule), holes, granule)?;
    let window = l2_window(k, arena)?;
    phase.done("arena");

    let shared = SharedHome::agree(bundle, weights, ranks, config)?;
    let mut homes = Vec::with_capacity(r.homes.len());
    for (h, home) in r.homes.iter().enumerate() {
        homes.push(place(home, shared.get(&h).map(|s| (s.host, s.device)), arena, ranks.symmetric())?);
    }
    let addrs: Vec<u64> = homes.iter().map(|p| p.device).collect();
    phase.done("homes");

    let mut maps = Vec::new();
    for rec in &r.image {
        apply(bundle, rec, arena, &addrs, &mut maps)?;
    }
    let tables = cuda::alloc(maps.len() + 8)?;
    cuda::htod(tables, &maps)?;
    let replay = tables + maps.len() as u64;
    phase.done("image");

    let staged: BTreeSet<usize> = (0..r.homes.len()).filter(|&h| r.homes[h].staged).collect();
    let unstaged = |hs: &[usize]| !hs.iter().any(|h| staged.contains(h));
    cuda::sync_all()?;
    code.loads(&r.image, arena, &addrs, config.block, unstaged)?;
    phase.done("loads");

    if let Some(w) = k.constant_records.as_ref().filter(|w| w.end > w.start) {
        let name = config.constant_records;
        let (at, n) = cuda::global(code.entry, name)?
            .ok_or_else(|| format!("constant record bytes and no global {name}"))?;
        if n as u64 > w.end - w.start {
            return Err(format!(
                "global {name}'s {n} bytes past the {} constant record bytes",
                w.end - w.start
            ));
        }
        cuda::dtod(at, arena + w.start, n)?;
    }
    let peers = ranks.peers(k.globals.peer_bytes as usize)?;
    for &m in &code.modules {
        put(m, &k.globals.fabric, &ranks.fabric_words(k.globals.multicast))?;
        put(m, &k.globals.peers, &peers)?;
    }

    let mut written: BTreeSet<usize> = (0..homes.len()).filter(|&h| homes[h].shared).collect();
    for i in bundle.inputs.inputs.iter().filter(|i| !staged.contains(&i.home) && !homes[i.home].shared) {
        if fill(bundle, weights, i, homes[i.home], len(&r.homes[i.home]), config)? {
            written.insert(i.home);
        }
    }
    code.loads(&r.image, arena, &addrs, config.block, |hs| {
        hs.iter().any(|h| written.contains(h)) && unstaged(hs)
    })?;
    let most = staged.iter().map(|&h| len(&r.homes[h])).max().unwrap_or(0);
    if most > 0 {
        println!("FREE rank={rank} before_staging={} staging={most}", cuda::free_memory()?);
        let staging = cuda::alloc(most as usize)?;
        for &h in &staged {
            cuda::memset8(staging, 0, most as usize)?;
            for i in bundle.inputs.inputs.iter().filter(|i| i.home == h) {
                fill(
                    bundle,
                    weights,
                    i,
                    Place { device: staging, host: None, shared: false },
                    len(&r.homes[h]),
                    config,
                )?;
            }
            let mut at = addrs.clone();
            at[h] = staging;
            code.loads(&r.image, arena, &at, config.block, |hs| hs.contains(&h))?;
        }
    }
    cuda::sync_all()?;
    ranks.barrier()?;
    phase.done("inputs");

    let params: Vec<u64> = k
        .params
        .iter()
        .map(|p| match *p {
            Param::Arena => arena,
            Param::Maps => tables,
            Param::Replay => replay,
            Param::Symmetric => ranks.symmetric(),
            Param::Progress => 0,
            Param::Home(h) => addrs[h],
        })
        .collect();
    let afters = r
        .image
        .iter()
        .filter_map(|x| match x {
            Record::After { from, to } => Some((
                addrs[to.0] + to.1.start,
                addrs[from.0] + from.1.start,
                (from.1.end - from.1.start) as usize,
            )),
            _ => None,
        })
        .collect();
    let done = cuda::event(code.stream)?;
    Ok(Loaded { code, arena, homes, params, window, replay, afters, done })
}

struct Phase {
    rank: usize,
    at: Instant,
}

impl Phase {
    fn done(&mut self, name: &str) {
        println!("BOOT rank={} phase={name} s={:.2}", self.rank, self.at.elapsed().as_secs_f64());
        self.at = Instant::now();
    }
}

struct Code {
    stream: Stream,
    entry: Module,
    modules: Vec<Module>,
    entries: Vec<(Function, u32)>,
    next: Option<Function>,
    loaders: Vec<Function>,
    split_clusters: u32,
}

impl Code {
    fn load<S: Served>(bundle: &Bundle<S>, rank: usize, split_clusters: u32) -> Result<Code, String> {
        let k = &bundle.reference.kernel;
        cuda::init(rank)?;
        let stream = cuda::stream()?;
        let (_, fault) = cuda::host_alloc(8)?;
        cuda::set_limit(
            Limit::CU_LIMIT_STACK_SIZE,
            cuda::limit(Limit::CU_LIMIT_STACK_SIZE)?.max(k.stack as usize),
        )?;
        let entry = cuda::module(&bundle.cubin("entry.cubin")?)?;
        let entries: Vec<(Function, u32)> = k
            .entries
            .iter()
            .map(|(name, grid)| Ok((cuda::function(entry, name)?, *grid)))
            .collect::<Result<_, String>>()?;
        let mut modules = vec![entry];
        let next = match &k.next {
            Some(name) => {
                let m = cuda::module(&bundle.cubin("next.cubin")?)?;
                modules.push(m);
                Some(cuda::function(m, name)?)
            }
            None => None,
        };
        let mut files: Vec<(&str, Module)> = Vec::new();
        let mut loaders = Vec::new();
        for (file, name) in &k.loaders {
            let m = match files.iter().find(|(f, _)| f == file) {
                Some(&(_, m)) => m,
                None => {
                    let m = cuda::module(&bundle.cubin(file)?)?;
                    files.push((file, m));
                    m
                }
            };
            loaders.push(cuda::function(m, name)?);
        }
        modules.extend(files.iter().map(|&(_, m)| m));
        for &m in &modules {
            put(m, &k.globals.fault, &fault.to_le_bytes())?;
        }
        for &(f, _) in &entries {
            let fixed = cuda::get_attribute(f, Attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
            if fixed as u32 > k.static_smem {
                return Err(format!(
                    "an entry of {fixed} static shared bytes, the bundle's {}",
                    k.static_smem
                ));
            }
            cuda::set_attribute(
                f,
                Attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                k.smem as i32,
            )?;
            if split(k) {
                cuda::set_attribute(f, Attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)?;
            }
        }
        Ok(Code { stream, entry, modules, entries, next, loaders, split_clusters })
    }

    fn loads(
        &self,
        image: &[Record],
        arena: u64,
        homes: &[u64],
        block: u64,
        pick: impl Fn(&[usize]) -> bool,
    ) -> Result<(), String> {
        for rec in image {
            let Record::Load { loader, threads, args } = rec else { continue };
            let named: Vec<usize> = args
                .iter()
                .filter_map(|a| {
                    if let Arg::Addr(Addr::Home(h, _)) | Arg::Aim(Addr::Home(h, _)) = a {
                        Some(*h)
                    } else {
                        None
                    }
                })
                .collect();
            if *threads == 0 || !pick(&named) {
                continue;
            }
            let words: Vec<u64> = args
                .iter()
                .map(|a| match *a {
                    Arg::Word(w) => w,
                    Arg::Addr(x) | Arg::Aim(x) => resolve(x, arena, homes),
                })
                .collect();
            let f = *self
                .loaders
                .get(*loader)
                .ok_or_else(|| format!("a load of loader {loader}, the bundle's {}", self.loaders.len()))?;
            cuda::launch_packed(f, threads.div_ceil(block) as u32, block as u32, self.stream, &words)?;
        }
        cuda::sync(self.stream)
    }

    fn launch(&self, k: &Kernel, params: &[u64], window: Option<Launch>, replay: u64) -> Result<(), String> {
        let with = |cluster: u32, more: Option<Launch>| -> Vec<Launch> {
            [(cluster > 1).then_some(Launch::Cluster(cluster)), window, more].into_iter().flatten().collect()
        };
        match (split(k), &self.entries[..]) {
            (true, &[(f, grid)]) => {
                let clustered = self.split_clusters * k.cluster;
                if grid <= clustered || !(grid - clustered).is_multiple_of(2) {
                    return Err(format!(
                        "a split launch of {grid} blocks: not {clustered} in clusters of {} and pairs past them",
                        k.cluster
                    ));
                }
                cuda::launch(f, clustered, k.threads, k.smem, self.stream, params, &with(k.cluster, None))?;
                cuda::launch(
                    f,
                    grid - clustered,
                    k.threads,
                    k.smem,
                    self.stream,
                    params,
                    &with(2, Some(Launch::Dependent)),
                )?;
            }
            (true, _) => {
                return Err(format!(
                    "a split launch (cluster {}) of {} entries",
                    k.cluster,
                    self.entries.len()
                ));
            }
            (false, _) => {
                let cooperative = k.cooperative.then_some(Launch::Cooperative);
                for &(f, grid) in &self.entries {
                    cuda::launch(
                        f,
                        grid,
                        k.threads,
                        k.smem,
                        self.stream,
                        params,
                        &with(k.cluster, cooperative),
                    )?;
                }
            }
        }
        if let Some(f) = self.next {
            cuda::launch(f, 1, 1, 0, self.stream, &[replay], &[])?;
        }
        Ok(())
    }
}

fn split(k: &Kernel) -> bool {
    k.cluster > 2
}

#[derive(Debug, PartialEq)]
pub struct Plan {
    pub pages: usize,
    pub window_pages: usize,
    pub holes: Vec<Range<u64>>,
}

pub fn plan(
    r: &Reference,
    rank: usize,
    free: u64,
    granule: u64,
    config: &LoaderConfig,
) -> Result<Plan, String> {
    let (other, slack) = (placed(r, granule)?, config.slack);
    let lanes = *r.served.model.batches.last().ok_or("a bundle of no decode widths")?;
    let bytes = free.checked_sub(other + slack).ok_or_else(|| {
        format!("the kernel's {other} B and {slack} B of slack past the GPU's {free} B free")
    })?;
    let (pages, window_pages) = pages_within(&r.served.pool, bytes).ok_or_else(|| {
        format!("{bytes} B of pools hold no sequence of {} positions", r.served.model.context)
    })?;
    let slots = lanes + 1;
    if slots > r.served.pool.slots {
        return Err(format!("{lanes} lanes past the bundle's {} slots", r.served.pool.slots));
    }
    let used: u64 = r.served.pool.states.iter().map(|s| state_bytes(s, pages, window_pages)).sum();
    println!(
        "POOL rank={rank} free={free} other={other} pools={used} pages={pages} window_pages={window_pages} slots={slots} (bundle {} {} {})",
        r.served.pool.pages, r.served.pool.window_pages, r.served.pool.slots
    );
    Ok(Plan { pages, window_pages, holes: holes(r, pages, window_pages)? })
}

fn state_bytes(s: &PoolState, pages: usize, window_pages: usize) -> u64 {
    s.bytes * if s.unit == Unit::Page { pages } else { window_pages } as u64
}

fn arena(span: u64, holes: &[Range<u64>], granule: u64) -> Result<u64, String> {
    let arena = cuda::reserve(span as usize, granule as usize)?;
    for s in segments(span, holes, granule) {
        let (at, n) = (arena + s.start, (s.end - s.start) as usize);
        cuda::map(at, n, cuda::create(n, false)?)?;
        cuda::memset8(at, 0, n)?;
    }
    cuda::sync_all()?;
    Ok(arena)
}

fn l2_window(k: &Kernel, arena: u64) -> Result<Option<Launch>, String> {
    match &k.l2 {
        Some(l2) if l2.end > l2.start => {
            cuda::set_limit(Limit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, (l2.end - l2.start) as usize)?;
            Ok(Some(Launch::Window(arena + l2.start, (l2.end - l2.start) as usize)))
        }
        _ => Ok(None),
    }
}

#[derive(Clone, Copy)]
struct Place {
    device: u64,
    host: Option<*mut u8>,
    shared: bool,
}

fn place(h: &Home, shared: Option<(*mut u8, u64)>, arena: u64, symmetric: u64) -> Result<Place, String> {
    let device = |device| Ok(Place { device, host: None, shared: false });
    let host = |(host, device), shared| Ok(Place { device, host: Some(host), shared });
    match (h.staged, h.level, shared) {
        (true, _, _) => device(0),
        (false, Level::Hbm, _) => device(arena + h.at.start),
        (false, Level::Peer, _) => device(symmetric + h.at.start),
        (false, Level::Host, None) => host(host_home(len(h))?, false),
        (false, Level::Host, Some(at)) => host(at, true),
    }
}

fn file_home<S: Served>(
    bundle: &Bundle<S>,
    weights: Option<&Checkpoint>,
    h: usize,
    home: &Home,
) -> Result<Option<(PathBuf, Range<u64>)>, String> {
    let mut on = bundle.inputs.inputs.iter().filter(|i| i.home == h);
    let (Level::Host, false, Some(i), None) = (home.level, home.staged, on.next(), on.next()) else {
        return Ok(None);
    };
    if !matches!(i.fill, Fill::File { .. }) || i.at != (0..len(home)) {
        return Ok(None);
    }
    match source(bundle, weights, &i.fill)? {
        Source::File(path, span) if span.end - span.start == len(home) => Ok(Some((path, span))),
        _ => Ok(None),
    }
}

fn shareable<S: Served>(r: &Reference<S>, h: usize) -> bool {
    let named = |a: &Arg| matches!(a, Arg::Addr(Addr::Home(x, _)) | Arg::Aim(Addr::Home(x, _)) if *x == h);
    let handed = r.kernel.params.iter().any(|p| matches!(p, Param::Home(x) if *x == h));
    let written = r.image.iter().any(|x| match x {
        Record::Init { home, .. } | Record::Map { home, .. } => *home == h,
        Record::After { to, .. } => to.0 == h,
        Record::Load { args, .. } => args.iter().any(named),
        Record::Write { .. } | Record::Patch { .. } => false,
    });
    !handed && !written
}

fn apply<S: Served>(
    bundle: &Bundle<S>,
    rec: &Record,
    arena: u64,
    homes: &[u64],
    maps: &mut Vec<u8>,
) -> Result<(), String> {
    match rec {
        Record::Write { at, blob } => cuda::htod(arena + at, &bundle.blob(blob)?),
        Record::Patch { at, to } => cuda::htod(arena + at, &resolve(*to, arena, homes).to_le_bytes()),
        Record::Init { home, word } => {
            cuda::memset32(homes[*home], *word, (len(&bundle.reference.homes[*home]) / 4) as usize)
        }
        Record::Map { home, at, place, elem, dims, strides, tile, swizzle, corners } => {
            let map = cuda::tensor_map(
                *elem,
                homes[*home] + at,
                dims,
                strides,
                tile,
                *swizzle,
                corners.as_ref().map(|c| (&c.lower[..], &c.upper[..])),
            )?;
            match place {
                Some(p) => cuda::htod(homes[*home] + p, &map),
                None => {
                    maps.extend_from_slice(&map);
                    Ok(())
                }
            }
        }
        Record::After { from, to } if from.1.end - from.1.start != to.1.end - to.1.start => {
            Err(format!("an after-replay copy of {from:?} into {to:?}"))
        }
        Record::Load { .. } | Record::After { .. } => Ok(()),
    }
}

fn len(h: &Home) -> u64 {
    h.at.end - h.at.start
}

pub fn resolve(a: Addr, arena: u64, homes: &[u64]) -> u64 {
    match a {
        Addr::Home(h, add) => homes[h] + add,
        Addr::Arena(add) => arena + add,
    }
}

fn put(m: Module, name: &str, bytes: &[u8]) -> Result<bool, String> {
    match cuda::global(m, name)? {
        Some((at, n)) if bytes.len() <= n => cuda::htod(at, bytes).map(|()| true),
        Some((_, n)) => Err(format!("{} bytes into global {name} of {n}", bytes.len())),
        None => Ok(false),
    }
}

fn host_home(bytes: u64) -> Result<(*mut u8, u64), String> {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    let whole = bytes.max(1).next_multiple_of(page) as usize;
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            whole,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(format!("mmap of a {bytes} B host home: {}", std::io::Error::last_os_error()));
    }
    Ok((p.cast(), unsafe { cuda::host_register(p.cast(), whole)? }))
}

struct SharedHome {
    fd: OwnedFd,
    host: *mut u8,
    device: u64,
}

impl SharedHome {
    fn agree<S: Served>(
        bundle: &Bundle<S>,
        weights: Option<&Checkpoint>,
        ranks: &Ranks,
        config: &LoaderConfig,
    ) -> Result<BTreeMap<usize, SharedHome>, String> {
        let r = &bundle.reference;
        let mut wanted = BTreeMap::new();
        for (h, home) in r.homes.iter().enumerate() {
            if let Some((path, span)) = file_home(bundle, weights, h, home)?.filter(|_| shareable(r, h)) {
                wanted.insert(format!("{} {span:?}", path.display()), (h, path, span));
            }
        }
        let made = match ranks.rank() {
            0 => Some(
                wanted
                    .values()
                    .map(|(_, path, span)| SharedHome::fill(path, span, config))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            _ => None,
        };
        let handed = made
            .as_ref()
            .map(|m| {
                Ok::<_, String>((
                    wanted.keys().cloned().collect(),
                    m.iter()
                        .map(|s| s.fd.try_clone().map_err(|e| e.to_string()))
                        .collect::<Result<_, _>>()?,
                ))
            })
            .transpose()?;
        let (keys, fds) = ranks.hand_out(handed)?;
        let mut shared = BTreeMap::new();
        match made {
            Some(made) => shared.extend(wanted.values().map(|w| w.0).zip(made)),
            None => {
                for (key, fd) in keys.iter().zip(fds) {
                    if let Some((h, _, span)) = wanted.get(key) {
                        shared.insert(*h, SharedHome::map(fd, span.end - span.start)?);
                    }
                }
            }
        }
        if shared.len() < wanted.len() {
            println!(
                "BOOT rank={} private_homes={}",
                ranks.rank(),
                wanted
                    .values()
                    .filter(|w| !shared.contains_key(&w.0))
                    .map(|w| w.0.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        Ok(shared)
    }

    fn fill(path: &Path, span: &Range<u64>, config: &LoaderConfig) -> Result<SharedHome, String> {
        let len = span.end - span.start;
        let fd = unsafe { libc::memfd_create(c"hyperserve-home".as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(format!("memfd_create: {}", std::io::Error::last_os_error()));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        std::fs::File::from(fd.try_clone().map_err(|e| e.to_string())?)
            .set_len(SharedHome::whole(len) as u64)
            .map_err(|e| format!("a {len} B shared home: {e}"))?;
        let home = SharedHome::map(fd, len)?;
        read(path, span.clone(), unsafe { std::slice::from_raw_parts_mut(home.host, len as usize) }, config)?;
        Ok(home)
    }

    fn map(fd: OwnedFd, len: u64) -> Result<SharedHome, String> {
        let whole = SharedHome::whole(len);
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                whole,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(format!("mmap of a {len} B shared home: {}", std::io::Error::last_os_error()));
        }
        let device = unsafe { cuda::host_register(p.cast(), whole)? };
        Ok(SharedHome { fd, host: p.cast(), device })
    }

    fn whole(len: u64) -> usize {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        len.max(1).next_multiple_of(page) as usize
    }
}

enum Source {
    Zeros,
    Bytes(Vec<u8>),
    File(PathBuf, Range<u64>),
}

fn source<S: Served>(
    bundle: &Bundle<S>,
    weights: Option<&Checkpoint>,
    fill: &Fill,
) -> Result<Source, String> {
    Ok(match fill {
        Fill::Zeros => Source::Zeros,
        Fill::Blob { blob } => Source::Bytes(bundle.blob(blob)?),
        Fill::Rope(r) => Source::Bytes(rope(r)),
        Fill::Checkpoint { transform, tensors, cut, laid } => {
            Source::Bytes(checkpoint(weights)?.laid(transform, tensors, cut.as_ref(), laid)?)
        }
        Fill::File { tensor } => {
            let (path, span) = checkpoint(weights)?.span(tensor)?;
            Source::File(path, span)
        }
    })
}

fn checkpoint(weights: Option<&Checkpoint>) -> Result<&Checkpoint, String> {
    weights.ok_or_else(|| "a checkpoint fill and no weights".to_string())
}

fn rope(r: &Rope) -> Vec<u8> {
    let dim = r.dim as usize;
    let mut freqs: Vec<f32> = (0..dim / 2).map(|i| 1.0 / r.base.powf((2 * i) as f32 / dim as f32)).collect();
    if r.original_seq_len > 0 {
        let corrected = |rotations: f32| {
            dim as f32 * (r.original_seq_len as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * r.base.ln())
        };
        let low = corrected(r.beta_fast).floor().max(0.0);
        let high = corrected(r.beta_slow).ceil().min(dim as f32 - 1.0);
        for (i, f) in freqs.iter_mut().enumerate() {
            let smooth = 1.0 - ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            *f = *f / r.factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut out = Vec::with_capacity(r.rows as usize * dim * 4);
    for p in 0..r.rows {
        let angles: Vec<(f32, f32)> = freqs.iter().map(|f| (p as f32 * f).sin_cos()).collect();
        out.extend(angles.iter().flat_map(|&(_, c)| c.to_le_bytes()));
        out.extend(angles.iter().flat_map(|&(s, _)| s.to_le_bytes()));
    }
    out
}

fn fill<S: Served>(
    bundle: &Bundle<S>,
    weights: Option<&Checkpoint>,
    i: &Input,
    home: Place,
    home_len: u64,
    config: &LoaderConfig,
) -> Result<bool, String> {
    let n = (i.at.end - i.at.start) as usize;
    if i.at.end > home_len {
        return Err(format!("input {}: bytes {:?} past its home's {home_len}", i.name, i.at));
    }
    let source = source(bundle, weights, &i.fill)?;
    let got = match &source {
        Source::Zeros => return Ok(false),
        Source::Bytes(b) => b.len(),
        Source::File(_, span) => (span.end - span.start) as usize,
    };
    if got != n {
        return Err(format!("input {}: {got} B filled, its {n}", i.name));
    }
    let host = home.host.map(|p| unsafe { std::slice::from_raw_parts_mut(p.add(i.at.start as usize), n) });
    match (source, host) {
        (Source::Bytes(b), Some(out)) => out.copy_from_slice(&b),
        (Source::Bytes(b), None) => cuda::htod(home.device + i.at.start, &b)?,
        (Source::File(path, span), Some(out)) => read(&path, span, out, config)?,
        (Source::File(path, span), None) => {
            let mut b = vec![0; n];
            read(&path, span, &mut b, config)?;
            cuda::htod(home.device + i.at.start, &b)?;
        }
        (Source::Zeros, _) => unreachable!("zeros write nothing"),
    }
    Ok(true)
}

fn read(path: &Path, span: Range<u64>, out: &mut [u8], config: &LoaderConfig) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    let (chunk, threads) = (config.read_chunk, config.read_threads);
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut parts: Vec<Vec<(u64, &mut [u8])>> = (0..threads).map(|_| Vec::new()).collect();
    for (c, part) in out.chunks_mut(chunk).enumerate() {
        parts[c % threads].push((span.start + (c * chunk) as u64, part));
    }
    std::thread::scope(|s| {
        let jobs: Vec<_> = parts
            .into_iter()
            .map(|part| s.spawn(|| part.into_iter().try_for_each(|(at, chunk)| f.read_exact_at(chunk, at))))
            .collect();
        jobs.into_iter()
            .try_for_each(|j| j.join().unwrap_or_else(|_| Err(std::io::Error::other("a reader panicked"))))
    })
    .map_err(|e| format!("{}: {e}", path.display()))
}

pub fn pages_within(pool: &Pool, bytes: u64) -> Option<(usize, usize)> {
    let cost = |u: Unit| pool.states.iter().filter(|s| s.unit == u).map(|s| s.bytes).sum::<u64>();
    let rest = bytes.checked_sub(pool.window_pages as u64 * cost(Unit::Window))?;
    let pages = (rest / cost(Unit::Page).max(1)).min(pool.pages as u64) as usize;
    (pages > pool.entries).then_some((pages, pool.window_pages))
}

fn placed(r: &Reference, granule: u64) -> Result<u64, String> {
    let whole = |b: u64| b.next_multiple_of(granule);
    let pools: BTreeSet<usize> = r.served.pool.states.iter().map(|s| s.home).collect();
    let declared: u64 = pools.iter().map(|&h| r.homes.get(h).map_or(0, len)).sum();
    let arena = r.kernel.workspace.checked_sub(declared).ok_or("pool homes past the arena")? + granule;
    let staging = r.homes.iter().filter(|h| h.staged).map(len).max().unwrap_or(0);
    let maps = r.image.iter().filter(|x| matches!(x, Record::Map { place: None, .. })).count() as u64;
    Ok(arena
        + 2 * granule * pools.len() as u64
        + whole(staging)
        + whole(size_of::<cuda::Map>() as u64 * maps + 8))
}

fn holes(r: &Reference, pages: usize, window_pages: usize) -> Result<Vec<Range<u64>>, String> {
    let homes: BTreeSet<usize> = r.served.pool.states.iter().map(|s| s.home).collect();
    homes
        .into_iter()
        .map(|h| {
            let home = r
                .homes
                .get(h)
                .ok_or_else(|| format!("a pool in home {h}, the bundle's {}", r.homes.len()))?;
            let real: u64 = r
                .served
                .pool
                .states
                .iter()
                .filter(|s| s.home == h)
                .map(|s| state_bytes(s, pages, window_pages))
                .sum();
            if home.level != Level::Hbm || home.staged || real > len(home) {
                return Err(format!(
                    "pool home {h}: {:?}, staged {}, {real} B in its {}",
                    home.level,
                    home.staged,
                    len(home)
                ));
            }
            let tail = home.at.start + real..home.at.end;
            let over = |a: u64, n: u64| a < tail.end && tail.start < a + n;
            let reaches = r.image.iter().any(|x| match x {
                Record::Write { at, blob } => over(*at, blob.len),
                Record::Patch { at, .. } => over(*at, 8),
                Record::Init { home: m, .. } => *m == h,
                Record::Map { home: m, place, .. } => {
                    *m == h && place.is_some_and(|p| over(home.at.start + p, size_of::<cuda::Map>() as u64))
                }
                Record::Load { args, .. } => {
                    args.iter().any(|a| matches!(a, Arg::Addr(Addr::Home(x, _)) if *x == h))
                }
                Record::After { from, to } => from.0 == h || to.0 == h,
            });
            match reaches {
                true => Err(format!("the image reaches pool home {h}'s tail past its {real} B")),
                false => Ok(tail),
            }
        })
        .collect()
}

pub fn segments(span: u64, holes: &[Range<u64>], granule: u64) -> Vec<Range<u64>> {
    let mut cut: Vec<Range<u64>> = holes
        .iter()
        .map(|h| h.start.next_multiple_of(granule)..h.end / granule * granule)
        .filter(|h| h.start < h.end)
        .collect();
    cut.sort_by_key(|h| h.start);
    let mut out = Vec::new();
    let mut at = 0;
    for h in cut {
        if h.start > at {
            out.push(at..h.start);
        }
        at = at.max(h.end);
    }
    if at < span {
        out.push(at..span);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(page: u64, window: u64) -> Pool {
        Pool {
            page_rows: 256,
            pages: 1 << 20,
            window_pages: 1 << 20,
            slots: 9,
            window_keep: 0,
            entries: 5,
            window: 3,
            states: vec![
                PoolState { home: 0, unit: Unit::Page, bytes: page },
                PoolState { home: 1, unit: Unit::Window, bytes: window },
            ],
        }
    }

    #[test]
    fn pools_take_the_bundles_window_pages_and_the_rest_in_pages() {
        let p = Pool { window_pages: 10, pages: 1000, ..pool(100, 10) };
        assert_eq!(pages_within(&p, 10 * 10 + 520 * 100 + 99), Some((520, 10)));
        assert_eq!(pages_within(&p, 10 * 10 + 5 * 100), None, "no more than one sequence's pages");
        assert_eq!(pages_within(&p, 10 * 10 - 1), None, "no room for the window pages");
        assert_eq!(pages_within(&Pool { pages: 7, ..p }, 1 << 40), Some((7, 10)), "at most the bundle's");
    }

    #[test]
    fn segments_skip_whole_granules_of_holes() {
        let g = 10;
        assert_eq!(segments(100, &[], g), vec![0..100]);
        assert_eq!(segments(100, &[25..75], g), vec![0..30, 70..100]);
        assert_eq!(segments(100, &[61..69], g), vec![0..100], "no whole granule");
        assert_eq!(segments(100, &[70..100, 0..20], g), vec![20..70]);
    }

    fn rope_of(json: &serde_json::Value) -> Vec<u8> {
        match serde_json::from_value(json.clone()) {
            Ok(Fill::Rope(r)) => rope(&r),
            other => panic!("{json}: not a rope fill: {other:?}"),
        }
    }

    #[test]
    fn rope_tables_match_the_exporters() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rope.json");
        let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for t in fixture["tables"].as_array().unwrap() {
            let table = rope_of(&t["fill"]);
            assert_eq!(table.len(), 1_048_582 * 64 * 4);
            for w in t["windows"].as_array().unwrap() {
                let hex = w["bytes"].as_str().unwrap();
                let want: Vec<u8> = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .collect();
                let at = w["row"].as_u64().unwrap() as usize * 64 * 4;
                assert!(table[at..at + want.len()] == want[..], "{} at row {}", t["name"], w["row"]);
            }
        }
    }

    #[derive(serde::Deserialize)]
    struct Bare {}

    impl Served for Bare {
        fn ranks(&self) -> usize {
            1
        }
    }

    fn reference(params: serde_json::Value, image: serde_json::Value) -> Reference<Bare> {
        let json = serde_json::json!({
            "format": "hyperserve-bundle-2", "point": "test",
            "kernel": { "entries": [["k", 1]], "threads": 256, "cluster": 1, "smem": 0, "static_smem": 0, "cooperative": false, "stack": 0, "workspace": 0, "symmetric": 0,
                "l2": null, "homew": null, "params": params, "next": null, "loaders": [],
                "globals": { "fault": "f", "fabric": "g", "peers": "p", "peer_bytes": 0, "multicast": false } },
            "homes": [{ "level": "host", "at": { "start": 0, "end": 4096 }, "staged": false }, { "level": "hbm", "at": { "start": 0, "end": 64 }, "staged": false }],
            "image": image,
        });
        serde_json::from_value(json).expect("a reference")
    }

    #[test]
    fn host_homes_are_shared_when_reached_only_by_a_patch() {
        let patch = serde_json::json!({ "patch": { "at": 0, "to": { "home": [0, 0] } } });
        assert!(shareable(&reference(serde_json::json!(["arena"]), serde_json::json!([patch])), 0));
        let writers = [
            serde_json::json!({ "init": { "home": 0, "word": 0 } }),
            serde_json::json!({ "map": { "home": 0, "at": 0, "place": null, "elem": 0, "dims": [16], "strides": [], "tile": [16], "swizzle": 0 } }),
            serde_json::json!({ "map": { "home": 1, "at": 0, "place": 0, "elem": 0, "dims": [16], "strides": [], "tile": [16], "swizzle": 0 } }),
            serde_json::json!({ "after": { "from": [0, { "start": 0, "end": 8 }], "to": [1, { "start": 0, "end": 8 }] } }),
            serde_json::json!({ "after": { "from": [1, { "start": 0, "end": 8 }], "to": [0, { "start": 0, "end": 8 }] } }),
            serde_json::json!({ "load": { "loader": 0, "threads": 1, "args": [{ "aim": { "home": [0, 0] } }] } }),
        ];
        let verdicts: Vec<bool> = writers
            .into_iter()
            .map(|w| shareable(&reference(serde_json::json!(["arena"]), serde_json::json!([patch, w])), 0))
            .collect();
        assert_eq!(
            verdicts,
            [false, false, true, true, false, false],
            "the other home's map and a copy out of it leave it unwritten"
        );
        assert!(
            !shareable(
                &reference(serde_json::json!(["arena", { "home": 0 }]), serde_json::json!([patch])),
                0
            ),
            "a kernel param"
        );
    }

    #[test]
    fn addresses_resolve_to_homes_or_the_arena() {
        let homes = [0x1000, 0, 0x9000];
        assert_eq!(resolve(Addr::Home(2, 8), 0x100, &homes), 0x9008);
        assert_eq!(resolve(Addr::Arena(8), 0x100, &homes), 0x108);
    }
}
