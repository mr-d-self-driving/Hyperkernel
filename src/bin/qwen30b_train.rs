use std::collections::HashMap;
use std::fs::{self, File};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use hyperserve::bundle;
use hyperserve::gpu::Pinned;
use hyperserve::gpu::cuda::{self, Attribute, Launch, Module};
use hyperserve::gpu::ranks::{self, Ranks, RanksConfig};
use hyperserve::weights::Checkpoint;
use serde::Deserialize;

struct TrainConfig {
    format: &'static str,
    chunk: usize,
    device_limit_ns: u64,
    spin: Duration,
    exit_hang: u8,
    ranks: RanksConfig,
}

impl Default for TrainConfig {
    fn default() -> TrainConfig {
        TrainConfig {
            format: "qwen30b-train-1",
            chunk: 256 << 20,
            device_limit_ns: 20_000_000_000,
            spin: Duration::from_millis(2),
            exit_hang: 3,
            ranks: RanksConfig { margin: 0, ..RanksConfig::default() },
        }
    }
}

#[derive(Parser, Clone)]
struct Flags {
    #[arg(long)]
    bundle: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokens: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    cubin: Option<PathBuf>,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    #[arg(long, default_value_t = 60, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    steps: usize,
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    #[arg(long, default_value_t = 120)]
    hang_seconds: u64,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    dry: bool,
    #[arg(long)]
    unpack: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Reference {
    format: String,
    model: Model,
    peak_flops_per_gpu: f64,
    kernel: Kernel,
    group_bytes: u64,
    data: Region,
    control: Region,
    maps_bytes: u64,
    host_record_bytes: u64,
    words: Words,
    maps: Vec<Map>,
    zeroed: Vec<[u64; 2]>,
    outputs: Vec<Output>,
    tensors: Vec<TensorFact>,
}

#[derive(Deserialize)]
struct Model {
    name: String,
    ranks: usize,
    batch: u64,
    seq: u64,
    flops_per_token: f64,
}

#[derive(Deserialize)]
struct Kernel {
    entry: String,
    grid: u32,
    threads: u32,
    smem: u32,
    static_smem: u32,
    cooperative: bool,
    cluster: u32,
    params: Vec<String>,
    group_words: String,
    group_table: String,
    multicast: bool,
}

#[derive(Deserialize)]
struct Region {
    at: [u64; 2],
}

#[derive(Deserialize)]
struct Words {
    go: usize,
    stop: usize,
    timeout_ns: usize,
    bounds: usize,
    address: usize,
    watchdog: usize,
    host_record: usize,
    range: usize,
    head: usize,
    all: usize,
}

#[derive(Deserialize)]
struct Map {
    at: u64,
    elem: String,
    dims: Vec<u64>,
    strides: Vec<u64>,
    #[serde(rename = "box")]
    extent: Vec<u32>,
    swizzle: u32,
}

#[derive(Deserialize)]
struct Output {
    name: String,
    at: u64,
    dtype: String,
}

#[derive(Deserialize)]
struct TensorFact {
    name: String,
    dtype: String,
    shape: Vec<u64>,
}

#[derive(Deserialize)]
struct RankFile {
    rank: usize,
    fills: Vec<Fill>,
    inputs: Vec<Input>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum Fill {
    Repeat { at: u64, elem: Vec<u8>, n: u64 },
    Blob { at: u64, blob: [u64; 2] },
    Checkpoint { reads: Vec<[u64; 3]>, targets: Vec<Target> },
}

#[derive(Deserialize)]
struct Target {
    at: u64,
    chain: Vec<String>,
    store: String,
    elems: u64,
    fnv: String,
}

#[derive(Deserialize)]
struct Input {
    at: u64,
    columns: u64,
    rows_of_batch: u64,
    rows: Vec<[u64; 2]>,
    span: u64,
    offset: u64,
    below: Option<u64>,
}

fn main() -> ExitCode {
    let (flags, config) = (Flags::parse(), TrainConfig::default());
    let run = || -> Result<u8, String> {
        if let Some(dir) = &flags.unpack {
            fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            for f in ["entry.cubin", "entry.sass", "reference.json"] {
                fs::write(dir.join(f), bundle::file(&flags.bundle, f)?)
                    .map_err(|e| format!("{}: {e}", dir.join(f).display()))?;
            }
            return Ok(0);
        }
        if flags.dry {
            return dry(&flags, &config);
        }
        match ranks::rank()? {
            Some(me) => rank(&flags, &config, me),
            None => {
                fs::create_dir_all(&flags.out).map_err(|e| format!("{}: {e}", flags.out.display()))?;
                ranks::spawn(read_reference(&flags.bundle, &config)?.model.ranks, &flags.out)?;
                Ok(0)
            }
        }
    };
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("qwen30b_train: {e}");
            ExitCode::FAILURE
        }
    }
}

fn read_reference(dir: &Path, config: &TrainConfig) -> Result<Reference, String> {
    let r: Reference = serde_json::from_slice(&bundle::file(dir, "reference.json")?)
        .map_err(|e| format!("reference.json: {e}"))?;
    if r.format != config.format {
        return Err(format!("{}: format {}, this loader reads {}", dir.display(), r.format, config.format));
    }
    Ok(r)
}

fn read_rank(dir: &Path, rank: usize) -> Result<RankFile, String> {
    let name = format!("rank{rank}.json");
    let f: RankFile =
        serde_json::from_slice(&bundle::file(dir, &name)?).map_err(|e| format!("{name}: {e}"))?;
    if f.rank != rank {
        return Err(format!("{name} is rank {}'s", f.rank));
    }
    Ok(f)
}

struct Sources {
    files: Vec<File>,
    tensors: Vec<(usize, u64, usize, bool)>,
}

impl Sources {
    fn open(r: &Reference, dir: &Path) -> Result<Sources, String> {
        let ck = Checkpoint::open(dir)?;
        let (mut files, mut seen, mut tensors) = (Vec::new(), HashMap::new(), Vec::new());
        for t in &r.tensors {
            let (dtype, shape) = ck.describe(&t.name)?;
            if !dtype.eq_ignore_ascii_case(&t.dtype) || shape != &t.shape[..] {
                return Err(format!(
                    "{}: tensor {} is {dtype} {shape:?}, the bundle wants {} {:?}: not the pinned checkpoint",
                    dir.display(),
                    t.name,
                    t.dtype,
                    t.shape
                ));
            }
            let (path, span) = ck.span(&t.name)?;
            let file = match seen.get(&path) {
                Some(&i) => i,
                None => {
                    files.push(File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?);
                    seen.insert(path, files.len() - 1);
                    files.len() - 1
                }
            };
            let (size, wide) = match t.dtype.as_str() {
                "bf16" => (2, false),
                "f32" => (4, true),
                d => return Err(format!("tensor {}: dtype {d}", t.name)),
            };
            tensors.push((file, span.start, size, wide));
        }
        Ok(Sources { files, tensors })
    }
}

fn fnv_add(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

fn round_bf16(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let e = ((x.abs().to_bits() >> 52) as i32 - 1023).max(-126);
    let quantum = 2f64.powi(e - 7);
    let r = (x / quantum).round_ties_even() * quantum;
    if r.abs() > f64::from(f32::MAX) * (1.0 - 2f64.powi(-25)) { f64::INFINITY.copysign(x) } else { r }
}

fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return (x.to_bits() >> 16) as u16;
    }
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn any_odd(src: &[u8], wide: bool) -> bool {
    let mut odd = false;
    if wide {
        for c in src.chunks_exact(4) {
            let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let e = v & 0x7f80_0000;
            odd |= (e == 0x7f80_0000) | ((e == 0) & (v & 0x7f_ffff != 0));
        }
    } else {
        for c in src.chunks_exact(2) {
            let v = u16::from_le_bytes([c[0], c[1]]);
            let e = v & 0x7f80;
            odd |= (e == 0x7f80) | ((e == 0) & (v & 0x7f != 0));
        }
    }
    odd
}

fn convert(src: &[u8], wide: bool, t: &Target, odd: bool, out: &mut Vec<u8>) -> Result<(), String> {
    let store_wide = match t.store.as_str() {
        "f32" => true,
        "bf16" => false,
        d => return Err(format!("store dtype {d}")),
    };
    let chain_wide = t.chain.iter().all(|d| d == "f32");
    if !odd && !wide && !store_wide {
        out.extend_from_slice(src);
        return Ok(());
    }
    if !odd && !wide && store_wide {
        let at = out.len();
        out.resize(at + src.len() * 2, 0);
        for (o, c) in out[at..].chunks_exact_mut(4).zip(src.chunks_exact(2)) {
            o[2] = c[0];
            o[3] = c[1];
        }
        return Ok(());
    }
    if !odd && wide && store_wide && chain_wide {
        out.extend_from_slice(src);
        return Ok(());
    }
    let size = if wide { 4 } else { 2 };
    for c in src.chunks_exact(size) {
        let mut x = if wide {
            f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64
        } else {
            bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64
        };
        for d in &t.chain {
            x = match d.as_str() {
                "f32" => x as f32 as f64,
                "bf16" => round_bf16(x),
                d => return Err(format!("dtype {d}")),
            };
        }
        match store_wide {
            true => out.extend_from_slice(&(x as f32).to_le_bytes()),
            false => out.extend_from_slice(&f32_to_bf16(x as f32).to_le_bytes()),
        }
    }
    Ok(())
}

fn make_checkpoint(
    s: &Sources,
    reads: &[[u64; 3]],
    targets: &[Target],
    chunk: usize,
    sink: &mut dyn FnMut(usize, u64, &[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let fnv0 = 0xcbf2_9ce4_8422_2325;
    let mut parts: Vec<(Vec<u8>, u64, u64)> = targets.iter().map(|_| (Vec::new(), 0, fnv0)).collect();
    let mut buf = Vec::new();
    for &[t, off, n] in reads {
        let (file, start, size, wide) = s.tensors[t as usize];
        buf.resize(n as usize * size, 0);
        s.files[file]
            .read_exact_at(&mut buf, start + off * size as u64)
            .map_err(|e| format!("reading tensor {}: {e}", t))?;
        let odd = any_odd(&buf, wide);
        for (i, tg) in targets.iter().enumerate() {
            let p = &mut parts[i];
            convert(&buf, wide, tg, odd, &mut p.0)?;
            if p.0.len() >= chunk {
                p.2 = fnv_add(p.2, &p.0);
                sink(i, p.1, &p.0)?;
                p.1 += p.0.len() as u64;
                p.0.clear();
            }
        }
    }
    for (i, tg) in targets.iter().enumerate() {
        let p = &mut parts[i];
        p.2 = fnv_add(p.2, &p.0);
        sink(i, p.1, &p.0)?;
        let bytes = p.1 + p.0.len() as u64;
        let size = if tg.store == "bf16" { 2 } else { 4 };
        if bytes != tg.elems * size {
            return Err(format!(
                "a fill at +{:#x}: {bytes} bytes, the bundle states {} elements",
                tg.at, tg.elems
            ));
        }
        let got = format!("{:016x}", p.2);
        if got != tg.fnv {
            return Err(format!(
                "a fill at +{:#x}: digest {got}, the bundle's {}: not the pinned checkpoint",
                tg.at, tg.fnv
            ));
        }
    }
    Ok(())
}

fn parallel(
    n: usize,
    threads: usize,
    job: impl Fn(usize) -> Result<(), String> + Sync,
) -> Result<(), String> {
    let next = AtomicUsize::new(0);
    let err = Mutex::new(None);
    thread::scope(|sc| {
        for _ in 0..threads.max(1).min(n.max(1)) {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n || err.lock().unwrap_or_else(|p| p.into_inner()).is_some() {
                        return;
                    }
                    if let Err(e) = job(i) {
                        *err.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
                    }
                }
            });
        }
    });
    err.into_inner().unwrap_or_else(|p| p.into_inner()).map_or(Ok(()), Err)
}

fn threads_of(flags: &Flags) -> usize {
    flags.threads.unwrap_or_else(|| (thread::available_parallelism().map_or(8, |n| n.get()) / 8).max(1))
}

fn dry(flags: &Flags, config: &TrainConfig) -> Result<u8, String> {
    let r = read_reference(&flags.bundle, config)?;
    let s = Sources::open(&r, &flags.model)?;
    println!("checkpoint: {} tensors at the bundle's dtypes and shapes", r.tensors.len());
    let threads = flags.threads.unwrap_or_else(|| thread::available_parallelism().map_or(8, |n| n.get()));
    for q in 0..r.model.ranks {
        let t0 = Instant::now();
        let f = read_rank(&flags.bundle, q)?;
        let bytes = AtomicUsize::new(0);
        parallel(f.fills.len(), threads, |i| match &f.fills[i] {
            Fill::Checkpoint { reads, targets } => {
                make_checkpoint(&s, reads, targets, config.chunk, &mut |_, _, b| {
                    bytes.fetch_add(b.len(), Ordering::Relaxed);
                    Ok(())
                })
            }
            _ => Ok(()),
        })?;
        check_tokens(&f, &flags.tokens)?;
        println!(
            "rank {q}: {} fills, {:.1} GB from the checkpoint match the digests ({:.0} s)",
            f.fills.len(),
            bytes.load(Ordering::Relaxed) as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );
    }
    println!("dry check passed");
    Ok(0)
}

fn tokens(file: &File, ids: u64, i: &Input, step: u64) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut row = vec![0u8; i.columns as usize * 4];
    for &[start, n] in &i.rows {
        for j in start..start + n {
            let w = step * i.rows_of_batch + j;
            let at = (w * (i.span - 1) + i.offset) % ids;
            let first = (ids - at).min(i.columns) as usize;
            file.read_exact_at(&mut row[..first * 4], at * 4).map_err(|e| format!("the token file: {e}"))?;
            if first < i.columns as usize {
                file.read_exact_at(&mut row[first * 4..], 0).map_err(|e| format!("the token file: {e}"))?;
            }
            if let Some(b) = i.below {
                for c in row.chunks_exact(4) {
                    let id = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64;
                    if id >= b {
                        return Err(format!(
                            "the token file: id {id} at or past {b}: not this model's tokens"
                        ));
                    }
                }
            }
            out.extend_from_slice(&row);
        }
    }
    Ok(out)
}

fn token_file(path: &Path) -> Result<(File, u64), String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let ids = f.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len() / 4;
    Ok((f, ids))
}

fn check_tokens(f: &RankFile, path: &Path) -> Result<(), String> {
    let (file, ids) = token_file(path)?;
    for i in &f.inputs {
        tokens(&file, ids, i, 0)?;
    }
    Ok(())
}

fn put(m: Module, name: &str, bytes: &[u8]) -> Result<(), String> {
    let (at, n) = cuda::global(m, name)?.ok_or_else(|| format!("the kernel has no global {name}"))?;
    cuda::htod(at, &bytes[..bytes.len().min(n)])
}

fn map_elem(e: &str) -> Result<u32, String> {
    use cuda::MapElem;
    Ok(match e {
        "bf16" => MapElem::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 as u32,
        "f32" => MapElem::CU_TENSOR_MAP_DATA_TYPE_FLOAT32 as u32,
        "f16" => MapElem::CU_TENSOR_MAP_DATA_TYPE_FLOAT16 as u32,
        "i32" => MapElem::CU_TENSOR_MAP_DATA_TYPE_INT32 as u32,
        "u8" => MapElem::CU_TENSOR_MAP_DATA_TYPE_UINT8 as u32,
        e => return Err(format!("a map's element {e}")),
    })
}

fn swizzle_of(bytes: u32) -> Result<u32, String> {
    use cuda::MapSwizzle;
    Ok(match bytes {
        0 => MapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE as u32,
        32 => MapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B as u32,
        64 => MapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B as u32,
        128 => MapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B as u32,
        b => return Err(format!("a map's swizzle {b}")),
    })
}

fn wait(
    end: &cuda::Event,
    limit: Duration,
    rank: usize,
    step: Option<usize>,
    config: &TrainConfig,
) -> Result<(), String> {
    let start = Instant::now();
    while !end.reached().map_err(|e| {
        format!(
            "rank {rank}: device fault {}: {e}",
            step.map_or("at the first launch".to_string(), |s| format!("at step {s}"))
        )
    })? {
        let waited = start.elapsed();
        if waited > limit {
            println!(
                "HANG: rank {rank}: {} did not finish within {} s. No result: this kernel has shown intermittent hangs like this one (GPU power falls and no step completes). The run's timing is void; run it again.",
                step.map_or("the first launch".to_string(), |s| format!("step {s}")),
                limit.as_secs()
            );
            std::process::exit(config.exit_hang as i32);
        }
        match waited < config.spin {
            true => thread::yield_now(),
            false => thread::sleep(Duration::from_millis(1)),
        }
    }
    Ok(())
}

fn rank(flags: &Flags, config: &TrainConfig, me: usize) -> Result<u8, String> {
    let r = read_reference(&flags.bundle, config)?;
    let k = &r.kernel;
    let f = read_rank(&flags.bundle, me)?;
    let t_start = Instant::now();
    let ranks = Ranks::join(me, r.group_bytes, &config.ranks)?;
    if ranks.world() != r.model.ranks {
        return Err(format!("{} ranks, the bundle's {}", ranks.world(), r.model.ranks));
    }
    let stream = cuda::stream()?;
    let image = match &flags.cubin {
        Some(p) => fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?,
        None => bundle::file(&flags.bundle, "entry.cubin")?,
    };
    let module = cuda::module(&image)?;
    let func = cuda::function(module, &k.entry)?;
    let fixed = cuda::get_attribute(func, Attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
    if fixed as u32 > k.static_smem {
        return Err(format!("the module's {fixed} static shared bytes, the bundle's {}", k.static_smem));
    }
    cuda::set_attribute(func, Attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem as i32)?;
    let (_, table_bytes) = cuda::global(module, &k.group_table)?
        .ok_or_else(|| format!("the kernel has no global {}", k.group_table))?;
    put(module, &k.group_words, &ranks.fabric_words(k.multicast))?;
    put(module, &k.group_table, &ranks.peers(table_bytes)?)?;

    let symmetric = ranks.symmetric();
    let (data, control) = (symmetric + r.data.at[0], symmetric + r.control.at[0]);
    let maps_at = cuda::alloc(r.maps_bytes as usize)?;
    cuda::memset8(maps_at, 0, r.maps_bytes as usize)?;
    let params: Vec<u64> = k
        .params
        .iter()
        .map(|p| match p.as_str() {
            "data" => Ok(data),
            "control" => Ok(control),
            "maps" => Ok(maps_at),
            "zero" => Ok(0),
            p => Err(format!("a parameter {p}")),
        })
        .collect::<Result<_, String>>()?;
    let mut with = Vec::new();
    if k.cooperative {
        with.push(Launch::Cooperative);
    }
    if k.cluster > 1 {
        with.push(Launch::Cluster(k.cluster));
    }
    if me == 0 {
        println!(
            "launch: grid {} x {} threads, cluster {}, cooperative {}, dynamic smem {} + static {}, regs {}",
            k.grid,
            k.threads,
            k.cluster,
            k.cooperative,
            k.smem,
            fixed,
            cuda::get_attribute(func, Attribute::CU_FUNC_ATTRIBUTE_NUM_REGS)?
        );
    }
    let launch = |limit: Duration, step: Option<usize>| -> Result<f64, String> {
        let t = Instant::now();
        let _start = cuda::event(stream)?;
        cuda::launch(func, k.grid, k.threads, k.smem, stream, &params, &with)?;
        let end = cuda::event(stream)?;
        wait(&end, limit, me, step, config)?;
        Ok(t.elapsed().as_secs_f64())
    };

    cuda::sync_all()?;
    ranks.barrier()?;
    launch(Duration::from_secs(900), None)?;
    let mut word = [0u8; 8];
    cuda::dtoh(&mut word, control + 8 * r.words.address as u64)?;
    let reported = u64::from_le_bytes(word);
    if reported != data {
        return Err(format!("the first launch reported address {reported:#x}, the data is at {data:#x}"));
    }
    let mut map_bytes = Vec::with_capacity(r.maps.len() * 128);
    for m in &r.maps {
        map_bytes.extend_from_slice(&cuda::tensor_map(
            map_elem(&m.elem)?,
            data + m.at,
            &m.dims,
            &m.strides,
            &m.extent,
            swizzle_of(m.swizzle)?,
            None,
        )?);
    }
    if !map_bytes.is_empty() {
        cuda::htod(maps_at, &map_bytes)?;
    }
    for &[at, bytes] in &r.zeroed {
        cuda::memset8(data + at, 0, bytes as usize)?;
    }
    cuda::sync_all()?;

    let sources = Sources::open(&r, &flags.model)?;
    let blobs = bundle::file(&flags.bundle, "blobs.bin")?;
    let t_fill = Instant::now();
    parallel(f.fills.len(), threads_of(flags), |i| {
        cuda::bind()?;
        match &f.fills[i] {
            Fill::Checkpoint { reads, targets } => {
                make_checkpoint(&sources, reads, targets, config.chunk, &mut |t, off, b| {
                    if b.is_empty() { Ok(()) } else { cuda::htod(data + targets[t].at + off, b) }
                })
            }
            Fill::Repeat { at, elem, n } => {
                let total = elem.len() * *n as usize;
                if elem.iter().all(|&b| b == 0) {
                    return cuda::memset8(data + at, 0, total);
                }
                let whole = (config.chunk / elem.len().max(1)).max(1);
                let run: Vec<u8> = elem.iter().copied().cycle().take(whole * elem.len()).collect();
                let mut done = 0;
                while done < total {
                    let m = run.len().min(total - done);
                    cuda::htod(data + at + done as u64, &run[..m])?;
                    done += m;
                }
                Ok(())
            }
            Fill::Blob { .. } => Ok(()),
        }
    })?;
    for fill in &f.fills {
        if let Fill::Blob { at, blob } = fill {
            let b = usize::try_from(blob[0])
                .ok()
                .zip(usize::try_from(blob[1]).ok())
                .and_then(|(a, n)| blobs.get(a..a.checked_add(n)?));
            cuda::htod(
                data + at,
                b.ok_or_else(|| format!("blob {}+{} past blobs.bin's {} B", blob[0], blob[1], blobs.len()))?,
            )?;
        }
    }
    cuda::sync_all()?;
    if me == 0 {
        println!(
            "filled in {:.0} s; {:.0} s since the start",
            t_fill.elapsed().as_secs_f64(),
            t_start.elapsed().as_secs_f64()
        );
    }

    let host_record = Pinned::new(r.host_record_bytes.max(8) as usize)?;
    cuda::memset8(control, 0, r.words.all * 8)?;
    let mut head = vec![0u64; r.words.head];
    head[r.words.go] = 1;
    head[r.words.timeout_ns] = config.device_limit_ns;
    head[r.words.host_record] = host_record.device();
    cuda::htod(control, &head.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    cuda::sync_all()?;

    let (tokens_file, ids) = token_file(&flags.tokens)?;
    let total = flags.warmup + flags.steps;
    let (mut walls, mut limit) = (Vec::with_capacity(flags.steps), Duration::from_secs(flags.hang_seconds));
    let mut last = String::new();
    let mut lines = (me == 0)
        .then(|| File::create(flags.out.join("steps.csv")))
        .transpose()
        .map_err(|e| format!("steps.csv: {e}"))?;
    for s in 0..total {
        for i in &f.inputs {
            cuda::htod(data + i.at, &tokens(&tokens_file, ids, i, s as u64)?)?;
        }
        ranks.barrier()?;
        let wall = launch(limit, Some(s))?;
        if me == 0 {
            let mut parts = Vec::new();
            for o in &r.outputs {
                let size = if o.dtype == "bf16" { 2 } else { 4 };
                let mut b = [0u8; 4];
                cuda::dtoh(&mut b[..size], data + o.at)?;
                let x = if size == 2 {
                    bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))
                } else {
                    f32::from_le_bytes(b)
                };
                parts.push(format!("{} {x:.6}", o.name));
            }
            last = parts.join(", ");
            if let Some(file) = &mut lines {
                use std::io::Write;
                writeln!(file, "{s},{:.3},{}", 1e3 * wall, parts.join(","))
                    .map_err(|e| format!("steps.csv: {e}"))?;
            }
            if s < 2 || s == flags.warmup || (s + 1) % flags.log_every.max(1) == 0 || s + 1 == total {
                println!(
                    "step {s}: {:.3} ms{}; {last}",
                    1e3 * wall,
                    if s < flags.warmup { " (warmup)" } else { "" }
                );
            }
        }
        let mut raw = vec![0u8; r.words.head * 8];
        cuda::dtoh(&mut raw, control)?;
        let w: Vec<u64> =
            raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
        if w[r.words.watchdog] != 0 || w[r.words.stop] != 0 {
            println!(
                "HANG: rank {me}: the kernel's own {} s limit tripped at step {s}. No result: this kernel has shown intermittent hangs like this one. The run's timing is void; run it again.",
                config.device_limit_ns / 1_000_000_000
            );
            std::process::exit(config.exit_hang as i32);
        }
        if w[r.words.bounds] != 0 || w[r.words.range] != 0 {
            return Err(format!("rank {me}: step {s}: the kernel recorded an out-of-bounds access"));
        }
        if s >= flags.warmup {
            walls.push(wall);
        }
        limit = limit.max(Duration::from_secs_f64(3.0 * wall));
    }

    walls.sort_by(f64::total_cmp);
    let median = walls[walls.len() / 2];
    let (ns, top) = ((median * 1e9) as u64, u64::MAX);
    let (least, most) = (ranks.min(ns)?, top - ranks.min(top - ns)?);
    if me == 0 {
        let m = &r.model;
        let tokens = (m.batch * m.seq) as f64;
        let mfu = |t: f64| 100.0 * m.flops_per_token * tokens / t / (m.ranks as f64 * r.peak_flops_per_gpu);
        println!(
            "model {}; {} ranks; batch {} x {}; {} warmup + {} timed steps",
            m.name,
            m.ranks,
            m.batch,
            m.seq,
            flags.warmup,
            walls.len()
        );
        println!("last outputs: {last}");
        println!(
            "step ms: median {:.3} (rank 0; min {:.3}, max {:.3}); ranks' medians {:.3} to {:.3}",
            1e3 * median,
            1e3 * walls[0],
            1e3 * walls[walls.len() - 1],
            least as f64 / 1e6,
            most as f64 / 1e6
        );
        println!("tokens/s per GPU: {:.0}", tokens / median / m.ranks as f64);
        println!(
            "MFU: {:.2}% of {:.2} PFLOP/s x {} (ranks: {:.2}% to {:.2}%)",
            mfu(median),
            r.peak_flops_per_gpu / 1e15,
            m.ranks,
            mfu(most as f64 / 1e9),
            mfu(least as f64 / 1e9)
        );
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
    ranks.barrier()?;
    Ok(0)
}
