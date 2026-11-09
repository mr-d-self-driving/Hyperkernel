use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde::de::DeserializeOwned;

pub struct Bundle<S = Dsv> {
    pub dir: PathBuf,
    pub reference: Reference<S>,
    pub inputs: Inputs,
    pub blobs: Vec<u8>,
}

impl<S: Served> Bundle<S> {
    pub fn read(dir: &Path, rank: usize) -> Result<Bundle<S>, String> {
        let reference: Reference<S> = serde_json::from_slice(&file(dir, "reference.json")?)
            .map_err(|e| format!("reference.json: {e}"))?;
        let name = format!("rank{rank}.json");
        let inputs = serde_json::from_slice(&file(dir, &name)?).map_err(|e| format!("{name}: {e}"))?;
        Ok(Bundle { dir: dir.to_path_buf(), reference, inputs, blobs: file(dir, "blobs.bin")? })
    }

    pub fn blob(&self, blob: &Blob) -> Result<Vec<u8>, String> {
        span(&self.blobs, blob).map(<[u8]>::to_vec)
    }

    pub fn cubin(&self, name: &str) -> Result<Vec<u8>, String> {
        file(&self.dir, name)
    }
}

fn span<'a>(blobs: &'a [u8], blob: &Blob) -> Result<&'a [u8], String> {
    let at = usize::try_from(blob.at)
        .ok()
        .zip(usize::try_from(blob.len).ok())
        .and_then(|(at, n)| blobs.get(at..at.checked_add(n)?));
    at.ok_or_else(|| format!("blob {}+{} past blobs.bin's {} B", blob.at, blob.len, blobs.len()))
}

pub fn file(dir: &Path, name: &str) -> Result<Vec<u8>, String> {
    let (plain, packed) = (dir.join(name), dir.join(format!("{name}.zst")));
    if !plain.exists() && packed.exists() {
        let mut out = Vec::new();
        let read = fs::File::open(&packed).map_err(|e| e.to_string());
        let decoded = read.and_then(|f| {
            ruzstd::decoding::StreamingDecoder::new(std::io::BufReader::new(f)).map_err(|e| e.to_string())
        });
        return decoded
            .and_then(|mut d| std::io::Read::read_to_end(&mut d, &mut out).map_err(|e| e.to_string()))
            .map(|_| out)
            .map_err(|e| format!("{}: {e}", packed.display()));
    }
    fs::read(&plain).map_err(|e| format!("{}: {e}", plain.display()))
}

#[derive(Deserialize)]
pub struct Reference<S = Dsv> {
    pub format: Format,
    pub point: String,
    pub kernel: Kernel,
    pub homes: Vec<Home>,
    pub image: Vec<Record>,
    #[serde(flatten)]
    pub served: S,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    #[serde(rename = "hyperserve-bundle-2")]
    Bundle2,
}

pub trait Served: DeserializeOwned {
    fn ranks(&self) -> usize;
}

#[derive(Deserialize)]
pub struct Dsv {
    pub model: Model,
    pub rings: Rings,
    pub pool: Pool,
}

impl Served for Dsv {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
    pub vocab: usize,
    pub context: usize,
    pub batches: Vec<usize>,
    pub chunks: Vec<usize>,
    pub block: usize,
    pub noise: u32,
    pub pad: u32,
}

#[derive(Deserialize)]
pub struct Kernel {
    pub entries: Vec<(String, u32)>,
    pub threads: u32,
    pub cluster: u32,
    pub smem: u32,
    pub static_smem: u32,
    pub cooperative: bool,
    pub stack: u64,
    pub workspace: u64,
    pub symmetric: u64,
    pub l2: Option<Range<u64>>,
    #[serde(rename = "homew")]
    pub constant_records: Option<Range<u64>>,
    pub params: Vec<Param>,
    pub next: Option<String>,
    pub loaders: Vec<(String, String)>,
    pub globals: Globals,
}

#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Param {
    Arena,
    Maps,
    Replay,
    Symmetric,
    Progress,
    Home(usize),
}

#[derive(Deserialize)]
pub struct Globals {
    pub fault: String,
    pub fabric: String,
    pub peers: String,
    pub peer_bytes: u64,
    pub multicast: bool,
    #[serde(default)]
    pub rings: String,
    #[serde(default)]
    pub ctl: String,
}

#[derive(Deserialize)]
pub struct Home {
    pub level: Level,
    pub at: Range<u64>,
    pub staged: bool,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Hbm,
    Peer,
    Host,
}

#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Addr {
    Home(usize, u64),
    Arena(u64),
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Record {
    Write {
        at: u64,
        blob: Blob,
    },
    Patch {
        at: u64,
        to: Addr,
    },
    Init {
        home: usize,
        word: u32,
    },
    Map {
        home: usize,
        at: u64,
        place: Option<u64>,
        elem: u32,
        dims: Vec<u64>,
        strides: Vec<u64>,
        tile: Vec<u32>,
        swizzle: u32,
        #[serde(default)]
        corners: Option<Corners>,
    },
    Load {
        loader: usize,
        threads: u64,
        args: Vec<Arg>,
    },
    After {
        from: (usize, Range<u64>),
        to: (usize, Range<u64>),
    },
}

#[derive(Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Corners {
    pub lower: Vec<i32>,
    pub upper: Vec<i32>,
}

#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Arg {
    Word(u64),
    Addr(Addr),
    Aim(Addr),
}

#[derive(Deserialize, Clone, Copy, Debug)]
pub struct Blob {
    pub at: u64,
    pub len: u64,
}

#[derive(Deserialize)]
pub struct Rings {
    pub slots: [usize; 2],
    pub messages: Vec<Message>,
    pub commits_head: usize,
    pub logprobs: usize,
}

impl Rings {
    pub fn message(&self, name: Name) -> Result<&Message, String> {
        self.messages
            .iter()
            .find(|m| name.is(m))
            .ok_or_else(|| format!("the bundle's rings have no {}", name.text()))
    }

    pub fn reply(&self, id: u32) -> Result<&Message, String> {
        self.messages
            .iter()
            .find(|m| m.ring == 1 && m.id == id)
            .ok_or_else(|| format!("a reply of id {id:#x} the bundle has no layout for"))
    }
}

#[derive(Deserialize)]
pub struct Message {
    pub name: String,
    pub id: u32,
    pub ring: usize,
    pub bytes: usize,
    pub fields: Vec<Field>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Name {
    End,
    Pages,
    Seat,
    Prompt,
    Start,
    Idle,
    Last,
}

impl Name {
    pub fn text(self) -> &'static str {
        match self {
            Name::End => "end",
            Name::Pages => "pages",
            Name::Seat => "seat",
            Name::Prompt => "prompt",
            Name::Start => "start",
            Name::Idle => "idle",
            Name::Last => "last",
        }
    }

    pub fn ring(self) -> usize {
        match self {
            Name::End | Name::Pages | Name::Seat | Name::Prompt | Name::Start => 0,
            Name::Idle | Name::Last => 1,
        }
    }

    pub fn of(m: &Message) -> Option<Name> {
        let all = [Name::End, Name::Pages, Name::Seat, Name::Prompt, Name::Start, Name::Idle, Name::Last];
        all.into_iter().find(|n| n.is(m))
    }

    fn is(self, m: &Message) -> bool {
        m.ring == self.ring() && m.name == self.text()
    }
}

#[derive(Deserialize)]
pub struct Field {
    pub name: String,
    pub at: usize,
    pub kind: Kind,
    pub dtype: Dtype,
    pub n: usize,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    One,
    Array,
    Span,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    I32,
    U32,
    F32,
    Bf16,
    U8,
    U64,
}

impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::Bf16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::U64 => 8,
        }
    }
}

#[derive(Deserialize)]
pub struct Pool {
    pub page_rows: usize,
    pub pages: usize,
    pub window_pages: usize,
    pub slots: usize,
    pub window_keep: usize,
    pub entries: usize,
    pub window: usize,
    pub states: Vec<PoolState>,
}

#[derive(Deserialize)]
pub struct PoolState {
    pub home: usize,
    pub unit: Unit,
    pub bytes: u64,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Page,
    Window,
}

#[derive(Deserialize)]
pub struct Ports {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Deserialize, Debug)]
pub struct Port {
    pub name: String,
    pub home: usize,
    pub at: Range<u64>,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

impl Ports {
    pub fn input(&self, name: &str, bytes: usize) -> Result<&Port, String> {
        port(&self.inputs, name, bytes)
    }

    pub fn output(&self, name: &str, bytes: usize) -> Result<&Port, String> {
        port(&self.outputs, name, bytes)
    }
}

fn port<'a>(ports: &'a [Port], name: &str, bytes: usize) -> Result<&'a Port, String> {
    let p = ports.iter().find(|p| p.name == name).ok_or_else(|| format!("the bundle has no port {name}"))?;
    match bytes as u64 <= p.at.end - p.at.start {
        true => Ok(p),
        false => Err(format!("{bytes} B of {name}, its port's {}", p.at.end - p.at.start)),
    }
}

#[derive(Deserialize)]
pub struct Inputs {
    pub rank: usize,
    pub inputs: Vec<Input>,
}

#[derive(Deserialize)]
pub struct Input {
    pub name: String,
    pub home: usize,
    pub at: Range<u64>,
    pub fill: Fill,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Fill {
    Zeros,
    Blob {
        blob: Blob,
    },
    Rope(Rope),
    File {
        tensor: String,
    },
    Checkpoint {
        transform: Transform,
        tensors: Vec<String>,
        cut: Option<Cut>,
        #[serde(flatten)]
        laid: Laid,
    },
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Laid {
    #[serde(default)]
    pub prefix: Option<u64>,
    #[serde(default)]
    pub width: Option<u64>,
    #[serde(default)]
    pub rows: Option<Vec<u64>>,
    #[serde(default)]
    pub round: Option<Dtype>,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub rows: u64,
    pub dim: u64,
    pub base: f32,
    pub original_seq_len: u64,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case", tag = "name")]
pub enum Transform {
    Stack,
    Transpose,
    VectorF32 { n: u64 },
    OnePlusF32 { n: u64 },
    HcMix { d: u64 },
    ProductF32,
    Columns { j: u64, of: u64 },
    Experts { experts: u64 },
    View { parts: Vec<View> },
    Band { rows: u64, samples: u64 },
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct View {
    pub tensor: usize,
    pub at: u64,
    pub dims: Vec<u64>,
    pub strides: Vec<i64>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Cut {
    pub whole: Vec<u64>,
    pub dim: usize,
    pub runs: Vec<Range<u64>>,
}

#[cfg(test)]
mod tests {
    use super::{Blob, file, span};

    #[test]
    fn a_zst_file_reads_as_its_plain_file() {
        let dir = std::env::temp_dir().join(format!("hs-bundle-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bytes: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let packed =
            ruzstd::encoding::compress_to_vec(&bytes[..], ruzstd::encoding::CompressionLevel::Fastest);
        std::fs::write(dir.join("blobs.bin.zst"), &packed).unwrap();
        std::fs::write(dir.join("entry.cubin"), b"plain").unwrap();
        let (blobs, plain, none) =
            (file(&dir, "blobs.bin"), file(&dir, "entry.cubin"), file(&dir, "next.cubin"));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(blobs.unwrap(), bytes);
        assert_eq!(plain.unwrap(), b"plain");
        assert!(none.is_err());
    }

    #[test]
    fn a_map_is_tiled_or_im2col_by_its_corners() {
        let map = |corners: serde_json::Value| serde_json::json!({"map": {"home": 1, "at": 0, "place": null, "elem": 9, "dims": [8, 4, 4, 2], "strides": [16, 64, 256], "tile": [8, 16], "swizzle": 3, "corners": corners}});
        let im2col: super::Record =
            serde_json::from_value(map(serde_json::json!({"lower": [-1, -1], "upper": [1, 1]}))).unwrap();
        assert!(
            matches!(im2col, super::Record::Map { corners: Some(super::Corners { ref lower, .. }), .. } if lower == &[-1, -1])
        );
        let mut tiled = map(serde_json::Value::Null);
        tiled["map"].as_object_mut().unwrap().remove("corners");
        assert!(matches!(serde_json::from_value(tiled).unwrap(), super::Record::Map { corners: None, .. }));
    }

    #[test]
    fn a_blob_past_the_blobs_is_refused() {
        let blobs = [1u8, 2, 3, 4];
        assert_eq!(span(&blobs, &Blob { at: 1, len: 3 }).unwrap(), [2, 3, 4]);
        assert!(span(&blobs, &Blob { at: 2, len: 3 }).is_err());
        assert!(span(&blobs, &Blob { at: u64::MAX, len: 2 }).is_err());
    }
}
