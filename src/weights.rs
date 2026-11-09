use std::collections::HashMap;
use std::fs::{self, File};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::thread;

use serde::Deserialize;

use crate::bundle::{Cut, Dtype, Laid, Transform, View};

pub struct Checkpoint {
    files: Vec<(PathBuf, File)>,
    tensors: HashMap<String, Tensor>,
}

struct Tensor {
    file: usize,
    dtype: String,
    shape: Vec<u64>,
    at: Range<u64>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Checkpoint, String> {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        if paths.is_empty() {
            return Err(format!(
                "{}: no .safetensors files (a checkpoint directory holds them)",
                dir.display()
            ));
        }
        paths.sort();
        let mut c = Checkpoint { files: Vec::new(), tensors: HashMap::new() };
        for path in paths {
            let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            for (name, t) in header(&file, c.files.len()).map_err(|e| format!("{}: {e}", path.display()))? {
                if c.tensors.insert(name.clone(), t).is_some() {
                    return Err(format!("{}: `{name}` is in another file too", path.display()));
                }
            }
            c.files.push((path, file));
        }
        Ok(c)
    }

    pub fn span(&self, name: &str) -> Result<(PathBuf, Range<u64>), String> {
        let t = self.tensor(name)?;
        Ok((self.files[t.file].0.clone(), t.at.clone()))
    }

    pub fn laid(
        &self,
        transform: &Transform,
        tensors: &[String],
        cut: Option<&Cut>,
        laid: &Laid,
    ) -> Result<Vec<u8>, String> {
        if *laid == Laid::default() {
            return self.fill(transform, tensors, cut);
        }
        let reshaped = laid.prefix.is_some() || laid.width.is_some() || laid.rows.is_some();
        if reshaped && cut.is_some() {
            return Err(format!("{tensors:?}: a fill of reshaped rows cut"));
        }
        let floats = matches!(
            transform,
            Transform::VectorF32 { .. }
                | Transform::OnePlusF32 { .. }
                | Transform::HcMix { .. }
                | Transform::ProductF32
        ) || tensors.iter().all(|n| self.tensor(n).is_ok_and(|t| t.dtype == "F32"));
        let e = if floats {
            4
        } else {
            same(&tensors.iter().map(|n| self.tensor(n)).collect::<Result<Vec<_>, _>>()?)?
        };
        let mut b = self.fill(transform, tensors, cut)?;
        if let Some(k) = laid.prefix {
            let row = self.row(transform, tensors, e)?;
            b.resize(k as usize * row, 0);
        }
        if let Some(w) = laid.width {
            b = widened(b, self.row(transform, tensors, e)?, w as usize * e)?;
        }
        if let Some(p) = &laid.rows {
            b = permuted(b, p)?;
        }
        match (laid.round, floats) {
            (None, _) => Ok(b),
            (Some(to), true) => rounded(&b, to),
            (Some(to), false) => Err(format!("{tensors:?}: no F32 values to round to {to:?}")),
        }
    }

    fn row(&self, transform: &Transform, tensors: &[String], e: usize) -> Result<usize, String> {
        let t = self.tensor(tensors.first().ok_or("a fill of no tensor")?)?;
        match transform {
            Transform::Transpose => Ok(tensors
                .iter()
                .map(|n| self.tensor(n).map(|t| t.shape.first().copied().unwrap_or(0)))
                .sum::<Result<u64, String>>()? as usize
                * e),
            Transform::Stack => Ok(t.shape[1.min(t.shape.len())..].iter().product::<u64>() as usize * e),
            other => Err(format!("a prefix of a {other:?} fill")),
        }
    }

    pub fn describe(&self, name: &str) -> Result<(&str, &[u64]), String> {
        let t = self.tensor(name)?;
        Ok((&t.dtype, &t.shape))
    }

    pub fn fill(
        &self,
        transform: &Transform,
        tensors: &[String],
        cut: Option<&Cut>,
    ) -> Result<Vec<u8>, String> {
        let ts = || tensors.iter().map(|n| self.tensor(n)).collect::<Result<Vec<_>, _>>();
        match transform {
            Transform::Stack => self.stacked(&ts()?, cut),
            Transform::Transpose => self.transposed(&ts()?, cut),
            Transform::Experts { experts } => self.experts(tensors, *experts, cut),
            Transform::VectorF32 { n } => part(self.vector(&ts()?, *n)?, 4, cut),
            Transform::OnePlusF32 { n } => part(self.one_plus(&ts()?, *n)?, 4, cut),
            Transform::HcMix { d } => part(self.hc_mix(&ts()?, *d)?, 4, cut),
            Transform::ProductF32 => part(self.product(&ts()?)?, 4, cut),
            Transform::Columns { j, of } => {
                let ts = ts()?;
                part(self.columns(&ts, *j, *of)?, same(&ts)?, cut)
            }
            Transform::Band { rows, samples } => {
                let ts = ts()?;
                part(self.band(one(&ts, 0)?, *rows, *samples)?, same(&ts)?, cut)
            }
            Transform::View { parts } => match cut {
                None => self.view(&ts()?, parts),
                Some(c) => Err(format!("a view cut as {c:?}")),
            },
        }
    }

    fn tensor(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors.get(name).ok_or_else(|| format!("the checkpoint has no tensor `{name}`"))
    }

    fn read(&self, t: &Tensor, at: u64, buf: &mut [u8]) -> Result<(), String> {
        let (path, file) = &self.files[t.file];
        file.read_exact_at(buf, t.at.start + at).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn raw(&self, t: &Tensor) -> Result<Vec<u8>, String> {
        self.concat(&[t], 0..t.at.end - t.at.start)
    }

    fn concat(&self, ts: &[&Tensor], range: Range<u64>) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; (range.end - range.start) as usize];
        let mut at = 0;
        for t in ts {
            let len = t.at.end - t.at.start;
            let (s, e) = (range.start.max(at), range.end.min(at + len));
            if s < e {
                self.read(t, s - at, &mut out[(s - range.start) as usize..(e - range.start) as usize])?;
            }
            at += len;
        }
        Ok(out)
    }

    fn stacked(&self, ts: &[&Tensor], cut: Option<&Cut>) -> Result<Vec<u8>, String> {
        let e = same(ts)?;
        let rest = &ts[0].shape[1..];
        if ts.iter().any(|t| t.shape.is_empty() || &t.shape[1..] != rest) {
            return Err(format!(
                "tensors of shapes {:?} stack no rows",
                ts.iter().map(|t| &t.shape).collect::<Vec<_>>()
            ));
        }
        let total: u64 = ts.iter().map(|t| t.at.end - t.at.start).sum();
        match cut {
            Some(c) if c.dim == 0 => {
                fits(c, total as usize / e)?;
                let row = total / c.whole[0].max(1);
                let mut out = Vec::new();
                for r in &c.runs {
                    out.extend(self.concat(ts, r.start * row..r.end * row)?);
                }
                Ok(out)
            }
            c => part(self.concat(ts, 0..total)?, e, c),
        }
    }

    fn transposed(&self, ts: &[&Tensor], cut: Option<&Cut>) -> Result<Vec<u8>, String> {
        let e = same(ts)?;
        let ins = &ts[0].shape[1.min(ts[0].shape.len())..];
        if ins.is_empty() || ts.iter().any(|t| &t.shape[1.min(t.shape.len())..] != ins) {
            return Err(format!(
                "tensors of shapes {:?} are no [out, in..] stack",
                ts.iter().map(|t| &t.shape).collect::<Vec<_>>()
            ));
        }
        let (k, n): (u64, u64) = (ins.iter().product(), ts.iter().map(|t| t.shape[0]).sum());
        if let Some(c) = cut {
            fits(c, (k * n) as usize)?;
            if c.whole != [k, n] {
                return Err(format!("a transposed [{k}, {n}] cut as {:?}", c.whole));
            }
        }
        let runs = |d: usize, all: u64| match cut {
            Some(c) if c.dim == d => c.runs.clone(),
            _ => vec![0..all],
        };
        let (cols, outs) = (runs(0, k), runs(1, n));
        let (kl, nl) = (
            cols.iter().map(|r| r.end - r.start).sum::<u64>() as usize,
            outs.iter().map(|r| r.end - r.start).sum::<u64>() as usize,
        );
        let mut out = vec![0u8; kl * nl * e];
        let mut col = 0;
        for r in &outs {
            let mut at = 0;
            for t in ts {
                let (s, end) = (r.start.max(at), r.end.min(at + t.shape[0]));
                if s < end {
                    let rows = (end - s) as usize;
                    let mut src = vec![0u8; rows * k as usize * e];
                    self.read(t, (s - at) * k * e as u64, &mut src)?;
                    let mut kk0 = 0;
                    for c in &cols {
                        let c = c.start as usize..c.end as usize;
                        let w = c.len();
                        transpose(
                            &src,
                            k as usize,
                            c,
                            &mut out[kk0 * nl * e..(kk0 + w) * nl * e],
                            nl,
                            col,
                            e,
                        )?;
                        kk0 += w;
                    }
                    col += rows;
                }
                at += t.shape[0];
            }
        }
        Ok(out)
    }

    fn experts(&self, names: &[String], experts: u64, cut: Option<&Cut>) -> Result<Vec<u8>, String> {
        let parts = |x: u64| {
            names
                .iter()
                .map(|n| self.tensor(&n.replace("{x}", &x.to_string())))
                .collect::<Result<Vec<_>, _>>()
        };
        let xs: Vec<u64> = match cut {
            Some(c) if c.dim == 0 => c.runs.iter().flat_map(|r| r.clone()).collect(),
            _ => (0..experts).collect(),
        };
        if let Some(c) = cut
            && (c.whole.first() != Some(&experts) || xs.iter().any(|&x| x >= experts))
        {
            return Err(format!("{experts} experts cut as {c:?}"));
        }
        let inner = cut.filter(|c| c.dim > 0).map(|c| Cut {
            whole: c.whole[1..].to_vec(),
            dim: c.dim - 1,
            runs: c.runs.clone(),
        });
        let threads = thread::available_parallelism().map_or(8, |n| n.get()).clamp(1, 32);
        let chunk = xs.len().div_ceil(threads).max(1);
        let pieces: Vec<Result<Vec<Vec<u8>>, String>> = thread::scope(|s| {
            let jobs: Vec<_> = xs
                .chunks(chunk)
                .map(|xs| s.spawn(|| xs.iter().map(|&x| self.stacked(&parts(x)?, inner.as_ref())).collect()))
                .collect();
            jobs.into_iter()
                .map(|j| j.join().unwrap_or_else(|_| Err("an expert's read panicked".into())))
                .collect()
        });
        let each: Vec<Vec<u8>> =
            pieces.into_iter().collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
        if each.iter().any(|b| b.len() != each[0].len()) {
            return Err(format!("experts {names:?} of unlike shapes"));
        }
        Ok(each.concat())
    }

    fn vector(&self, ts: &[&Tensor], n: u64) -> Result<Vec<u8>, String> {
        let v = self.floats(one(ts, 0)?)?;
        if v.is_empty() || !(n as usize).is_multiple_of(v.len()) {
            return Err(format!("{} elements fill no {n}", v.len()));
        }
        Ok(bytes(v.iter().copied().cycle().take(n as usize)))
    }

    fn one_plus(&self, ts: &[&Tensor], n: u64) -> Result<Vec<u8>, String> {
        let v: Vec<f32> = self.floats(one(ts, 0)?)?.iter().map(|x| 1.0 + x).collect();
        if v.is_empty() || !(n as usize).is_multiple_of(v.len()) {
            return Err(format!("{} elements fill no {n}", v.len()));
        }
        Ok(bytes(v.iter().copied().cycle().take(n as usize)))
    }

    fn hc_mix(&self, ts: &[&Tensor], d: u64) -> Result<Vec<u8>, String> {
        let t = one(ts, 0)?;
        let [mix, sd] = t.shape[..] else { return Err(format!("an mHC mix of shape {:?}", t.shape)) };
        let (mix, sd, d) = (mix as usize, sd as usize, d as usize);
        let s = sd / d.max(1);
        if d == 0 || s * d != sd || s * (s + 2) != mix {
            return Err(format!("an mHC mix [{mix}, {sd}] of no s(s + 2) mixes over d = {d}"));
        }
        let f = self.floats(t)?;
        let mut v = vec![0f32; f.len()];
        for (m, row) in f.chunks_exact(sd).enumerate() {
            for (i, &x) in row.iter().enumerate() {
                v[((i % d) * s + i / d) * mix + m] = x;
            }
        }
        Ok(bytes(v))
    }

    fn product(&self, ts: &[&Tensor]) -> Result<Vec<u8>, String> {
        let (a, b) = (self.floats(one(ts, 0)?)?, self.floats(one(ts, 1)?)?);
        if a.len() != b.len() {
            return Err(format!("a product of {} and {} elements", a.len(), b.len()));
        }
        Ok(bytes(a.iter().zip(&b).map(|(x, y)| x * y)))
    }

    fn columns(&self, ts: &[&Tensor], j: u64, of: u64) -> Result<Vec<u8>, String> {
        let t = one(ts, 0)?;
        let [_, cols] = t.shape[..] else { return Err(format!("columns of shape {:?}", t.shape)) };
        if of == 0 || cols % of != 0 || j >= of {
            return Err(format!("column block {j} of {of} of {cols} columns"));
        }
        let width = (cols / of) as usize * elem(&t.dtype)?;
        Ok(self
            .raw(t)?
            .chunks_exact(width * of as usize)
            .flat_map(|r| &r[j as usize * width..(j as usize + 1) * width])
            .copied()
            .collect())
    }

    fn band(&self, t: &Tensor, rows: u64, samples: u64) -> Result<Vec<u8>, String> {
        let (e, groups) = (elem(&t.dtype)?, t.shape.first().copied().unwrap_or(0) as usize);
        let taps = t.shape[1.min(t.shape.len())..].iter().product::<u64>() as usize;
        let (rows, samples) = (rows as usize, samples as usize);
        if groups == 0 || taps == 0 || taps - 1 + rows > samples {
            return Err(format!("a band of {rows} rows over {samples} samples of filters {:?}", t.shape));
        }
        let h = self.raw(t)?;
        let mut out = vec![0u8; groups * samples * rows * e];
        for g in 0..groups {
            for j in 0..taps {
                let tap = &h[(g * taps + taps - 1 - j) * e..][..e];
                for r in 0..rows {
                    let at = ((g * samples + r + j) * rows + r) * e;
                    out[at..at + e].copy_from_slice(tap);
                }
            }
        }
        Ok(out)
    }

    fn view(&self, ts: &[&Tensor], parts: &[View]) -> Result<Vec<u8>, String> {
        let e = same(ts)?;
        let rows = parts.first().and_then(|p| p.dims.first()).copied().ok_or("a view of no parts")?;
        for p in parts {
            let t =
                ts.get(p.tensor).ok_or_else(|| format!("a view of tensor {} of {}", p.tensor, ts.len()))?;
            if p.dims.first() != Some(&rows) || !reaches(p, (t.at.end - t.at.start) / e as u64) {
                return Err(format!("a view {p:?} of {rows} rows over a tensor of shape {:?}", t.shape));
            }
        }
        let raws = ts.iter().map(|t| self.raw(t)).collect::<Result<Vec<_>, _>>()?;
        let row = parts.iter().map(|p| p.dims[1..].iter().product::<u64>()).sum::<u64>() as usize * e;
        let mut out = vec![0u8; rows as usize * row];
        let threads = thread::available_parallelism().map_or(8, |n| n.get()).clamp(1, 32);
        let chunk = (rows as usize).div_ceil(threads).max(1) * row;
        thread::scope(|s| {
            for (c, rows) in out.chunks_mut(chunk.max(1)).enumerate() {
                let raws = &raws;
                s.spawn(move || {
                    for (i, dst) in rows.chunks_mut(row.max(1)).enumerate() {
                        let r = (c * chunk / row.max(1) + i) as i64;
                        let mut at = 0;
                        for p in parts {
                            at += gather(&raws[p.tensor], p, r, e, &mut dst[at..]);
                        }
                    }
                });
            }
        });
        Ok(out)
    }

    fn floats(&self, t: &Tensor) -> Result<Vec<f32>, String> {
        let raw = self.raw(t)?;
        match t.dtype.as_str() {
            "F32" => Ok(raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()),
            "BF16" => Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                .collect()),
            d => Err(format!("a {d} tensor read as floats")),
        }
    }
}

fn header(file: &File, index: usize) -> Result<Vec<(String, Tensor)>, String> {
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let mut len = [0u8; 8];
    file.read_exact_at(&mut len, 0).map_err(|e| e.to_string())?;
    let len = u64::from_le_bytes(len);
    if 8 + len > size {
        return Err(format!("a header of {len} bytes"));
    }
    let mut head = vec![0u8; len as usize];
    file.read_exact_at(&mut head, 8).map_err(|e| e.to_string())?;
    let head: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&head).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (name, v) in head.into_iter().filter(|(n, _)| n != "__metadata__") {
        let t: Entry = serde_json::from_value(v).map_err(|e| format!("`{name}`: {e}"))?;
        let at = 8 + len + t.data_offsets[0]..8 + len + t.data_offsets[1];
        let bytes = t.shape.iter().product::<u64>() * elem(&t.dtype)? as u64;
        if at.end < at.start || at.end - at.start != bytes || at.end > size {
            return Err(format!("`{name}`: bytes {at:?}, its shape's {bytes}, the file's {size}"));
        }
        out.push((name, Tensor { file: index, dtype: t.dtype, shape: t.shape, at }));
    }
    Ok(out)
}

fn elem(dtype: &str) -> Result<usize, String> {
    Ok(match dtype {
        "F64" | "I64" => 8,
        "F32" | "I32" => 4,
        "BF16" | "F16" | "I16" => 2,
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "I8" | "U8" | "BOOL" => 1,
        d => return Err(format!("a stored {d}")),
    })
}

fn one<'a>(ts: &[&'a Tensor], i: usize) -> Result<&'a Tensor, String> {
    ts.get(i).copied().ok_or_else(|| format!("a fill of {} tensors read as having {}", ts.len(), i + 1))
}

fn same(ts: &[&Tensor]) -> Result<usize, String> {
    let t = ts.first().ok_or("a fill of no tensor")?;
    if let Some(o) = ts.iter().find(|o| o.dtype != t.dtype) {
        return Err(format!("a stack of {} and {}", t.dtype, o.dtype));
    }
    elem(&t.dtype)
}

fn permuted(b: Vec<u8>, rows: &[u64]) -> Result<Vec<u8>, String> {
    let row = b.len() / rows.len().max(1);
    let mut seen = vec![false; rows.len()];
    if row * rows.len() != b.len()
        || rows.iter().any(|&i| i as usize >= rows.len() || std::mem::replace(&mut seen[i as usize], true))
    {
        return Err(format!("{} bytes in rows permuted by {} indices", b.len(), rows.len()));
    }
    Ok(rows.iter().flat_map(|&i| &b[i as usize * row..(i as usize + 1) * row]).copied().collect())
}

fn widened(b: Vec<u8>, row: usize, width: usize) -> Result<Vec<u8>, String> {
    if row == 0 || width < row || !b.len().is_multiple_of(row) {
        return Err(format!("{} bytes in rows of {row} widened to {width}", b.len()));
    }
    Ok(b.chunks_exact(row)
        .flat_map(|r| r.iter().copied().chain(std::iter::repeat_n(0, width - row)))
        .collect())
}

fn rounded(b: &[u8], to: Dtype) -> Result<Vec<u8>, String> {
    let half = |x: f32| {
        let bits = x.to_bits();
        if x.is_nan() {
            ((bits >> 16) | 0x40) as u16
        } else {
            ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        }
    };
    let values = b.as_chunks::<4>().0.iter().map(|w| half(f32::from_le_bytes(*w)));
    match to {
        Dtype::Bf16 => Ok(values.flat_map(u16::to_le_bytes).collect()),
        Dtype::F32 => Ok(values.flat_map(|h| f32::from_bits(u32::from(h) << 16).to_le_bytes()).collect()),
        other => Err(format!("F32 values rounded to {other:?}")),
    }
}

fn bytes(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn fits(c: &Cut, n: usize) -> Result<(), String> {
    let ok = c.whole.iter().product::<u64>() as usize == n
        && c.dim < c.whole.len()
        && c.runs.iter().all(|r| r.start <= r.end && r.end <= c.whole[c.dim]);
    match ok {
        true => Ok(()),
        false => Err(format!("a whole of {n} elements cut as {c:?}")),
    }
}

fn part(whole: Vec<u8>, e: usize, cut: Option<&Cut>) -> Result<Vec<u8>, String> {
    let Some(c) = cut else { return Ok(whole) };
    fits(c, whole.len() / e)?;
    let inner = c.whole[c.dim + 1..].iter().product::<u64>() as usize * e;
    let (outer, len) = (c.whole[..c.dim].iter().product::<u64>() as usize, c.whole[c.dim] as usize);
    let mut out = Vec::new();
    for o in 0..outer {
        for r in &c.runs {
            out.extend_from_slice(
                &whole[(o * len + r.start as usize) * inner..(o * len + r.end as usize) * inner],
            );
        }
    }
    Ok(out)
}

fn reaches(p: &View, elements: u64) -> bool {
    let (lo, hi) = p.dims.iter().zip(&p.strides).fold((p.at as i128, p.at as i128), |(lo, hi), (&d, &s)| {
        let span = (d as i128 - 1) * s as i128;
        (lo + span.min(0), hi + span.max(0))
    });
    p.dims.len() == p.strides.len() && p.dims.iter().all(|&d| d > 0) && lo >= 0 && hi < elements as i128
}

fn gather(src: &[u8], p: &View, r: i64, e: usize, dst: &mut [u8]) -> usize {
    let (dims, strides) = (&p.dims[1..], &p.strides[1..]);
    let n: u64 = dims.iter().product();
    let (inner, step) = (dims.last().copied().unwrap_or(1) as usize, strides.last().copied().unwrap_or(1));
    let mut index = vec![0u64; dims.len()];
    let mut w = 0;
    for _ in 0..n / inner as u64 {
        let base = p.at as i64
            + r * p.strides[0]
            + index.iter().zip(strides).map(|(&i, &s)| i as i64 * s).sum::<i64>();
        match step {
            1 => {
                dst[w..w + inner * e].copy_from_slice(&src[base as usize * e..(base as usize + inner) * e]);
                w += inner * e;
            }
            _ => {
                for j in 0..inner as i64 {
                    let at = (base + j * step) as usize * e;
                    dst[w..w + e].copy_from_slice(&src[at..at + e]);
                    w += e;
                }
            }
        }
        for k in (0..dims.len().saturating_sub(1)).rev() {
            index[k] += 1;
            if index[k] < dims[k] {
                break;
            }
            index[k] = 0;
        }
    }
    w
}

fn transpose(
    src: &[u8],
    k: usize,
    cols: Range<usize>,
    dst: &mut [u8],
    stride: usize,
    col: usize,
    e: usize,
) -> Result<(), String> {
    fn tiles<W>(src: &[u8], k: usize, cols: Range<usize>, dst: &mut [u8], stride: usize, col: usize) {
        let (e, t) = (size_of::<W>(), 32);
        let rows = src.len() / (k * e);
        for k0 in cols.clone().step_by(t) {
            for j0 in (0..rows).step_by(t) {
                for kk in k0..(k0 + t).min(cols.end) {
                    let d = (kk - cols.start) * stride + col;
                    for j in j0..(j0 + t).min(rows) {
                        let s = (j * k + kk) * e;
                        dst[(d + j) * e..(d + j + 1) * e].copy_from_slice(&src[s..s + e]);
                    }
                }
            }
        }
    }
    match e {
        1 => tiles::<u8>(src, k, cols, dst, stride, col),
        2 => tiles::<u16>(src, k, cols, dst, stride, col),
        4 => tiles::<u32>(src, k, cols, dst, stride, col),
        8 => tiles::<u64>(src, k, cols, dst, stride, col),
        e => return Err(format!("a transpose of {e}-byte elements")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(tag: &str, files: &[&[(&str, &str, &[u64], Vec<u8>)]]) -> Checkpoint {
        let dir = std::env::temp_dir().join(format!("hyperserve-weights-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for (i, ts) in files.iter().enumerate() {
            let (mut head, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
            head.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
            for (name, dtype, shape, b) in ts.iter() {
                head.insert(name.to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + b.len()]}));
                data.extend(b);
            }
            let head = serde_json::to_vec(&head).unwrap();
            let mut f = (head.len() as u64).to_le_bytes().to_vec();
            f.extend(head);
            f.extend(data);
            fs::write(dir.join(format!("model-{i}.safetensors")), f).unwrap();
        }
        Checkpoint::open(&dir).unwrap()
    }

    fn cut(whole: &[u64], dim: usize, runs: &[Range<u64>]) -> Cut {
        Cut { whole: whole.to_vec(), dim, runs: runs.to_vec() }
    }

    fn names(n: &[&str]) -> Vec<String> {
        n.iter().map(|s| s.to_string()).collect()
    }

    fn u16s(v: impl IntoIterator<Item = u16>) -> Vec<u8> {
        v.into_iter().flat_map(u16::to_le_bytes).collect()
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    #[test]
    fn stack() {
        let c = checkpoint(
            "stack",
            &[&[("a", "U8", &[2, 3], (0..6).collect())], &[("b", "U8", &[1, 3], vec![10, 11, 12])]],
        );
        let t = names(&["a", "b"]);
        assert_eq!(c.fill(&Transform::Stack, &t, None).unwrap(), [0, 1, 2, 3, 4, 5, 10, 11, 12]);
        let rows = cut(&[3, 3], 0, &[2..3, 0..1]);
        assert_eq!(c.fill(&Transform::Stack, &t, Some(&rows)).unwrap(), [10, 11, 12, 0, 1, 2]);
        let span = cut(&[3, 3], 0, &[1..3]);
        assert_eq!(c.fill(&Transform::Stack, &t, Some(&span)).unwrap(), [3, 4, 5, 10, 11, 12]);
        let cols = cut(&[3, 3], 1, &[2..3, 0..1]);
        assert_eq!(c.fill(&Transform::Stack, &t, Some(&cols)).unwrap(), [2, 0, 5, 3, 12, 10]);
        let g = checkpoint("stack-groups", &[&[("w", "BF16", &[6, 2], u16s(0..12))]]);
        let groups = cut(&[3, 2, 2], 0, &[1..2]);
        assert_eq!(g.fill(&Transform::Stack, &names(&["w"]), Some(&groups)).unwrap(), u16s(4..8));
        let inner = cut(&[3, 2, 2], 2, &[1..2]);
        assert_eq!(
            g.fill(&Transform::Stack, &names(&["w"]), Some(&inner)).unwrap(),
            u16s([1, 3, 5, 7, 9, 11])
        );
    }

    #[test]
    fn a_laid_fill_permutes_its_rows_and_rounds_its_floats() {
        let v = [1.0f32, 1.00390625, -2.5, 3.0e-3, 7.0, 0.5];
        let c =
            checkpoint("laid", &[&[("w", "F32", &[3, 2], v.iter().flat_map(|x| x.to_le_bytes()).collect())]]);
        let w = names(&["w"]);
        let bf16 = |x: f32| ((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) as u16;
        let rows = [2u64, 0, 1];
        let want: Vec<u16> = [4, 5, 0, 1, 2, 3].iter().map(|&i| bf16(v[i])).collect();
        let laid = |prefix: Option<u64>, rows: Option<&[u64]>, round: Option<Dtype>| Laid {
            prefix,
            width: None,
            rows: rows.map(<[u64]>::to_vec),
            round,
        };
        assert_eq!(
            c.laid(&Transform::Stack, &w, None, &laid(None, Some(&rows), Some(Dtype::Bf16))).unwrap(),
            u16s(want.clone())
        );
        let widened: Vec<f32> = want.iter().map(|&h| f32::from_bits(u32::from(h) << 16)).collect();
        assert_eq!(
            f32s(&c.laid(&Transform::Stack, &w, None, &laid(None, Some(&rows), Some(Dtype::F32))).unwrap()),
            widened
        );
        assert_eq!(
            c.laid(&Transform::Stack, &w, None, &Laid::default()).unwrap(),
            c.fill(&Transform::Stack, &w, None).unwrap()
        );
        assert!(c.laid(&Transform::Stack, &w, None, &laid(None, Some(&[0, 0, 1]), None)).is_err());
        assert!(
            c.laid(&Transform::Stack, &w, Some(&cut(&[3, 2], 0, &[0..1])), &laid(None, Some(&rows), None))
                .is_err()
        );
        let first = c
            .laid(
                &Transform::Stack,
                &w,
                Some(&cut(&[3, 2], 0, &[0..1])),
                &laid(None, None, Some(Dtype::Bf16)),
            )
            .unwrap();
        assert_eq!(first, u16s([bf16(v[0]), bf16(v[1])]));
        let padded: Vec<f32> = [v[0], v[2], v[4], v[1], v[3], v[5], 0.0, 0.0, 0.0].to_vec();
        assert_eq!(
            f32s(&c.laid(&Transform::Transpose, &w, None, &laid(Some(3), None, None)).unwrap()),
            padded
        );
        assert_eq!(
            f32s(&c.laid(&Transform::Transpose, &w, None, &laid(Some(1), None, None)).unwrap()),
            padded[..3].to_vec()
        );
        let wide = |width: u64, round: Option<Dtype>| Laid { width: Some(width), round, ..Laid::default() };
        let widened: Vec<f32> = [v[0], v[2], v[4], 0.0, 0.0, v[1], v[3], v[5], 0.0, 0.0].to_vec();
        assert_eq!(f32s(&c.laid(&Transform::Transpose, &w, None, &wide(5, None)).unwrap()), widened);
        assert_eq!(
            c.laid(&Transform::Transpose, &w, None, &wide(5, Some(Dtype::Bf16))).unwrap(),
            u16s(widened.iter().map(|&x| bf16(x)))
        );
        assert!(c.laid(&Transform::Transpose, &w, None, &wide(2, None)).is_err());
    }

    #[test]
    fn band() {
        let (groups, taps, rows, samples) = (2, 3, 4, 8);
        let h: Vec<u16> = (0..groups * taps).map(|i| 100 + i as u16).collect();
        let c = checkpoint("band", &[&[("h", "BF16", &[groups as u64, 1, taps as u64], u16s(h.clone()))]]);
        let mut want = vec![0u16; groups * samples * rows];
        for g in 0..groups {
            for s in 0..samples {
                for t in 0..rows {
                    if s >= t && s - t < taps {
                        want[(g * samples + s) * rows + t] = h[g * taps + taps - 1 - (s - t)];
                    }
                }
            }
        }
        let band = |rows: u64, samples: u64| c.fill(&Transform::Band { rows, samples }, &names(&["h"]), None);
        assert_eq!(band(rows as u64, samples as u64).unwrap(), u16s(want));
        assert!(band(rows as u64, 5).is_err());
    }

    #[test]
    fn transpose() {
        let c = checkpoint(
            "transpose",
            &[&[
                ("a", "BF16", &[2, 3], u16s([0, 1, 2, 10, 11, 12])),
                ("b", "BF16", &[1, 3], u16s([20, 21, 22])),
            ]],
        );
        let t = names(&["a", "b"]);
        assert_eq!(c.fill(&Transform::Transpose, &t, None).unwrap(), u16s([0, 10, 20, 1, 11, 21, 2, 12, 22]));
        let outs = cut(&[3, 3], 1, &[1..3]);
        assert_eq!(c.fill(&Transform::Transpose, &t, Some(&outs)).unwrap(), u16s([10, 20, 11, 21, 12, 22]));
        let ins = cut(&[3, 3], 0, &[2..3, 0..1]);
        assert_eq!(c.fill(&Transform::Transpose, &t, Some(&ins)).unwrap(), u16s([2, 12, 22, 0, 10, 20]));
        let at = |o: u16, i: u16| 100 * o + i;
        let big = checkpoint(
            "transpose-tiles",
            &[&[
                ("p", "BF16", &[70, 40], u16s((0..70).flat_map(|o| (0..40).map(move |i| at(o, i))))),
                ("q", "BF16", &[30, 40], u16s((70..100).flat_map(|o| (0..40).map(move |i| at(o, i))))),
            ]],
        );
        let t = names(&["p", "q"]);
        assert_eq!(
            big.fill(&Transform::Transpose, &t, None).unwrap(),
            u16s((0..40).flat_map(|i| (0..100).map(move |o| at(o, i))))
        );
        let outs = cut(&[40, 100], 1, &[60..80, 0..5]);
        let want = (0..40).flat_map(|i| (60..80).chain(0..5).map(move |o| at(o, i)));
        assert_eq!(big.fill(&Transform::Transpose, &t, Some(&outs)).unwrap(), u16s(want));
        let ins = cut(&[40, 100], 0, &[33..40, 1..2]);
        let want = (33..40).chain(1..2).flat_map(|i| (0..100).map(move |o| at(o, i)));
        assert_eq!(big.fill(&Transform::Transpose, &t, Some(&ins)).unwrap(), u16s(want));
    }

    #[test]
    fn vector() {
        let c = checkpoint(
            "vector",
            &[&[
                ("v", "BF16", &[2], u16s([0x3f80, 0xc000])),
                ("f", "F32", &[1], 0.5f32.to_le_bytes().to_vec()),
            ]],
        );
        let t = names(&["v"]);
        let n6 = Transform::VectorF32 { n: 6 };
        assert_eq!(f32s(&c.fill(&n6, &t, None).unwrap()), [1., -2., 1., -2., 1., -2.]);
        assert_eq!(f32s(&c.fill(&n6, &t, Some(&cut(&[3, 2], 1, &[1..2]))).unwrap()), [-2., -2., -2.]);
        assert_eq!(f32s(&c.fill(&n6, &t, Some(&cut(&[3, 2], 0, &[1..2]))).unwrap()), [1., -2.]);
        assert_eq!(f32s(&c.fill(&Transform::VectorF32 { n: 3 }, &names(&["f"]), None).unwrap()), [0.5; 3]);
        assert!(c.fill(&Transform::VectorF32 { n: 3 }, &t, None).is_err());
        assert_eq!(f32s(&c.fill(&Transform::OnePlusF32 { n: 4 }, &t, None).unwrap()), [2.0, -1.0, 2.0, -1.0]);
        assert_eq!(f32s(&c.fill(&Transform::OnePlusF32 { n: 1 }, &names(&["f"]), None).unwrap()), [1.5]);
        assert!(c.fill(&Transform::OnePlusF32 { n: 3 }, &t, None).is_err());
    }

    #[test]
    fn hc_mix() {
        let v: Vec<u8> = (0..8)
            .flat_map(|m| (0..4).map(move |i| (10 * m + i) as f32))
            .flat_map(f32::to_le_bytes)
            .collect();
        let c = checkpoint("hc", &[&[("hc", "F32", &[8, 4], v)]]);
        let got = f32s(&c.fill(&Transform::HcMix { d: 2 }, &names(&["hc"]), None).unwrap());
        let want: Vec<f32> =
            [0, 2, 1, 3].iter().flat_map(|&i| (0..8).map(move |m| (10 * m + i) as f32)).collect();
        assert_eq!(got, want);
        let rows = cut(&[4, 8], 0, &[1..2]);
        assert_eq!(
            f32s(&c.fill(&Transform::HcMix { d: 2 }, &names(&["hc"]), Some(&rows)).unwrap()),
            want[8..16]
        );
        assert!(c.fill(&Transform::HcMix { d: 4 }, &names(&["hc"]), None).is_err());
    }

    #[test]
    fn product() {
        let b: Vec<u8> = [2f32, -3.].iter().flat_map(|x| x.to_le_bytes()).collect();
        let c = checkpoint(
            "product",
            &[&[("a", "BF16", &[1, 2], u16s([0x3fc0, 0x4000])), ("b", "F32", &[1, 2], b)]],
        );
        assert_eq!(f32s(&c.fill(&Transform::ProductF32, &names(&["a", "b"]), None).unwrap()), [3., -6.]);
        assert_eq!(
            f32s(
                &c.fill(&Transform::ProductF32, &names(&["a", "b"]), Some(&cut(&[1, 2], 1, &[1..2])))
                    .unwrap()
            ),
            [-6.]
        );
    }

    #[test]
    fn columns() {
        let c = checkpoint("columns", &[&[("w", "U8", &[2, 6], (0..12).collect())]]);
        assert_eq!(c.fill(&Transform::Columns { j: 1, of: 3 }, &names(&["w"]), None).unwrap(), [2, 3, 8, 9]);
        assert_eq!(
            c.fill(&Transform::Columns { j: 2, of: 3 }, &names(&["w"]), Some(&cut(&[2, 2], 0, &[1..2])))
                .unwrap(),
            [10, 11]
        );
        assert!(c.fill(&Transform::Columns { j: 0, of: 4 }, &names(&["w"]), None).is_err());
    }

    #[test]
    fn experts() {
        let at = |x: u8, p: u8, c: u8| 100u8.wrapping_mul(x) + 10 * p + c;
        let ts: Vec<(String, Vec<u8>)> = (0..3)
            .flat_map(|x| {
                (0..2)
                    .map(move |p| (format!("e.{x}.w{}", [1, 3][p as usize]), vec![at(x, p, 0), at(x, p, 1)]))
            })
            .collect();
        let ts: Vec<(&str, &str, &[u64], Vec<u8>)> =
            ts.iter().map(|(n, b)| (n.as_str(), "U8", &[1u64, 2][..], b.clone())).collect();
        let c = checkpoint("experts", &[&ts[..3], &ts[3..]]);
        let (e, t) = (Transform::Experts { experts: 3 }, names(&["e.{x}.w1", "e.{x}.w3"]));
        let all: Vec<u8> =
            (0..3).flat_map(|x| (0..2).flat_map(move |p| (0..2).map(move |c| at(x, p, c)))).collect();
        assert_eq!(c.fill(&e, &t, None).unwrap(), all);
        assert_eq!(
            c.fill(&e, &t, Some(&cut(&[3, 2, 2], 0, &[2..3, 0..1]))).unwrap(),
            [&all[8..12], &all[0..4]].concat()
        );
        assert_eq!(
            c.fill(&e, &t, Some(&cut(&[3, 2, 2], 1, &[1..2]))).unwrap(),
            [at(0, 1, 0), at(0, 1, 1), at(1, 1, 0), at(1, 1, 1), at(2, 1, 0), at(2, 1, 1)]
        );
        assert_eq!(
            c.fill(&e, &t, Some(&cut(&[3, 2, 2], 2, &[1..2]))).unwrap(),
            [at(0, 0, 1), at(0, 1, 1), at(1, 0, 1), at(1, 1, 1), at(2, 0, 1), at(2, 1, 1)]
        );
        assert!(c.fill(&Transform::Experts { experts: 4 }, &t, None).is_err());
    }

    #[test]
    fn view() {
        let c = checkpoint(
            "view",
            &[&[("w", "BF16", &[6, 2], u16s(0..12)), ("h", "BF16", &[2, 3], u16s([20, 21, 22, 23, 24, 25]))]],
        );
        let v = |tensor, at, dims: &[u64], strides: &[i64]| View {
            tensor,
            at,
            dims: dims.to_vec(),
            strides: strides.to_vec(),
        };
        let t = names(&["w", "h"]);
        let fill = |parts: Vec<View>| c.fill(&Transform::View { parts }, &t, None);
        assert_eq!(
            fill(vec![v(0, 0, &[2, 6], &[1, 2])]).unwrap(),
            u16s([0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11])
        );
        let heads = |part: u64| v(0, part * 2, &[2, 2, 1], &[1, 6, 2]);
        assert_eq!(fill((0..3).map(heads).collect()).unwrap(), u16s([0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11]));
        assert_eq!(fill(vec![v(1, 2, &[3, 2], &[-1, 3])]).unwrap(), u16s([22, 25, 21, 24, 20, 23]));
        assert_eq!(
            fill(vec![v(1, 0, &[2, 3], &[3, 1]), v(0, 0, &[2, 1], &[1, 1])]).unwrap(),
            u16s([20, 21, 22, 0, 23, 24, 25, 1])
        );
        assert!(fill(vec![v(1, 0, &[3, 3], &[3, 1])]).is_err());
        assert!(fill(vec![v(1, 0, &[2, 3], &[-1, 1])]).is_err());
        assert!(fill(vec![v(1, 0, &[2, 3], &[3, 1]), v(0, 0, &[3, 1], &[1, 1])]).is_err());
        assert!(
            c.fill(
                &Transform::View { parts: vec![v(0, 0, &[2, 1], &[1, 1])] },
                &t,
                Some(&cut(&[2], 0, &[0..1]))
            )
            .is_err()
        );
    }

    #[test]
    fn span() {
        let c = checkpoint("span", &[&[("a", "U8", &[4], vec![1; 4]), ("b", "U8", &[2, 3], vec![2; 6])]]);
        let (path, at) = c.span("b").unwrap();
        let f = fs::read(&path).unwrap();
        let head = 8 + u64::from_le_bytes(f[..8].try_into().unwrap());
        assert_eq!(at, head + 4..head + 10);
        assert_eq!(f[at.start as usize..at.end as usize], [2; 6]);
        assert!(c.span("c").is_err());
    }
}
