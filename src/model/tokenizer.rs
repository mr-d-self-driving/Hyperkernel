use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use fancy_regex::Regex;
use serde_json::Value;

#[derive(Clone, Copy, Debug)]
pub struct TokenizerConfig {
    pub memo_min: usize,
    pub memo_cap: usize,
}

impl Default for TokenizerConfig {
    fn default() -> TokenizerConfig {
        TokenizerConfig { memo_min: 256, memo_cap: 1 << 30 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Special {
    Skip,
    Keep,
}

#[derive(Clone, Default)]
struct Token {
    bytes: Vec<u8>,
    special: bool,
}

struct Added {
    content: String,
    id: u32,
}

pub struct Tokenizer {
    tokens: Vec<Token>,
    byte_ids: [u32; 256],
    merges: HashMap<(u32, u32), (u32, u32)>,
    added: Vec<Vec<Added>>,
    splits: Vec<Regex>,
    bos: Option<u32>,
    memo_min: usize,
    memo: Mutex<Memo>,
}

fn fail<T>(why: impl Into<String>) -> Result<T, String> {
    Err(format!("tokenizer: {}", why.into()))
}

fn gpt2_chars() -> [char; 256] {
    let mut chars = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..256u32 {
        let printable =
            (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable {
            b
        } else {
            next += 1;
            next - 1
        };
        chars[b as usize] = char::from_u32(c).expect("a byte character");
    }
    chars
}

fn splits(pre: &Value) -> Result<Vec<Regex>, String> {
    let regex = |pattern: &str| Regex::new(pattern).or_else(|e| fail(format!("pattern {pattern:?}: {e}")));
    match pre["type"].as_str() {
        None if pre.is_null() => Ok(Vec::new()),
        Some("Sequence") => {
            let mut out = Vec::new();
            for p in pre["pretokenizers"].as_array().map_or(&[][..], Vec::as_slice) {
                out.extend(splits(p)?);
            }
            Ok(out)
        }
        Some("Split") => {
            if pre["behavior"] != "Isolated" || pre["invert"] == true {
                return fail(format!(
                    "Split behavior {} invert {} (Isolated, not inverted, only)",
                    pre["behavior"], pre["invert"]
                ));
            }
            match (pre["pattern"]["Regex"].as_str(), pre["pattern"]["String"].as_str()) {
                (Some(p), _) => Ok(vec![regex(p)?]),
                (None, Some(p)) => Ok(vec![regex(&fancy_regex::escape(p))?]),
                (None, None) => fail("Split without a pattern"),
            }
        }
        Some("ByteLevel") if pre["add_prefix_space"] == true => fail("ByteLevel add_prefix_space"),
        Some("ByteLevel") if pre["use_regex"] == true => {
            Ok(vec![regex(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")?])
        }
        Some("ByteLevel") => Ok(Vec::new()),
        other => fail(format!("pre-tokenizer {other:?}")),
    }
}

fn leading_special(post: &Value) -> Option<u32> {
    match post["type"].as_str()? {
        "Sequence" => post["processors"].as_array()?.iter().find_map(leading_special),
        "TemplateProcessing" => {
            let name = post["single"].as_array()?.first()?["SpecialToken"]["id"].as_str()?;
            post["special_tokens"][name]["ids"].as_array()?.first()?.as_u64().map(|id| id as u32)
        }
        _ => None,
    }
}

fn merge_pair(m: &Value) -> Result<(String, String), String> {
    match m {
        Value::String(s) => s
            .split_once(' ')
            .map(|(l, r)| (l.to_owned(), r.to_owned()))
            .ok_or_else(|| format!("tokenizer: merge {s:?}")),
        Value::Array(p) if p.len() == 2 => {
            Ok((p[0].as_str().unwrap_or("").to_owned(), p[1].as_str().unwrap_or("").to_owned()))
        }
        _ => fail(format!("merge {m}")),
    }
}

impl Tokenizer {
    pub fn open(path: &Path, config: &TokenizerConfig) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            format!(
                "--tokenizer {}: {e} (the checkpoint's tokenizer.json, or pass --tokenizer FILE)",
                path.display()
            )
        })?;
        Tokenizer::from_json(&text, config).map_err(|e| format!("--tokenizer {}: {e}", path.display()))
    }

    pub fn from_json(text: &str, config: &TokenizerConfig) -> Result<Tokenizer, String> {
        let json: Value = serde_json::from_str(text).or_else(|e| fail(format!("tokenizer.json: {e}")))?;
        let model = &json["model"];
        if model["type"] != "BPE" {
            return fail(format!("model type {} (BPE only)", model["type"]));
        }
        let normalizer = &json["normalizer"];
        if !(normalizer.is_null()
            || (normalizer["type"] == "Sequence"
                && normalizer["normalizers"].as_array().is_some_and(Vec::is_empty)))
        {
            return fail(format!("normalizer {} (none only)", normalizer["type"]));
        }
        let Some(vocab) = model["vocab"].as_object() else { return fail("model.vocab is not a map") };
        let id_of = |token: &str| vocab.get(token).and_then(Value::as_u64).map(|id| id as u32);
        let chars = gpt2_chars();
        let char_bytes: HashMap<char, u8> = chars.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let bytes_of = |token: &str| {
            token
                .chars()
                .map(|c| char_bytes.get(&c).copied())
                .collect::<Option<Vec<u8>>>()
                .unwrap_or_else(|| token.as_bytes().to_vec())
        };

        let mut tokens: Vec<Token> = Vec::new();
        let mut put = |id: u32, token: Token| {
            if tokens.len() <= id as usize {
                tokens.resize(id as usize + 1, Token::default());
            }
            tokens[id as usize] = token;
        };
        for (token, id) in vocab {
            let Some(id) = id.as_u64() else { return fail(format!("vocab {token:?}: not an id")) };
            put(id as u32, Token { bytes: bytes_of(token), special: false });
        }
        let mut added: Vec<Vec<Added>> = (0..256).map(|_| Vec::new()).collect();
        for a in json["added_tokens"].as_array().map_or(&[][..], Vec::as_slice) {
            let (Some(content), Some(id)) = (a["content"].as_str(), a["id"].as_u64()) else {
                return fail(format!("added token {a}"));
            };
            put(id as u32, Token { bytes: bytes_of(content), special: a["special"] == true });
            if let Some(&b0) = content.as_bytes().first() {
                added[b0 as usize].push(Added { content: content.to_owned(), id: id as u32 });
            }
        }
        for list in &mut added {
            list.sort_by_key(|a| Reverse(a.content.len()));
        }

        let mut byte_ids = [0; 256];
        for (b, c) in chars.iter().enumerate() {
            byte_ids[b] =
                id_of(&c.to_string()).ok_or_else(|| format!("tokenizer: byte {b:#04x} has no token"))?;
        }
        let mut merges = HashMap::new();
        for (rank, m) in model["merges"].as_array().map_or(&[][..], Vec::as_slice).iter().enumerate() {
            let (l, r) = merge_pair(m)?;
            let (Some(li), Some(ri), Some(mi)) = (id_of(&l), id_of(&r), id_of(&format!("{l}{r}"))) else {
                return fail(format!("merge {l:?} {r:?}: a token not in the vocabulary"));
            };
            merges.entry((li, ri)).or_insert((rank as u32, mi));
        }
        Ok(Tokenizer {
            tokens,
            byte_ids,
            merges,
            added,
            splits: splits(&json["pre_tokenizer"])?,
            bos: leading_special(&json["post_processor"]),
            memo_min: config.memo_min,
            memo: Mutex::new(Memo { cap: config.memo_cap, ..Memo::default() }),
        })
    }

    pub fn id(&self, content: &str) -> Option<u32> {
        let first = *content.as_bytes().first()?;
        self.added[first as usize].iter().find(|a| a.content == content).map(|a| a.id)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn encode(&self, text: &str, bos: bool) -> Result<Vec<u32>, String> {
        let mut out: Vec<u32> = self.bos.filter(|_| bos).into_iter().collect();
        let bytes = text.as_bytes();
        let (mut plain, mut at) = (0, 0);
        while at < bytes.len() {
            match self.added[bytes[at] as usize]
                .iter()
                .find(|a| bytes[at..].starts_with(a.content.as_bytes()))
            {
                Some(a) => {
                    out.extend(self.encode_plain(&text[plain..at])?);
                    out.push(a.id);
                    at += a.content.len();
                    plain = at;
                }
                None => at += 1,
            }
        }
        out.extend(self.encode_plain(&text[plain..])?);
        Ok(out)
    }

    pub fn bytes(&self, id: u32, special: Special) -> &[u8] {
        match self.tokens.get(id as usize) {
            Some(t) if !(t.special && special == Special::Skip) => &t.bytes,
            Some(_) | None => &[],
        }
    }

    pub fn decode(&self, ids: &[u32], special: Special) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.bytes(id, special)).copied().collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<u32>, String> {
        if text.len() < self.memo_min {
            return self.encode_pieces(text);
        }
        if let Some(ids) = self.memo().get(text) {
            return Ok(ids);
        }
        let ids = self.encode_pieces(text)?;
        self.memo().put(text.to_owned(), ids.clone());
        Ok(ids)
    }

    fn memo(&self) -> std::sync::MutexGuard<'_, Memo> {
        self.memo.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn encode_pieces(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut pieces = vec![text];
        for re in &self.splits {
            let mut next = Vec::new();
            for piece in pieces {
                let mut last = 0;
                for m in re.find_iter(piece) {
                    let m = m.or_else(|e| fail(format!("split: {e}")))?;
                    next.extend([&piece[last..m.start()], m.as_str()]);
                    last = m.end();
                }
                next.push(&piece[last..]);
            }
            pieces = next.into_iter().filter(|p| !p.is_empty()).collect();
        }
        Ok(pieces.into_iter().flat_map(|p| self.merge(p)).collect())
    }

    fn merge(&self, piece: &str) -> Vec<u32> {
        let mut word: Vec<u32> = piece.bytes().map(|b| self.byte_ids[b as usize]).collect();
        let n = word.len();
        if n < 2 {
            return word;
        }
        let (mut prev, mut next): (Vec<usize>, Vec<usize>) =
            ((0..n).map(|i| i.wrapping_sub(1)).collect(), (1..=n).collect());
        let pair = |word: &[u32], l: usize, r: usize| {
            self.merges.get(&(word[l], word[r])).map(|&(rank, id)| Reverse((rank, l, r, id)))
        };
        let mut heap: BinaryHeap<_> = (1..n).filter_map(|r| pair(&word, r - 1, r)).collect();
        while let Some(Reverse((rank, l, r, id))) = heap.pop() {
            if next[l] != r || pair(&word, l, r).is_none_or(|Reverse(p)| p.0 != rank) {
                continue;
            }
            word[l] = id;
            next[l] = next[r];
            next[r] = usize::MAX;
            if next[l] < n {
                prev[next[l]] = l;
                heap.extend(pair(&word, l, next[l]));
            }
            if prev[l] < n {
                heap.extend(pair(&word, prev[l], l));
            }
        }
        std::iter::successors(Some(0), |&i| Some(next[i]).filter(|&j| j < n)).map(|i| word[i]).collect()
    }
}

#[derive(Default)]
struct Memo {
    new: HashMap<String, Vec<u32>>,
    old: HashMap<String, Vec<u32>>,
    new_bytes: usize,
    cap: usize,
}

impl Memo {
    fn get(&mut self, text: &str) -> Option<Vec<u32>> {
        if let Some(ids) = self.new.get(text) {
            return Some(ids.clone());
        }
        let (text, ids) = self.old.remove_entry(text)?;
        self.put(text, ids.clone());
        Some(ids)
    }

    fn put(&mut self, text: String, ids: Vec<u32>) {
        let bytes = text.len() + 4 * ids.len();
        if self.new_bytes + bytes > self.cap / 2 {
            self.old = std::mem::take(&mut self.new);
            self.new_bytes = 0;
        }
        if self.new.insert(text, ids).is_none() {
            self.new_bytes += bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draws(seed: u64) -> impl Iterator<Item = usize> {
        std::iter::successors(Some(seed), |x| {
            Some(x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
        })
        .skip(1)
        .map(|x| (x >> 33) as usize)
    }

    fn abcde() -> Tokenizer {
        let mut symbols: Vec<String> = "abcde".chars().map(String::from).collect();
        let mut merges: Vec<String> =
            (0..25).map(|k| (k * 7) % 25).map(|p| format!("{} {}", symbols[p / 5], symbols[p % 5])).collect();
        symbols.extend(merges.iter().map(|m| m.replace(' ', "")));
        let mut pick = draws(7);
        while merges.len() < 300 {
            let (l, r) = (
                symbols[pick.next().unwrap() % symbols.len()].clone(),
                symbols[pick.next().unwrap() % symbols.len()].clone(),
            );
            if !symbols.contains(&format!("{l}{r}")) {
                merges.push(format!("{l} {r}"));
                symbols.push(format!("{l}{r}"));
            }
        }
        let mut vocab: serde_json::Map<String, Value> =
            gpt2_chars().iter().enumerate().map(|(i, c)| (c.to_string(), Value::from(i))).collect();
        for s in &symbols[5..] {
            vocab.insert(s.clone(), Value::from(vocab.len()));
        }
        Tokenizer::from_json(
            &serde_json::json!({ "model": { "type": "BPE", "vocab": vocab, "merges": merges } }).to_string(),
            &TokenizerConfig::default(),
        )
        .unwrap()
    }

    fn text(seed: u64, len: usize) -> String {
        draws(seed).take(len).map(|x| char::from(b"abcde"[x % 5])).collect()
    }

    fn scanned(tok: &Tokenizer, piece: &str) -> Vec<u32> {
        let mut word: Vec<u32> = piece.bytes().map(|b| tok.byte_ids[b as usize]).collect();
        loop {
            let best = word
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| tok.merges.get(&(w[0], w[1])).map(|&(rank, id)| (rank, i, id)))
                .min();
            let Some((_, i, id)) = best else { return word };
            word[i] = id;
            word.remove(i + 1);
        }
    }

    #[test]
    fn the_heap_merge_is_the_scan_merge() {
        let tok = abcde();
        let pieces =
            ["", "a", "aaaaaaaaaaa", "abababababab", "eeeeeeeeeeeeeeeeeeeee", "\u{1F600}abc\u{4E16}"]
                .map(String::from);
        for piece in pieces.into_iter().chain((0..300).map(|i| text(i, i as usize % 97 + 1))) {
            assert_eq!(tok.merge(&piece), scanned(&tok, &piece), "{piece:?}");
        }
        assert!(tok.merge(&text(1, 64)).len() < 40);
    }

    #[test]
    fn a_long_piece_merges_in_n_log_n() {
        let (tok, piece) = (abcde(), text(3, 1 << 20));
        let t = std::time::Instant::now();
        let ids = tok.merge(&piece);
        assert!(t.elapsed().as_secs() < 5, "{:?}", t.elapsed());
        assert_eq!(tok.decode(&ids, Special::Keep), piece);
    }
}
