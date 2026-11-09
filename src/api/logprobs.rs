use serde_json::{Map, Value, json};

use crate::engine::Token;
use crate::model::tokenizer::{Special, Tokenizer};

pub struct Logprobs<'t> {
    pub tok: &'t Tokenizer,
    pub n: usize,
    pub as_ids: bool,
    pub floor: f32,
    pub context: usize,
}

impl Logprobs<'_> {
    pub fn completion(&self, tokens: &[Token], from: usize, offset: usize) -> Value {
        let (mut text_offset, mut token_logprobs, mut texts, mut top) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut at = offset;
        for i in from..tokens.len() {
            let entries: Map<String, Value> =
                self.entries(tokens, i).into_iter().map(|(text, x)| (text, json!(x))).collect();
            let (text, x) = self.sampled(tokens, i);
            text_offset.push(at);
            at += text.chars().count();
            token_logprobs.push(json!(x));
            texts.push(text);
            top.push(Value::Object(entries));
        }
        json!({ "text_offset": text_offset, "token_logprobs": token_logprobs, "tokens": texts, "top_logprobs": top })
    }

    pub fn chat(&self, tokens: &[Token], from: usize) -> Value {
        let one = |(text, x): (String, f64)| json!({ "token": text, "logprob": x, "bytes": text.as_bytes() });
        let content: Vec<Value> = (from..tokens.len())
            .map(|i| {
                let top: Vec<Value> = self.entries(tokens, i).into_iter().take(self.n).map(one).collect();
                let mut c = one(self.sampled(tokens, i));
                c["top_logprobs"] = json!(top);
                c
            })
            .collect();
        json!({ "content": content })
    }

    fn sampled(&self, tokens: &[Token], i: usize) -> (String, f64) {
        let t = &tokens[i];
        (self.text(tokens, i, t.id), self.floored(t.logprobs.first().map_or(self.floor, |e| e.1)))
    }

    fn entries(&self, tokens: &[Token], i: usize) -> Vec<(String, f64)> {
        let t = &tokens[i];
        let top = t.logprobs.iter().skip(1).take(self.n).filter(|&&(id, _)| id != t.id);
        std::iter::once(self.sampled(tokens, i))
            .chain(top.map(|&(id, x)| (self.text(tokens, i, id), self.floored(x))))
            .collect()
    }

    fn text(&self, tokens: &[Token], i: usize, id: u32) -> String {
        if self.as_ids {
            return format!("token_id:{id}");
        }
        let text = self.tok.decode(&[id], Special::Keep);
        if !text.ends_with('\u{FFFD}') {
            return text;
        }
        let context: Vec<u32> = tokens[i.saturating_sub(self.context)..i].iter().map(|t| t.id).collect();
        self.corrected(&context, id)
    }

    fn corrected(&self, context: &[u32], id: u32) -> String {
        let decode = |ids: &[u32]| self.tok.decode(ids, Special::Keep);
        for k in 1..=context.len() {
            let context = &context[context.len() - k..];
            let full = decode(&[context, &[id]].concat());
            if full.ends_with('\u{FFFD}') {
                continue;
            }
            let clean = context.len()
                - context.iter().rev().take_while(|&&c| decode(&[c]).ends_with('\u{FFFD}')).count();
            let prefix = decode(&context[..clean]);
            if let Some(rest) = full.strip_prefix(prefix.as_str()) {
                return rest.to_owned();
            }
            let common: usize =
                prefix.chars().zip(full.chars()).take_while(|(a, b)| a == b).map(|(a, _)| a.len_utf8()).sum();
            return full[common..].to_owned();
        }
        String::new()
    }

    fn floored(&self, x: f32) -> f64 {
        f64::from(x.max(self.floor))
    }
}
