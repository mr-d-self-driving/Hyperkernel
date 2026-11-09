use std::io::Write;
use std::mem;
use std::sync::mpsc::Receiver;

use serde_json::{Value, json};

use crate::api::http::{self, Response};
use crate::api::logprobs::Logprobs;
use crate::api::metrics::{Ended, Life};
use crate::api::{Server, unix_secs};
use crate::engine::{End, Out, Token};
use crate::error::Error;
use crate::model::detok::Detok;
use crate::model::reply::{Delta, Reader};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Whole,
    Events { usage: bool, continuous: bool },
}

pub enum Endpoint<'s> {
    Completion { echo: String },
    Chat(Reader<'s>),
}

pub struct Ask<'s> {
    pub endpoint: Endpoint<'s>,
    pub stream: Stream,
    pub detok: Detok<'s>,
    pub logprobs: Option<Logprobs<'s>>,
    pub prompt: Vec<u32>,
}

enum Finish {
    Stop(Value),
    Length,
}

#[derive(Default)]
struct Sent {
    deltas: usize,
    tokens: usize,
    chars: usize,
}

pub struct Reply<'s> {
    pub out: Receiver<Out>,
    server: &'s Server,
    ask: Ask<'s>,
    head: Value,
    deltas: Vec<Value>,
    tokens: Vec<Token>,
    sent: Sent,
    life: Life,
}

impl<'s> Reply<'s> {
    pub fn new(server: &'s Server, out: Receiver<Out>, ask: Ask<'s>, limit: usize) -> Reply<'s> {
        let (prefix, opening) = match &ask.endpoint {
            Endpoint::Completion { echo } => ("cmpl", (!echo.is_empty()).then(|| json!(echo))),
            Endpoint::Chat(_) => ("chatcmpl", Some(json!({ "role": "assistant", "content": "" }))),
        };
        let head = json!({ "id": server.id(prefix), "created": unix_secs(), "model": server.config.name });
        Reply {
            out,
            server,
            ask,
            head,
            deltas: opening.into_iter().collect(),
            tokens: Vec::new(),
            sent: Sent::default(),
            life: Life::new(limit),
        }
    }

    pub fn streams(&self) -> bool {
        self.ask.stream != Stream::Whole
    }

    pub fn start(&mut self, w: &mut impl Write) -> bool {
        !self.streams() || (http::write_events(w) && self.send(w, None))
    }

    pub fn write(&mut self, out: Out, w: &mut impl Write) -> bool {
        match out {
            Out::Seated { at, cached } => {
                self.life.seat(at, cached);
                true
            }
            Out::Token(t) => match self.token(t) {
                None => self.send(w, None),
                Some(stop) => self.end(Ok(Finish::Stop(json!(stop))), w),
            },
            Out::End(End::Stop) => {
                let reason = self
                    .tokens
                    .last()
                    .filter(|t| t.id != self.server.eos)
                    .map_or(Value::Null, |t| json!(t.id));
                self.end(Ok(Finish::Stop(reason)), w)
            }
            Out::End(End::Length) => self.end(Ok(Finish::Length), w),
            Out::End(End::Refused(why)) => {
                self.end(Err(Error::new(503, "server_error", format!("the request was refused: {why}"))), w)
            }
        }
    }

    fn token(&mut self, t: Token) -> Option<String> {
        let (text, stop) = self.ask.detok.push(t.id);
        self.tokens.push(t);
        self.server.requests.token(&self.life, &self.tokens);
        self.read(&text);
        stop
    }

    fn read(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.ask.endpoint {
            Endpoint::Completion { .. } => self.deltas.push(json!(text)),
            Endpoint::Chat(reader) => {
                let deltas = reader.push(text);
                self.chat_deltas(deltas);
            }
        }
    }

    fn chat_deltas(&mut self, deltas: Vec<Delta>) {
        for d in deltas {
            let delta = match d {
                Delta::Content(t) => json!({ "content": t }),
                Delta::Reasoning(t) => json!({ "reasoning_content": t }),
                Delta::Call { name, arguments } => {
                    let index = self.deltas.iter().filter(|d| d.get("tool_calls").is_some()).count();
                    json!({ "tool_calls": [{ "index": index, "id": self.server.id("chatcmpl-tool"), "type": "function", "function": { "name": name, "arguments": arguments } }] })
                }
            };
            self.deltas.push(delta);
        }
    }

    fn end(&mut self, end: Result<Finish, Error>, w: &mut impl Write) -> bool {
        let ended = match &end {
            Ok(Finish::Stop(_)) => Ended::Stop,
            Ok(Finish::Length) => Ended::Length,
            Err(_) => Ended::Error,
        };
        self.server.requests.end(ended, &self.life, self.ask.prompt.len(), &self.tokens);
        if end.is_ok() {
            let rest = self.ask.detok.finish();
            self.read(&rest);
            if let Endpoint::Chat(reader) = &mut self.ask.endpoint {
                let deltas = reader.finish();
                self.chat_deltas(deltas);
            }
            self.deltas.push(match self.ask.endpoint {
                Endpoint::Completion { .. } => json!(""),
                Endpoint::Chat(_) => json!({}),
            });
        }
        self.send(w, Some(end));
        false
    }

    fn send(&mut self, w: &mut impl Write, end: Option<Result<Finish, Error>>) -> bool {
        let Stream::Events { usage, continuous } = self.ask.stream else {
            match end {
                None => return true,
                Some(Ok(finish)) => Response::json(200, &self.whole(&finish)).write(w),
                Some(Err(e)) => Response::from(e).write(w),
            }
            return false;
        };
        let finish = match end {
            None => None,
            Some(Ok(finish)) => Some(finish),
            Some(Err(e)) => {
                http::write_event(w, &e.json());
                http::write_done(w);
                return false;
            }
        };
        let mut chunks = self.chunks(finish.as_ref());
        if continuous {
            chunks.iter_mut().for_each(|v| v["usage"] = self.usage());
        }
        let sent = chunks.iter().all(|v| http::write_event(w, v));
        if finish.is_none() {
            return sent;
        }
        if sent && usage {
            let mut v = self.event(false, json!([]));
            v["usage"] = self.usage();
            http::write_event(w, &v);
        }
        http::write_done(w);
        false
    }

    fn chunks(&mut self, finish: Option<&Finish>) -> Vec<Value> {
        let Sent { deltas: from, tokens, chars } = self.sent;
        if from == self.deltas.len() {
            return Vec::new();
        }
        let mut logprobs =
            if tokens < self.tokens.len() { self.logprobs(tokens, chars) } else { Value::Null };
        let last = self.deltas.len() - 1;
        let chunks = (from..=last)
            .map(|i| {
                self.event(
                    false,
                    json!([self.chunk(
                        &self.deltas[i],
                        mem::take(&mut logprobs),
                        finish.filter(|_| i == last).map(|f| self.finish(f))
                    )]),
                )
            })
            .collect();
        let text: usize =
            self.deltas[from..].iter().filter_map(Value::as_str).map(|t| t.chars().count()).sum();
        self.sent = Sent { deltas: self.deltas.len(), tokens: self.tokens.len(), chars: chars + text };
        chunks
    }

    fn whole(&self, finish: &Finish) -> Value {
        let logprobs = self.logprobs(0, 0);
        let (finish_reason, stop_reason) = self.finish(finish);
        let (choice, nulls) = match &self.ask.endpoint {
            Endpoint::Completion { .. } => {
                let text: String = self.deltas.iter().filter_map(Value::as_str).collect();
                let mut choice = completion_choice(&text, logprobs, Some((finish_reason, stop_reason)));
                choice["token_ids"] = json!(self.tokens.iter().map(|t| t.id).collect::<Vec<_>>());
                choice["prompt_token_ids"] = json!(self.ask.prompt);
                choice["prompt_logprobs"] = Value::Null;
                choice["routed_experts"] = Value::Null;
                (
                    choice,
                    &["service_tier", "system_fingerprint", "kv_transfer_params", "ec_transfer_params"][..],
                )
            }
            Endpoint::Chat(_) => {
                let choice = json!({ "index": 0, "message": message(&self.deltas), "logprobs": logprobs, "finish_reason": finish_reason, "stop_reason": stop_reason, "token_ids": null, "routed_experts": null });
                (
                    choice,
                    &[
                        "service_tier",
                        "system_fingerprint",
                        "prompt_logprobs",
                        "prompt_token_ids",
                        "prompt_text",
                        "kv_transfer_params",
                        "ec_transfer_params",
                    ][..],
                )
            }
        };
        let mut v = self.event(true, json!([choice]));
        if let Endpoint::Completion { .. } = self.ask.endpoint {
            v["prompt_token_ids"] = json!(self.ask.prompt);
        }
        v["usage"] = self.usage();
        v["usage"]["prompt_tokens_details"] = Value::Null;
        v["usage"]["completion_tokens_details"] = Value::Null;
        for &k in nulls {
            v[k] = Value::Null;
        }
        v
    }

    fn chunk(&self, delta: &Value, logprobs: Value, finish: Option<(&str, Value)>) -> Value {
        match self.ask.endpoint {
            Endpoint::Completion { .. } => {
                completion_choice(delta.as_str().unwrap_or_default(), logprobs, finish)
            }
            Endpoint::Chat(_) => {
                let mut choice = json!({ "index": 0, "delta": delta, "logprobs": logprobs, "finish_reason": null, "token_ids": null });
                if let Some((finish_reason, stop_reason)) = finish {
                    choice["finish_reason"] = json!(finish_reason);
                    choice["stop_reason"] = stop_reason;
                }
                choice
            }
        }
    }

    fn finish(&self, finish: &Finish) -> (&'static str, Value) {
        let called = self.deltas.iter().any(|d| d.get("tool_calls").is_some());
        match finish {
            Finish::Stop(reason) => (if called { "tool_calls" } else { "stop" }, reason.clone()),
            Finish::Length => ("length", Value::Null),
        }
    }

    fn logprobs(&self, from: usize, offset: usize) -> Value {
        match (&self.ask.logprobs, &self.ask.endpoint) {
            (None, _) => Value::Null,
            (Some(l), Endpoint::Completion { .. }) => l.completion(&self.tokens, from, offset),
            (Some(l), Endpoint::Chat(_)) => l.chat(&self.tokens, from),
        }
    }

    fn usage(&self) -> Value {
        let (prompt, made) = (self.ask.prompt.len(), self.tokens.len());
        json!({ "prompt_tokens": prompt, "completion_tokens": made, "total_tokens": prompt + made })
    }

    fn event(&self, whole: bool, choices: Value) -> Value {
        let object = match (&self.ask.endpoint, whole) {
            (Endpoint::Completion { .. }, _) => "text_completion",
            (Endpoint::Chat(_), true) => "chat.completion",
            (Endpoint::Chat(_), false) => "chat.completion.chunk",
        };
        let mut v = self.head.clone();
        v["object"] = json!(object);
        v["choices"] = choices;
        v
    }
}

fn completion_choice(text: &str, logprobs: Value, finish: Option<(&str, Value)>) -> Value {
    let (finish_reason, stop_reason) = finish.map_or((Value::Null, Value::Null), |(f, s)| (json!(f), s));
    json!({ "index": 0, "text": text, "logprobs": logprobs, "finish_reason": finish_reason, "stop_reason": stop_reason, "prompt_token_ids": null, "token_ids": null })
}

fn message(deltas: &[Value]) -> Value {
    let joined = |field: &str| {
        let text: String = deltas.iter().filter_map(|d| d[field].as_str()).collect();
        if text.is_empty() { Value::Null } else { json!(text) }
    };
    let calls: Vec<Value> = deltas
        .iter()
        .filter_map(|d| d["tool_calls"].as_array())
        .flatten()
        .map(|c| json!({ "id": c["id"], "type": c["type"], "function": c["function"] }))
        .collect();
    let mut message = json!({ "role": "assistant", "content": joined("content"), "refusal": null, "annotations": null, "audio": null, "function_call": null, "reasoning_content": joined("reasoning_content") });
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    message
}
