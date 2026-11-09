use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use hyperserve::api::metrics::MetricsConfig;
use hyperserve::api::{Server, ServerConfig};
use hyperserve::engine::{End, Job, Metrics, Out, Sampling, StepMetrics, Token};
use hyperserve::model::tokenizer::{Tokenizer, TokenizerConfig};

fn eos_id() -> u32 {
    301
}

fn hello_world() -> &'static str {
    "Hello world, Hello world"
}

fn name() -> String {
    ServerConfig::default().name
}

fn tok() -> Tokenizer {
    Tokenizer::from_json(
        &fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer.json"))
            .unwrap(),
        &TokenizerConfig::default(),
    )
    .unwrap()
}

fn script(text: &str, eos: bool) -> Vec<u32> {
    let mut s = tok().encode(text, false).unwrap();
    if eos {
        s.push(eos_id());
    }
    s
}

#[derive(Clone, Debug)]
struct Seen {
    prompt: Vec<u32>,
    sampling: Sampling,
    limit: usize,
    stop: Vec<u32>,
    sent: usize,
}

struct Up {
    addr: SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Up {
    fn seen(&self, n: usize) -> Vec<Seen> {
        until("the engine's ends", || self.seen.lock().unwrap().len() >= n);
        self.seen.lock().unwrap().clone()
    }
}

fn up_with(script: Vec<u32>, context: usize, pace: Duration) -> Up {
    up_keyed(script, context, pace, None)
}

fn up_keyed(script: Vec<u32>, context: usize, pace: Duration, key: Option<&str>) -> Up {
    let (queue, jobs) = mpsc::channel::<Job>();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    let metrics = Arc::new(Metrics::new(MetricsConfig::default().iteration_tokens));
    let m = metrics.clone();
    thread::spawn(move || {
        let mut distinct: Vec<u32> = Vec::new();
        for &t in &script {
            if !distinct.contains(&t) {
                distinct.push(t);
            }
        }
        for j in jobs {
            let r = match j {
                Job::Run(r) => r,
                Job::Reset(answer) => {
                    answer.send(7).unwrap();
                    continue;
                }
            };
            m.record(StepMetrics {
                running: 1,
                prompt_tokens: r.prompt.len() as u64,
                ..StepMetrics::default()
            });
            let _ = r.out.send(Out::Seated { at: Instant::now(), cached: 0 });
            let mut sent = 0;
            let end = loop {
                if sent == r.limit {
                    break Some(End::Length);
                }
                thread::sleep(pace);
                let id = script[sent % script.len()];
                let mut logprobs = Vec::new();
                if r.sampling.logprobs > 0 {
                    logprobs.push((id, -0.5));
                    logprobs.push((id, -0.5));
                    logprobs.extend(
                        distinct.iter().filter(|&&t| t != id).enumerate().map(|(i, &t)| (t, -1.5 - i as f32)),
                    );
                }
                if r.out.send(Out::Token(Token { id, logprobs, at: Instant::now() })).is_err() {
                    break None;
                }
                sent += 1;
                m.record(StepMetrics { running: 1, generation_tokens: 1, ..StepMetrics::default() });
                if r.stop.contains(&id) {
                    break Some(End::Stop);
                }
            };
            m.record(StepMetrics::default());
            if let Some(end) = end {
                let _ = r.out.send(Out::End(end));
            }
            log.lock().unwrap().push(Seen {
                prompt: r.prompt,
                sampling: r.sampling,
                limit: r.limit,
                stop: r.stop,
                sent,
            });
        }
    });
    let server = Arc::new(
        Server::new(tok(), context, queue, metrics, key.map(str::to_owned), ServerConfig::default()).unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || server.serve(listener));
    Up { addr, seen }
}

fn up() -> Up {
    up_with(script(hello_world(), true), 64, Duration::ZERO)
}

fn send(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    (head.split(' ').nth(1).unwrap().parse().unwrap(), body.to_owned())
}

fn post(addr: SocketAddr, path: &str, body: Value) -> (u16, Value) {
    let (status, text) = send(addr, "POST", path, &body.to_string());
    (status, serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}")))
}

fn events(addr: SocketAddr, path: &str, body: Value) -> Vec<Value> {
    let (status, text) = send(addr, "POST", path, &body.to_string());
    assert_eq!(status, 200, "{text}");
    let data: Vec<&str> =
        text.split("\n\n").filter(|e| !e.is_empty()).map(|e| e.strip_prefix("data: ").unwrap()).collect();
    assert_eq!(data.last(), Some(&"[DONE]"));
    data[..data.len() - 1].iter().map(|e| serde_json::from_str(e).unwrap()).collect()
}

fn completions() -> &'static str {
    "/v1/completions"
}

fn chat() -> &'static str {
    "/v1/chat/completions"
}

#[test]
fn health_and_models() {
    let u = up();
    assert_eq!(send(u.addr, "GET", "/health", "").0, 200);
    let (status, body) = send(u.addr, "GET", "/v1/models", "");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(status, 200);
    assert_eq!(v["data"][0]["id"], name());
    assert_eq!(v["data"][0]["max_model_len"], 64);
    assert_eq!(send(u.addr, "GET", "/v2/nothing", "").0, 404);
    assert_eq!(send(u.addr, "GET", completions(), "").0, 405);
}

#[test]
fn completion_runs_to_eos() {
    let u = up();
    let (status, v) =
        post(u.addr, completions(), json!({ "model": name(), "prompt": "Hi", "temperature": 0 }));
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], hello_world());
    assert_eq!(
        (&v["choices"][0]["finish_reason"], &v["choices"][0]["stop_reason"]),
        (&json!("stop"), &Value::Null)
    );
    assert_eq!(
        v["usage"],
        json!({ "prompt_tokens": 3, "completion_tokens": 13, "total_tokens": 16, "prompt_tokens_details": null, "completion_tokens_details": null })
    );
    assert_eq!(v["choices"][0]["token_ids"].as_array().unwrap().last(), Some(&json!(eos_id())));
    assert_eq!(v["prompt_token_ids"], json!([300, b'H', b'i']));
    let seen = u.seen(1)[0].clone();
    assert_eq!(
        (seen.prompt, seen.limit, seen.stop),
        (vec![300, b'H' as u32, b'i' as u32], 61, vec![eos_id()])
    );
    assert_eq!(
        (seen.sampling.temperature, seen.sampling.top_p, seen.sampling.top_k, seen.sampling.logprobs),
        (0.0, 1.0, 0, 0)
    );
}

#[test]
fn completion_stops_at_max_tokens_and_echoes() {
    let u = up();
    let (_, v) = post(u.addr, completions(), json!({ "prompt": [300, 72], "max_tokens": 3, "echo": true }));
    assert_eq!(v["choices"][0]["text"], "<｜begin▁of▁sentence｜>HHello wor");
    assert_eq!(
        (&v["choices"][0]["finish_reason"], &v["choices"][0]["stop_reason"]),
        (&json!("length"), &Value::Null)
    );
    assert_eq!(v["usage"]["completion_tokens"], 3);
}

#[test]
fn completion_stream() {
    let u = up();
    let ev = events(
        u.addr,
        completions(),
        json!({ "prompt": "Hi", "stream": true, "stream_options": { "include_usage": true } }),
    );
    let text: String = ev.iter().filter_map(|e| e["choices"][0]["text"].as_str()).collect();
    assert_eq!(text, hello_world());
    assert!(ev.len() > 3, "{ev:?}");
    let n = ev.len();
    assert_eq!(ev[n - 2]["choices"][0]["finish_reason"], "stop");
    assert_eq!(ev[n - 1]["choices"], json!([]));
    assert_eq!(ev[n - 1]["usage"]["completion_tokens"], 13);
}

#[test]
fn stop_strings_and_stop_tokens() {
    let u = up();
    let choice = |body: Value| {
        let (status, v) = post(u.addr, completions(), body);
        assert_eq!(status, 200, "{v}");
        v
    };
    let v = choice(json!({ "prompt": "Hi", "stop": ["xyz", ", H"] }));
    assert_eq!(
        (&v["choices"][0]["text"], &v["choices"][0]["finish_reason"], &v["choices"][0]["stop_reason"]),
        (&json!("Hello world"), &json!("stop"), &json!(", H"))
    );
    assert_eq!(v["usage"]["completion_tokens"], 8);
    let v = choice(json!({ "prompt": "Hi", "stop": ", H", "include_stop_str_in_output": true }));
    assert_eq!(v["choices"][0]["text"], "Hello world, H");
    let wo = tok().encode(" wo", false).unwrap();
    assert_eq!(wo.len(), 1);
    let v = choice(json!({ "prompt": "Hi", "stop_token_ids": [wo[0]] }));
    assert_eq!((&v["choices"][0]["text"], &v["choices"][0]["stop_reason"]), (&json!("Hello"), &json!(wo[0])));
    assert_eq!(v["usage"]["completion_tokens"], 2);
    let v = choice(json!({ "prompt": "Hi", "ignore_eos": true, "max_tokens": 20 }));
    assert_eq!(
        (&v["choices"][0]["finish_reason"], &v["usage"]["completion_tokens"]),
        (&json!("length"), &json!(20))
    );
    let ev = events(u.addr, completions(), json!({ "prompt": "Hi", "stream": true, "stop": [" world"] }));
    let last = &ev.last().unwrap()["choices"][0];
    assert_eq!((&last["finish_reason"], &last["stop_reason"]), (&json!("stop"), &json!(" world")));
    assert_eq!(ev.iter().filter_map(|e| e["choices"][0]["text"].as_str()).collect::<String>(), "Hello");
    let msgs = json!([{ "role": "user", "content": "Hi" }]);
    let (_, v) = post(u.addr, chat(), json!({ "messages": msgs, "stop": [" world"] }));
    assert_eq!(
        (&v["choices"][0]["message"]["content"], &v["choices"][0]["stop_reason"]),
        (&json!("Hello"), &json!(" world"))
    );
    let ev = events(u.addr, chat(), json!({ "messages": msgs, "stream": true, "max_completion_tokens": 2 }));
    let last = &ev.last().unwrap()["choices"][0];
    assert_eq!((&last["finish_reason"], &last["stop_reason"]), (&json!("length"), &Value::Null));
}

#[test]
fn chat_and_chat_stream() {
    let u = up();
    let msgs = json!([{ "role": "system", "content": "s" }, { "role": "user", "content": [{ "type": "text", "text": "Hi" }] }]);
    let (status, v) =
        post(u.addr, chat(), json!({ "model": name(), "messages": msgs, "max_completion_tokens": 32 }));
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(
        v["choices"][0]["message"],
        json!({ "role": "assistant", "content": hello_world(), "refusal": null, "annotations": null, "audio": null, "function_call": null, "reasoning_content": null })
    );
    assert_eq!(v["usage"]["prompt_tokens"], 8);
    assert_eq!(u.seen(1)[0].limit, 32);

    let ev = events(u.addr, chat(), json!({ "messages": msgs, "stream": true }));
    assert_eq!(ev[0]["choices"][0]["delta"], json!({ "role": "assistant", "content": "" }));
    let text: String = ev.iter().filter_map(|e| e["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(text, hello_world());
    assert_eq!(
        ev.last().unwrap()["choices"][0],
        json!({ "index": 0, "delta": {}, "logprobs": null, "finish_reason": "stop", "stop_reason": null, "token_ids": null })
    );
    assert_eq!(ev.last().unwrap()["object"], "chat.completion.chunk");
}

#[test]
fn sampling_reaches_the_engine() {
    let u = up();
    let (s, v) = post(
        u.addr,
        completions(),
        json!({ "prompt": "Hi", "max_tokens": 2, "top_p": 0.9, "top_k": 5, "seed": 3 }),
    );
    assert_eq!(s, 200, "{v}");
    let (s, _) = post(
        u.addr,
        completions(),
        json!({ "prompt": "Hi", "max_tokens": 2, "temperature": 0.7, "top_k": -1, "seed": -1, "logprobs": 2 }),
    );
    assert_eq!(s, 200);
    let seen = u.seen(2);
    assert_eq!(seen[0].sampling, Sampling { temperature: 1.0, top_p: 0.9, top_k: 5, seed: 3, logprobs: 0 });
    assert_eq!(
        seen[1].sampling,
        Sampling { temperature: 0.7, top_p: 1.0, top_k: 0, seed: u64::MAX, logprobs: 2 }
    );
}

#[test]
fn refusals_name_the_field() {
    let u = up_with(script(hello_world(), true), 24, Duration::ZERO);
    let refused = |path: &str, body: Value, status: u16, param: Option<&str>| {
        let (s, v) = post(u.addr, path, body.clone());
        assert_eq!(s, status, "{body} -> {v}");
        assert_eq!(v["error"]["code"], status);
        assert!(v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()));
        if let Some(p) = param {
            assert_eq!(v["error"]["param"], p, "{body} -> {v}");
        }
    };
    for (field, value) in [
        ("temperature", json!(-1)),
        ("top_p", json!(0)),
        ("top_k", json!(17)),
        ("n", json!(2)),
        ("prompt_logprobs", json!(1)),
        ("presence_penalty", json!(0.5)),
        ("suffix", json!("x")),
        ("seed", json!("x")),
        ("stop", json!(3)),
    ] {
        let mut body = json!({ "prompt": "Hi" });
        body[field] = value;
        refused(completions(), body, 400, Some(field));
    }
    refused(completions(), json!({ "prompt": ["a", "b"] }), 400, Some("prompt"));
    refused(completions(), json!({ "prompt": [] }), 400, Some("prompt"));
    refused(completions(), json!({ "prompt": "Hi", "max_tokens": 0 }), 400, Some("max_tokens"));
    refused(completions(), json!({ "prompt": "Hi", "max_tokens": 30 }), 400, None);
    refused(completions(), json!({ "prompt": "x".repeat(30) }), 400, None);
    refused(completions(), json!({ "prompt": [999] }), 400, Some("prompt"));
    refused(completions(), json!({ "model": "other", "prompt": "Hi" }), 404, None);
    refused(
        completions(),
        json!({ "prompt": "Hi", "stream_options": { "include_usage": true } }),
        400,
        Some("stream_options"),
    );
    let user = json!([{ "role": "user", "content": "Hi" }]);
    refused(chat(), json!({ "messages": user, "top_logprobs": 2 }), 400, Some("top_logprobs"));
    refused(chat(), json!({ "messages": user, "tools": [{ "type": "function" }] }), 400, Some("tools"));
    refused(chat(), json!({ "messages": [{ "role": "tool" }] }), 400, Some("messages"));
    refused(
        chat(),
        json!({ "messages": [{ "role": "user", "content": [{ "type": "image_url" }] }] }),
        400,
        Some("messages"),
    );
    refused(
        chat(),
        json!({ "messages": user, "response_format": { "type": "json_object" } }),
        400,
        Some("response_format"),
    );
    refused(chat(), json!({ "messages": user, "functions": [{ "name": "f" }] }), 400, Some("functions"));
    assert_eq!(send(u.addr, "POST", completions(), "{not json").0, 400);
    let (s, v) = post(
        u.addr,
        completions(),
        json!({ "prompt": "Hi", "temperature": 0.0, "top_p": 1, "n": 1, "seed": 7, "max_tokens": 2, "logit_bias": {}, "stream_options": {} }),
    );
    assert_eq!(s, 200, "{v}");
    assert_eq!(u.seen(1).len(), 1);
}

#[test]
fn requests_queue_one_at_a_time() {
    let u = up_with(script(hello_world(), false), 64, Duration::ZERO);
    let addr = u.addr;
    let runs: Vec<_> = (0..6)
        .map(|_| {
            thread::spawn(move || post(addr, completions(), json!({ "prompt": "Hi", "max_tokens": 40 })))
        })
        .collect();
    for r in runs {
        let (s, v) = r.join().unwrap();
        assert_eq!(s, 200, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 40);
        assert!(v["choices"][0]["text"].as_str().unwrap().starts_with(hello_world()));
    }
}

fn tool_request(stream: bool) -> Value {
    json!({ "messages": [{ "role": "user", "content": "Weather?" }], "stream": stream, "chat_template_kwargs": { "thinking": true },
            "tools": [{ "type": "function", "function": { "name": "get_weather", "parameters": { "type": "object", "properties": { "city": { "type": "string" } } } } }] })
}

#[test]
fn tool_calls_whole_and_streamed() {
    let calling = "Weather first.</think>Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Oslo</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let u = up_with(script(calling, true), 8192, Duration::ZERO);
    let (status, v) = post(u.addr, chat(), tool_request(false));
    assert_eq!(status, 200, "{v}");
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(choice["message"]["reasoning_content"], "Weather first.");
    assert_eq!(choice["message"]["content"], "Checking.");
    let calls = choice["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(
        calls[0]["function"],
        json!({ "name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 3}" })
    );
    assert_eq!(calls[1]["function"]["arguments"], "{\"city\": \"Oslo\"}");
    assert_ne!(calls[0]["id"], calls[1]["id"]);

    let ev = events(u.addr, chat(), tool_request(true));
    let deltas: Vec<&Value> = ev.iter().map(|e| &e["choices"][0]["delta"]).collect();
    let text = |field: &str| deltas.iter().filter_map(|d| d[field].as_str()).collect::<String>();
    assert_eq!(text("reasoning_content"), "Weather first.");
    assert_eq!(text("content"), "Checking.");
    let calls: Vec<&Value> =
        deltas.iter().filter_map(|d| d["tool_calls"].as_array()).map(|c| &c[0]).collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        (calls[0]["index"].as_u64(), calls[0]["type"].as_str(), calls[0]["function"]["name"].as_str()),
        (Some(0), Some("function"), Some("get_weather"))
    );
    assert!(calls[0]["id"].is_string());
    assert_eq!(calls[0]["function"]["arguments"].as_str(), Some("{\"city\": \"Paris\", \"days\": 3}"));
    assert_eq!(calls[1]["index"], 1);
    assert_eq!(ev.last().unwrap()["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn tool_round_trip_renders_and_malformed_calls_are_content() {
    let broken = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\">1</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let u = up_with(script(broken, true), 8192, Duration::ZERO);
    let body = json!({ "messages": [
        { "role": "user", "content": "Weather?" },
        { "role": "assistant", "content": "", "tool_calls": [{ "id": "c0", "type": "function", "function": { "name": "f", "arguments": "{\"x\": 1}" } }] },
        { "role": "tool", "tool_call_id": "c0", "content": "sunny" }],
        "tools": [{ "type": "function", "function": { "name": "f" } }] });
    let (status, v) = post(u.addr, chat(), body);
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["choices"][0]["message"]["content"], broken);
    assert!(v["choices"][0]["message"]["tool_calls"].is_null());
}

fn choice_request(choice: Value, thinking: bool) -> Value {
    json!({ "messages": [{ "role": "user", "content": "Weather?" }], "tool_choice": choice, "chat_template_kwargs": { "thinking": thinking },
            "tools": [{ "type": "function", "function": { "name": "get_weather" } }] })
}

#[test]
fn tool_choice() {
    let rest = "<｜DSML｜ parameter name=\"city\" string=\"true\">Rome</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let u = up_with(script(rest, true), 8192, Duration::ZERO);
    let (status, v) = post(
        u.addr,
        chat(),
        choice_request(json!({ "type": "function", "function": { "name": "get_weather" } }), false),
    );
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert!(v["choices"][0]["message"]["content"].is_null());
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"],
        json!({ "name": "get_weather", "arguments": "{\"city\": \"Rome\"}" })
    );
    let u = up_with(script(&format!("get_weather\">\n{rest}"), true), 8192, Duration::ZERO);
    let (_, v) = post(u.addr, chat(), choice_request(json!("required"), false));
    assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather");
    let block = format!("\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n{rest}");
    let u = up_with(script(&block, true), 8192, Duration::ZERO);
    let (_, none) = post(u.addr, chat(), choice_request(json!("none"), false));
    let (_, auto) = post(u.addr, chat(), choice_request(json!("auto"), false));
    assert_eq!(none["choices"][0]["message"]["content"], block);
    assert_eq!(none["choices"][0]["finish_reason"], "stop");
    assert_eq!(auto["choices"][0]["finish_reason"], "tool_calls");
    assert!(none["usage"]["prompt_tokens"].as_u64() < auto["usage"]["prompt_tokens"].as_u64());

    let refused = |body: Value| {
        let (status, v) = post(u.addr, chat(), body);
        assert_eq!((status, &v["error"]["param"]), (400, &json!("tool_choice")), "{v}");
    };
    refused(choice_request(json!("required"), true));
    refused(choice_request(json!({ "type": "function", "function": { "name": "other" } }), false));
    refused(choice_request(json!("sometimes"), false));
    refused(json!({ "messages": [{ "role": "user", "content": "Hi" }], "tool_choice": "required" }));
}

fn refusal(status: u16, v: &Value) -> (String, Value) {
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["type"], "BadRequestError", "{v}");
    assert_eq!(v["error"]["code"], 400, "{v}");
    (v["error"]["message"].as_str().unwrap().to_owned(), v["error"]["param"].clone())
}

fn overflow(context: usize, m: usize, p: usize) -> String {
    format!(
        "This model's maximum context length is {context} tokens. However, you requested {m} output tokens and your prompt contains {p} input tokens, for a total of {} tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value={p})",
        p + m
    )
}

#[test]
fn overflows_are_vllms_400() {
    let u = up();
    for stream in [false, true] {
        let (status, v) = post(
            u.addr,
            completions(),
            json!({ "prompt": "x".repeat(60), "max_tokens": 10, "stream": stream }),
        );
        assert_eq!(refusal(status, &v), (overflow(64, 10, 61), json!("input_tokens")));
    }
    let (status, v) = post(
        u.addr,
        chat(),
        json!({ "messages": [{ "role": "user", "content": "x".repeat(60) }], "max_completion_tokens": 10, "stream": true }),
    );
    let (message, param) = refusal(status, &v);
    assert!(message.starts_with("This model's maximum context length is 64 tokens. However, you requested 10 output tokens and your prompt contains "), "{message}");
    assert_eq!(param, "input_tokens");
    let (status, v) = post(u.addr, completions(), json!({ "prompt": "x".repeat(64) }));
    assert_eq!(refusal(status, &v).0, overflow(64, 0, 65));
    let (status, v) = post(u.addr, completions(), json!({ "prompt": "x".repeat(63) }));
    let (message, param) = refusal(status, &v);
    assert_eq!(
        message,
        "The decoder prompt (length 64) plus the number of requested output tokens (at least 1) is longer than the maximum model length of 64. Make sure that `max_model_len` is no smaller than the number of text tokens (prompt + requested output tokens)."
    );
    assert!(param.is_null());
    let (status, v) = post(u.addr, completions(), json!({ "prompt": "Hi", "max_tokens": 100 }));
    assert_eq!(refusal(status, &v), ("max_tokens=100 cannot be greater than max_model_len=64. Please request fewer output tokens. (parameter=max_tokens, value=100)".to_owned(), json!("max_tokens")));
    let (status, v) = post(
        u.addr,
        chat(),
        json!({ "messages": [{ "role": "user", "content": "Hi" }], "max_completion_tokens": 100 }),
    );
    assert_eq!(refusal(status, &v).1, "max_completion_tokens");
    let (status, v) = post(
        u.addr,
        completions(),
        json!({ "prompt": "x".repeat(53), "max_tokens": 10, "ignore_eos": true }),
    );
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 10);
    let (status, v) = post(u.addr, completions(), json!({ "prompt": "x".repeat(9) }));
    assert_eq!(status, 200, "{v}");
    assert_eq!(u.seen(2).iter().map(|s| s.limit).collect::<Vec<_>>(), [10, 54]);
}

#[test]
fn logprobs_completion_shape_matches_vllm() {
    let u = up();
    let (status, v) = post(
        u.addr,
        completions(),
        json!({ "prompt": "Hi", "max_tokens": 3, "ignore_eos": true, "logprobs": 2 }),
    );
    assert_eq!(status, 200, "{v}");
    let lp = &v["choices"][0]["logprobs"];
    assert_eq!(lp["tokens"], json!(["Hello", " wo", "r"]));
    assert_eq!(lp["token_logprobs"], json!([-0.5, -0.5, -0.5]));
    assert_eq!(lp["text_offset"], json!([0, 5, 8]));
    assert_eq!(lp["top_logprobs"][0], json!({ "Hello": -0.5, " wo": -1.5 }));
    assert_eq!(lp["top_logprobs"][1], json!({ " wo": -0.5, "Hello": -1.5 }));
    let (_, v) = post(u.addr, completions(), json!({ "prompt": "Hi", "max_tokens": 2, "logprobs": 0 }));
    assert_eq!(v["choices"][0]["logprobs"]["top_logprobs"], json!([{ "Hello": -0.5 }, { " wo": -0.5 }]));
    let (_, v) = post(
        u.addr,
        completions(),
        json!({ "prompt": "Hi", "max_tokens": 1, "logprobs": 1, "return_tokens_as_token_ids": true }),
    );
    let hello = tok().encode("Hello", false).unwrap()[0];
    assert_eq!(v["choices"][0]["logprobs"]["tokens"], json!([format!("token_id:{hello}")]));
    let (_, v) = post(u.addr, completions(), json!({ "prompt": "Hi", "logprobs": 1 }));
    let tokens = v["choices"][0]["logprobs"]["tokens"].as_array().unwrap().clone();
    assert_eq!(tokens.len(), 13);
    assert_eq!(tokens[12], "<｜end▁of▁sentence｜>");
    let ev = events(u.addr, completions(), json!({ "prompt": "Hi", "logprobs": 1, "stream": true }));
    let streamed: Vec<Value> = ev
        .iter()
        .filter_map(|e| e["choices"][0]["logprobs"]["tokens"].as_array())
        .flatten()
        .cloned()
        .collect();
    assert_eq!(streamed, tokens);
}

#[test]
fn logprobs_chat_shape_matches_vllm() {
    let u = up();
    let msgs = json!([{ "role": "user", "content": "Hi" }]);
    let (status, v) = post(
        u.addr,
        chat(),
        json!({ "messages": msgs, "max_completion_tokens": 2, "logprobs": true, "top_logprobs": 2 }),
    );
    assert_eq!(status, 200, "{v}");
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap().clone();
    let entry = |t: &str, x: f64| json!({ "token": t, "logprob": x, "bytes": t.as_bytes() });
    let mut first = entry("Hello", -0.5);
    first["top_logprobs"] = json!([entry("Hello", -0.5), entry(" wo", -1.5)]);
    assert_eq!(content[0], first);
    assert_eq!((content.len(), &content[1]["token"]), (2, &json!(" wo")));
    let (_, v) =
        post(u.addr, chat(), json!({ "messages": msgs, "max_completion_tokens": 1, "logprobs": true }));
    assert_eq!(v["choices"][0]["logprobs"]["content"][0]["top_logprobs"], json!([]));
    let (_, whole) = post(u.addr, chat(), json!({ "messages": msgs, "logprobs": true, "top_logprobs": 1 }));
    let ev = events(
        u.addr,
        chat(),
        json!({ "messages": msgs, "logprobs": true, "top_logprobs": 1, "stream": true }),
    );
    let streamed: Vec<Value> = ev
        .iter()
        .filter_map(|e| e["choices"][0]["logprobs"]["content"].as_array())
        .flatten()
        .cloned()
        .collect();
    assert_eq!(&json!(streamed), &whole["choices"][0]["logprobs"]["content"]);
}

#[test]
fn logprobs_refusals_as_vllm() {
    let u = up();
    let (status, v) = post(u.addr, completions(), json!({ "prompt": "Hi", "logprobs": 17 }));
    assert_eq!(refusal(status, &v), ("Requested sample logprobs of 17, which is greater than max allowed: 16 (parameter=logprobs, value=17)".to_owned(), json!("logprobs")));
    let msgs = json!([{ "role": "user", "content": "Hi" }]);
    let (status, v) = post(u.addr, chat(), json!({ "messages": msgs, "logprobs": true, "top_logprobs": 25 }));
    assert_eq!(
        refusal(status, &v).0,
        "Requested sample logprobs of 25, which is greater than max allowed: 16 (parameter=logprobs, value=25)"
    );
    let (status, v) = post(u.addr, chat(), json!({ "messages": msgs, "top_logprobs": 2 }));
    assert_eq!(
        refusal(status, &v),
        (
            "when using `top_logprobs`, `logprobs` must be set to true. (parameter=top_logprobs)".to_owned(),
            json!("top_logprobs")
        )
    );
    let (status, v) = post(u.addr, completions(), json!({ "prompt": "Hi", "logprobs": "x" }));
    assert_eq!(refusal(status, &v).0, "`logprobs` must be an integer. (parameter=logprobs, value=x)");
    let (status, v) = post(u.addr, completions(), json!({ "prompt": "Hi", "logprobs": -2 }));
    assert_eq!(
        refusal(status, &v).0,
        "`logprobs` must be a positive value or -1. (parameter=logprobs, value=-2)"
    );
    let (status, v) = post(u.addr, completions(), json!({ "prompt": "Hi", "logprobs": 1, "echo": true }));
    assert_eq!((status, &v["error"]["param"]), (400, &json!("echo")), "{v}");
}

#[test]
fn logprobs_texts_of_a_character_cut_between_tokens() {
    let u = up_with(script("é!", true), 64, Duration::ZERO);
    let (_, v) = post(u.addr, completions(), json!({ "prompt": "Hi", "max_tokens": 3, "logprobs": 0 }));
    assert_eq!(v["choices"][0]["logprobs"]["tokens"], json!(["", "é", "!"]));
    assert_eq!(v["choices"][0]["text"], "é!");
}

#[test]
fn stream_options_and_chunk_shapes_as_vllm() {
    let u = up();
    let ev = events(
        u.addr,
        completions(),
        json!({ "prompt": "Hi", "stream": true, "stream_options": { "include_usage": true, "continuous_usage_stats": true } }),
    );
    let mut last = 0;
    for e in &ev {
        let used = e["usage"]["completion_tokens"].as_u64().unwrap_or_else(|| panic!("no usage in {e}"));
        assert!(used >= last, "{ev:?}");
        assert_eq!(e["usage"]["prompt_tokens"], 3);
        last = used;
    }
    assert_eq!(last, 13);
    for e in &ev[..ev.len() - 1] {
        let c = e["choices"][0].as_object().unwrap();
        assert!(
            c.contains_key("stop_reason")
                && c.contains_key("prompt_token_ids")
                && c.contains_key("token_ids"),
            "{e}"
        );
    }
    let ev = events(
        u.addr,
        completions(),
        json!({ "prompt": "Hi", "stream": true, "stream_options": { "include_usage": true } }),
    );
    assert_eq!(ev.iter().filter(|e| !e["usage"].is_null()).count(), 1);
}

fn open(addr: SocketAddr, body: Value) -> TcpStream {
    let mut s = TcpStream::connect(addr).unwrap();
    let body = body.to_string();
    write!(
        s,
        "POST {} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        completions(),
        body.len()
    )
    .unwrap();
    s
}

fn until(what: &str, done: impl Fn() -> bool) {
    let t = Instant::now();
    while !done() {
        assert!(t.elapsed() < Duration::from_secs(10), "{what}: not within 10 s");
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn a_client_gone_ends_its_request() {
    let u = up_with(script(hello_world(), false), 4096, Duration::from_millis(5));
    let c = open(u.addr, json!({ "prompt": "Hi", "max_tokens": 1000, "stream": true }));
    thread::sleep(Duration::from_millis(100));
    drop(c);
    let sent = u.seen(1)[0].sent;
    assert!(sent < 200, "{sent} tokens");
    let c = open(u.addr, json!({ "prompt": "Hi", "max_tokens": 1000 }));
    thread::sleep(Duration::from_millis(100));
    c.shutdown(Shutdown::Write).unwrap();
    let sent = u.seen(2)[1].sent;
    assert!(sent < 200, "{sent} tokens");
}

fn raw(addr: SocketAddr, request: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    s.write_all(request.as_bytes()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

#[test]
fn an_api_key_guards_every_route_but_health() {
    let u = up_keyed(script(hello_world(), true), 64, Duration::ZERO, Some("sesame"));
    let get = |path: &str, auth: &str| raw(u.addr, &format!("GET {path} HTTP/1.1\r\nHost: x\r\n{auth}\r\n"));
    assert!(get("/health", "").starts_with("HTTP/1.1 200"));
    assert!(get("/v1/models", "").starts_with("HTTP/1.1 401"));
    assert!(get("/metrics", "Authorization: Bearer open\r\n").starts_with("HTTP/1.1 401"));
    assert!(get("/v1/models", "Authorization: Bearer sesame\r\n").starts_with("HTTP/1.1 200"));
    assert!(
        get("/v1/models", "Authorization: bearer sesame\r\n").starts_with("HTTP/1.1 200"),
        "the scheme is case-insensitive"
    );
    let unsent = raw(
        u.addr,
        &format!("POST {} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n", completions(), 8 << 20),
    );
    assert!(unsent.starts_with("HTTP/1.1 401"), "refused before its body: {unsent}");
    assert_eq!(send(up().addr, "GET", "/v1/models", "").0, 200);
}

#[test]
fn a_request_trickled_in_is_cut_off_at_its_deadline() {
    let u = up();
    let mut s = TcpStream::connect(u.addr).unwrap();
    s.write_all(b"GET /v1/models HTTP/1.1\r\n").unwrap();
    s.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(40) && s.write_all(b"X-Slow: 1\r\n").is_ok() {
        match s.read(&mut [0u8; 512]) {
            Ok(_) => break,
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            Err(_) => break,
        }
    }
    assert!(started.elapsed() < Duration::from_secs(36), "held {:?}", started.elapsed());
}

#[test]
fn an_unread_body_is_refused_and_the_connection_closed() {
    let u = up();
    let big = raw(
        u.addr,
        &format!(
            "POST {} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{{\"prompt\":",
            completions(),
            17 << 20
        ),
    );
    assert!(big.starts_with("HTTP/1.1 413") && big.contains("Connection: close"), "{big}");
    let chunked = raw(
        u.addr,
        &format!(
            "POST {} HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n",
            completions()
        ),
    );
    assert!(chunked.starts_with("HTTP/1.1 411"), "{chunked}");
}

#[test]
fn connections_past_the_cap_are_refused() {
    let u = up();
    let held: Vec<TcpStream> =
        (0..ServerConfig::default().http.connections).map(|_| TcpStream::connect(u.addr).unwrap()).collect();
    let mut over = TcpStream::connect(u.addr).unwrap();
    let mut out = String::new();
    over.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 503") && out.contains("connections are open"), "{out}");
    drop(held);
    until("a slot frees", || send(u.addr, "GET", "/health", "").0 == 200);
}

#[test]
fn an_engine_refusal_is_a_503() {
    let (queue, jobs) = mpsc::channel::<Job>();
    thread::spawn(move || {
        for j in jobs {
            if let Job::Run(r) = j {
                let _ = r.out.send(Out::End(End::Refused("the queue is full".into())));
            }
        }
    });
    let server = Arc::new(
        Server::new(
            tok(),
            64,
            queue,
            Arc::new(Metrics::new(MetricsConfig::default().iteration_tokens)),
            None,
            ServerConfig::default(),
        )
        .unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || server.serve(listener));
    let (status, v) = post(addr, completions(), json!({ "prompt": "Hi" }));
    assert_eq!(status, 503, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("the queue is full"), "{v}");
    let (_, text) = send(addr, "POST", completions(), &json!({ "prompt": "Hi", "stream": true }).to_string());
    assert!(text.contains("the queue is full") && text.ends_with("data: [DONE]\n\n"), "{text}");
}

fn sample(text: &str, name: &str, labels: &str) -> f64 {
    let head = format!("{name}{{{labels}");
    let line = text.lines().find(|l| l.starts_with(&head)).unwrap_or_else(|| panic!("no {head} in\n{text}"));
    line.rsplit(' ').next().unwrap().parse().unwrap()
}

#[test]
fn metrics_are_vllms_and_move() {
    let histograms = [
        "request_prompt_tokens",
        "request_generation_tokens",
        "iteration_tokens_total",
        "request_max_num_generation_tokens",
        "request_params_n",
        "request_params_max_tokens",
        "time_to_first_token_seconds",
        "inter_token_latency_seconds",
        "request_time_per_output_token_seconds",
        "e2e_request_latency_seconds",
        "request_queue_time_seconds",
        "request_inference_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "request_num_preemptions",
        "request_prefill_kv_computed_tokens",
    ];
    let u = up();
    let (status, before) = send(u.addr, "GET", "/metrics", "");
    assert_eq!(status, 200);
    let model = format!("engine=\"0\",model_name=\"{}\"", name());
    for (name, kind) in [
        ("num_requests_running", "gauge"),
        ("num_requests_waiting", "gauge"),
        ("num_requests_waiting_by_reason", "gauge"),
        ("engine_sleep_state", "gauge"),
        ("kv_cache_usage_perc", "gauge"),
        ("mm_cache_queries_total", "counter"),
        ("mm_cache_hits_total", "counter"),
        ("num_preemptions_total", "counter"),
        ("prompt_tokens_total", "counter"),
        ("prompt_tokens_by_source_total", "counter"),
        ("prompt_tokens_cached_total", "counter"),
        ("generation_tokens_total", "counter"),
        ("prefix_cache_queries_total", "counter"),
        ("prefix_cache_hits_total", "counter"),
        ("external_prefix_cache_queries_total", "counter"),
        ("external_prefix_cache_hits_total", "counter"),
        ("top_p_clipped_total", "counter"),
        ("request_success_total", "counter"),
        ("estimated_flops_per_gpu_total", "counter"),
        ("estimated_read_bytes_per_gpu_total", "counter"),
        ("estimated_write_bytes_per_gpu_total", "counter"),
    ]
    .into_iter()
    .chain(histograms.map(|h| (h, "histogram")))
    {
        assert!(before.contains(&format!("\n# TYPE vllm:{name} {kind}\n")), "{name}: {before}");
        assert!(before.contains(&format!("# HELP vllm:{name} ")), "{name}");
    }
    assert!(
        before.lines().filter(|l| !l.starts_with('#')).all(|l| l.starts_with("vllm:") && l.contains(&model)),
        "{before}"
    );
    assert!(
        !before.contains("spec_decode") && !before.contains("tool_call_parser"),
        "no drafter, no tool parser counted: {before}"
    );
    let families =
        before.lines().filter(|l| l.starts_with("# TYPE ") && !l.ends_with("_created gauge")).count();
    assert_eq!(families, 21 + histograms.len(), "the old server's families, no more");
    let count = |t: &str, h: &str| sample(t, &format!("vllm:{h}_count"), "");
    let sum = |t: &str, h: &str| sample(t, &format!("vllm:{h}_sum"), "");
    assert!(histograms.iter().all(|h| count(&before, h) == 0.0));
    assert_eq!(sample(&before, "vllm:engine_sleep_state", "sleep_state=\"awake\""), 1.0);
    let stops = |t: &str| sample(t, "vllm:request_success_total", "finished_reason=\"stop\"");
    let lengths = |t: &str| sample(t, "vllm:request_success_total", "finished_reason=\"length\"");
    assert_eq!((stops(&before), lengths(&before)), (0.0, 0.0));

    post(u.addr, completions(), json!({ "prompt": "Hi", "max_tokens": 1 }));
    post(u.addr, completions(), json!({ "prompt": "Hi" }));
    let (_, after) = send(u.addr, "GET", "/metrics", "");
    assert_eq!((stops(&after), lengths(&after)), (1.0, 1.0), "{after}");
    let made = script(hello_world(), true).len() as f64 + 1.0;
    assert_eq!(sample(&after, "vllm:generation_tokens_total", ""), made);
    assert!(sample(&after, "vllm:prompt_tokens_total", "") > 0.0);
    assert_eq!(sample(&after, "vllm:num_requests_running", ""), 0.0);
    for h in [
        "time_to_first_token_seconds",
        "e2e_request_latency_seconds",
        "request_queue_time_seconds",
        "request_inference_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "request_time_per_output_token_seconds",
        "request_prompt_tokens",
        "request_generation_tokens",
        "request_params_max_tokens",
        "request_num_preemptions",
        "request_prefill_kv_computed_tokens",
    ] {
        assert_eq!(count(&after, h), 2.0, "{h}: {after}");
        assert_eq!(sample(&after, &format!("vllm:{h}_bucket"), &format!("{model},le=\"+Inf\"")), 2.0, "{h}");
    }
    assert!((1.0..=made - 2.0).contains(&count(&after, "inter_token_latency_seconds")), "{after}");
    assert_eq!(sum(&after, "request_generation_tokens"), made);
    assert_eq!(
        sum(&after, "request_prompt_tokens"),
        sum(&after, "request_prefill_kv_computed_tokens"),
        "nothing cached"
    );
    assert_eq!(sum(&after, "request_num_preemptions"), 0.0);
    assert_eq!(
        sum(&after, "request_params_max_tokens"),
        u.seen(2).iter().map(|s| s.limit as f64).sum::<f64>(),
        "max_tokens as resolved"
    );
    assert_eq!(sample(&after, "vllm:request_num_preemptions_bucket", &format!("{model},le=\"1.0\"")), 2.0);
    for h in ["request_max_num_generation_tokens", "request_params_n"] {
        assert_eq!(count(&after, h), 0.0, "vLLM observes {h} only past n=1");
    }

    let u = up_with(script(hello_world(), false), 4096, Duration::from_millis(5));
    let c = open(u.addr, json!({ "prompt": "Hi", "max_tokens": 1000, "stream": true }));
    let running = || sample(&send(u.addr, "GET", "/metrics", "").1, "vllm:num_requests_running", "");
    until("a request running", || running() == 1.0);
    drop(c);
    until("its end", || running() == 0.0);
}

#[test]
fn metrics_of_a_drafting_engine() {
    let (queue, _jobs) = mpsc::channel::<Job>();
    let metrics = Arc::new(Metrics::new(MetricsConfig::default().iteration_tokens));
    let mut step =
        StepMetrics { spec_tokens: 2, top_p_clipped: 3, prefill_tokens: 9, ..StepMetrics::default() };
    step.verified(1);
    step.verified(2);
    metrics.record(step);
    let server = Server::new(tok(), 64, queue, metrics, None, ServerConfig::default()).unwrap();
    let text = hyperserve::api::metrics::render(&server);
    for (name, v) in [
        ("spec_decode_num_drafts", 2.0),
        ("spec_decode_num_draft_tokens", 4.0),
        ("spec_decode_num_accepted_tokens", 3.0),
        ("top_p_clipped", 3.0),
    ] {
        assert!(text.contains(&format!("\n# TYPE vllm:{name}_total counter\n")), "{name}: {text}");
        assert_eq!(sample(&text, &format!("vllm:{name}_total"), ""), v, "{name}");
    }
    assert!(text.contains("\n# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter\n"));
    let pos = |i: usize| {
        sample(&text, "vllm:spec_decode_num_accepted_tokens_per_pos_total", &format!("position=\"{i}\""))
    };
    assert_eq!((pos(0), pos(1)), (2.0, 1.0));
    assert!(!text.contains("position=\"2\""), "{text}");
    let model = format!("engine=\"0\",model_name=\"{}\"", name());
    assert_eq!(sample(&text, "vllm:iteration_tokens_total_bucket", &format!("{model},le=\"8.0\"")), 0.0);
    assert_eq!(sample(&text, "vllm:iteration_tokens_total_bucket", &format!("{model},le=\"16.0\"")), 1.0);
    assert_eq!(sample(&text, "vllm:iteration_tokens_total_sum", ""), 9.0);
}

#[test]
fn reset_prefix_cache_answers_once_done() {
    let u = up();
    let (status, v) = post(u.addr, "/reset_prefix_cache", json!({}));
    assert_eq!((status, &v), (200, &json!({ "reset": true, "dropped": "7 pages freed" })));
    let (status, _) = send(u.addr, "GET", "/reset_prefix_cache", "");
    assert_eq!(status, 405);
}
