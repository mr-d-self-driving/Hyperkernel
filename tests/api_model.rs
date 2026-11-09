use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use hyperserve::error::Error;
use hyperserve::json::{Json, Member};
use hyperserve::model;
use hyperserve::model::detok::Detok;
use hyperserve::model::grammar::GrammarConfig;
use hyperserve::model::reply::{Calls, Delta, Opening, Reader};
use hyperserve::model::template::{Conversation, Mode};
use hyperserve::model::tokenizer::{Special, Tokenizer, TokenizerConfig};

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn bos_id() -> u32 {
    300
}

fn eos_id() -> u32 {
    301
}

fn tok() -> Tokenizer {
    Tokenizer::from_json(&fixture("tokenizer.json"), &TokenizerConfig::default()).unwrap()
}

fn member(key: &str, value: Json) -> Member {
    Member { key: key.to_owned(), value }
}

fn case(n: usize) -> Result<String, Error> {
    let input = Json::parse(&fixture(&format!("dsv41/test_input_{n}.json"))).unwrap();
    let (body, mode) = match &input {
        Json::Array(_) => (Json::Object(vec![member("messages", input.clone())]), Mode::Chat),
        case => {
            let mode = if case.get("thinking_mode").as_str() == Some("thinking") {
                Mode::Thinking
            } else {
                Mode::Chat
            };
            (
                Json::Object(
                    ["messages", "tools", "reasoning_effort"]
                        .into_iter()
                        .map(|k| member(k, case.get(k).clone()))
                        .collect(),
                ),
                mode,
            )
        }
    };
    Ok(Conversation::parse(&in_mode(body, mode))?.render(&GrammarConfig::default()))
}

fn in_mode(body: Json, mode: Mode) -> Json {
    let Json::Object(mut members) = body else { panic!("{body}: not an object") };
    members.push(member(
        "chat_template_kwargs",
        Json::Object(vec![member("thinking", Json::Bool(mode == Mode::Thinking))]),
    ));
    Json::Object(members)
}

#[test]
fn golden_cases() {
    for n in 1..=4 {
        assert_eq!(case(n).unwrap(), fixture(&format!("dsv41/test_output_{n}.txt")), "case {n}");
    }
    assert_eq!(case(5).unwrap_err().param.as_deref(), Some("messages"));
}

fn render(body: &str, mode: Mode) -> String {
    Conversation::parse(&in_mode(Json::parse(body).unwrap(), mode)).unwrap().render(&GrammarConfig::default())
}

#[test]
fn system_messages() {
    let effort_88 = "<｜System｜>Reasoning Effort: 88 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
    let first = r#"{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "hello"}]}"#;
    assert_eq!(
        render(first, Mode::Chat),
        "<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>"
    );
    let mid = r#"{"reasoning_effort": 88, "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, {"role": "system", "content": "mid sys"}]}"#;
    let want = format!(
        "<｜begin▁of▁sentence｜>{effort_88}sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜System｜>mid sys<｜Assistant｜><think>"
    );
    assert_eq!(render(mid, Mode::Thinking), want);
}

#[test]
fn thinking_dropped_before_the_last_user_message() {
    let body = r#"{"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, {"role": "user", "content": "q2"}]}"#;
    let p = render(body, Mode::Thinking);
    assert!(p.contains("<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"), "{p}");
    assert!(!p.contains("r1"));
    assert!(p.ends_with("<｜User｜>q2<｜Assistant｜><think>"));
}

#[test]
fn task_token() {
    let body = r#"{"messages": [{"role": "user", "content": "classify me", "task": "query"}]}"#;
    assert_eq!(render(body, Mode::Chat), "<｜begin▁of▁sentence｜><｜User｜>classify me<｜query｜>");
}

#[test]
fn tool_call_in_an_assistant_message() {
    let lookup = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let body = r#"{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "reasoning_content": "  reason  ", "content": "summary",
        "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": "{\"query\":\"value\",\"limit\":2}"}}]}]}"#;
    let want = format!(
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>question<｜Assistant｜><think>  reason  </think>summary\n\n{lookup}<｜end▁of▁sentence｜>"
    );
    assert_eq!(render(body, Mode::Thinking), want);
}

#[test]
fn arguments_not_an_object() {
    let body = r#"{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "f", "arguments": "{oops"}}, {"function": {"name": "g", "arguments": "\"{\\\"x\\\": [1.5e-7, 1e16, 100.0, -0.0]}\""}}]}]}"#;
    let p = render(body, Mode::Chat);
    assert!(p.contains("<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"arguments\" string=\"true\">{oops</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"g\">"), "{p}");
    assert!(p.contains("<｜DSML｜ parameter name=\"x\" string=\"false\">[1.5e-07, 1e+16, 100.0, -0.0]</｜DSML｜ parameter>"), "{p}");
}

#[test]
fn tool_results_join_the_user_message_in_call_order() {
    let body = r#"{"tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "b": 1, "a": "é\n"}}}],
        "messages": [{"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}, {"id": "y", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "y", "content": "Y"}, {"role": "tool", "tool_call_id": "x", "content": [{"type": "text", "text": "X"}]},
        {"role": "user", "content": "more"}]}"#;
    let p = render(body, Mode::Chat);
    assert!(p.starts_with("<｜begin▁of▁sentence｜><｜System｜>\n\n## Tools\n\n"), "{p}");
    assert!(
        p.contains(
            "\n{\"name\": \"f\", \"parameters\": {\"type\": \"object\", \"b\": 1, \"a\": \"é\\n\"}}\n"
        ),
        "{p}"
    );
    assert!(p.contains("<｜DSML｜ invoke name=\"f\">\n\n</｜DSML｜ invoke>"), "{p}");
    assert!(p.ends_with("<｜end▁of▁sentence｜><｜User｜><tool_result>X</tool_result>\n\n<tool_result>Y</tool_result>\n\nmore<｜Assistant｜></think>"), "{p}");
}

#[test]
fn template_refusals() {
    let refused = |body: &str| Conversation::parse(&Json::parse(body).unwrap()).unwrap_err().param;
    assert_eq!(
        refused(r#"{"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "medium"}"#)
            .as_deref(),
        Some("reasoning_effort")
    );
    assert_eq!(refused(r#"{"messages": [{"role": "user", "content": "q"}], "tools": [{"type": "function", "function": {"name": "a::b"}}]}"#).as_deref(), Some("tools"));
    assert_eq!(refused(r#"{"messages": [{"role": "tool"}]}"#).as_deref(), Some("messages"));
    assert_eq!(refused(r#"{"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}]}"#).as_deref(), Some("messages"));
}

#[test]
fn json_keeps_key_order_and_reads_as_serde_json() {
    let text = r#"{"z": [1, -2, 3.5, -0.0, 1e300, 18446744073709551615, -9223372036854775808, true, false, null],
        "a": {"s": "tab\t quote\" é 🚀 日本", "e": "", "n": {}, "l": []}, "dup": 1, "m": "x", "dup": [2]}"#;
    let json = Json::parse(text).unwrap();
    assert_eq!(Value::from(&json), serde_json::from_str::<Value>(text).unwrap());
    let keys: Vec<&str> = json.members().iter().map(|m| m.key.as_str()).collect();
    assert_eq!(keys, ["z", "a", "dup", "m"]);
    assert_eq!(json.get("dup"), &Json::Array(vec![Json::Number(2.into())]));
    for bad in ["", "[1,", "{\"a\" 1}", "nul", "[1] x"] {
        assert!(Json::parse(bad).is_err(), "{bad:?}");
    }
}

#[derive(Debug, Default, PartialEq)]
struct Read {
    reasoning: String,
    content: String,
    calls: Vec<String>,
}

fn read(opening: Opening, calls: Calls, pieces: &[&str]) -> Read {
    let g = GrammarConfig::default();
    let mut reader = Reader::new(&g, opening, calls);
    let mut deltas: Vec<Delta> = pieces.iter().flat_map(|p| reader.push(p)).collect();
    deltas.extend(reader.finish());
    let mut out = Read::default();
    for d in deltas {
        match d {
            Delta::Reasoning(t) => out.reasoning.push_str(&t),
            Delta::Content(t) => out.content.push_str(&t),
            Delta::Call { name, arguments } => out.calls.push(format!("{name} {arguments}")),
        }
    }
    out
}

fn chars(text: &str) -> Vec<&str> {
    text.char_indices().map(|(i, c)| &text[i..i + c.len_utf8()]).collect()
}

#[test]
fn reads_as_the_encoder_parses() {
    let cases: Value = serde_json::from_str(&fixture("dsv41/parse.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let opening = if case["mode"] == "thinking" { Opening::Reasoning } else { Opening::Content };
        let text = case["completion"].as_str().unwrap();
        let want = Read {
            reasoning: case["reasoning_content"].as_str().unwrap().to_owned(),
            content: case["content"].as_str().unwrap().to_owned(),
            calls: case["tool_calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| format!("{} {}", c["name"].as_str().unwrap(), c["arguments"].as_str().unwrap()))
                .collect(),
        };
        assert_eq!(read(opening, Calls::Read, &[text]), want, "{text:?}");
        assert_eq!(read(opening, Calls::Read, &chars(text)), want, "{text:?} a character at a time");
    }
}

fn call() -> &'static str {
    "<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\" string=\"false\">1</｜DSML｜ parameter>\n</｜DSML｜ invoke>"
}

#[test]
fn broken_calls_are_content() {
    let call = call();
    let good = format!("Hi\n\n<｜DSML｜ calls>\n{call}\n");
    let bad = "<｜DSML｜ invoke name=\"g\">\n<｜DSML｜ parameter name=\"y\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let text = format!("{good}{bad}");
    for pieces in [vec![text.as_str()], chars(&text)] {
        let r = read(Opening::Content, Calls::Read, &pieces);
        assert_eq!(r.calls, ["f {\"x\": 1}"]);
        assert_eq!(r.content, format!("Hi\n{bad}"));
    }
    let first = format!("\n\n<｜DSML｜ calls>\n{bad}");
    for pieces in [vec![first.as_str()], chars(&first)] {
        assert_eq!(
            read(Opening::Content, Calls::Read, &pieces),
            Read { content: first.clone(), ..Read::default() }
        );
    }
    let cut = format!("Hi\n\n<｜DSML｜ calls>\n{}", &call[..40]);
    assert_eq!(
        read(Opening::Content, Calls::Read, &[&cut]),
        Read { content: cut.clone(), ..Read::default() }
    );
    let twice = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\" string=\"true\">1</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"x\" string=\"true\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    assert_eq!(read(Opening::Content, Calls::Read, &[twice]).calls, Vec::<String>::new());
    let after = format!("\n\n<｜DSML｜ calls>\n{call}\n</｜DSML｜ calls> and more");
    let r = read(Opening::Content, Calls::Read, &[&after]);
    assert_eq!((r.calls.len(), r.content.as_str()), (1, " and more"));
}

#[test]
fn ignored_calls_and_unclosed_reasoning() {
    let call = call();
    let text = format!("x\n\n<｜DSML｜ calls>\n{call}\n</｜DSML｜ calls>");
    assert_eq!(
        read(Opening::Content, Calls::Ignored, &chars(&text)),
        Read { content: text.clone(), ..Read::default() }
    );
    assert_eq!(
        read(Opening::Reasoning, Calls::Read, &["still </thi", "nking"]),
        Read { reasoning: "still </thinking".into(), ..Read::default() }
    );
}

#[test]
fn merges_by_rank_and_round_trips() {
    let t = tok();
    let ids = t.encode("Hello world", true).unwrap();
    assert_eq!(ids, [bos_id(), 259, 261, b'r' as u32, b'l' as u32, b'd' as u32]);
    assert_eq!(t.decode(&ids, Special::Skip), "Hello world");
    for text in ["héllo  wörld\n\n 123456 ok?!", "你好，世界 ありがとう", "a\tb \r\n", "  trailing  "]
    {
        assert_eq!(t.decode(&t.encode(text, false).unwrap(), Special::Keep), text);
    }
}

#[test]
fn added_tokens_match_verbatim() {
    let t = tok();
    let ids = t.encode("<｜User｜>Hello<｜Assistant｜></think>", false).unwrap();
    assert_eq!(ids, [302, 259, 303, 305]);
    assert_eq!(t.decode(&ids, Special::Skip), "Hello</think>");
    assert_eq!(t.decode(&[eos_id()], Special::Skip), "");
    assert_eq!(t.decode(&ids, Special::Keep), "<｜User｜>Hello<｜Assistant｜></think>");
    assert_eq!(t.id("<｜end▁of▁sentence｜>"), Some(eos_id()));
}

#[test]
fn refuses_what_it_does_not_implement() {
    let mut v: Value = serde_json::from_str(&fixture("tokenizer.json")).unwrap();
    v["normalizer"] = json!({ "type": "NFC" });
    assert!(Tokenizer::from_json(&v.to_string(), &TokenizerConfig::default()).is_err());
    v["normalizer"] = Value::Null;
    v["model"]["type"] = "Unigram".into();
    assert!(Tokenizer::from_json(&v.to_string(), &TokenizerConfig::default()).is_err());
}

#[test]
fn reads_the_released_files_form() {
    let mut v: Value = serde_json::from_str(&fixture("tokenizer.json")).unwrap();
    v["normalizer"] = json!({ "type": "Sequence", "normalizers": [] });
    v["model"]["vocab"]["<｜▁pad▁｜>"] = 262.into();
    let t = Tokenizer::from_json(&v.to_string(), &TokenizerConfig::default()).unwrap();
    assert_eq!(t.decode(&[262], Special::Keep), "<｜▁pad▁｜>");
    assert_eq!(t.encode("Hello world", true).unwrap()[..2], [bos_id(), 259]);
}

#[test]
fn detok_waits_for_whole_characters() {
    let t = tok();
    let mut d = Detok::new(&t, Special::Keep, vec![eos_id()], Vec::new(), false);
    let e = "é".as_bytes();
    assert_eq!(d.push(e[0] as u32), (String::new(), None));
    assert_eq!(d.push(e[1] as u32), ("é".into(), None));
    assert_eq!(d.push(eos_id()), (String::new(), None));
    assert_eq!(d.finish(), "");
}

fn push_text(t: &Tokenizer, d: &mut Detok, text: &str) -> (String, Option<String>) {
    let mut sent = String::new();
    for id in t.encode(text, false).unwrap() {
        let (text, stop) = d.push(id);
        sent.push_str(&text);
        if stop.is_some() {
            return (sent, stop);
        }
    }
    (sent, None)
}

#[test]
fn stop_strings_found_across_tokens_and_cut() {
    let t = tok();
    let mut d = Detok::new(&t, Special::Skip, Vec::new(), vec!["lo wo".into(), "zzz".into()], false);
    assert_eq!(push_text(&t, &mut d, "Hel"), (String::new(), None));
    assert_eq!(push_text(&t, &mut d, "lo world"), ("Hel".into(), Some("lo wo".into())));
    assert_eq!(d.finish(), "");

    let mut d = Detok::new(&t, Special::Skip, Vec::new(), vec!["zz".into()], false);
    assert_eq!(push_text(&t, &mut d, "abz"), ("ab".into(), None));
    assert_eq!(d.finish(), "z");
    let mut d = Detok::new(&t, Special::Skip, Vec::new(), vec!["wo".into()], true);
    assert_eq!(push_text(&t, &mut d, "Hello world"), ("Hello wo".into(), Some("wo".into())));
    assert_eq!(d.finish(), "");
}

fn agent_tokenizer() -> Tokenizer {
    match std::env::var_os("DSV41_TOKENIZER") {
        Some(path) => Tokenizer::from_json(
            &fs::read_to_string(&path).expect("DSV41_TOKENIZER: a tokenizer.json"),
            &TokenizerConfig::default(),
        )
        .unwrap(),
        None => tok(),
    }
}

fn unicode() -> &'static str {
    "\n\n日本語のテキスト、ひらがなとカタカナ。中文文本：模型的上下文。한국어 문장입니다. Ünïcödé façade naïve café — “quotes” ‘single’ … emoji 🚀🔥👩‍💻 and flags 🇯🇵; math ∑∫√∞ ≠ ≤; Ελληνικά, русский текст, עברית, العربية.\n\
    Special-looking text that is not a token: <｜User｜ (unclosed) </ think> <tool_result> ｜DSML ｜ <|EOT| <｜end▁of▁sentence\n\n"
}

fn corpus() -> String {
    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.unwrap().path();
            if path.is_dir() {
                sources(&path, out);
            } else if path.extension().is_some_and(|x| x == "rs") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    sources(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
    files.sort();
    let mut out = String::new();
    for f in files {
        out.push_str(&fs::read_to_string(&f).unwrap_or_else(|e| panic!("{}: {e}", f.display())));
        out.push_str(unicode());
    }
    out
}

struct Text {
    corpus: String,
    at: usize,
}

impl Text {
    fn take(&mut self, bytes: usize) -> String {
        let c = &self.corpus;
        let mut start = self.at % c.len();
        while !c.is_char_boundary(start) {
            start += 1;
        }
        let mut end = (start + bytes).min(c.len());
        while !c.is_char_boundary(end) {
            end -= 1;
        }
        self.at = end + 7919;
        c[start..end].to_owned()
    }
}

struct Agent {
    text: Text,
    messages: Vec<Value>,
    turn: usize,
}

impl Agent {
    fn new(system_bytes: usize) -> Agent {
        let mut text = Text { corpus: corpus(), at: 0 };
        let messages = vec![
            json!({"role": "system", "content": text.take(system_bytes)}),
            json!({"role": "user", "content": text.take(1200)}),
        ];
        Agent { text, messages, turn: 0 }
    }

    fn next(&mut self, bytes: usize) {
        self.turn += 1;
        let t = &mut self.text;
        let reasoning = t.take(bytes / 8);
        let content = t.take(bytes / 16);
        match self.turn % 4 {
            0 => {
                self.messages
                    .push(json!({"role": "assistant", "content": content, "reasoning_content": reasoning}));
                self.messages.push(json!({"role": "user", "content": [{"type": "text", "text": t.take(bytes / 4)}, {"type": "text", "text": t.take(bytes / 8)}]}));
            }
            n => {
                let id = |k: usize| format!("call_{}_{k}", self.turn);
                let code = t.take(bytes / 8);
                let calls = json!([
                    {"id": id(0), "type": "function", "function": {"name": "edit", "arguments": json!({"path": "src/api/json.rs", "old": code, "new": "x"}).to_string()}},
                    {"id": id(1), "type": "function", "function": {"name": "run", "arguments": json!({"command": format!("cargo test {n}"), "timeout_s": 60.5}).to_string()}}
                ]);
                let content = if n == 2 { Value::Null } else { Value::String(content) };
                self.messages.push(json!({"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls}));
                self.messages
                    .push(json!({"role": "tool", "tool_call_id": id(1), "content": t.take(bytes / 2)}));
                self.messages
                    .push(json!({"role": "tool", "tool_call_id": id(0), "content": t.take(bytes / 16)}));
            }
        }
    }

    fn body(&self, thinking: bool, tools: bool, extra: &Value) -> Json {
        let mut body = json!({"messages": self.messages, "chat_template_kwargs": {"thinking": thinking}});
        if tools {
            body["tools"] = json!([
                {"type": "function", "function": {"name": "edit", "description": "Replace text in a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "run", "description": "Run a shell command. Ünïcödé 説明.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "number"}}}}}
            ]);
        }
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        Json::parse(&body.to_string()).unwrap()
    }
}

fn ids(tok: &Tokenizer, body: &Json) -> Vec<u32> {
    tok.encode(&model::chat(&GrammarConfig::default(), body).map_err(|e| e.message).unwrap().prompt, false)
        .unwrap()
}

#[test]
fn memoized_ids_are_a_fresh_tokenizers() {
    let warm = agent_tokenizer();
    for (thinking, tools) in [(true, true), (false, true), (true, false), (false, false)] {
        let mut agent = Agent::new(3000);
        for turn in 0..8 {
            let body = agent.body(thinking, tools, &json!({}));
            let want = ids(&agent_tokenizer(), &body);
            assert_eq!(ids(&warm, &body), want, "thinking {thinking} tools {tools} turn {turn}");
            assert_eq!(ids(&warm, &body), want, "thinking {thinking} tools {tools} turn {turn}, again");
            agent.next(6000);
        }
    }
    let mut agent = Agent::new(3000);
    for _ in 0..5 {
        agent.next(6000);
    }
    agent.messages.push(json!({"role": "user", "content": "go on"}));
    for extra in [
        json!({"tool_choice": "required"}),
        json!({"tool_choice": {"type": "function", "function": {"name": "run"}}}),
        json!({"add_generation_prompt": false}),
    ] {
        let body = agent.body(false, true, &extra);
        assert_eq!(ids(&warm, &body), ids(&agent_tokenizer(), &body), "{extra}");
    }
    agent.messages.push(json!({"role": "assistant", "content": "Partial ünïcödé answer 日本"}));
    let body =
        agent.body(true, true, &json!({"continue_final_message": true, "add_generation_prompt": false}));
    let p = model::chat(&GrammarConfig::default(), &body).unwrap().prompt;
    assert!(p.ends_with("Partial ünïcödé answer 日本"), "{}", &p[p.len() - 60..]);
    assert_eq!(ids(&warm, &body), ids(&agent_tokenizer(), &body));
}

#[test]
fn edge_texts_encode_alike_cold_and_warm() {
    let warm = agent_tokenizer();
    let at_memo_min = "x".repeat(TokenizerConfig::default().memo_min);
    let unicode = unicode();
    let texts = [
        String::new(),
        "<｜User｜>".into(),
        "<｜begin▁of▁sentence｜><｜User｜><｜Assistant｜></think><｜end▁of▁sentence｜>".into(),
        format!("{at_memo_min}<｜User｜>{at_memo_min}x<think>{}", &at_memo_min[1..]),
        format!("{unicode}<｜User｜{unicode}<<｜User｜>>｜DSML｜｜DSML｜ calls>{unicode}</think>"),
        corpus(),
    ];
    for (i, t) in texts.iter().enumerate() {
        let want = agent_tokenizer().encode(t, true).unwrap();
        assert_eq!(warm.encode(t, true).unwrap(), want, "text {i}");
        assert_eq!(warm.encode(t, true).unwrap(), want, "text {i}, again");
    }
}
