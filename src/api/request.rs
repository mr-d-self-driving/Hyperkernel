use std::sync::mpsc;

use crate::api::Server;
use crate::api::logprobs::Logprobs;
use crate::api::write::{Ask, Endpoint, Reply, Stream};
use crate::engine::{Request, Sampling};
use crate::error::Error;
use crate::json::Json;
use crate::model;
use crate::model::detok::Detok;
use crate::model::tokenizer::Special;

struct Served {
    field: &'static str,
    ok: fn(&Json) -> bool,
    serves: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Completions,
    Chat,
}

pub fn parse<'s>(server: &'s Server, api: Api, body: &Json) -> Result<(Request, Reply<'s>), Error> {
    let name = &server.config.name;
    match body.get("model") {
        Json::Null => {}
        Json::String(m) if m == name => {}
        m => {
            return Err(Error::new(
                404,
                "not_found_error",
                format!("The model {m} does not exist (this server serves \"{name}\")."),
            ));
        }
    }
    if !body.is_object() {
        return Err(Error::bad("the body is not a JSON object"));
    }
    let served = [
        Served { field: "n", ok: |v| v.as_f64() == Some(1.0), serves: "one choice per request (n 1)" },
        Served {
            field: "best_of",
            ok: |v| v.as_f64() == Some(1.0),
            serves: "one choice per request (best_of 1)",
        },
        Served { field: "use_beam_search", ok: |v| v.as_bool() == Some(false), serves: "no beam search" },
        Served { field: "min_p", ok: |v| v.as_f64() == Some(0.0), serves: "no min_p" },
        Served { field: "presence_penalty", ok: |v| v.as_f64() == Some(0.0), serves: "no penalties" },
        Served { field: "frequency_penalty", ok: |v| v.as_f64() == Some(0.0), serves: "no penalties" },
        Served { field: "repetition_penalty", ok: |v| v.as_f64() == Some(1.0), serves: "no penalties" },
        Served {
            field: "logit_bias",
            ok: |v| {
                v.as_array().is_some_and(<[Json]>::is_empty)
                    || (v.is_object() && v.members().is_empty())
                    || v.as_str() == Some("")
            },
            serves: "no logit bias",
        },
        Served { field: "min_tokens", ok: |v| v.as_f64() == Some(0.0), serves: "no min_tokens" },
        Served { field: "prompt_logprobs", ok: |_| false, serves: "no prompt logprobs" },
        Served {
            field: "allowed_token_ids",
            ok: |v| v.as_array().is_some_and(<[Json]>::is_empty),
            serves: "every token allowed",
        },
        Served {
            field: "bad_words",
            ok: |v| v.as_array().is_some_and(<[Json]>::is_empty),
            serves: "no bad words",
        },
        Served {
            field: "logits_processors",
            ok: |v| v.as_array().is_some_and(<[Json]>::is_empty),
            serves: "no logits processors",
        },
        Served {
            field: "response_format",
            ok: |v| v.get("type").as_str() == Some("text"),
            serves: "unconstrained text (response_format text)",
        },
        Served { field: "guided_json", ok: |_| false, serves: "unconstrained text" },
        Served { field: "guided_regex", ok: |_| false, serves: "unconstrained text" },
        Served { field: "guided_choice", ok: |_| false, serves: "unconstrained text" },
        Served { field: "guided_grammar", ok: |_| false, serves: "unconstrained text" },
        Served { field: "structured_outputs", ok: |_| false, serves: "unconstrained text" },
    ];
    for s in &served {
        let v = body.get(s.field);
        if !v.is_null() && !(s.ok)(v) {
            return Err(Error::field(
                s.field,
                format!("{} {v} is not supported: this server serves {}", s.field, s.serves),
            ));
        }
    }
    let tok = &server.tok;
    let max_field = if api == Api::Chat && !body.get("max_completion_tokens").is_null() {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    let max_tokens = match body.get(max_field) {
        Json::Null => None,
        v => match v.as_u64() {
            Some(n) if n > 0 => Some(n as usize),
            _ => return Err(Error::field(max_field, format!("{max_field} {v}: a positive integer"))),
        },
    };
    let mut stops: Vec<String> = match body.get("stop") {
        Json::Null => Vec::new(),
        Json::String(s) => vec![s.clone()],
        Json::Array(a) => a
            .iter()
            .map(|s| s.as_str().map(str::to_owned))
            .collect::<Option<_>>()
            .ok_or_else(|| Error::field("stop", "stop: a string or a list of strings"))?,
        v => return Err(Error::field("stop", format!("stop {v}: a string or a list of strings"))),
    };
    stops.retain(|s| !s.is_empty());
    let mut stop: Vec<u32> = match body.get("stop_token_ids") {
        Json::Null => Vec::new(),
        Json::Array(a) => a
            .iter()
            .map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok()))
            .collect::<Option<_>>()
            .ok_or_else(|| Error::field("stop_token_ids", "stop_token_ids: a list of token ids"))?,
        v => return Err(Error::field("stop_token_ids", format!("stop_token_ids {v}: a list of token ids"))),
    };
    if !body.flag("ignore_eos", false)? {
        stop.push(server.eos);
    }
    let special = if body.flag("skip_special_tokens", true)? { Special::Skip } else { Special::Keep };
    let include_stop = body.flag("include_stop_str_in_output", false)?;
    let stream = stream(body)?;
    let nucleus = server.config.nucleus;
    let mut sampling = sampling(body, nucleus)?;
    let top = logprobs(body, api, tok.vocab_size(), nucleus)?;
    let as_ids = body.flag("return_tokens_as_token_ids", false)?;
    let (prompt, endpoint) = match api {
        Api::Completions => completion(server, body, top.is_some())?,
        Api::Chat => chat(server, body)?,
    };
    if prompt.is_empty() {
        return Err(Error::field("prompt", "the prompt is empty"));
    }
    if let Some(&t) = prompt.iter().find(|&&t| t as usize >= tok.vocab_size()) {
        return Err(Error::field(
            "prompt",
            format!("token id {t} is past the vocabulary ({})", tok.vocab_size()),
        ));
    }
    let limit = length_check(server.context, prompt.len(), max_tokens, max_field)?;
    sampling.logprobs = top.map_or(0, |n| n.max(1) as u32);
    let logprobs = top.map(|n| Logprobs {
        tok,
        n,
        as_ids,
        floor: server.config.logprob_floor,
        context: server.config.logprob_context,
    });
    let detok = Detok::new(tok, special, stop.clone(), stops, include_stop);
    let (out, outs) = mpsc::channel();
    let ask = Ask { endpoint, stream, detok, logprobs, prompt: prompt.clone() };
    Ok((Request { prompt, sampling, limit, stop, out }, Reply::new(server, outs, ask, limit)))
}

fn chat<'s>(server: &'s Server, body: &Json) -> Result<(Vec<u32>, Endpoint<'s>), Error> {
    if body.get("functions").as_array().is_some_and(|t| !t.is_empty()) {
        return Err(Error::field("functions", "functions is not supported: send tools"));
    }
    let model::Chat { prompt, reader } = model::chat(&server.config.grammar, body)?;
    Ok((server.tok.encode(&prompt, false).map_err(|e| Error::field("messages", e))?, Endpoint::Chat(reader)))
}

fn completion<'s>(
    server: &'s Server,
    body: &Json,
    logprobs: bool,
) -> Result<(Vec<u32>, Endpoint<'s>), Error> {
    if !body.get("suffix").is_null() && body.get("suffix").as_str() != Some("") {
        return Err(Error::field("suffix", "suffix is not supported"));
    }
    let echo = body.get("echo").as_bool() == Some(true);
    if echo && logprobs {
        return Err(Error::field("echo", "echo with logprobs (the prompt's logprobs) is not supported"));
    }
    let (ids, text) = completion_prompt(server, body.get("prompt"))?;
    let echo = match text {
        _ if !echo => String::new(),
        Some(text) => text.to_owned(),
        None => server.tok.decode(&ids, Special::Keep),
    };
    Ok((ids, Endpoint::Completion { echo }))
}

fn stream(body: &Json) -> Result<Stream, Error> {
    let options = body.get("stream_options");
    if !body.flag("stream", false)? {
        if !options.members().is_empty() {
            return Err(Error::validation(
                "Stream options can only be defined when `stream=True`.",
                Some("stream_options"),
                None,
            ));
        }
        return Ok(Stream::Whole);
    }
    let usage = options.flag("include_usage", false)?;
    Ok(Stream::Events { usage, continuous: usage && options.flag("continuous_usage_stats", false)? })
}

fn completion_prompt<'j>(server: &Server, v: &'j Json) -> Result<(Vec<u32>, Option<&'j str>), Error> {
    let one = match v {
        Json::Array(a) if a.len() == 1 && a[0].as_number().is_none() => &a[0],
        v => v,
    };
    match one {
        Json::String(text) => {
            Ok((server.tok.encode(text, true).map_err(|e| Error::field("prompt", e))?, Some(text)))
        }
        Json::Array(a) if a.iter().all(|t| t.as_number().is_some()) => {
            match a.iter().map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok())).collect() {
                Some(ids) => Ok((ids, None)),
                None => Err(Error::field("prompt", "prompt: token ids are integers from 0 to 2^32 - 1")),
            }
        }
        Json::Array(_) => {
            Err(Error::field("prompt", "a list of prompts: this server takes one prompt per request"))
        }
        Json::Null => Err(Error::field("prompt", "prompt is missing")),
        v => Err(Error::field("prompt", format!("prompt {v}: a string or a list of token ids"))),
    }
}

fn length_check(
    context: usize,
    prompt: usize,
    max_tokens: Option<usize>,
    field: &str,
) -> Result<usize, Error> {
    let m = max_tokens.unwrap_or(0);
    if m > context {
        return Err(Error::validation(
            format!(
                "{field}={m} cannot be greater than max_model_len={context}. Please request fewer output tokens."
            ),
            Some(field),
            Some(m.to_string()),
        ));
    }
    if prompt > context - m {
        return Err(Error::validation(
            format!(
                "This model's maximum context length is {context} tokens. However, you requested {m} output tokens and your prompt contains {prompt} input tokens, for a total of {} tokens. Please reduce the length of the input prompt or the number of requested output tokens.",
                prompt + m
            ),
            Some("input_tokens"),
            Some(prompt.to_string()),
        ));
    }
    match max_tokens {
        Some(m) => Ok(m),
        None if prompt >= context => Err(Error::validation(
            format!(
                "The decoder prompt (length {prompt}) plus the number of requested output tokens (at least 1) is longer than the maximum model length of {context}. Make sure that `max_model_len` is no smaller than the number of text tokens (prompt + requested output tokens)."
            ),
            None,
            None,
        )),
        None => Ok(context - prompt),
    }
}

fn logprobs(body: &Json, api: Api, vocab: usize, nucleus: usize) -> Result<Option<usize>, Error> {
    let count = |field: &'static str| -> Result<Option<i64>, Error> {
        match body.get(field) {
            Json::Null => Ok(None),
            v => match v.as_i64().or_else(|| v.as_f64().filter(|x| x.fract() == 0.0).map(|x| x as i64)) {
                Some(n) => Ok(Some(n)),
                None => Err(Error::validation(
                    format!("`{field}` must be an integer."),
                    Some(field),
                    Some(v.as_str().map_or_else(|| v.to_string(), str::to_owned)),
                )),
            },
        }
    };
    let (n, field) = if api == Api::Chat {
        let top = count("top_logprobs")?;
        if !body.flag("logprobs", false)? {
            if top.is_some_and(|n| n != 0) {
                return Err(Error::validation(
                    "when using `top_logprobs`, `logprobs` must be set to true.",
                    Some("top_logprobs"),
                    None,
                ));
            }
            return Ok(None);
        }
        (top.unwrap_or(0), "top_logprobs")
    } else {
        match count("logprobs")? {
            Some(n) => (n, "logprobs"),
            None => return Ok(None),
        }
    };
    if n < -1 {
        return Err(Error::validation(
            format!("`{field}` must be a positive value or -1."),
            Some(field),
            Some(n.to_string()),
        ));
    }
    let n = if n == -1 { vocab } else { n as usize };
    if n > nucleus {
        return Err(Error::validation(
            format!("Requested sample logprobs of {n}, which is greater than max allowed: {nucleus}"),
            Some("logprobs"),
            Some(n.to_string()),
        ));
    }
    Ok(Some(n))
}

fn sampling(body: &Json, nucleus: usize) -> Result<Sampling, Error> {
    let seed = match body.get("seed") {
        Json::Null => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64),
        v => match (v.as_i64(), v.as_u64()) {
            (Some(s), _) => s as u64,
            (None, Some(s)) => s,
            (None, None) => return Err(Error::field("seed", "seed: an integer")),
        },
    };
    let number = |field: &'static str, default: f64, ok: fn(f64) -> bool, what: &str| -> Result<f64, Error> {
        match body.get(field) {
            Json::Null => Ok(default),
            v => v
                .as_f64()
                .filter(|&x| ok(x))
                .ok_or_else(|| Error::field(field, format!("{field} {v}: {what}"))),
        }
    };
    let temperature = number("temperature", 1.0, |x| x >= 0.0 && x.is_finite(), "a number at least 0")?;
    let top_p = number("top_p", 1.0, |x| x > 0.0 && x <= 1.0, "a number in (0, 1]")?;
    let top_k = match body.get("top_k") {
        Json::Null => 0,
        v if v.as_i64() == Some(-1) => 0,
        v => match v.as_u64() {
            Some(k) if k as usize <= nucleus => k as u32,
            Some(k) => {
                return Err(Error::field(
                    "top_k",
                    format!(
                        "top_k {k} is not supported: this server draws from each row's {nucleus} largest logits"
                    ),
                ));
            }
            None => return Err(Error::field("top_k", format!("top_k {v}: -1, 0 or a positive integer"))),
        },
    };
    Ok(Sampling { temperature: temperature as f32, top_p: top_p as f32, top_k, seed, logprobs: 0 })
}
