pub mod http;
pub mod logprobs;
pub mod metrics;
pub mod request;
pub mod write;

use std::net::{Shutdown, TcpListener, TcpStream};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::api::http::{HttpConfig, Request, Response};
use crate::api::metrics::MetricsConfig;
use crate::api::request::Api;
use crate::api::write::Reply;
use crate::engine::{self, End, Job, Out};
use crate::error::Error;
use crate::model::grammar::GrammarConfig;
use crate::model::tokenizer::Tokenizer;

pub struct ServerConfig {
    pub name: String,
    pub nucleus: usize,
    pub logprob_floor: f32,
    pub logprob_context: usize,
    pub http: HttpConfig,
    pub metrics: MetricsConfig,
    pub grammar: GrammarConfig,
}

impl Default for ServerConfig {
    fn default() -> ServerConfig {
        ServerConfig {
            name: "deepseek-v4.1-flash".into(),
            nucleus: 16,
            logprob_floor: -9999.0,
            logprob_context: 4,
            http: HttpConfig::default(),
            metrics: MetricsConfig::default(),
            grammar: GrammarConfig::default(),
        }
    }
}

pub struct Server {
    pub tok: Tokenizer,
    pub context: usize,
    pub eos: u32,
    queue: Sender<Job>,
    engine: Arc<engine::Metrics>,
    requests: metrics::Requests,
    started: u64,
    next_id: AtomicU64,
    key: Option<String>,
    open: AtomicUsize,
    pub config: ServerConfig,
}

enum Answer<'s> {
    Now(Response),
    Reply(Box<Reply<'s>>),
}

pub fn unix_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Server {
    pub fn new(
        tok: Tokenizer,
        context: usize,
        queue: Sender<Job>,
        engine: Arc<engine::Metrics>,
        key: Option<String>,
        config: ServerConfig,
    ) -> Result<Server, String> {
        let eos = config.grammar.eos.as_str();
        let eos = tok.id(eos).ok_or_else(|| format!("the tokenizer has no {eos}"))?;
        let requests = metrics::Requests::new(context, &config.metrics);
        Ok(Server {
            tok,
            context,
            eos,
            queue,
            engine,
            requests,
            started: unix_secs(),
            next_id: AtomicU64::new(0),
            key,
            open: AtomicUsize::new(0),
            config,
        })
    }

    pub fn serve(self: Arc<Server>, listener: TcpListener) {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let connections = self.config.http.connections;
            if self.open.fetch_add(1, Ordering::Relaxed) >= connections {
                self.open.fetch_sub(1, Ordering::Relaxed);
                Response::from(Error::new(
                    503,
                    "server_error",
                    format!("{connections} connections are open: retry later"),
                ))
                .write(&mut stream);
                let _ = stream.shutdown(Shutdown::Write);
                continue;
            }
            let server = self.clone();
            thread::spawn(move || {
                let _ = panic::catch_unwind(AssertUnwindSafe(|| server.connection(stream)));
                server.open.fetch_sub(1, Ordering::Relaxed);
            });
        }
    }

    pub fn id(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    fn connection(&self, mut s: TcpStream) {
        let config = &self.config.http;
        let _ = s.set_write_timeout(Some(config.write));
        let answer = match Request::read(&mut s, self.key.as_deref(), config) {
            Ok(Some(req)) => self.route(&req),
            Ok(None) => return,
            Err(e) => Err(e),
        };
        match answer {
            Ok(Answer::Now(r)) => r.write(&mut s),
            Ok(Answer::Reply(reply)) => relay(&mut s, *reply, config),
            Err(e) => Response::from(e).write(&mut s),
        }
        http::close(&mut s, config);
    }

    fn route(&self, req: &Request) -> Result<Answer<'_>, Error> {
        match req.path.as_str() {
            "/health" => Ok(Answer::Now(Response::text("text/plain", ""))),
            "/v1/models" => Ok(Answer::Now(Response::json(200, &self.models()))),
            "/metrics" => Ok(Answer::Now(Response::text(
                "text/plain; version=0.0.4; charset=utf-8",
                metrics::render(self),
            ))),
            "/reset_prefix_cache" => self.reset(req).map(Answer::Now),
            "/v1/completions" => self.generate(req, Api::Completions).map(|r| Answer::Reply(Box::new(r))),
            "/v1/chat/completions" => self.generate(req, Api::Chat).map(|r| Answer::Reply(Box::new(r))),
            path => Err(Error::new(404, "not_found_error", format!("{} {path}: no such route", req.method))),
        }
    }

    fn models(&self) -> Value {
        let name = &self.config.name;
        let model = json!({ "id": name, "object": "model", "created": self.started, "owned_by": "hyperkernels", "root": name, "parent": null, "max_model_len": self.context });
        json!({ "object": "list", "data": [model] })
    }

    fn reset(&self, req: &Request) -> Result<Response, Error> {
        req.post()?;
        let (answer, freed) = mpsc::channel();
        self.ask(Job::Reset(answer))?;
        let freed = freed.recv().map_err(|_| gone())?;
        Ok(Response::json(200, &json!({ "reset": true, "dropped": format!("{freed} pages freed") })))
    }

    fn generate(&self, req: &Request, api: Api) -> Result<Reply<'_>, Error> {
        req.post()?;
        let (request, reply) = request::parse(self, api, &req.json()?)?;
        self.ask(Job::Run(request))?;
        Ok(reply)
    }

    fn ask(&self, job: Job) -> Result<(), Error> {
        self.queue.send(job).map_err(|_| gone())
    }
}

fn gone() -> Error {
    Error::new(503, "server_error", "the engine is gone")
}

fn relay(s: &mut TcpStream, mut reply: Reply, config: &HttpConfig) {
    if !reply.start(s) {
        return;
    }
    let mut probed = Instant::now();
    loop {
        if probed.elapsed() >= config.poll {
            if http::gone(s) {
                return;
            }
            probed = Instant::now();
        }
        let out = match reply.out.recv_timeout(config.poll) {
            Ok(out) => out,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                Out::End(End::Refused("the engine dropped the request".into()))
            }
        };
        if !reply.write(out, s) {
            return;
        }
    }
}
