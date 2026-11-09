use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::error::Error;
use crate::json::Json;

pub struct HttpConfig {
    pub max_head: usize,
    pub max_body: usize,
    pub read: Duration,
    pub write: Duration,
    pub linger: Duration,
    pub connections: usize,
    pub poll: Duration,
}

impl Default for HttpConfig {
    fn default() -> HttpConfig {
        HttpConfig {
            max_head: 64 << 10,
            max_body: 16 << 20,
            read: Duration::from_secs(30),
            write: Duration::from_secs(30),
            linger: Duration::from_secs(1),
            connections: 256,
            poll: Duration::from_millis(50),
        }
    }
}

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Request {
    pub fn read(s: &mut TcpStream, key: Option<&str>, config: &HttpConfig) -> Result<Option<Request>, Error> {
        let mut s = Timed::new(s, config.read);
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        let (method, path, bearer, len, expect, start) = loop {
            let n = s.read(&mut chunk).map_err(|e| Error::bad(format!("reading the request: {e}")))?;
            if n == 0 {
                return if buf.is_empty() {
                    Ok(None)
                } else {
                    Err(Error::bad("the request ended inside its headers"))
                };
            }
            buf.extend_from_slice(&chunk[..n]);
            let mut headers = [httparse::EMPTY_HEADER; 64];
            let mut req = httparse::Request::new(&mut headers);
            match req.parse(&buf) {
                Ok(httparse::Status::Complete(start)) => {
                    let header = |name: &str| {
                        req.headers
                            .iter()
                            .find(|h| h.name.eq_ignore_ascii_case(name))
                            .map(|h| String::from_utf8_lossy(h.value).into_owned())
                    };
                    if header("transfer-encoding").is_some() {
                        return Err(Error::new(
                            411,
                            "invalid_request_error",
                            "chunked request bodies are not supported: send Content-Length",
                        ));
                    }
                    let len = match header("content-length") {
                        Some(v) => v
                            .trim()
                            .parse::<usize>()
                            .map_err(|_| Error::bad(format!("Content-Length {v:?}")))?,
                        None => 0,
                    };
                    let bearer = header("authorization").and_then(|a| {
                        a.split_once(' ')
                            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                            .map(|(_, k)| k.trim().to_owned())
                    });
                    let expect = header("expect").is_some_and(|e| e.eq_ignore_ascii_case("100-continue"));
                    let path = req.path.unwrap_or("");
                    let path = path.split_once('?').map_or(path, |(p, _)| p);
                    break (req.method.unwrap_or("").to_owned(), path.to_owned(), bearer, len, expect, start);
                }
                Ok(httparse::Status::Partial) if buf.len() < config.max_head => {}
                Ok(httparse::Status::Partial) => {
                    return Err(Error::new(413, "invalid_request_error", "headers too large"));
                }
                Err(e) => return Err(Error::bad(format!("malformed request: {e}"))),
            }
        };
        if let Some(key) = key
            && path != "/health"
            && !bearer.is_some_and(|given| same(key.as_bytes(), given.as_bytes()))
        {
            return Err(Error::new(
                401,
                "authentication_error",
                "send the server's API key as Authorization: Bearer KEY",
            ));
        }
        if len > config.max_body {
            return Err(Error::new(
                413,
                "invalid_request_error",
                format!("a body of {len} bytes (at most {})", config.max_body),
            ));
        }
        if expect && len > buf.len() - start {
            let _ = s.s.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        }
        let mut body = buf[start..].to_vec();
        body.truncate(len);
        while body.len() < len {
            let n = s.read(&mut chunk).map_err(|e| Error::bad(format!("reading the body: {e}")))?;
            if n == 0 {
                return Err(Error::bad("the request ended inside its body"));
            }
            body.extend_from_slice(&chunk[..n.min(len - body.len())]);
        }
        Ok(Some(Request { method, path, body }))
    }

    pub fn post(&self) -> Result<(), Error> {
        if self.method == "POST" {
            Ok(())
        } else {
            Err(Error::new(405, "invalid_request_error", format!("{} {}: POST only", self.method, self.path)))
        }
    }

    pub fn json(&self) -> Result<Json, Error> {
        serde_json::from_slice(&self.body).map_err(|e| Error::bad(format!("the body is not JSON: {e}")))
    }
}

impl From<Error> for Response {
    fn from(e: Error) -> Response {
        Response::json(e.status, &e.json())
    }
}

impl Response {
    pub fn json(status: u16, v: &Value) -> Response {
        Response { status, content_type: "application/json", body: v.to_string().into_bytes() }
    }

    pub fn text(content_type: &'static str, body: impl Into<Vec<u8>>) -> Response {
        Response { status: 200, content_type, body: body.into() }
    }

    pub fn write(&self, w: &mut impl Write) {
        let reason = match self.status {
            200 => "OK",
            400 => "Bad Request",
            401 => "Unauthorized",
            404 => "Not Found",
            405 => "Method Not Allowed",
            411 => "Length Required",
            413 => "Payload Too Large",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Error",
        };
        let head = format!(
            "HTTP/1.1 {} {reason}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.status,
            self.content_type,
            self.body.len()
        );
        let _ = w.write_all(head.as_bytes()).and_then(|()| w.write_all(&self.body)).and_then(|()| w.flush());
    }
}

pub fn write_events(w: &mut impl Write) -> bool {
    w.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").and_then(|()| w.flush()).is_ok()
}

pub fn write_event(w: &mut impl Write, v: &Value) -> bool {
    w.write_all(format!("data: {v}\n\n").as_bytes()).and_then(|()| w.flush()).is_ok()
}

pub fn write_done(w: &mut impl Write) {
    let _ = w.write_all(b"data: [DONE]\n\n").and_then(|()| w.flush());
}

pub fn close(s: &mut TcpStream, config: &HttpConfig) {
    let _ = s.shutdown(Shutdown::Write);
    let mut s = Timed::new(s, config.linger);
    let (mut chunk, mut drained) = ([0u8; 8192], 0);
    while drained < config.max_head {
        match s.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => drained += n,
        }
    }
}

struct Timed<'s> {
    s: &'s mut TcpStream,
    until: Instant,
}

impl<'s> Timed<'s> {
    fn new(s: &'s mut TcpStream, within: Duration) -> Timed<'s> {
        Timed { s, until: Instant::now() + within }
    }

    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = self.until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "out of time"));
        }
        self.s.set_read_timeout(Some(left))?;
        self.s.read(buf)
    }
}

fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |d, (x, y)| d | (x ^ y)) == 0
}

pub fn gone(s: &TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() {
        return false;
    }
    let gone = match s.peek(&mut [0u8; 1]) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => !matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted),
    };
    let _ = s.set_nonblocking(false);
    gone
}
