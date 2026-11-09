use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    pub param: Option<String>,
}

impl Error {
    pub fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Error {
        Error { status, kind, message: message.into(), param: None }
    }

    pub fn bad(message: impl Into<String>) -> Error {
        Error::new(400, "invalid_request_error", message)
    }

    pub fn field(param: &str, message: impl Into<String>) -> Error {
        Error { param: Some(param.to_owned()), ..Error::bad(message) }
    }

    pub fn validation(message: impl Into<String>, param: Option<&str>, value: Option<String>) -> Error {
        let extras: Vec<String> = param
            .map(|p| format!("parameter={p}"))
            .into_iter()
            .chain(value.map(|v| format!("value={v}")))
            .collect();
        let message = message.into();
        let message = if extras.is_empty() { message } else { format!("{message} ({})", extras.join(", ")) };
        Error { status: 400, kind: "BadRequestError", message, param: param.map(str::to_owned) }
    }

    pub fn json(&self) -> Value {
        json!({ "error": { "message": self.message, "type": self.kind, "param": self.param, "code": self.status } })
    }
}
