use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::{self, Write};

use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Number, Value};

use crate::error::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Json>),
    Object(Vec<Member>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub key: String,
    pub value: Json,
}

impl Json {
    pub fn parse(text: &str) -> serde_json::Result<Json> {
        serde_json::from_str(text)
    }

    pub fn get(&self, key: &str) -> &Json {
        self.members().iter().find(|m| m.key == key).map_or(&Json::Null, |m| &m.value)
    }

    pub fn members(&self) -> &[Member] {
        if let Json::Object(m) = self { m } else { &[] }
    }

    pub fn is_null(&self) -> bool {
        *self == Json::Null
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Json::Object(_))
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::String(s) = self { Some(s) } else { None }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self { Some(*b) } else { None }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        if let Json::Array(a) = self { Some(a) } else { None }
    }

    pub fn as_number(&self) -> Option<&Number> {
        if let Json::Number(n) = self { Some(n) } else { None }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_number().and_then(Number::as_u64)
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_number().and_then(Number::as_i64)
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.as_number().and_then(Number::as_f64)
    }

    pub fn flag(&self, field: &str, default: bool) -> Result<bool, Error> {
        match self.get(field) {
            Json::Null => Ok(default),
            Json::Bool(b) => Ok(*b),
            v => Err(Error::field(field, format!("{field} {v}: a boolean"))),
        }
    }

    pub fn python(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Number(n) => match n.as_f64() {
                Some(f) if n.is_f64() => write_float(f, out),
                _ => out.push_str(&n.to_string()),
            },
            Json::String(s) => write_quoted(s, out),
            Json::Array(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Object(members) => {
                out.push('{');
                for (i, m) in members.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_quoted(&m.key, out);
                    out.push_str(": ");
                    m.value.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_quoted(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_float(x: f64, out: &mut String) {
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').expect("Rust writes {:e} with an exponent");
    let exp: i32 = exp.parse().expect("Rust writes a decimal exponent");
    if (-4..16).contains(&exp) {
        let fixed = x.to_string();
        out.push_str(&fixed);
        if !fixed.contains('.') {
            out.push_str(".0");
        }
    } else {
        let _ = write!(out, "{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", Value::from(self))
    }
}

impl From<&Json> for Value {
    fn from(j: &Json) -> Value {
        match j {
            Json::Null => Value::Null,
            Json::Bool(b) => Value::Bool(*b),
            Json::Number(n) => Value::Number(n.clone()),
            Json::String(s) => Value::String(s.clone()),
            Json::Array(items) => Value::Array(items.iter().map(Value::from).collect()),
            Json::Object(members) => {
                Value::Object(members.iter().map(|m| (m.key.clone(), Value::from(&m.value))).collect())
            }
        }
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("JSON")
    }

    fn visit_unit<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_bool<E>(self, b: bool) -> Result<Json, E> {
        Ok(Json::Bool(b))
    }

    fn visit_u64<E>(self, n: u64) -> Result<Json, E> {
        Ok(Json::Number(n.into()))
    }

    fn visit_i64<E>(self, n: i64) -> Result<Json, E> {
        Ok(Json::Number(n.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, n: f64) -> Result<Json, E> {
        Number::from_f64(n).map(Json::Number).ok_or_else(|| E::custom(format!("{n}: not a JSON number")))
    }

    fn visit_str<E>(self, s: &str) -> Result<Json, E> {
        Ok(Json::String(s.to_owned()))
    }

    fn visit_string<E>(self, s: String) -> Result<Json, E> {
        Ok(Json::String(s))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<Json>()? {
            items.push(item);
        }
        Ok(Json::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let (mut members, mut at): (Vec<Member>, HashMap<String, usize>) = (Vec::new(), HashMap::new());
        while let Some((key, value)) = map.next_entry::<String, Json>()? {
            match at.entry(key) {
                Entry::Occupied(e) => members[*e.get()].value = value,
                Entry::Vacant(e) => {
                    members.push(Member { key: e.key().clone(), value });
                    e.insert(members.len() - 1);
                }
            }
        }
        Ok(Json::Object(members))
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Json, D::Error> {
        d.deserialize_any(JsonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeated_key_keeps_its_first_place_and_last_value() {
        let j = Json::parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(j.members().iter().map(|m| m.key.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(j.get("a").as_u64(), Some(3));
    }

    #[test]
    fn a_wide_object_parses_in_linear_time() {
        let text =
            format!("{{{}}}", (0..200_000).map(|i| format!("\"k{i}\": {i}")).collect::<Vec<_>>().join(", "));
        let t = std::time::Instant::now();
        assert_eq!(Json::parse(&text).unwrap().members().len(), 200_000);
        assert!(t.elapsed().as_secs() < 2, "{:?}", t.elapsed());
    }
}
