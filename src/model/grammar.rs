use crate::json::Json;

pub struct GrammarConfig {
    pub bos: String,
    pub eos: String,
    pub system: String,
    pub user: String,
    pub assistant: String,
    pub latest_reminder: String,
    pub think: String,
    pub end_think: String,
    pub calls: String,
    pub invoke: String,
    pub parameter: String,
    pub block_start: String,
    pub block_open: String,
    pub block_end: String,
    pub invoke_open: String,
    pub invoke_end: String,
    pub parameter_open: String,
    pub value_end: String,
    pub next_parameter: String,
    pub next_invoke_end: String,
    pub string: String,
    pub not_string: String,
}

impl Default for GrammarConfig {
    fn default() -> GrammarConfig {
        let (calls, invoke, parameter) = ("｜DSML｜ calls", "｜DSML｜ invoke", "｜DSML｜ parameter");
        GrammarConfig {
            bos: "<｜begin▁of▁sentence｜>".into(),
            eos: "<｜end▁of▁sentence｜>".into(),
            system: "<｜System｜>".into(),
            user: "<｜User｜>".into(),
            assistant: "<｜Assistant｜>".into(),
            latest_reminder: "<｜latest_reminder｜>".into(),
            think: "<think>".into(),
            end_think: "</think>".into(),
            calls: calls.into(),
            invoke: invoke.into(),
            parameter: parameter.into(),
            block_start: format!("\n\n<{calls}"),
            block_open: format!("\n\n<{calls}>"),
            block_end: format!("\n</{calls}>"),
            invoke_open: format!("\n<{invoke}"),
            invoke_end: format!("</{invoke}>"),
            parameter_open: format!("<{parameter}"),
            value_end: format!("/{parameter}"),
            next_parameter: format!(">\n<{parameter}"),
            next_invoke_end: format!(">\n</{invoke}>"),
            string: "\" string=\"true\">".into(),
            not_string: "\" string=\"false\">".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub key: String,
    pub value: String,
    pub string: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invoke {
    pub name: String,
    pub params: Vec<Param>,
}

pub enum Item {
    Call(Invoke),
    End,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Miss {
    Wait,
    Broken,
}

impl Param {
    pub fn of(key: &str, value: &Json) -> Param {
        match value.as_str() {
            Some(s) => Param { key: key.to_owned(), value: s.to_owned(), string: true },
            None => Param { key: key.to_owned(), value: value.python(), string: false },
        }
    }

    fn write(&self, g: &GrammarConfig) -> String {
        format!(
            "<{} name=\"{}\" string=\"{}\">{}</{}>",
            g.parameter, self.key, self.string, self.value, g.parameter
        )
    }

    fn read(g: &GrammarConfig, body: &str) -> Option<Param> {
        let rest = body.strip_prefix(" name=\"")?.strip_suffix('<')?;
        let (key, marker, value) = split_first(rest, &[g.string.as_str(), g.not_string.as_str()])?;
        Some(Param { key: key.to_owned(), value: value.to_owned(), string: marker == g.string })
    }

    fn json(&self) -> String {
        if self.string { Json::String(self.value.clone()).python() } else { self.value.clone() }
    }
}

impl Invoke {
    fn write(&self, g: &GrammarConfig) -> String {
        let params: Vec<String> = self.params.iter().map(|p| p.write(g)).collect();
        format!("<{} name=\"{}\">\n{}\n</{}>", g.invoke, self.name, params.join("\n"), g.invoke)
    }

    pub fn arguments(&self) -> String {
        let members: Vec<String> = self
            .params
            .iter()
            .map(|p| format!("{}: {}", Json::String(p.key.clone()).python(), p.json()))
            .collect();
        format!("{{{}}}", members.join(", "))
    }
}

impl GrammarConfig {
    pub fn write_block<'c>(&self, calls: impl IntoIterator<Item = &'c Invoke>) -> String {
        let calls: Vec<String> = calls.into_iter().map(|c| c.write(self)).collect();
        format!("{}\n{}{}", self.block_open, calls.join("\n"), self.block_end)
    }

    pub fn forced_block(&self, name: Option<&str>) -> String {
        match name {
            Some(name) => format!("{}{} name=\"{name}\">\n", self.block_open, self.invoke_open),
            None => format!("{}{} name=\"", self.block_open, self.invoke_open),
        }
    }

    pub fn read_first(&self, text: &str) -> Result<(Item, usize), Miss> {
        literal(text, &[self.block_open.as_str()])?;
        let (item, n) = self.read_item(&text[self.block_open.len()..])?;
        Ok((item, self.block_open.len() + n))
    }

    pub fn read_item(&self, text: &str) -> Result<(Item, usize), Miss> {
        let mut rest = text;
        if literal_in(&mut rest, &[self.invoke_open.as_str(), self.block_end.as_str()])? == self.block_end {
            return Ok((Item::End, text.len() - rest.len()));
        }
        let (head, mut next) =
            until_in(&mut rest, &[self.parameter_open.as_str(), self.invoke_end.as_str()])?;
        let name = head
            .trim_start()
            .strip_prefix("name=\"")
            .and_then(|h| h.strip_suffix("\">\n"))
            .ok_or(Miss::Broken)?;
        let mut params: Vec<Param> = Vec::new();
        while next == self.parameter_open {
            let (body, _) = until_in(&mut rest, &[self.value_end.as_str()])?;
            let param = Param::read(self, body)
                .filter(|p| params.iter().all(|q| q.key != p.key))
                .ok_or(Miss::Broken)?;
            params.push(param);
            next = if literal_in(&mut rest, &[self.next_parameter.as_str(), self.next_invoke_end.as_str()])?
                == self.next_parameter
            {
                self.parameter_open.as_str()
            } else {
                self.invoke_end.as_str()
            };
        }
        Ok((Item::Call(Invoke { name: name.to_owned(), params }), text.len() - rest.len()))
    }
}

fn literal<'l>(text: &str, literals: &[&'l str]) -> Result<&'l str, Miss> {
    match literals.iter().find(|l| text.starts_with(**l)) {
        Some(&l) => Ok(l),
        None if literals.iter().any(|l| l.starts_with(text)) => Err(Miss::Wait),
        None => Err(Miss::Broken),
    }
}

fn literal_in<'l>(rest: &mut &str, literals: &[&'l str]) -> Result<&'l str, Miss> {
    let l = literal(rest, literals)?;
    *rest = &rest[l.len()..];
    Ok(l)
}

fn until_in<'t, 'l>(rest: &mut &'t str, markers: &[&'l str]) -> Result<(&'t str, &'l str), Miss> {
    let (before, marker, after) = split_first(rest, markers).ok_or(Miss::Wait)?;
    *rest = after;
    Ok((before, marker))
}

fn split_first<'t, 'l>(text: &'t str, markers: &[&'l str]) -> Option<(&'t str, &'l str, &'t str)> {
    let (at, marker) = markers.iter().filter_map(|&m| text.find(m).map(|at| (at, m))).min()?;
    Some((&text[..at], marker, &text[at + marker.len()..]))
}
