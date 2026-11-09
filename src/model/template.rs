use std::collections::HashMap;

use crate::error::Error;
use crate::json::Json;

use crate::model::grammar::{GrammarConfig, Invoke, Param};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Chat,
    Thinking,
}

impl Mode {
    fn parse(v: &Json) -> Result<Mode, Error> {
        Ok(if v.flag("thinking", false)? || v.flag("enable_thinking", false)? {
            Mode::Thinking
        } else {
            Mode::Chat
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Effort(u8);

impl Default for Effort {
    fn default() -> Effort {
        Effort(75)
    }
}

impl Effort {
    fn parse(v: &Json) -> Result<Effort, Error> {
        let refuse = || {
            Error::field(
                "reasoning_effort",
                "reasoning_effort: an integer from 1 to 100, or \"low\" (50), \"high\" (75) or \"max\" (100)",
            )
        };
        if v.is_null() {
            return Ok(Effort::default());
        }
        let budget = match v.as_str() {
            Some("low") => Some(50),
            Some("high") => Some(75),
            Some("max") => Some(100),
            Some(_) => None,
            None => v.as_number().and_then(|n| n.as_u64()).and_then(|n| u8::try_from(n).ok()),
        };
        match budget {
            Some(b @ 1..=100) => Ok(Effort(b)),
            _ => Err(refuse()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Action,
    Query,
    Authority,
    Domain,
    Title,
    ReadUrl,
}

impl Task {
    fn named() -> [(&'static str, Task); 6] {
        [
            ("action", Task::Action),
            ("query", Task::Query),
            ("authority", Task::Authority),
            ("domain", Task::Domain),
            ("title", Task::Title),
            ("read_url", Task::ReadUrl),
        ]
    }

    fn parse(v: &Json) -> Result<Option<Task>, Error> {
        if v.is_null() {
            return Ok(None);
        }
        match Task::named().into_iter().find(|(name, _)| v.as_str() == Some(name)) {
            Some((_, task)) => Ok(Some(task)),
            None => Err(Error::field(
                "messages",
                format!("task {}: action, query, authority, domain, title or read_url", v.python()),
            )),
        }
    }

    fn token(self) -> String {
        let (name, _) = Task::named().into_iter().find(|(_, t)| *t == self).expect("every task is named");
        format!("<｜{name}｜>")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    function: Json,
}

impl Tool {
    fn parse(v: &Json) -> Result<Tool, Error> {
        let function = v.get("function");
        if !(v.get("type").is_null() || v.get("type").as_str() == Some("function")) {
            return Err(Error::field(
                "tools",
                format!("a tool of type {}: this server serves functions", v.get("type").python()),
            ));
        }
        if !v.get("namespace").is_null() || !function.get("namespace").is_null() {
            return Err(Error::field("tools", "tool namespaces are not supported"));
        }
        match function.get("name").as_str() {
            Some(name) if !name.is_empty() && !name.contains("::") => Ok(Tool { function: function.clone() }),
            _ => {
                Err(Error::field("tools", format!("tool {}: a function with a name (no \"::\")", v.python())))
            }
        }
    }

    pub fn name(&self) -> &str {
        self.function.get("name").as_str().expect("a tool is parsed with a string name")
    }

    fn list(v: &Json) -> Result<Vec<Tool>, Error> {
        if v.is_null() {
            return Ok(Vec::new());
        }
        let Some(list) = v.as_array() else {
            return Err(Error::field("tools", format!("tools {}: a list of tools", v.python())));
        };
        list.iter().map(Tool::parse).collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub id: Option<String>,
    pub invoke: Invoke,
}

impl Call {
    fn parse(v: &Json) -> Result<Call, Error> {
        let function = v.get("function");
        if !v.get("namespace").is_null() || !function.get("namespace").is_null() {
            return Err(Error::field("messages", "tool namespaces are not supported"));
        }
        let (Some(name), Some(arguments)) =
            (function.get("name").as_str(), function.get("arguments").as_str())
        else {
            return Err(Error::field(
                "messages",
                format!("tool call {}: a function with a name and its arguments as a string", v.python()),
            ));
        };
        let parsed = Json::parse(arguments).ok();
        let parsed = if let Some(Json::String(inner)) = &parsed { Json::parse(inner).ok() } else { parsed };
        let params = match &parsed {
            Some(Json::Object(members)) => members.iter().map(|m| Param::of(&m.key, &m.value)).collect(),
            _ => vec![Param::of("arguments", &Json::String(arguments.to_owned()))],
        };
        Ok(Call { id: string_field(v, "id")?, invoke: Invoke { name: name.to_owned(), params } })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Block {
    Text(String),
    Result { call_id: Option<String>, content: String },
}

impl Block {
    fn text(&self) -> String {
        match self {
            Block::Text(text) => text.clone(),
            Block::Result { content, .. } => format!("<tool_result>{content}</tool_result>"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Turn {
    System { content: String, tools: Vec<Tool> },
    User(Vec<Block>),
    Reminder(String),
    Assistant { reasoning: String, content: String, calls: Vec<Call> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub turn: Turn,
    pub task: Option<Task>,
}

fn content_parts(m: &Json) -> Result<Vec<String>, Error> {
    let v = m.get("content");
    if v.is_null() {
        return Ok(Vec::new());
    }
    if let Some(s) = v.as_str() {
        return Ok(vec![s.to_owned()]);
    }
    let Some(list) = v.as_array() else {
        return Err(Error::field(
            "messages",
            format!("content {}: a string or a list of text parts", v.python()),
        ));
    };
    list.iter()
        .map(|part| match (part.get("type").as_str(), part.get("text").as_str()) {
            (Some("text"), Some(text)) => Ok(text.to_owned()),
            _ => Err(Error::field(
                "messages",
                format!("a content part {}: this server serves text only", part.python()),
            )),
        })
        .collect()
}

fn content_text(m: &Json) -> Result<String, Error> {
    Ok(content_parts(m)?.join("\n\n"))
}

fn string_field(m: &Json, field: &str) -> Result<Option<String>, Error> {
    let v = m.get(field);
    match v.as_str() {
        Some(s) => Ok(Some(s.to_owned())),
        None if v.is_null() => Ok(None),
        None => Err(Error::field("messages", format!("{field} {}: a string", v.python()))),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub mode: Mode,
    pub effort: Effort,
    pub ending: Ending,
    pub choice: Choice,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    Auto,
    None,
    Required,
    Named(String),
}

impl Choice {
    fn parse(v: &Json) -> Result<Choice, Error> {
        if v.is_null() {
            return Ok(Choice::Auto);
        }
        match (v.as_str(), v.get("type").as_str(), v.get("function").get("name").as_str()) {
            (Some("auto"), _, _) => Ok(Choice::Auto),
            (Some("none"), _, _) => Ok(Choice::None),
            (Some("required"), _, _) => Ok(Choice::Required),
            (None, Some("function"), Some(name)) => Ok(Choice::Named(name.to_owned())),
            _ => Err(Error::field(
                "tool_choice",
                format!(
                    "tool_choice {}: none, auto, required or {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}",
                    v.python()
                ),
            )),
        }
    }

    pub fn forced(&self, g: &GrammarConfig) -> String {
        match self {
            Choice::Auto | Choice::None => String::new(),
            Choice::Required => g.forced_block(None),
            Choice::Named(name) => g.forced_block(Some(name)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    Open,
    Bare,
    Continue,
}

impl Ending {
    fn parse(body: &Json) -> Result<Ending, Error> {
        if body.flag("continue_final_message", false)? {
            Ok(Ending::Continue)
        } else if body.flag("add_generation_prompt", true)? {
            Ok(Ending::Open)
        } else {
            Ok(Ending::Bare)
        }
    }
}

impl Conversation {
    pub fn parse(body: &Json) -> Result<Conversation, Error> {
        let mode = Mode::parse(body.get("chat_template_kwargs"))?;
        let ending = Ending::parse(body)?;
        let Some(list) = body.get("messages").as_array() else {
            return Err(Error::field("messages", "messages: a list of messages"));
        };
        if list.is_empty() {
            return Err(Error::field("messages", "messages is empty"));
        }
        let mut messages: Vec<Message> = Vec::new();
        for m in list {
            let (message, joins) = Message::parse(m)?;
            joins.onto(&mut messages, message);
        }
        let choice = Choice::parse(body.get("tool_choice"))?;
        let request_tools = Tool::list(body.get("tools"))?;
        if !request_tools.is_empty() && choice != Choice::None {
            match &mut messages[0].turn {
                Turn::System { tools, .. } if tools.is_empty() => *tools = request_tools,
                Turn::System { .. } => {
                    return Err(Error::field(
                        "tools",
                        "tools both in the request and on the first system message",
                    ));
                }
                Turn::User(_) | Turn::Reminder(_) | Turn::Assistant { .. } => messages.insert(
                    0,
                    Message {
                        turn: Turn::System { content: String::new(), tools: request_tools },
                        task: None,
                    },
                ),
            }
        }
        if ending == Ending::Continue
            && !matches!(messages.last().map(|m| &m.turn), Some(Turn::Assistant { .. }))
        {
            return Err(Error::field(
                "continue_final_message",
                "continue_final_message needs an assistant message last",
            ));
        }
        let effort = Effort::parse(body.get("reasoning_effort"))?;
        let mut conversation = Conversation { messages, mode, effort, ending, choice };
        conversation.sort_results();
        conversation.checked()
    }

    fn checked(self) -> Result<Conversation, Error> {
        let refuse = |why: &str| Err(Error::field("tool_choice", why));
        match &self.choice {
            Choice::Auto | Choice::None => Ok(self),
            Choice::Required | Choice::Named(_) if self.mode == Mode::Thinking => {
                refuse("a forced tool_choice needs thinking off: thinking mode reasons before the calls")
            }
            Choice::Required | Choice::Named(_) if self.ending != Ending::Open || !self.opens_turn() => {
                refuse("a forced tool_choice needs the prompt to open the assistant's turn")
            }
            Choice::Required if self.tools().next().is_none() => {
                refuse("tool_choice required: the request has no tools")
            }
            Choice::Named(name) if self.tools().all(|t| t.name() != name) => {
                refuse(&format!("tool_choice names {name:?}: the request has no such tool"))
            }
            Choice::Required | Choice::Named(_) => Ok(self),
        }
    }

    fn sort_results(&mut self) {
        let mut order: HashMap<String, usize> = HashMap::new();
        for m in &mut self.messages {
            match &mut m.turn {
                Turn::Assistant { calls, .. } if !calls.is_empty() => {
                    order = calls
                        .iter()
                        .enumerate()
                        .filter_map(|(i, c)| c.id.clone().map(|id| (id, i)))
                        .collect();
                }
                Turn::User(blocks) => {
                    let is_result = |b: &&mut Block| matches!(b, Block::Result { .. });
                    let mut results: Vec<Block> =
                        blocks.iter_mut().filter(is_result).map(|b| b.clone()).collect();
                    if results.len() < 2 || order.is_empty() {
                        continue;
                    }
                    results.sort_by_key(|b| match b {
                        Block::Result { call_id: Some(id), .. } => order.get(id).map_or(0, |&i| i),
                        Block::Result { call_id: None, .. } | Block::Text(_) => 0,
                    });
                    for (place, result) in blocks.iter_mut().filter(is_result).zip(results) {
                        *place = result;
                    }
                }
                Turn::Assistant { .. } | Turn::System { .. } | Turn::Reminder(_) => {}
            }
        }
    }

    fn opens_turn(&self) -> bool {
        let last = self.messages.len() - 1;
        self.messages[last].task.is_none() && self.asks(last)
    }

    pub fn tools(&self) -> impl Iterator<Item = &Tool> {
        self.messages.iter().flat_map(|m| match &m.turn {
            Turn::System { tools, .. } => tools.as_slice(),
            Turn::User(_) | Turn::Reminder(_) | Turn::Assistant { .. } => &[],
        })
    }

    fn asks(&self, i: usize) -> bool {
        matches!(self.messages[i].turn, Turn::User(_))
            || (i > 0 && matches!(self.messages[i].turn, Turn::System { .. }))
    }

    fn reasons(&self, turn: usize) -> bool {
        let last_ask = (0..self.messages.len()).rev().find(|&i| self.asks(i));
        self.mode == Mode::Thinking && (self.tools().next().is_some() || last_ask.is_none_or(|a| turn > a))
    }

    pub fn render(&self, g: &GrammarConfig) -> String {
        let mut out = g.bos.clone();
        out.push_str(&self.preamble(g));
        for i in 0..self.messages.len() {
            out.push_str(&self.message(g, i));
            out.push_str(&self.handover(g, i));
        }
        out + &self.choice.forced(g)
    }

    fn preamble(&self, g: &GrammarConfig) -> String {
        let thinking = self.mode == Mode::Thinking;
        let system = thinking || matches!(self.messages[0].turn, Turn::System { .. });
        let effort = if thinking {
            format!(
                "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the reasoning)\n\n",
                self.effort.0
            )
        } else {
            String::new()
        };
        format!("{}{effort}", if system { g.system.as_str() } else { "" })
    }

    fn message(&self, g: &GrammarConfig, i: usize) -> String {
        match &self.messages[i].turn {
            Turn::System { content, tools } => {
                format!("{}{content}{}", if i > 0 { g.system.as_str() } else { "" }, tools_prompt(g, tools))
            }
            Turn::User(blocks) => {
                let blocks: Vec<String> = blocks.iter().map(Block::text).collect();
                format!("{}{}", g.user, blocks.join("\n\n"))
            }
            Turn::Reminder(content) => format!("{}{content}", g.latest_reminder),
            Turn::Assistant { reasoning, content, calls } => {
                let answers_task = i > 0 && self.messages[i - 1].task.is_some();
                let reasoning = if self.reasons(i) && !answers_task {
                    format!("{reasoning}{}", g.end_think)
                } else {
                    String::new()
                };
                let calls = if calls.is_empty() {
                    String::new()
                } else {
                    g.write_block(calls.iter().map(|c| &c.invoke))
                };
                let continued = self.ending == Ending::Continue && i == self.messages.len() - 1;
                format!("{reasoning}{content}{calls}{}", if continued { "" } else { g.eos.as_str() })
            }
        }
    }

    fn handover(&self, g: &GrammarConfig, i: usize) -> String {
        let next = self.messages.get(i + 1);
        if next.is_some_and(|n| !matches!(n.turn, Turn::Assistant { .. } | Turn::Reminder(_))) {
            return String::new();
        }
        let think = |on: bool| if on { g.think.as_str() } else { g.end_think.as_str() };
        match self.messages[i].task {
            Some(Task::Action) => {
                format!("{}{}{}", g.assistant, think(self.mode == Mode::Thinking), Task::Action.token())
            }
            Some(task) => task.token(),
            None if self.asks(i) && (next.is_some() || self.ending == Ending::Open) => {
                format!("{}{}", g.assistant, think(self.reasons(i + 1)))
            }
            None => String::new(),
        }
    }
}

fn tools_prompt(g: &GrammarConfig, tools: &[Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let schemas: Vec<String> = tools.iter().map(|t| t.function.python()).collect();
    let GrammarConfig { calls, invoke, parameter, think, end_think, .. } = g;
    format!(
        "\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{calls}>\" block like the following:\n\n\
         <{calls}>\n<{invoke} name=\"$TOOL_NAME\">\n<{parameter} name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{parameter}>\n...\n</{invoke}>\n\
         <{invoke} name=\"$TOOL_NAME2\">\n...\n</{invoke}>\n</{calls}>\n\n\
         String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
         If thinking_mode is enabled (triggered by {think}), you MUST output your complete reasoning inside {think}...{end_think} BEFORE any tool calls or final response.\n\n\
         Otherwise, output directly after {end_think} with tool calls or final response.\n\n### Available Tool Schemas\n\n{}\n\n\
         You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n",
        schemas.join("\n")
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Joins {
    AnyUser,
    UntaskedUser,
    Nothing,
}

impl Joins {
    fn onto(self, messages: &mut Vec<Message>, message: Message) {
        let into = match (self, messages.last_mut()) {
            (Joins::AnyUser, Some(Message { turn: Turn::User(before), .. })) => before,
            (Joins::UntaskedUser, Some(Message { turn: Turn::User(before), task: None })) => before,
            _ => return messages.push(message),
        };
        if let Turn::User(blocks) = message.turn {
            into.extend(blocks);
        }
    }
}

impl Message {
    fn parse(m: &Json) -> Result<(Message, Joins), Error> {
        let task = Task::parse(m.get("task"))?;
        let (turn, joins) = match m.get("role").as_str() {
            Some("system" | "developer") => (
                Turn::System { content: content_text(m)?, tools: Tool::list(m.get("tools"))? },
                Joins::Nothing,
            ),
            Some("user") if m.get("content").is_null() => {
                (Turn::User(vec![Block::Text(String::new())]), Joins::UntaskedUser)
            }
            Some("user") => {
                (Turn::User(content_parts(m)?.into_iter().map(Block::Text).collect()), Joins::UntaskedUser)
            }
            Some("tool") => {
                if m.get("content").is_null() {
                    return Err(Error::field("messages", "a tool message without content"));
                }
                let result =
                    Block::Result { call_id: string_field(m, "tool_call_id")?, content: content_text(m)? };
                (Turn::User(vec![result]), Joins::AnyUser)
            }
            Some("latest_reminder") => (Turn::Reminder(content_text(m)?), Joins::Nothing),
            Some("assistant") => {
                let calls = m.get("tool_calls");
                let calls = match calls.as_array() {
                    Some(list) => list.iter().map(Call::parse).collect::<Result<_, _>>()?,
                    None if calls.is_null() => Vec::new(),
                    None => {
                        return Err(Error::field(
                            "messages",
                            format!("tool_calls {}: a list of tool calls", calls.python()),
                        ));
                    }
                };
                let reasoning = string_field(m, "reasoning_content")?;
                (
                    Turn::Assistant {
                        reasoning: reasoning.unwrap_or_default(),
                        content: content_text(m)?,
                        calls,
                    },
                    Joins::Nothing,
                )
            }
            other => {
                return Err(Error::field(
                    "messages",
                    format!("role {other:?}: system, user, assistant, tool or latest_reminder"),
                ));
            }
        };
        Ok((Message { turn, task }, joins))
    }
}
