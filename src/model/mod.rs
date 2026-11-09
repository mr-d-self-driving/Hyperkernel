pub mod cli;
pub mod detok;
pub mod grammar;
pub mod reply;
pub mod template;
pub mod tokenizer;

use crate::error::Error;
use crate::json::Json;
use crate::model::grammar::GrammarConfig;
use crate::model::reply::{Calls, Opening, Reader};
use crate::model::template::{Choice, Conversation};

pub struct Chat<'g> {
    pub prompt: String,
    pub reader: Reader<'g>,
}

pub fn chat<'g>(g: &'g GrammarConfig, body: &Json) -> Result<Chat<'g>, Error> {
    let conversation = Conversation::parse(body)?;
    let prompt = conversation.render(g);
    Ok(Chat { reader: reader(g, &conversation, &prompt), prompt })
}

fn reader<'g>(g: &'g GrammarConfig, conversation: &Conversation, prompt: &str) -> Reader<'g> {
    let opening = if prompt.ends_with(&g.think) { Opening::Reasoning } else { Opening::Content };
    let calls = if conversation.choice != Choice::None && conversation.tools().next().is_some() {
        Calls::Read
    } else {
        Calls::Ignored
    };
    let mut reader = Reader::new(g, opening, calls);
    let primed = reader.push(&conversation.choice.forced(g));
    debug_assert!(primed.is_empty(), "{primed:?}");
    reader
}
