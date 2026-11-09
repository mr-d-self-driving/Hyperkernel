use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Ask {
    Write { code: usize, name: String, at: usize },
    Read { code: usize, name: String, start: usize, end: usize },
    Launch { code: usize },
    Finish,
    Randn { code: usize, name: String, stream: usize },
}

pub struct Link(pub UnixStream);

impl Link {
    pub fn ask(&self, ask: &Ask, payload: &[u8]) -> Result<Vec<u8>, String> {
        self.send(&serde_json::to_vec(ask).map_err(|e| e.to_string())?)?;
        self.send(payload)?;
        replied(self.recv()?)
    }

    pub fn serve(
        &self,
        mut answer: impl FnMut(Ask, Vec<u8>) -> Result<Vec<u8>, String>,
    ) -> Result<(), String> {
        loop {
            let Ok(head) = self.recv() else { return Ok(()) };
            let ask: Ask = serde_json::from_slice(&head).map_err(|e| format!("an ask: {e}"))?;
            let payload = self.recv()?;
            self.send(&reply(answer(ask, payload)))?;
        }
    }

    fn send(&self, bytes: &[u8]) -> Result<(), String> {
        let mut w = &self.0;
        w.write_all(&(bytes.len() as u64).to_le_bytes())
            .and_then(|()| w.write_all(bytes))
            .map_err(|e| format!("the link: {e}"))
    }

    fn recv(&self) -> Result<Vec<u8>, String> {
        let mut r = &self.0;
        let mut len = [0; 8];
        r.read_exact(&mut len).map_err(|e| format!("the link: {e}"))?;
        let mut bytes = vec![0; u64::from_le_bytes(len) as usize];
        r.read_exact(&mut bytes).map_err(|e| format!("the link: {e}"))?;
        Ok(bytes)
    }
}

fn reply(r: Result<Vec<u8>, String>) -> Vec<u8> {
    match r {
        Ok(bytes) => [&[0u8][..], &bytes].concat(),
        Err(e) => [&[1u8][..], e.as_bytes()].concat(),
    }
}

fn replied(b: Vec<u8>) -> Result<Vec<u8>, String> {
    match b.split_first() {
        Some((0, bytes)) => Ok(bytes.to_vec()),
        Some((_, why)) => Err(String::from_utf8_lossy(why).into_owned()),
        None => Err("an empty reply".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{Ask, Link, replied, reply};
    use std::os::unix::net::UnixStream;

    #[test]
    fn a_reply_reads_as_it_was_written() {
        assert_eq!(replied(reply(Ok(vec![1, 2]))), Ok(vec![1, 2]));
        assert_eq!(replied(reply(Err("no".into()))), Err("no".into()));
    }

    #[test]
    fn an_ask_is_answered_across_the_link() {
        let (a, b) = UnixStream::pair().unwrap();
        let server = std::thread::spawn(move || {
            Link(b).serve(|ask, payload| match ask {
                Ask::Write { at, .. } => Ok(vec![at as u8, payload.len() as u8]),
                Ask::Read { start, end, .. } => Ok(vec![7; end - start]),
                _ => Err("refused".into()),
            })
        });
        let link = Link(a);
        assert_eq!(link.ask(&Ask::Write { code: 0, name: "latents".into(), at: 3 }, &[9; 5]), Ok(vec![3, 5]));
        assert_eq!(
            link.ask(&Ask::Read { code: 0, name: "frames".into(), start: 2, end: 6 }, &[]),
            Ok(vec![7; 4])
        );
        assert_eq!(link.ask(&Ask::Finish, &[]), Err("refused".into()));
        drop(link);
        assert_eq!(server.join().unwrap(), Ok(()));
    }
}
