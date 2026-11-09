use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use crate::bundle::Pool;
use crate::gpu::Command;

#[derive(Debug, PartialEq)]
pub enum Wait {
    Later,
    Never(String),
}

pub struct Cache {
    rows: usize,
    entries: usize,
    window: usize,
    keep: bool,
    pages: Units,
    windows: Units,
    lanes: Vec<Place>,
    nodes: BTreeMap<usize, Node>,
    edges: HashMap<(Option<usize>, Box<[u32]>), usize>,
    ids: usize,
    clock: u64,
}

#[derive(Default)]
struct Place {
    pages: Vec<u32>,
    window: Vec<Option<u32>>,
    promise: usize,
    upto: usize,
    sent: Vec<i32>,
    sent_upto: usize,
}

struct Node {
    parent: Option<usize>,
    tokens: Box<[u32]>,
    page: u32,
    window: Option<u32>,
    children: usize,
    used: u64,
}

struct Units {
    held: Vec<u32>,
    free: Vec<u32>,
}

impl Units {
    fn new(n: usize) -> Units {
        let mut held = vec![0; n.max(1)];
        held[0] = 1;
        Units { held, free: (1..n as u32).rev().collect() }
    }

    fn take(&mut self) -> Option<u32> {
        let u = self.free.pop()?;
        self.held[u as usize] = 1;
        Some(u)
    }

    fn share(&mut self, u: u32) {
        self.held[u as usize] += 1;
    }

    fn release(&mut self, u: u32) {
        let h = &mut self.held[u as usize];
        *h -= 1;
        if *h == 0 {
            self.free.push(u);
        }
    }

    fn once(&self, u: u32) -> bool {
        self.held[u as usize] == 1
    }
}

impl Cache {
    pub fn new(pool: &Pool, pages: usize, window_pages: usize, lanes: usize) -> Cache {
        let row = pool.entries + pool.window;
        let place = || Place { sent: vec![0; row], ..Place::default() };
        Cache {
            rows: pool.page_rows,
            entries: pool.entries,
            window: pool.window,
            keep: pool.window_keep > 0,
            pages: Units::new(pages),
            windows: Units::new(window_pages),
            lanes: (0..lanes).map(|_| place()).collect(),
            nodes: BTreeMap::new(),
            edges: HashMap::new(),
            ids: 0,
            clock: 0,
        }
    }

    pub fn admit(&mut self, lane: usize, prompt: &[u32], extent: usize) -> Result<usize, Wait> {
        let p = self.rows;
        let extent = extent.min(self.entries * p);
        if prompt.len() > extent {
            return Err(Wait::Never(format!(
                "a prompt of {} tokens past a row's {} pages of {p}",
                prompt.len(),
                self.entries
            )));
        }
        let path = self.find(prompt);
        let need = extent.div_ceil(p) - path.len();
        if self.spare() + self.freeable(&path) < need {
            let live = self.lanes.iter().any(|l| !l.pages.is_empty() || l.promise > 0);
            if !live || need + path.len() > self.pages.held.len() - 1 {
                return Err(Wait::Never(format!(
                    "{need} pages more than its {} of a pool of {}",
                    path.len(),
                    self.pages.held.len() - 1
                )));
            }
            return Err(Wait::Later);
        }
        while self.spare() < need {
            assert!(self.evict_page(&path), "admission counted pages the tree does not free");
        }
        let now = self.tick();
        for &n in &path {
            let node = self.nodes.get_mut(&n).expect("a live node");
            node.used = now;
            self.pages.share(node.page);
            self.lanes[lane].pages.push(node.page);
        }
        if let Some(&last) = path.last() {
            let w = self.node(last).window.expect("a resume's window page");
            self.windows.share(w);
            self.lanes[lane].window = vec![None; path.len() - 1];
            self.lanes[lane].window.push(Some(w));
        }
        self.lanes[lane].promise = need;
        Ok(path.len() * p)
    }

    pub fn hold(&mut self, lane: usize, written: Range<usize>) -> Result<(), String> {
        let p = self.rows;
        let last = (written.end - 1) / p;
        if last >= self.entries || last >= self.window {
            return Err(format!(
                "position {} past a row's {} pages of {p}",
                written.end - 1,
                self.entries.min(self.window)
            ));
        }
        while self.lanes[lane].pages.len() <= last {
            let place = &mut self.lanes[lane];
            let page = if place.promise > 0 {
                place.promise -= 1;
                self.pages.take()
            } else if self.spare() > 0 {
                self.pages.take()
            } else if self.evict_page(&[]) {
                continue;
            } else {
                return Err("no page free".into());
            };
            self.lanes[lane].pages.push(page.expect("a free page"));
        }
        let from = ((written.start + 1) / p).saturating_sub(1);
        let place = &mut self.lanes[lane];
        for w in place.window.iter_mut().take(from).filter_map(Option::take) {
            self.windows.release(w);
        }
        if place.window.len() <= last {
            place.window.resize(last + 1, None);
        }
        for k in from..=last {
            while self.lanes[lane].window[k].is_none() {
                match self.windows.take() {
                    Some(w) => self.lanes[lane].window[k] = Some(w),
                    None if self.evict_window() => {}
                    None => return Err("no window page free".into()),
                }
            }
        }
        Ok(())
    }

    pub fn run_to(&mut self, lane: usize, upto: usize) {
        self.lanes[lane].upto = upto;
    }

    pub fn upto(&self, lane: usize) -> usize {
        self.lanes[lane].upto
    }

    pub fn pages(&mut self, lane: usize, gap: usize) -> Vec<Command> {
        let place = &self.lanes[lane];
        let pages = place.pages.iter().map(|&p| p as i32).chain(std::iter::repeat(0)).take(self.entries);
        let window = place
            .window
            .iter()
            .map(|w| w.map_or(0, |w| w as i32))
            .chain(std::iter::repeat(0))
            .take(self.window);
        let row: Vec<i32> = pages.chain(window).collect();
        let place = &mut self.lanes[lane];
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for k in (0..row.len()).filter(|&k| row[k] != place.sent[k]) {
            match runs.last_mut() {
                Some((_, end)) if k < *end + gap => *end = k + 1,
                _ => runs.push((k, k + 1)),
            }
        }
        if runs.is_empty() && place.upto == place.sent_upto {
            return Vec::new();
        }
        if runs.is_empty() {
            runs.push((0, 0));
        }
        let upto = place.upto as u32;
        let commands = runs
            .into_iter()
            .map(|(a, b)| Command::Pages { lane: lane as u32, from: a as u32, upto, row: row[a..b].to_vec() })
            .collect();
        (place.sent, place.sent_upto) = (row, place.upto);
        commands
    }

    pub fn finish(&mut self, lane: usize, tokens: &[u32]) {
        self.remember(lane, tokens);
        let place = &mut self.lanes[lane];
        for p in place.pages.drain(..) {
            self.pages.release(p);
        }
        for w in place.window.drain(..).flatten() {
            self.windows.release(w);
        }
        (place.promise, place.upto, place.sent_upto) = (0, 0, 0);
    }

    pub fn pool(&self) -> usize {
        self.pages.held.len() - 1
    }

    pub fn held(&self) -> usize {
        self.lanes.iter().map(|l| l.pages.len()).sum()
    }

    pub fn reset(&mut self) -> usize {
        let free = self.pages.free.len();
        for node in std::mem::take(&mut self.nodes).into_values() {
            self.pages.release(node.page);
            if let Some(w) = node.window {
                self.windows.release(w);
            }
        }
        self.edges.clear();
        self.pages.free.len() - free
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn spare(&self) -> usize {
        self.pages.free.len() - self.lanes.iter().map(|l| l.promise).sum::<usize>()
    }

    fn node(&self, n: usize) -> &Node {
        &self.nodes[&n]
    }

    fn find(&self, prompt: &[u32]) -> Vec<usize> {
        let (mut path, mut best, mut parent) = (Vec::new(), 0, None);
        for block in prompt.chunks_exact(self.rows) {
            if (path.len() + 1) * self.rows >= prompt.len() {
                break;
            }
            let Some(&n) = self.edges.get(&(parent, block.into())) else { break };
            path.push(n);
            if self.node(n).window.is_some() {
                best = path.len();
            }
            parent = Some(n);
        }
        path.truncate(best);
        path
    }

    fn freeable(&self, keep: &[usize]) -> usize {
        self.nodes.iter().filter(|(n, node)| !keep.contains(n) && self.pages.once(node.page)).count()
    }

    fn lru(&self, keep: &[usize], pick: impl Fn(&Node) -> bool) -> Option<usize> {
        self.nodes
            .iter()
            .filter(|(n, node)| !keep.contains(n) && pick(node))
            .min_by_key(|(_, node)| node.used)
            .map(|(&n, _)| n)
    }

    fn evict_page(&mut self, keep: &[usize]) -> bool {
        let Some(n) = self.lru(keep, |node| node.children == 0 && self.pages.once(node.page)) else {
            return false;
        };
        let node = self.nodes.remove(&n).expect("a live node");
        self.edges.remove(&(node.parent, node.tokens));
        if let Some(p) = node.parent {
            self.nodes.get_mut(&p).expect("a live parent").children -= 1;
        }
        self.pages.release(node.page);
        if let Some(w) = node.window {
            self.windows.release(w);
        }
        true
    }

    fn evict_window(&mut self) -> bool {
        let alone = |node: &Node| node.window.is_some_and(|w| self.windows.once(w));
        let interior = self.keep.then(|| self.lru(&[], |node| alone(node) && node.children > 0)).flatten();
        let Some(n) = interior.or_else(|| self.lru(&[], alone)) else { return false };
        let w = self.nodes.get_mut(&n).and_then(|node| node.window.take()).expect("a node's window page");
        self.windows.release(w);
        true
    }

    fn remember(&mut self, lane: usize, tokens: &[u32]) {
        let now = self.tick();
        let mut parent = None;
        for (k, block) in tokens.chunks_exact(self.rows).enumerate() {
            let key: (Option<usize>, Box<[u32]>) = (parent, block.into());
            let n = match self.edges.get(&key) {
                Some(&n) => n,
                None => {
                    let Some(&page) = self.lanes[lane].pages.get(k) else { return };
                    self.pages.share(page);
                    let node =
                        Node { parent, tokens: key.1.clone(), page, window: None, children: 0, used: now };
                    self.ids += 1;
                    let n = self.ids;
                    self.nodes.insert(n, node);
                    self.edges.insert(key, n);
                    if let Some(p) = parent {
                        self.nodes.get_mut(&p).expect("a live parent").children += 1;
                    }
                    n
                }
            };
            let held = self.lanes[lane].window.get(k).copied().flatten();
            let node = self.nodes.get_mut(&n).expect("a live node");
            node.used = now;
            if let (None, Some(w)) = (node.window, held) {
                node.window = Some(w);
                self.windows.share(w);
            }
            parent = Some(n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cache, Wait};
    use crate::bundle::Pool;

    fn pool(keep: usize) -> Pool {
        Pool {
            page_rows: 4,
            pages: 0,
            window_pages: 0,
            slots: 0,
            window_keep: keep,
            entries: 8,
            window: 8,
            states: Vec::new(),
        }
    }

    fn served(c: &mut Cache, lane: usize, tokens: &[u32], n: usize) -> usize {
        let cached = c.admit(lane, &tokens[..n], tokens.len() + 1).unwrap();
        c.hold(lane, cached..n).unwrap();
        c.hold(lane, tokens.len() - 1..tokens.len() + 1).unwrap();
        c.run_to(lane, tokens.len() + 1);
        c.finish(lane, &tokens[..tokens.len() - 1]);
        cached
    }

    #[test]
    fn a_prompt_resumes_from_the_longest_prefix_kept() {
        let mut c = Cache::new(&pool(0), 40, 20, 2);
        let a: Vec<u32> = (0..15).collect();
        assert_eq!(served(&mut c, 0, &a, 13), 0);
        assert_eq!(served(&mut c, 1, &a, 13), 12);
        let mut b = a[..9].to_vec();
        b.extend([99, 98, 97, 96]);
        assert_eq!(c.admit(0, &b, 14), Ok(0), "page 1 holds no window page: nothing to resume from");
    }

    #[test]
    fn pages_evicted_least_recently_used_and_a_prompt_too_long_never_fits() {
        let mut c = Cache::new(&pool(0), 7, 20, 2);
        let a: Vec<u32> = (0..15).collect();
        let b: Vec<u32> = (100..115).collect();
        served(&mut c, 0, &a, 13);
        served(&mut c, 0, &b, 13);
        assert_eq!(served(&mut c, 0, &b, 13), 12, "b evicted a, not itself");
        assert_eq!(served(&mut c, 0, &a, 13), 0);
        let long: Vec<u32> = (0..40).collect();
        assert!(matches!(c.admit(0, &long, 41), Err(Wait::Never(_))));
    }

    #[test]
    fn a_row_goes_as_its_changed_runs() {
        let mut c = Cache::new(&pool(0), 40, 20, 1);
        c.admit(0, &[1; 6], 8).unwrap();
        c.hold(0, 0..6).unwrap();
        let p = c.pages(0, crate::engine::EngineConfig::default().gap);
        assert_eq!(p.len(), 1);
        assert!(c.pages(0, crate::engine::EngineConfig::default().gap).is_empty(), "nothing changed");
        c.hold(0, 5..8).unwrap();
        c.run_to(0, 8);
        let p = c.pages(0, crate::engine::EngineConfig::default().gap);
        assert!(matches!(&p[..], [crate::gpu::Command::Pages { upto: 8, .. }]), "{p:?}");
    }
}
