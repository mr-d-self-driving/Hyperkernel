use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use super::{Share, cuda};

#[derive(Clone, Copy, Debug)]
pub struct RanksConfig {
    pub margin: u64,
    pub dial: Duration,
}

impl Default for RanksConfig {
    fn default() -> RanksConfig {
        RanksConfig { margin: 64 << 20, dial: Duration::from_secs(300) }
    }
}

pub struct Ranks {
    rank: usize,
    world: usize,
    links: Vec<UnixStream>,
    base: u64,
    stride: u64,
    multicast: u64,
}

pub fn spawn(world: usize, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let exe = std::env::current_exe().map_err(|e| format!("this executable: {e}"))?;
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let mut kids: Vec<Child> = Vec::new();
    for r in 0..world {
        let kid = Command::new(&exe)
            .args(&args)
            .env("HS_RANK", r.to_string())
            .env("HS_WORLD", world.to_string())
            .env("HS_DIR", dir)
            .spawn();
        kids.push(kid.map_err(|e| format!("rank {r}: {e}"))?);
    }
    let (first, status) = loop {
        if let Some(ended) =
            kids.iter_mut().enumerate().find_map(|(r, k)| k.try_wait().ok().flatten().map(|s| (r, s)))
        {
            break ended;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    for k in &mut kids {
        let _ = k.kill();
        let _ = k.wait();
    }
    match status.success() {
        true => Ok(()),
        false => Err(format!("rank {first} ended: {status}")),
    }
}

pub fn rank() -> Result<Option<usize>, String> {
    number("HS_RANK")
}

fn number(v: &str) -> Result<Option<usize>, String> {
    std::env::var(v)
        .ok()
        .map(|t| t.parse::<usize>().map_err(|_| format!("{v}={t}: not a number")))
        .transpose()
}

impl Ranks {
    pub fn join(rank: usize, symmetric: u64, config: &RanksConfig) -> Result<Ranks, String> {
        let world = number("HS_WORLD")?.unwrap_or(1);
        if rank >= world {
            return Err(format!("HS_RANK {rank} of HS_WORLD {world}"));
        }
        cuda::init(rank)?;
        let links = match world {
            1 => Vec::new(),
            _ => link(
                rank,
                world,
                &PathBuf::from(std::env::var_os("HS_DIR").ok_or("HS_DIR is not set")?),
                config.dial,
            )?,
        };
        let mut ranks = Ranks { rank, world, links, base: 0, stride: 0, multicast: 0 };
        ranks.fabric_map(symmetric, config.margin)?;
        Ok(ranks)
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world(&self) -> usize {
        self.world
    }

    fn to(&self, r: usize) -> &UnixStream {
        &self.links[if self.rank == 0 { r - 1 } else { 0 }]
    }

    fn gather<T: Wire>(&self, mine: T) -> Result<Vec<T>, String> {
        if self.rank != 0 {
            T::send(self.to(0), &[mine])?;
            return T::recv(self.to(0), self.world);
        }
        let mut all = vec![mine];
        for r in 1..self.world {
            all.extend(T::recv(self.to(r), 1)?);
        }
        (1..self.world).try_for_each(|r| T::send(self.to(r), &all))?;
        Ok(all)
    }

    fn broadcast<T: Wire>(&self, mine: Option<T>) -> Result<T, String> {
        match mine {
            Some(x) if self.rank == 0 => {
                (1..self.world).try_for_each(|r| T::send(self.to(r), std::slice::from_ref(&x))).map(|()| x)
            }
            _ if self.rank == 0 => Err("rank 0 broadcast nothing".into()),
            _ => T::recv(self.to(0), 1)?.pop().ok_or_else(|| "nothing from rank 0".into()),
        }
    }

    pub fn hand_out(
        &self,
        made: Option<(Vec<String>, Vec<OwnedFd>)>,
    ) -> Result<(Vec<String>, Vec<OwnedFd>), String> {
        let (keys, fds) = match made {
            Some((keys, fds)) => {
                (self.broadcast(Some(serde_json::to_vec(&keys).map_err(|e| e.to_string())?))?, Some(fds))
            }
            None => (self.broadcast(None)?, None),
        };
        let keys: Vec<String> =
            serde_json::from_slice(&keys).map_err(|e| format!("shared homes from rank 0: {e}"))?;
        let fds = match fds {
            Some(fds) => fds.into_iter().map(|f| self.broadcast(Some(f))).collect::<Result<Vec<_>, _>>()?,
            None => keys.iter().map(|_| self.broadcast(None)).collect::<Result<Vec<_>, _>>()?,
        };
        Ok((keys, fds))
    }

    pub fn barrier(&self) -> Result<(), String> {
        self.gather(Vec::new()).map(drop)
    }

    pub fn min(&self, v: u64) -> Result<u64, String> {
        Ok(self.words(v)?.into_iter().min().unwrap_or(v))
    }

    fn words(&self, v: u64) -> Result<Vec<u64>, String> {
        self.gather(v.to_le_bytes().to_vec())?
            .iter()
            .map(|b| {
                b.as_slice()
                    .try_into()
                    .map(u64::from_le_bytes)
                    .map_err(|_| "a gathered word of other bytes".to_string())
            })
            .collect()
    }

    pub fn send_share(&self, share: &Share) -> Result<(), String> {
        let bytes = serde_json::to_vec(share).map_err(|e| format!("a share: {e}"))?;
        self.broadcast(Some(bytes)).map(drop)
    }

    pub fn recv_share(&self) -> Result<Share, String> {
        serde_json::from_slice(&self.broadcast::<Vec<u8>>(None)?)
            .map_err(|e| format!("a share from rank 0: {e}"))
    }

    fn base(&self, r: usize) -> u64 {
        self.base + r as u64 * self.stride
    }

    pub fn symmetric(&self) -> u64 {
        self.base(self.rank)
    }

    pub fn fabric_words(&self, multicast: bool) -> Vec<u8> {
        [self.rank as u64, 0, if multicast { self.multicast } else { 0 }, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    pub fn peers(&self, bytes: usize) -> Result<Vec<u8>, String> {
        peer_table(self.rank, &(0..self.world).map(|r| self.base(r)).collect::<Vec<u64>>(), bytes)
    }

    fn fabric_map(&mut self, symmetric: u64, margin: u64) -> Result<(), String> {
        let offered =
            self.words(if self.world > 1 { cuda::multicast_granularity(self.world)? as u64 } else { 0 })?;
        let multicast = match offered.iter().min() {
            Some(&m) if m > 0 => offered.iter().copied().max().unwrap_or(0),
            _ => 0,
        };
        let align = (cuda::granularity(true)? as u64).max(multicast);
        let stride = (symmetric.next_multiple_of(256) + margin).next_multiple_of(align);
        let len = stride as usize;
        let own = cuda::create(len, true)?;
        self.base = cuda::reserve(len * self.world, align as usize)?;
        self.stride = stride;
        cuda::map(self.symmetric(), len, own)?;
        cuda::memset8(self.symmetric(), 0, len)?;
        cuda::sync_all()?;
        if self.world > 1 {
            for (r, theirs) in
                self.gather(cuda::export(own)?)?.into_iter().enumerate().filter(|&(r, _)| r != self.rank)
            {
                cuda::map(self.base(r), len, cuda::import(theirs.as_fd())?)
                    .map_err(|e| format!("rank {r}'s arena: {e}"))?;
            }
        }
        if multicast > 0 {
            let mc = match self.rank {
                0 => {
                    let mc = cuda::multicast_create(self.world, len)?;
                    self.broadcast(Some(cuda::export(mc)?))?;
                    mc
                }
                _ => cuda::import(self.broadcast::<OwnedFd>(None)?.as_fd())?,
            };
            cuda::multicast_add(mc)?;
            self.barrier()?;
            cuda::multicast_bind(mc, own, len)?;
            self.multicast = cuda::reserve(len, multicast as usize)?;
            cuda::map(self.multicast, len, mc)?;
        }
        println!(
            "FABRIC rank={} world={} stride={stride} multicast={}",
            self.rank,
            self.world,
            multicast > 0
        );
        self.barrier()
    }
}

fn link(rank: usize, world: usize, dir: &Path, dial: Duration) -> Result<Vec<UnixStream>, String> {
    let path = dir.join("ranks.sock");
    let io = |e: std::io::Error| format!("{}: {e}", path.display());
    if rank == 0 {
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).map_err(io)?;
        let mut links: Vec<Option<UnixStream>> = (1..world).map(|_| None).collect();
        for _ in 1..world {
            let (s, _) = listener.accept().map_err(io)?;
            let r =
                u64::from_le_bytes(recv(&s)?.try_into().map_err(|_| "a rank's hello is no rank")?) as usize;
            match r.checked_sub(1).and_then(|i| links.get_mut(i)) {
                Some(slot @ None) => *slot = Some(s),
                _ => return Err(format!("a hello from rank {r} twice, or past the world")),
            }
        }
        return Ok(links.into_iter().flatten().collect());
    }
    let start = Instant::now();
    let s = loop {
        match UnixStream::connect(&path) {
            Ok(s) => break s,
            Err(e) if start.elapsed() > dial => return Err(io(e)),
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    send(&s, &(rank as u64).to_le_bytes())?;
    Ok(vec![s])
}

trait Wire: Sized {
    fn send(s: &UnixStream, items: &[Self]) -> Result<(), String>;
    fn recv(s: &UnixStream, n: usize) -> Result<Vec<Self>, String>;
}

impl Wire for Vec<u8> {
    fn send(s: &UnixStream, items: &[Self]) -> Result<(), String> {
        items.iter().try_for_each(|b| send(s, b))
    }

    fn recv(s: &UnixStream, n: usize) -> Result<Vec<Self>, String> {
        (0..n).map(|_| recv(s)).collect()
    }
}

impl Wire for OwnedFd {
    fn send(s: &UnixStream, items: &[Self]) -> Result<(), String> {
        send_fds(s, &items.iter().map(|f| f.as_raw_fd()).collect::<Vec<RawFd>>())
    }

    fn recv(s: &UnixStream, n: usize) -> Result<Vec<Self>, String> {
        recv_fds(s, n)
    }
}

pub fn peer_table(rank: usize, bases: &[u64], bytes: usize) -> Result<Vec<u8>, String> {
    let n = bases.len();
    let mut b: Vec<u8> = bases.iter().flat_map(|a| a.to_le_bytes()).collect();
    b.extend(0u64.to_le_bytes());
    b.extend((rank as i32).to_le_bytes());
    (0..n).for_each(|_| b.extend((-1i32).to_le_bytes()));
    b.extend(std::iter::repeat_n(0u8, n));
    if b.len() > bytes {
        return Err(format!("a peer table of {} ranks is {} bytes, the kernel's {bytes}", n, b.len()));
    }
    b.resize(bytes, 0);
    Ok(b)
}

fn send(s: &UnixStream, bytes: &[u8]) -> Result<(), String> {
    let mut w = s;
    w.write_all(&(bytes.len() as u64).to_le_bytes())
        .and_then(|()| w.write_all(bytes))
        .map_err(|e| format!("a rank link: {e}"))
}

fn recv(s: &UnixStream) -> Result<Vec<u8>, String> {
    let mut r = s;
    let mut len = [0; 8];
    r.read_exact(&mut len).map_err(|e| format!("a rank link: {e}"))?;
    let mut bytes = vec![0; u64::from_le_bytes(len) as usize];
    r.read_exact(&mut bytes).map_err(|e| format!("a rank link: {e}"))?;
    Ok(bytes)
}

fn send_fds(s: &UnixStream, fds: &[RawFd]) -> Result<(), String> {
    let mut byte = [0u8; 1];
    let mut iov = libc::iovec { iov_base: byte.as_mut_ptr().cast(), iov_len: 1 };
    let data = std::mem::size_of_val(fds) as u32;
    let space = unsafe { libc::CMSG_SPACE(data) } as usize;
    let mut buf = vec![0u64; space.div_ceil(8)];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = buf.as_mut_ptr().cast();
    msg.msg_controllen = space as _;
    unsafe {
        let c = libc::CMSG_FIRSTHDR(&msg);
        (*c).cmsg_level = libc::SOL_SOCKET;
        (*c).cmsg_type = libc::SCM_RIGHTS;
        (*c).cmsg_len = libc::CMSG_LEN(data) as _;
        std::ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(c).cast::<RawFd>(), fds.len());
        if libc::sendmsg(s.as_raw_fd(), &msg, 0) != 1 {
            return Err(format!("sendmsg: {}", std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

fn recv_fds(s: &UnixStream, n: usize) -> Result<Vec<OwnedFd>, String> {
    let mut byte = [0u8; 1];
    let mut iov = libc::iovec { iov_base: byte.as_mut_ptr().cast(), iov_len: 1 };
    let data = (n * std::mem::size_of::<RawFd>()) as u32;
    let space = unsafe { libc::CMSG_SPACE(data) } as usize;
    let mut buf = vec![0u64; space.div_ceil(8)];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = buf.as_mut_ptr().cast();
    msg.msg_controllen = space as _;
    unsafe {
        if libc::recvmsg(s.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) != 1 {
            return Err(format!("recvmsg: {}", std::io::Error::last_os_error()));
        }
        let c = libc::CMSG_FIRSTHDR(&msg);
        if c.is_null()
            || (*c).cmsg_type != libc::SCM_RIGHTS
            || msg.msg_flags & libc::MSG_CTRUNC != 0
            || (*c).cmsg_len as usize != libc::CMSG_LEN(data) as usize
        {
            return Err(format!("recvmsg: not the {n} descriptors a peer sent"));
        }
        let p = libc::CMSG_DATA(c).cast::<RawFd>();
        Ok((0..n).map(|i| OwnedFd::from_raw_fd(p.add(i).read_unaligned())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_peer_table_of_four() {
        let t = peer_table(2, &[0x10, 0x20, 0x30, 0x40], 64).unwrap();
        assert_eq!(t.len(), 64);
        assert_eq!(u64::from_le_bytes(t[16..24].try_into().unwrap()), 0x30);
        assert_eq!(u64::from_le_bytes(t[32..40].try_into().unwrap()), 0);
        assert_eq!(i32::from_le_bytes(t[40..44].try_into().unwrap()), 2);
        assert!(t[44..60].chunks(4).all(|c| i32::from_le_bytes(c.try_into().unwrap()) == -1));
        assert!(t[60..64].iter().all(|&b| b == 0));
        assert!(peer_table(0, &[0; 4], 63).is_err());
        assert_eq!(peer_table(0, &[1], 32).unwrap().len(), 32);
    }

    #[test]
    fn frames_and_descriptors_over_a_pair() {
        let (a, b) = UnixStream::pair().unwrap();
        send(&a, b"share").unwrap();
        assert_eq!(recv(&b).unwrap(), b"share");
        let f = std::fs::File::open("/dev/null").unwrap();
        send_fds(&a, &[f.as_raw_fd(), f.as_raw_fd()]).unwrap();
        send(&a, b"after").unwrap();
        let got = recv_fds(&b, 2).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(recv(&b).unwrap(), b"after");
    }
}
