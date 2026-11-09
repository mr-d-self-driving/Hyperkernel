use std::path::PathBuf;

use hyperserve::bundle::{Blob, Bundle, Fill, Port, Record};
use hyperserve::evo2::{Config, Evo2};
use hyperserve::weights::Checkpoint;

fn dir() -> PathBuf {
    std::env::var_os("HS_EVO2")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300/evo2"), PathBuf::from)
}

fn inside(blob: &Blob, size: u64) -> bool {
    blob.at.checked_add(blob.len).is_some_and(|end| end <= size)
}

fn bytes(p: &Port) -> u64 {
    p.at.end - p.at.start
}

#[test]
fn both_programs_plan_on_one_b300() {
    let (free, granule, slack): (u64, u64, u64) = (275_921_829_888, 2 << 20, 8 << 30);
    let c = Config::default();
    let mut placed = 0;
    let mut programs = Vec::new();
    for program in [c.prefill, c.step] {
        let at = dir().join(program);
        let b = Bundle::<Evo2>::read(&at, 0).unwrap_or_else(|e| panic!("{}: {e}", at.display()));
        let (r, size) = (&b.reference, b.blobs.len() as u64);
        let (f, ports) = (&r.served.evo2, &r.served.ports);
        assert_eq!(r.served.model.ranks, 1, "{program}");
        let len = |h: usize| r.homes[h].at.end - r.homes[h].at.start;
        for x in &r.image {
            match x {
                Record::Write { blob, .. } => assert!(
                    inside(blob, size),
                    "{program}: a write's blob {blob:?} past blobs.bin's {size} B"
                ),
                Record::After { from, to } => assert!(
                    from.1.end <= len(from.0)
                        && to.1.end <= len(to.0)
                        && from.1.end - from.1.start == to.1.end - to.1.start,
                    "{program}: {x:?}"
                ),
                _ => {}
            }
        }
        for i in &b.inputs.inputs {
            assert!(i.at.end <= len(i.home), "{program}: input {} past its home", i.name);
            if let Fill::Blob { blob } = &i.fill {
                assert!(inside(blob, size), "{program}: input {}'s blob past blobs.bin", i.name);
            }
        }
        for p in ports.inputs.iter().chain(&ports.outputs) {
            assert!(p.at.end <= len(p.home), "{program}: port {} past its home", p.name);
        }
        for (name, n) in [("ids", 4 * f.rows), ("rope", 4 * f.rows * f.head), ("at", 4)] {
            ports.input(name, n).unwrap_or_else(|e| panic!("{program}: {e}"));
        }
        ports.output("logits", 4 * f.rows * f.vocab).unwrap_or_else(|e| panic!("{program}: {e}"));
        for j in 0..=f.layers {
            ports.output(&format!("x{j}"), 2 * f.rows * f.width).unwrap_or_else(|e| panic!("{program}: {e}"));
        }
        let cubins = r
            .kernel
            .loaders
            .iter()
            .map(|(c, _)| c.as_str())
            .chain(["entry.cubin"])
            .chain(r.kernel.next.as_ref().map(|_| "next.cubin"));
        for cubin in cubins {
            assert!(b.cubin(cubin).is_ok_and(|c| !c.is_empty()), "{program}: no {cubin}");
        }
        let staging = r.homes.iter().filter(|h| h.staged).map(|h| h.at.end - h.at.start).max().unwrap_or(0);
        let maps = r.image.iter().filter(|x| matches!(x, Record::Map { place: None, .. })).count() as u64;
        placed += [r.kernel.workspace, staging, 128 * maps + 8]
            .iter()
            .map(|b| b.next_multiple_of(granule))
            .sum::<u64>();
        programs.push(b);
    }
    let [p, s] = &programs[..] else { unreachable!() };
    let (f, g) = (&p.reference.served.evo2, &s.reference.served.evo2);
    assert_eq!((g.rows, f.vocab, f.cache, &f.states), (1, g.vocab, g.cache, &g.states));
    assert_eq!(f.cache, 1 << 20, "caches of the trained context");
    for name in &f.states {
        let (a, b) = (
            p.reference.served.ports.input(name, 0).unwrap(),
            s.reference.served.ports.input(name, 0).unwrap(),
        );
        assert_eq!(bytes(a), bytes(b), "state {name}");
    }
    assert!(placed + slack < free, "{placed} B of both programs past a B300's {free}");
    println!("evo2: both programs planned, {placed} B placed, {} states", f.states.len());
}

#[test]
#[ignore = "reads the converted checkpoint: HS_EVO2_WEIGHTS=DIR"]
fn every_weight_of_the_bundle_is_a_tensor_of_its_bytes() {
    let weights =
        std::env::var_os("HS_EVO2_WEIGHTS").expect("HS_EVO2_WEIGHTS: the converted checkpoint's directory");
    let ck = Checkpoint::open(std::path::Path::new(&weights)).unwrap_or_else(|e| panic!("{e}"));
    let c = Config::default();
    for program in [c.prefill, c.step] {
        let at = dir().join(program);
        let b = Bundle::<Evo2>::read(&at, 0).unwrap_or_else(|e| panic!("{}: {e}", at.display()));
        for i in &b.inputs.inputs {
            if let Fill::Checkpoint { transform, tensors, cut, laid } = &i.fill {
                let bytes = ck
                    .laid(transform, tensors, cut.as_ref(), laid)
                    .unwrap_or_else(|e| panic!("{program} {}: {e}", i.name));
                assert_eq!(bytes.len() as u64, i.at.end - i.at.start, "{program} {}", i.name);
            }
        }
    }
}
