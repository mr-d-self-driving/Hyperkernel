use std::path::PathBuf;

use hyperserve::bundle::{Bundle, Dtype, Fill, Kind};
use hyperserve::pi0::{Config, Pi0};
use hyperserve::weights::Checkpoint;

fn bundles(c: &Config) -> [(&'static str, usize); 2] {
    [("pi0fast", c.tokens), ("pi0fast-fixed60", c.fixed)]
}

fn dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300").join(name)
}

#[test]
#[ignore = "reads kernels/b300/pi0fast{,-fixed60}"]
fn the_pi0_bundles_boot_at_a_b300_and_carry_the_observation_and_token_messages() {
    let free: u64 = 275_921_829_888;
    let c = Config::default();
    for (name, tokens) in bundles(&c) {
        let b = Bundle::<Pi0>::read(&dir(name), 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        let r = &b.reference;
        assert_eq!((r.served.model.name.as_str(), r.served.model.ranks), ("pi0fast", 1), "{name}");
        assert!(r.kernel.workspace < free, "{name}: a {} B arena", r.kernel.workspace);
        assert!(
            !r.kernel.globals.rings.is_empty() && !r.kernel.globals.ctl.is_empty(),
            "{name}: no ring globals"
        );
        let rings = &r.served.rings;
        let req = rings
            .messages
            .iter()
            .find(|m| m.name == c.request && m.ring == 0)
            .unwrap_or_else(|| panic!("{name}: no {}", c.request));
        let span = |f: &str, dtype: Dtype, n: usize| {
            let x = req.fields.iter().find(|x| x.name == f).unwrap_or_else(|| panic!("{name}: no {f}"));
            assert_eq!((x.kind, x.dtype, x.n), (Kind::Span, dtype, n), "{name} {f}");
        };
        span("images", Dtype::F32, c.image_floats());
        span("ids", Dtype::I32, c.prompt);
        span("tokens", Dtype::I32, tokens);
        assert!(
            req.fields.iter().any(|x| x.name == "lens" && x.kind == Kind::One && x.dtype == Dtype::I32),
            "{name} lens"
        );
        assert!(rings.messages.iter().any(|m| m.name == c.reply && m.ring == 1), "{name}: no {}", c.reply);
        for cubin in r.kernel.loaders.iter().map(|(c, _)| c.as_str()).chain(["entry.cubin"]) {
            assert!(b.cubin(cubin).is_ok_and(|c| !c.is_empty()), "{name}: no {cubin}");
        }
    }
}

#[test]
#[ignore = "reads the published checkpoint: HS_PI0_WEIGHTS=DIR (lerobot/pi0fast-base@308eb844)"]
fn every_weight_of_the_pi0_bundles_is_laid_from_the_published_checkpoint_at_its_bytes() {
    let weights = std::env::var_os("HS_PI0_WEIGHTS").expect("HS_PI0_WEIGHTS: the checkpoint's directory");
    let ck = Checkpoint::open(std::path::Path::new(&weights)).unwrap_or_else(|e| panic!("{e}"));
    for (name, _) in bundles(&Config::default()) {
        let b = Bundle::<Pi0>::read(&dir(name), 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut laid = 0;
        for i in &b.inputs.inputs {
            if let Fill::Checkpoint { transform, tensors, cut, laid: l } = &i.fill {
                let bytes = ck
                    .laid(transform, tensors, cut.as_ref(), l)
                    .unwrap_or_else(|e| panic!("{name} {}: {e}", i.name));
                assert_eq!(bytes.len() as u64, i.at.end - i.at.start, "{name} {}", i.name);
                laid += 1;
            }
        }
        assert!(laid > 400, "{name}: {laid} weights laid");
    }
}
