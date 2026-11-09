use std::path::PathBuf;

use hyperserve::bundle::{Blob, Bundle, Dtype, Fill, Record};
use hyperserve::mg3::spec::Spec;
use hyperserve::mg3::{Config, Mg3, Port};

fn dir() -> PathBuf {
    std::env::var_os("HS_MG3")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300/mg3"), PathBuf::from)
}

fn bundle(name: &str) -> Bundle<Mg3> {
    let d = dir().join(name);
    Bundle::read(&d, 0).unwrap_or_else(|e| panic!("{}: {e}", d.display()))
}

fn inside(blob: &Blob, size: u64) -> bool {
    blob.at.checked_add(blob.len).is_some_and(|end| end <= size)
}

fn sized(b: &Bundle<Mg3>, code: usize, name: &str, bytes: usize, dtype: Dtype) -> Port {
    let p = b.reference.served.port(code, name).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        ((p.at.end - p.at.start) as usize, p.dtype),
        (bytes, dtype),
        "{} `{name}`",
        b.reference.served.model.name
    );
    p.clone()
}

fn whole(b: &Bundle<Mg3>, free: u64) {
    let (r, size, name) = (&b.reference, b.blobs.len() as u64, &b.reference.served.model.name);
    assert!(r.kernel.workspace < free, "{name}: a workspace of {} B", r.kernel.workspace);
    assert!(
        r.served.codes.n == 1 || r.served.codes.at + 8 <= r.kernel.workspace,
        "{name}: its code word past the arena"
    );
    for x in &r.image {
        if let Record::Write { blob, .. } = x {
            assert!(inside(blob, size), "{name}: a write's blob {blob:?} past blobs.bin's {size} B");
        }
    }
    for i in &b.inputs.inputs {
        let home = &r.homes[i.home];
        assert!(i.at.end <= home.at.end - home.at.start, "{name}: input {} past its home", i.name);
        if let Fill::Blob { blob } = &i.fill {
            assert!(inside(blob, size), "{name}: input {}'s blob past blobs.bin", i.name);
        }
    }
    for p in &r.served.ports {
        let home = &r.homes[p.home];
        assert!(
            p.at.end <= home.at.end - home.at.start && !home.staged,
            "{name}: port {} past its home or staged",
            p.name
        );
    }
    for cubin in r.kernel.loaders.iter().map(|(c, _)| c.as_str()).chain(["entry.cubin"]) {
        assert!(b.cubin(cubin).is_ok_and(|c| !c.is_empty()), "{name}: no {cubin}");
    }
}

fn denoised(b: &Bundle<Mg3>, s: &Spec, prompt: usize, code: usize) {
    let latent = s.latent_bytes();
    sized(b, prompt, "context", s.text * s.text_dim * 2, Dtype::Bf16);
    sized(b, code, "rope", s.heads * s.turns * s.head * 4, Dtype::F32);
    sized(b, code, "poses", s.latents() * 24 * 4, Dtype::F32);
    sized(b, code, "positions", s.rows() * s.head / 2 * 4, Dtype::I32);
    sized(b, code, "keyboard_windows", s.latents() * s.window() * s.key_cols * 2, Dtype::Bf16);
    sized(b, code, "mouse_windows", s.rows() * s.mouse_cols * 2, Dtype::Bf16);
    sized(b, code, "latents", s.rows() * s.row_bytes(), Dtype::Bf16);
    let noise = b.reference.served.port(code, "noise").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(noise.dtype, Dtype::Bf16);
    let history = b.reference.served.port(code, "l0.history").unwrap_or_else(|e| panic!("{e}"));
    assert!(history.at.end - history.at.start >= (s.turns * latent) as u64);
    if s.memory() > 0 {
        sized(b, code, "memory", s.memory() * latent, Dtype::Bf16);
    }
}

#[test]
fn every_mg3_bundle_holds_what_its_loop_writes_and_reads() {
    let (cfg, free) = (Config::default(), 275_921_829_888);
    let (chunk, first, vae) = (bundle("mg3chunk"), bundle("mg3first"), bundle("mg3vae"));
    [&chunk, &first, &vae].into_iter().for_each(|b| whole(b, free));
    let (c, f) = (&chunk.reference.served.model, &first.reference.served.model);
    denoised(&chunk, c, 0, 1);
    denoised(&first, f, 0, 1);
    assert!(chunk.reference.served.port(1, "l0.past").is_ok());
    sized(&first, 1, "image_latent", c.latent_bytes(), Dtype::Bf16);
    let latents = vae.reference.served.port(0, "latents").unwrap_or_else(|e| panic!("{e}"));
    assert!(latents.at.end - latents.at.start >= (c.rows() * c.row_bytes()) as u64);
    sized(&vae, 0, "frames", cfg.chunk_bytes(), Dtype::Bf16);
    let gpu0 = chunk.reference.kernel.workspace + first.reference.kernel.workspace;
    assert!(gpu0 < free, "the DiT's two kernels' {gpu0} B on one GPU");
    println!("{}: mg3chunk, mg3first and mg3vae planned", dir().display());
}
