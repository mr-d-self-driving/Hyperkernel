use std::fs;
use std::path::{Path, PathBuf};

use hyperserve::bundle::{Blob, Bundle, Dsv, Fill, Record};
use hyperserve::gpu::load;

fn bundles() -> Vec<PathBuf> {
    let root = std::env::var_os("HS_BUNDLES").expect("HS_BUNDLES: a directory of bundles");
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .expect("HS_BUNDLES readable")
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|d| ["reference.json", "reference.json.zst"].iter().any(|f| d.join(f).is_file()))
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no bundle under {}", Path::new(&root).display());
    dirs
}

fn inside(blob: &Blob, size: u64) -> bool {
    blob.at.checked_add(blob.len).is_some_and(|end| end <= size)
}

#[test]
#[ignore = "reads exported bundles: HS_BUNDLES=DIR"]
fn every_rank_plans_at_a_b300() {
    let (free, granule) = (275_921_829_888, 2 << 20);
    for dir in bundles() {
        let ranks = Bundle::<Dsv>::read(&dir, 0)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .reference
            .served
            .model
            .ranks;
        for rank in 0..ranks {
            let b: Bundle =
                Bundle::read(&dir, rank).unwrap_or_else(|e| panic!("{} rank {rank}: {e}", dir.display()));
            let (r, size) = (&b.reference, b.blobs.len() as u64);
            let plan = load::plan(r, rank, free, granule, &load::LoaderConfig::default())
                .unwrap_or_else(|e| panic!("{} rank {rank}: {e}", dir.display()));
            assert!(plan.pages > 0 && plan.window_pages > 0, "{} rank {rank}: {plan:?}", dir.display());
            for x in &r.image {
                if let Record::Write { blob, .. } = x {
                    assert!(
                        inside(blob, size),
                        "{}: a write's blob {blob:?} past blobs.bin's {size} B",
                        dir.display()
                    );
                }
            }
            for i in &b.inputs.inputs {
                let home = &r.homes[i.home];
                assert!(
                    i.at.end <= home.at.end - home.at.start,
                    "{} rank {rank}: input {} past its home",
                    dir.display(),
                    i.name
                );
                if let Fill::Blob { blob } = &i.fill {
                    assert!(
                        inside(blob, size),
                        "{} rank {rank}: input {}'s blob past blobs.bin",
                        dir.display(),
                        i.name
                    );
                }
            }
            for cubin in r.kernel.loaders.iter().map(|(c, _)| c.as_str()).chain(["entry.cubin"]) {
                assert!(b.cubin(cubin).is_ok_and(|c| !c.is_empty()), "{}: no {cubin}", dir.display());
            }
        }
        println!("{}: {ranks} ranks planned", dir.display());
    }
}
