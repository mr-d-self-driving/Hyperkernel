import os
import shutil
import sys
from dataclasses import dataclass

import torch
from safetensors.torch import save_file


@dataclass(frozen=True)
class Config:
    manifest: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests", "esmfold.tsv")
    header: str = "# facebook/esmfold_v1 75a3841ee059 pytorch_model.bin as esmfold_safetensors.py writes it: each floating tensor F32 (name, dtype, shape, count, bytes)"


def kept(state):
    for name, t in state.items():
        if not t.is_floating_point() or name.endswith("rotary_embeddings.inv_freq"):
            continue
        yield name, t.to(torch.float32).contiguous()


def names(manifest):
    with open(manifest) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if not line.startswith("#")]
    return [(r[0], tuple(int(x) for x in r[2].split("x"))) for r in rows]


def main(cfg, argv):
    if argv[0] == "--manifest":
        state = torch.load(argv[1], map_location="cpu", mmap=True, weights_only=True)
        print(cfg.header)
        for name, t in kept(state):
            print(f"{name}\tF32\t{'x'.join(str(d) for d in t.shape)}\t1\t{t.numel() * 4}")
        return
    pth, config, out = argv
    state = dict(kept(torch.load(pth, map_location="cpu", mmap=True, weights_only=True)))
    tensors = {}
    for name, shape in names(cfg.manifest):
        t = state[name]
        assert tuple(t.shape) == shape, name
        tensors[name] = t
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    shutil.copy(config, os.path.join(out, "config.json"))
    print(f"{len(tensors)} tensors -> {out}/model.safetensors")


if __name__ == "__main__":
    main(Config(), sys.argv[1:])
