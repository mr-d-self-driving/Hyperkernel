import os
import sys
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


@dataclass(frozen=True)
class Config:
    summarizer: str = "p_policy_model.temporal_summarizer."
    excluded: frozenset = frozenset({"model/scalar_tensor_default", "model/p_scalar_tensor_default", "model/p_val_51", "model/p_pad"})
    norm: dict = field(default_factory=lambda: {"model/vision_model.vision._mean": 127.5, "model/vision_model.vision._std": 63.75})
    floats: tuple = (np.float16, np.float32)

    @property
    def matmuls(self):
        t = self.summarizer
        return {
            "model/p_val_13": (t + "_feats_encode.0.weight", (512, 512)),
            "model/p_val_16": (t + "_feats_encode.2.weight", (512, 512)),
            "model/p_val_29": (t + "transformer.0.attn.c_attn.weight", (512, 1536)),
            "model/p_val_58": (t + "transformer.0.attn.c_proj.weight", (512, 512)),
            "model/p_val_62": (t + "transformer.0.mlp.c_fc.weight", (512, 2048)),
            "model/p_val_64": (t + "transformer.0.mlp.c_proj.weight", (2048, 512)),
        }

    @property
    def pos(self):
        return ("model/p_add", self.summarizer + "pos")


def need(ok, *why):
    if not ok:
        raise SystemExit("openpilot_safetensors: " + " ".join(map(str, why)))


def tensors(cfg, path):
    m = onnx.load(path)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    by_out = {n.output[0]: n for n in m.graph.node}
    out, named = {}, set()
    for node_out, (name, shape) in cfg.matmuls.items():
        n = by_out.get(node_out)
        need(n is not None and n.op_type == "MatMul" and n.input[1] in inits, node_out, "is not a MatMul of an initializer")
        w = inits[n.input[1]]
        need(w.shape == shape, name, w.shape, "not", shape)
        out[name] = np.ascontiguousarray(w.T.astype(np.float32))
        named.add(n.input[1])
    pos = cfg.pos
    n = by_out.get(pos[0])
    need(n is not None and n.op_type == "Add" and n.input[1] in inits and inits[n.input[1]].shape == (1, 9, 512), pos[0], "is not an Add of a (1, 9, 512) initializer")
    out[pos[1]] = np.ascontiguousarray(inits[n.input[1]][0].astype(np.float32))
    named.add(n.input[1])
    for k, c in cfg.norm.items():
        need(k in inits and np.all(inits[k] == c), k, "is not", c)
    for k, v in inits.items():
        if k in named or k in cfg.excluded or k in cfg.norm or v.dtype not in cfg.floats:
            continue
        need(k.startswith("model/"), k, "is outside model/")
        out[k[len("model/"):]] = np.ascontiguousarray(v.astype(np.float32))
    return out


def main(cfg, path, dst):
    t = tensors(cfg, path)
    os.makedirs(dst, exist_ok=True)
    save_file(t, os.path.join(dst, "model.safetensors"))
    print(f"{len(t)} tensors -> {dst}/model.safetensors")


if __name__ == "__main__":
    main(Config(), sys.argv[1], sys.argv[2])
