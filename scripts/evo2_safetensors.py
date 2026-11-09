import io
import json
import os
import struct
import sys
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Config:
    dtypes: dict = field(default_factory=lambda: {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e5m2: "F8_E5M2",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    })


def save(dtypes, t, path):
    header, at = {}, 0
    for k, v in t.items():
        if v.dtype not in dtypes:
            raise SystemExit(f"{k}: no safetensors dtype for {v.dtype}")
        n = v.numel() * v.element_size()
        header[k] = {"dtype": dtypes[v.dtype], "shape": list(v.shape), "data_offsets": [at, at + n]}
        at += n
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for v in t.values():
            f.write(v.contiguous().reshape(-1).view(torch.uint8).numpy())


def main(cfg, pt, dst):
    torch.serialization.add_safe_globals([io.BytesIO])
    model = torch.load(pt, map_location="cpu", mmap=True, weights_only=True)
    t = {k: v for k, v in model.items() if isinstance(v, torch.Tensor) and not k.endswith("_extra_state")}
    os.makedirs(dst, exist_ok=True)
    save(cfg.dtypes, t, os.path.join(dst, "model.safetensors"))
    print(f"{len(t)} tensors -> {dst}/model.safetensors")


if __name__ == "__main__":
    main(Config(), sys.argv[1], sys.argv[2])
