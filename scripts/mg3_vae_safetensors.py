import os
import sys

import torch
from safetensors.torch import save_file


def main(pth, dst):
    model = torch.load(pth, map_location="cpu", mmap=True, weights_only=True)["state_dict"]["gen_model"]
    t = {k: v.contiguous().clone() for k, v in model.items() if k.startswith(("decoder.", "post_quant_conv."))}
    os.makedirs(dst, exist_ok=True)
    save_file(t, os.path.join(dst, "vae.safetensors"))
    print(f"{len(t)} tensors -> {dst}/vae.safetensors")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
