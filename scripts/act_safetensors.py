import os
import shutil
import sys
from dataclasses import dataclass

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet_safetensors import folded


@dataclass(frozen=True)
class Config:
    prefix: str = "model.backbone."


def constants(ck):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    cfg = PreTrainedConfig.from_pretrained(ck)
    cfg.pretrained_backbone_weights = None
    cfg.device = "cpu"
    m = ACTPolicy.from_pretrained(ck, config=cfg).eval().model.double()
    d = cfg.dim_model
    [(_, image)] = cfg.image_features.items()
    grid = [n // 32 for n in image.shape[1:]]
    out = {}
    with torch.no_grad():
        cam = m.encoder_cam_feat_pos_embed(torch.zeros(1, d, *grid, dtype=torch.float64))
        pos = torch.cat([m.encoder_1d_feature_pos_embed.weight, cam[0].flatten(1).T])
        for j, layer in enumerate(m.encoder.layers):
            w = layer.self_attn.in_proj_weight
            out[f"hk.encoder.layers.{j}.self_attn.pos_qkv"] = torch.cat([pos @ w[: 2 * d].T, torch.zeros(len(pos), d, dtype=pos.dtype)], 1)
        layer = m.decoder.layers[0]
        qpos = m.decoder_pos_embed.weight.unsqueeze(1)
        y = torch.zeros(cfg.chunk_size, 1, d, dtype=torch.float64)
        n1 = layer.norm1(y + layer.self_attn(y + qpos, y + qpos, value=y)[0])[:, 0]
        w, b = layer.multihead_attn.in_proj_weight, layer.multihead_attn.in_proj_bias
        out["hk.decoder.layers.0.norm1.out"] = n1
        out["hk.decoder.layers.0.multihead_attn.q"] = (n1 + qpos[:, 0]) @ w[:d].T + b[:d]
        out["hk.decoder.layers.0.multihead_attn.pos_k"] = pos @ w[d : 2 * d].T
    return {k: v.float().contiguous() for k, v in out.items()}


def main(cfg, ck, out, manifest=None):
    state = load_file(os.path.join(ck, "model.safetensors"))
    tensors = {k: t.contiguous() for k, t in state.items() if not k.startswith(cfg.prefix)}
    tensors.update(folded(state, cfg.prefix))
    tensors.update(constants(ck))
    tensors = dict(sorted(tensors.items()))
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    shutil.copy(os.path.join(ck, "config.json"), os.path.join(out, "config.json"))
    if manifest:
        with open(manifest, "w") as f:
            f.write("# lerobot/act_aloha_sim_transfer_cube_human @ ba73b2766f1371cdc133ca4efb97eb090d744625 as scripts/act_safetensors.py writes it: the backbone's BatchNorms folded into its convs, the tick's input-independent values hk.* (name, dtype, shape, count, bytes)\n")
            for k, t in tensors.items():
                f.write(f"{k}\tF32\t{'x'.join(str(d) for d in t.shape)}\t1\t{t.numel() * 4}\n")
    print(f"{len(tensors)} tensors -> {out}/model.safetensors")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Config(), a[0], a[1], a[3] if len(a) > 3 and a[2] == "--manifest" else None)
