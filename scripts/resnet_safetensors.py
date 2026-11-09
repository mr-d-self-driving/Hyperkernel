import torch


def folded(state, prefix="", eps=1e-5):
    out, norms = {}, set()
    for name, w in state.items():
        if not name.startswith(prefix) or not name.endswith(".weight") or w.dim() != 4:
            continue
        conv = name[: -len(".weight")]
        bn = conv[:-1] + "1" if conv.endswith("downsample.0") else conv.replace("conv", "bn")
        g, b, m, v = (state[f"{bn}.{k}"].double() for k in ("weight", "bias", "running_mean", "running_var"))
        scale = g / torch.sqrt(v + eps)
        out[f"{conv}.weight"] = (w.double() * scale[:, None, None, None]).float().contiguous()
        out[f"{conv}.bias"] = (b - m * scale).float().contiguous()
        norms |= {f"{bn}.{k}" for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")}
    for name, t in state.items():
        if name.startswith(prefix) and name not in norms and name not in out and not (name.endswith(".weight") and t.dim() == 4):
            out[name] = t.float().contiguous()
    return out
