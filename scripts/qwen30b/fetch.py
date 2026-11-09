import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer


@dataclass(frozen=True)
class Config:
    repo: str = "Qwen/Qwen3-30B-A3B"
    revision: str = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    data: str = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/"
    files: int = 14


def tokenize(cfg, tokenizer, out, tokens, cache):
    tk = Tokenizer.from_file(tokenizer)
    eos = tk.token_to_id("<|endoftext|>")
    assert eos is not None, "the tokenizer has no <|endoftext|>"
    os.makedirs(cache, exist_ok=True)
    n = 0
    with open(out + ".tmp", "wb") as f:
        for i in range(cfg.files):
            name = f"{i:03d}_00000.parquet"
            path = os.path.join(cache, name)
            if not os.path.exists(path):
                urllib.request.urlretrieve(cfg.data + name, path + ".part")
                os.replace(path + ".part", path)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text"]):
                for e in tk.encode_batch(batch.column("text").to_pylist(), add_special_tokens=False):
                    a = np.asarray(e.ids + [eos], dtype=np.uint32)
                    a.tofile(f)
                    n += len(a)
                if n >= tokens:
                    break
            os.remove(path)
            print(f"{name}: {n} tokens", flush=True)
            if n >= tokens:
                break
    if n < tokens:
        sys.exit(f"the sample holds {n} tokens, fewer than the {tokens} asked")
    os.replace(out + ".tmp", out)


def main(cfg):
    model, out, tokens, cache = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    print(f"checkpoint {cfg.repo} @ {cfg.revision[:8]} -> {model}", flush=True)
    snapshot_download(cfg.repo, revision=cfg.revision, local_dir=model, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
    if os.path.exists(out) and os.path.getsize(out) >= tokens * 4:
        print(f"tokens: {out} already holds {os.path.getsize(out) // 4}", flush=True)
    else:
        print(f"tokenizing FineWeb-Edu to {tokens} tokens -> {out}", flush=True)
        tokenize(cfg, os.path.join(model, "tokenizer.json"), out, tokens, cache)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        h.update(f.read(1 << 30))
    print(f"tokens: {os.path.getsize(out) // 4}; sha256 of the first GiB {h.hexdigest()}", flush=True)


if __name__ == "__main__":
    main(Config())
