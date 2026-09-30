#!/usr/bin/env python3
"""One-shot Colab environment rebuild for the LM track.

Colab loses pip packages on every runtime restart and loses /content entirely when the runtime is
recycled, so this script is written to be run at the top of every session and to be safely
re-runnable: each step is skipped when its output is already present.

    !python colab_setup.py

Deliberately depends on NOTHING in Google Drive. Drive filled up mid-session once and the project
folder was cleared while freeing space, taking the released checkpoint with it. Every input here
comes from either this git repo or a public HuggingFace download, so a fresh runtime rebuilds the
whole environment in roughly 20 minutes without touching Drive.

Produces:
    /content/poses_train          98,419 OpenASL pose pkls, flat, all splits
    /content/weights/…            authors' released checkpoint + google/mt5-base
    /content/pruned_traindev.pth  leak-free pruned checkpoint (26,025 ids, train+dev census)
"""
import io
import os
import sys
import zipfile

# Line-buffer stdout: these run under nohup with output redirected to a file, where
# Python block-buffers by default and progress is invisible for minutes at a time.
sys.stdout.reconfigure(line_buffering=True)

W = "/content/weights"
POSES = "/content/poses_train"
CKPT = "/content/pruned_traindev.pth"
ZIP = "/content/zip"
BASE = "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main/openasl_pose_format.zip."
PARTS = [f"{i:02d}" for i in range(8)]


def packages():
    # Reinstalled every session: a runtime restart keeps /content but drops site-packages.
    os.system("pip install -q portalocker rouge 'transformers>=4.45,<5' sentencepiece")


def archive():
    """Download the 8 zip parts in parallel, skipping any already complete.

    Whole-archive download beats per-member HTTP range extraction here. openasl_pose_fetch.py
    keeps a 4 MB read-ahead buffer that it discards on every non-adjacent seek, so pulling a
    sparse subset costs ~4.2 MB transferred per ~0.5 MB clip. Above roughly 20% density that
    exceeds the size of the archive itself, and we want every split anyway.
    """
    import concurrent.futures as cf

    import requests

    os.makedirs(ZIP, exist_ok=True)
    s = requests.Session()
    sizes = {p: int(s.head(BASE + p, allow_redirects=True).headers["Content-Length"]) for p in PARTS}
    print(f"[setup] archive {sum(sizes.values())/1e9:.1f} GB in {len(PARTS)} parts")

    def get(p):
        dst = f"{ZIP}/p{p}.zip"
        if os.path.exists(dst) and os.path.getsize(dst) == sizes[p]:
            return p, "cached"
        with requests.get(BASE + p, stream=True) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for c in r.iter_content(1 << 22):
                    f.write(c)
        return p, f"{os.path.getsize(dst)/1e9:.2f} GB"

    with cf.ThreadPoolExecutor(8) as ex:
        for p, msg in ex.map(get, PARTS):
            print(f"[setup]   part {p}: {msg}")
    return [f"{ZIP}/p{p}.zip" for p in PARTS]


class Concat(io.RawIOBase):
    """The 8 parts as one seekable stream, so the zip is read without materialising a 32 GB copy."""

    def __init__(self, paths):
        self.f = [open(p, "rb") for p in paths]
        self.sz = [os.path.getsize(p) for p in paths]
        self.off = [sum(self.sz[:i]) for i in range(len(self.sz))]
        self.total, self.pos = sum(self.sz), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, o, w=0):
        self.pos = {0: o, 1: self.pos + o, 2: self.total + o}[w]
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.total - self.pos
        out = b""
        while n > 0 and self.pos < self.total:
            i = max(j for j, o in enumerate(self.off) if o <= self.pos)
            self.f[i].seek(self.pos - self.off[i])
            b = self.f[i].read(min(n, self.sz[i] - (self.pos - self.off[i])))
            if not b:
                break
            out += b
            self.pos += len(b)
            n -= len(b)
        return out


def extract(parts):
    """Extract every pose pkl, validating against the size the archive records for each member.

    An existence check is not enough. A killed writer (an interrupted download, a recycled
    runtime) leaves a truncated or zero-byte file behind, and skipping it because the path
    exists hides the corruption until a DataLoader worker reaches that clip and dies with
    "EOFError: Ran out of input" -- which happened ~2250 steps into a training run. Comparing
    against ZipInfo.file_size costs nothing (the index is already in memory) and makes the
    step genuinely idempotent.
    """
    os.makedirs(POSES, exist_ok=True)
    z = zipfile.ZipFile(Concat(parts))
    members = [m for m in z.namelist() if m.endswith(".pkl")]
    written, repaired = 0, 0
    for m in members:
        # Flatten: the archive nests under pose-rtmpose-192/, train_adapt --poses wants flat.
        info = z.getinfo(m)
        dst = os.path.join(POSES, os.path.basename(m))
        if os.path.exists(dst):
            if os.path.getsize(dst) == info.file_size:
                continue
            repaired += 1          # present but wrong size -> truncated, rewrite it
        with z.open(m) as src, open(dst, "wb") as f:
            f.write(src.read())
        written += 1
        if written % 10000 == 0:
            print(f"[setup]   {written} written")
    print(f"[setup] poses: {len(os.listdir(POSES))} files "
          f"({written} written, of which {repaired} were corrupt and rewritten)")


def weights():
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(W, exist_ok=True)
    ckpt = hf_hub_download("ZechengLi19/Uni-Sign", "openasl_pose_only_slt.pth", local_dir=W)
    mt5 = snapshot_download("google/mt5-base", local_dir=f"{W}/mt5-base",
                            ignore_patterns=["*.h5", "*.msgpack", "*tf_model*", "*flax*"])
    print(f"[setup] released ckpt {os.path.getsize(ckpt)/1e6:.0f} MB; mt5-base at {mt5}")
    return ckpt, mt5


def prune(released, mt5):
    if os.path.exists(CKPT):
        print(f"[setup] pruned checkpoint cached ({os.path.getsize(CKPT)/1e6:.0f} MB)")
        return
    import json

    import torch
    sys.path.insert(0, "/content/SLTedge")
    from unisign.model import prune_and_save
    keep = json.load(open("/content/SLTedge/results/openasl_vocab_keep_ids_traindev.json"))
    prune_and_save(released, mt5, keep, CKPT)
    d = torch.load(CKPT, map_location="cpu")
    # bf16 to match the deployed artifact's 570 MB, so memory rows stay comparable
    torch.save({"model": {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                          for k, v in d["model"].items()},
                "keep_ids": d["keep_ids"]}, CKPT)
    print(f"[setup] pruned ({len(keep)} ids): {os.path.getsize(CKPT)/1e6:.0f} MB")


if __name__ == "__main__":
    packages()
    parts = archive()
    extract(parts)
    released, mt5 = weights()
    prune(released, mt5)
    print("\n[setup] ready:")
    print(f"  --poses {POSES}")
    print(f"  --ckpt  {CKPT}")
    print(f"  --mt5   {W}/mt5-base")
