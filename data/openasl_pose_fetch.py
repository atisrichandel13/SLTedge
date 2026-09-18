#!/usr/bin/env python3
"""Fetch individual Uni-Sign OpenASL pose pkls straight from HuggingFace without downloading the
32 GB archive.

The release is one zip split into 8 parts (openasl_pose_format.zip.00..07). A zip's index (central
directory) lives at the end, so this script exposes the 8 parts as one virtual file over HTTP range
requests, lets Python's zipfile read the index (~10 MB), and extracts only the members asked for.

    python data/openasl_pose_fetch.py --names "Ads-4j06eJY-00:07:37.233-00:07:47.200" --out data/openasl_ref_pose
    python data/openasl_pose_fetch.py --split test --limit 20 --out data/openasl_test_pose   # first 20 test clips
    python data/openasl_pose_fetch.py --list | head                                          # member names

Member names inside the archive look like pose-rtmpose-192/<clip>.pkl.
"""
import argparse
import gzip
import io
import os
import pickle
import sys
import zipfile

import requests

REPO = "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main/"
PARTS = [f"openasl_pose_format.zip.{i:02d}" for i in range(8)]


class HttpConcatFile(io.RawIOBase):
    """Read-only, seekable view over several HTTP objects concatenated, with a read-ahead buffer."""

    def __init__(self, urls, chunk=4 << 20):
        self.s = requests.Session()
        self.urls, self.sizes = [], []
        for u in urls:
            r = self.s.head(u, allow_redirects=True)
            r.raise_for_status()
            self.urls.append(r.url)
            self.sizes.append(int(r.headers["Content-Length"]))
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total = sum(self.sizes)
        self.pos = 0
        self.chunk = chunk
        self.buf, self.buf_start = b"", 0
        self.bytes_fetched = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.total + off}[whence]
        return self.pos

    def _fetch(self, start, end):  # [start, end)
        out = b""
        while start < end:
            i = max(j for j, o in enumerate(self.offsets) if o <= start)
            lo = start - self.offsets[i]
            hi = min(end - self.offsets[i], self.sizes[i]) - 1
            r = self.s.get(self.urls[i], headers={"Range": f"bytes={lo}-{hi}"})
            r.raise_for_status()
            out += r.content
            start += hi - lo + 1
        self.bytes_fetched += len(out)
        return out

    def read(self, n=-1):
        if n < 0:
            n = self.total - self.pos
        end = min(self.pos + n, self.total)
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            want = max(n, self.chunk)
            self.buf_start = self.pos
            self.buf = self._fetch(self.pos, min(self.pos + want, self.total))
        a = self.pos - self.buf_start
        data = self.buf[a:a + (end - self.pos)]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_archive():
    f = HttpConcatFile([REPO + p for p in PARTS])
    print(f"[fetch] archive {f.total/1e9:.1f} GB over {len(PARTS)} parts; reading index ...", file=sys.stderr)
    z = zipfile.ZipFile(f)
    print(f"[fetch] {len(z.namelist())} members, index cost {f.bytes_fetched/1e6:.1f} MB", file=sys.stderr)
    return f, z


def split_names(split, labels_dir):
    d = pickle.load(gzip.open(os.path.join(labels_dir, f"labels.{split}"), "rb"))
    return [k.replace(".mp4", "") for k in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=[], help="clip ids without .pkl, e.g. Ads-4j06eJY-00:07:37.233-00:07:47.200")
    ap.add_argument("--split", choices=["train", "dev", "test"], help="fetch every clip of a split (needs --labels-dir)")
    ap.add_argument("--labels-dir", default=None, help="dir with Uni-Sign data/OpenASL/labels.*")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/openasl_ref_pose")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    f, z = open_archive()
    if args.list:
        for n in z.namelist():
            print(n)
        return
    names = list(args.names)
    if args.split:
        names += split_names(args.split, args.labels_dir)
    if args.limit:
        names = names[:args.limit]
    by_base = {os.path.basename(n)[:-4]: n for n in z.namelist() if n.endswith(".pkl")}
    os.makedirs(args.out, exist_ok=True)
    ok, missing = 0, []
    for i, name in enumerate(names):
        m = by_base.get(name)
        if m is None:
            missing.append(name)
            continue
        dst = os.path.join(args.out, name + ".pkl")
        if os.path.exists(dst):
            ok += 1
            continue
        with z.open(m) as src, open(dst, "wb") as out:
            out.write(src.read())
        ok += 1
        if i % 20 == 0:
            print(f"[fetch] {i+1}/{len(names)} {name} ({f.bytes_fetched/1e6:.0f} MB fetched so far)", file=sys.stderr)
    print(f"[fetch] done: {ok} written to {args.out}, {len(missing)} not in archive, "
          f"{f.bytes_fetched/1e6:.1f} MB transferred", file=sys.stderr)
    if missing:
        print("[fetch] missing:", missing[:10], file=sys.stderr)


if __name__ == "__main__":
    main()
