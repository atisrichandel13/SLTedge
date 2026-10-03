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

Integrity (added 2026-10-03). Every file is written to <name>.pkl.part and renamed only after its
size matches the size recorded in the archive index, and a full-member read makes zipfile verify the
CRC. Resuming re-checks sizes rather than trusting that the path exists, which is what the original
did -- so an interrupted run used to leave a truncated pkl that was accepted silently on every later
run. Use --verify-only to audit an existing directory without downloading.
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
            got = len(r.content)
            want = hi - lo + 1
            # Advance by what ARRIVED, not by what was asked for. The original advanced by `want`,
            # so a short range response (a proxy trimming the body, a connection cut mid-body)
            # silently shifted every later byte and produced a corrupt member. Advancing by `got`
            # makes the loop re-request the remainder instead, which is self-correcting.
            if got == 0:
                raise IOError(f"empty range response for {self.urls[i]} bytes={lo}-{hi}")
            if got > want:
                # a server that ignores Range answers 200 with the whole object; taking that as the
                # requested slice would corrupt the stream in a way the zip CRC may not localise
                raise IOError(f"range ignored for {self.urls[i]}: asked {want} B, got {got} B "
                              f"(status {r.status_code}); refusing to guess the alignment")
            out += r.content
            start += got
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
    ap.add_argument("--verify-only", action="store_true",
                    help="check existing files against the archive's recorded sizes and report, "
                         "without downloading anything")
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

    # Fetch in ARCHIVE-OFFSET order, not label order. HttpConcatFile keeps a single 4 MB
    # read-ahead buffer and drops it on any non-adjacent seek, so a label-ordered walk of a
    # 32 GB archive pays a full 4 MB range request per ~0.5 MB member -- measured at 4.22 MB
    # transferred per clip on both the dev split (967 clips, 4069.5 MB) and the first 441 train
    # clips (1863 MB). Sorting by header_offset makes consecutive members buffer hits instead.
    names.sort(key=lambda n: z.getinfo(by_base[n]).header_offset if n in by_base else -1)

    os.makedirs(args.out, exist_ok=True)
    ok, missing, repaired, bad = 0, [], [], []
    for i, name in enumerate(names):
        m = by_base.get(name)
        if m is None:
            missing.append(name)
            continue
        info = z.getinfo(m)
        dst = os.path.join(args.out, name + ".pkl")
        # Resume must verify, not assume. The archive index carries each member's uncompressed
        # size, so a truncated file is detectable for free -- and WAS NOT detected before: the old
        # check was `os.path.exists(dst)`, which accepts a half-written file forever, and every
        # accuracy number in this project is scored on files fetched by this script.
        if os.path.exists(dst):
            have = os.path.getsize(dst)
            if have == info.file_size:
                ok += 1
                continue
            print(f"[fetch] REFETCH {name}: {have} B on disk, archive says {info.file_size} B",
                  file=sys.stderr)
            repaired.append(name)
        if args.verify_only:
            continue
        # Write to a temp name and rename. A pkl under its final name is the resume marker, so it
        # must never exist half-written -- the same discipline task1_rtmpose/09_batch_clips.py uses.
        tmp = dst + ".part"
        try:
            with z.open(m) as src, open(tmp, "wb") as out:
                out.write(src.read())   # full read, so zipfile verifies the member CRC
            got = os.path.getsize(tmp)
            if got != info.file_size:
                raise IOError(f"wrote {got} B but archive says {info.file_size} B")
            os.replace(tmp, dst)
        except Exception as e:                      # noqa: BLE001 -- report and keep going
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"[fetch] FAILED {name}: {type(e).__name__}: {e}", file=sys.stderr)
            bad.append(name)
            continue
        ok += 1
        if i % 20 == 0:
            print(f"[fetch] {i+1}/{len(names)} {name} ({f.bytes_fetched/1e6:.0f} MB fetched so far)", file=sys.stderr)
    print(f"[fetch] done: {ok} ok in {args.out}, {len(missing)} not in archive, "
          f"{len(repaired)} re-fetched (size mismatch), {len(bad)} failed, "
          f"{f.bytes_fetched/1e6:.1f} MB transferred", file=sys.stderr)
    if missing:
        print("[fetch] missing:", missing[:10], file=sys.stderr)
    if repaired:
        print("[fetch] size mismatches:", repaired[:10], file=sys.stderr)
    if bad:
        print("[fetch] failures:", bad[:10], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
