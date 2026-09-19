#!/usr/bin/env python3
"""Evict our own large files from the page cache without root (posix_fadvise DONTNEED).

Why: on the Orin Nano the GPU allocator (nvmap) draws only from MemFree and does not reclaim page
cache, so a TensorRT build can fail with "Cuda Runtime (out of memory)" while `free` shows GBs of
reclaimable cache. The cache is mostly our own weights / ONNX / engines, and we may drop those.

    python3 jetson/drop_file_cache.py [dirs...]     default: models weights data results
"""
import os
import sys


def drop(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return os.fstat(fd).st_size
    except OSError:
        return 0
    finally:
        os.close(fd)


def meminfo(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) // 1024
    return -1


DEFAULT_ROOTS = ["models", "weights", "data", "results"]


def main(roots=None):
    roots = roots or DEFAULT_ROOTS
    before = (meminfo("MemFree:"), meminfo("Cached:"))
    n = total = 0
    for root in roots:
        for d, _, files in os.walk(root):
            for fn in files:
                sz = drop(os.path.join(d, fn))
                if sz > 1 << 20:
                    n += 1; total += sz
    after = (meminfo("MemFree:"), meminfo("Cached:"))
    print(f"[cache] fadvise DONTNEED on {n} files ({total/1e9:.2f} GB) under {roots}: "
          f"MemFree {before[0]} -> {after[0]} MB, Cached {before[1]} -> {after[1]} MB")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
