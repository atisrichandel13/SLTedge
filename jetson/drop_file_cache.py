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


def force_reclaim(target_mb, cap_mb=6144, chunk_mb=256):
    """Make the kernel evict page cache by briefly allocating anonymous memory.

    fadvise DONTNEED only drops pages backed by files we name, and only when nothing else references
    them. When the cache is full of somebody else's files -- this board is shared, and a desktop
    session counts -- MemFree can sit at ~100 MB with 6 GB Cached and fadvise frees almost nothing.
    nvmap allocates from MemFree ONLY, so that state fails every model load with
    "NvMapMemAllocInternalTagged ... error 12" even though MemAvailable looks healthy.

    Dropping caches properly needs root (/proc/sys/vm/drop_caches) and we have none. Allocating and
    immediately freeing anonymous memory achieves it from userspace: the kernel reclaims clean page
    cache to satisfy the allocation. Stops as soon as MemFree reaches target_mb, never allocates past
    cap_mb, and refuses to start if MemAvailable cannot cover it -- overshooting here would invite the
    OOM killer onto a shared machine.
    """
    free0 = meminfo("MemFree:")
    if free0 >= target_mb:
        print(f"[cache] MemFree {free0} MB already >= target {target_mb} MB, no reclaim needed")
        return free0
    avail = meminfo("MemAvailable:")
    # Allocate roughly the TARGET, not the deficit: a deficit-sized allocation fits in memory that is
    # already free and so evicts nothing. To make the kernel drop page cache the allocation has to be
    # big enough that free memory alone cannot satisfy it.
    want = min(cap_mb, target_mb)
    if avail < want + 512:
        want = max(0, avail - 512)
        if want < chunk_mb:
            print(f"[cache] WARNING MemAvailable {avail} MB too low to force reclaim; skipping")
            return free0
        print(f"[cache] capping reclaim at {want} MB (MemAvailable {avail} MB)")
    blocks, got = [], 0
    try:
        while got < want:
            blocks.append(bytearray(chunk_mb << 20))   # bytearray is touched, so it is resident
            got += chunk_mb
    except MemoryError:
        print(f"[cache] MemoryError after {got} MB; releasing")
    finally:
        blocks.clear()
    free1 = meminfo("MemFree:")
    print(f"[cache] force-reclaim touched {got} MB: MemFree {free0} -> {free1} MB "
          f"(target {target_mb})")
    return free1


def main(argv=None):
    argv = list(argv or [])
    target = None
    roots = []
    for a in argv:
        if a.startswith("--target-free-mb="):
            target = int(a.split("=", 1)[1])
        else:
            roots.append(a)
    before = (meminfo("MemFree:"), meminfo("Cached:"))
    n = total = 0
    for root in (roots or DEFAULT_ROOTS):
        for d, _, files in os.walk(root):
            for fn in files:
                sz = drop(os.path.join(d, fn))
                if sz > 1 << 20:
                    n += 1; total += sz
    after = (meminfo("MemFree:"), meminfo("Cached:"))
    print(f"[cache] fadvise DONTNEED on {n} files ({total/1e9:.2f} GB) under "
          f"{roots or DEFAULT_ROOTS}: MemFree {before[0]} -> {after[0]} MB, "
          f"Cached {before[1]} -> {after[1]} MB")
    if target:
        force_reclaim(target)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
