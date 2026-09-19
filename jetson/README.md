# Jetson container for this project (pose owner)

Board: `jetson-lpcv-03` (lab LAN `192.168.1.73`, reachable through WireGuard), user `tgoyal`, **no
sudo**, docker group. Shared course board: other students' containers run on it, so never touch
containers you did not start. L4T R36.4.7, 15 W mode fixed, 6 CPUs, 7.4 GiB RAM, ~40 GB free.

The project runs in its own image `slt-jetson:25.06`, built by `jetson/Dockerfile` from
`nvcr.io/nvidia/pytorch:25.06-py3` (already pulled on the board). Same stack as the Phase-1 numbers:
Python 3.12.3, torch 2.8.0a0 nv25.06, TensorRT 10.11.0.33, CUDA 12.9, plus OpenCV, onnx,
onnxruntime (CPU), transformers 4.x, sentencepiece. The container runs as your UID, so files written
into the bind mount stay yours (a root container left `~/lpcv` root-owned once).

## Layout on the board

```
~/sign-lang-project/      this repo, rsynced from the Mac   -> /workspace in the container
~/.cache/slt-container/   HF / torch caches                 -> /home/tgoyal/.cache
```

## Commands (on the board)

```bash
jetson/run.sh build     # once
jetson/run.sh probe     # versions, GPU, power rails
jetson/run.sh up        # persistent container slt-work (engine builds, long runs)
jetson/run.sh exec python3 task1_rtmpose/03_infer_frames.py --help
jetson/run.sh shell     # interactive, throwaway
```

## Pulling results back

Use `jetson/pull_results.sh`, not a bare `rsync board:results/ results/`. The board holds a full
copy of the repo, so a plain pull overwrites `results/RESULTS.md` with the board's older version and
silently reverts the write-up (this happened on 2026-09-18). The script excludes `*.md` and `*.npz`.

## Memory on the shared board

The Orin Nano's 8 GB is one pool for CPU and GPU. GPU allocations (nvmap) take only from `MemFree`
and do not reclaim page cache, so a big TensorRT build or a 2 GB model load can fail with
`Cuda Runtime (out of memory)` while `free` shows gigabytes of "available" cache. Without root the
fix is to evict our own files from the cache first:

    python3 jetson/drop_file_cache.py            # fadvise DONTNEED on models/ weights/ data/ results/

`task3_mt5_onnx/03_build_engines.py` runs this automatically when it detects the Jetson. Also keep
`--workspace-gb 1` for engine builds and build one engine per process.

## Sync from the Mac

```bash
rsync -av --exclude .git --exclude __pycache__ ./ tgoyal@192.168.1.73:~/sign-lang-project/
```

Large artefacts (`weights/`, `models/*.onnx`, `data/test_frames/`) are not in git; copy them the same
way, or fetch them directly on the board (internet works there: PyPI, GitHub, Hugging Face).
