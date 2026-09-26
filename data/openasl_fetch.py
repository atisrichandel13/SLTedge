#!/usr/bin/env python3
"""Fetch OpenASL clips as cropped JPEG frames (queue P3, guide 2.3).

OpenASL is distributed by reference: the release gives YouTube ids, time ranges, signer bounding
boxes and sentences, not video. This script downloads only the seconds each clip needs, crops to the
signer box and writes frames, which is what the pose engines actually consume.

    # 5 test clips, each from a different YouTube video (signer proxy):
    python3 data/openasl_fetch.py --split test --n-clips 5 --out data/clips

    # >=300 calibration frames spread over >=5 training signers (INT8 later):
    python3 data/openasl_fetch.py --calib --n-signers 5 --frames-per-signer 70 --out data/calib_frames

Crop recipe matches `data/test_frames_meta.json` from the first clip (SESSION-LOG §3): the
normalised OpenASL box scaled to the frame and **clamped** to it, at native resolution and native
fps. That differs from OpenASL's own `prep/crop_video.py`, which squares the box, black-pads what
falls outside and resizes to 224 - we keep the extra pixels because RTMPose/RTMW do their own affine
to the model input, and black padding would change the pose input. Whatever changes here must change
for every clip, or latency and accuracy rows stop being comparable.

Needs `yt-dlp` and `ffmpeg` on PATH (or --yt-dlp / --ffmpeg). Expect dead links: OpenASL is 3 years
old and clips go private or get deleted, so the script walks candidates until enough succeed and
records every failure with its reason.
"""
import argparse
import gzip
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile

TSV_URL = "https://raw.githubusercontent.com/chevalierNoir/OpenASL/main/data/openasl-v1.0.tsv"
BBOX_URL = "https://raw.githubusercontent.com/chevalierNoir/OpenASL/main/data/bbox-v1.0.json"


def sh(cmd, timeout=900):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def fetch_meta(cache_dir):
    """OpenASL table + bbox json, cached. Returns (rows, vid2bbox)."""
    os.makedirs(cache_dir, exist_ok=True)
    tsv, bj = os.path.join(cache_dir, "openasl-v1.0.tsv"), os.path.join(cache_dir, "bbox-v1.0.json")
    for url, path in ((TSV_URL, tsv), (BBOX_URL, bj)):
        if not os.path.exists(path):
            import urllib.request
            print(f"[meta] downloading {os.path.basename(path)}")
            urllib.request.urlretrieve(url, path)
    with open(tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, l.rstrip("\n").split("\t"))) for l in f]
    return rows, json.load(open(bj))


def hms_to_s(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def probe(ffprobe, path):
    rc, out, err = sh([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=width,height,avg_frame_rate,nb_read_packets", "-count_packets",
                       "-of", "json", path])
    if rc:
        raise RuntimeError(f"ffprobe failed: {err.strip()[:200]}")
    st = json.loads(out)["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return int(st["width"]), int(st["height"]), round(fps, 3), int(st.get("nb_read_packets", 0))


def crop_xywh(bbox_norm, W, H):
    """Normalised OpenASL box -> integer crop, clamped to the frame (no black padding)."""
    x0, y0, x1, y1 = bbox_norm
    x0, x1 = int(round(x0 * W)), int(round(x1 * W))
    y0, y1 = int(round(y0 * H)), int(round(y1 * H))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise RuntimeError(f"degenerate crop {x0},{y0},{x1},{y1} in {W}x{H}")
    return [x0, y0, x1 - x0, y1 - y0]


def download_section(yt_dlp, yid, start_s, end_s, dest, height=720, cookies=None):
    """yt-dlp section download. Returns (ok, reason)."""
    cmd = [yt_dlp, "-q", "--no-warnings", "--no-playlist",
           "-f", f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b",
           "--download-sections", f"*{start_s:.3f}-{end_s:.3f}", "--force-keyframes-at-cuts",
           "--merge-output-format", "mp4", "-o", dest, f"https://www.youtube.com/watch?v={yid}"]
    if cookies:
        cmd[1:1] = ["--cookies-from-browser", cookies]
    try:
        rc, out, err = sh(cmd)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 10000:
        return True, "ok"
    msg = (err or out).strip().splitlines()
    msg = msg[-1][:160] if msg else f"exit {rc}"
    return False, msg


def extract_frames(ffmpeg, video, out_dir, crop, prefix="f", stride=1, limit=None, quality=2):
    os.makedirs(out_dir, exist_ok=True)
    x, y, w, h = crop
    vf = f"crop={w}:{h}:{x}:{y}"
    if stride > 1:
        vf += f",select=not(mod(n\\,{stride}))"
    cmd = [ffmpeg, "-v", "error", "-y", "-i", video, "-vf", vf, "-vsync", "0",
           "-q:v", str(quality)]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd += [os.path.join(out_dir, f"{prefix}_%04d.jpg")]
    rc, out, err = sh(cmd)
    if rc:
        raise RuntimeError(f"ffmpeg crop failed: {err.strip()[:200]}")
    return sorted(f for f in os.listdir(out_dir) if f.startswith(prefix + "_"))


def candidates(rows, vid2bbox, split, min_dur, max_dur, exclude_yids):
    out = []
    for r in rows:
        if r["split"] != split or r["vid"] not in vid2bbox or r["yid"] in exclude_yids:
            continue
        dur = hms_to_s(r["end"]) - hms_to_s(r["start"])
        if not (min_dur <= dur <= max_dur):
            continue
        out.append((r, dur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib", action="store_true", help="calibration-frame mode (train split, strided)")
    ap.add_argument("--n-clips", type=int, default=5)
    ap.add_argument("--n-signers", type=int, default=5, help="calib mode: distinct videos to sample")
    ap.add_argument("--frames-per-signer", type=int, default=70)
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--exclude-yid", action="append", default=["Ads-4j06eJY"],
                    help="already have this signer (the baseline clip); repeatable")
    ap.add_argument("--max-attempts", type=int, default=40, help="dead links are common; cap the walk")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="e.g. chrome - needed if YouTube demands sign-in")
    ap.add_argument("--yt-dlp", default=shutil.which("yt-dlp") or "yt-dlp")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    ap.add_argument("--cache", default=None, help="where to cache the tsv/bbox (default <out>/../openasl_meta)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = args.cache or os.path.join(os.path.dirname(os.path.abspath(args.out)), "openasl_meta")
    rows, vid2bbox = fetch_meta(cache)
    split = "train" if args.calib else args.split
    cands = candidates(rows, vid2bbox, split, args.min_dur, args.max_dur, set(args.exclude_yid))
    print(f"[meta] {len(rows)} rows, {len(cands)} {split} candidates with a bbox and duration "
          f"{args.min_dur}-{args.max_dur}s")

    # one clip per video, shuffled deterministically: distinct yid is our signer proxy (OpenASL has
    # no signer field, but a video is one signer in these news/vlog sources)
    by_yid = {}
    rnd = random.Random(args.seed)
    rnd.shuffle(cands)
    for r, dur in cands:
        by_yid.setdefault(r["yid"], (r, dur))
    order = list(by_yid.values())
    rnd.shuffle(order)

    want = args.n_signers if args.calib else args.n_clips
    os.makedirs(args.out, exist_ok=True)
    done, failures = [], []
    for r, dur in order:
        if len(done) >= want or len(done) + len(failures) >= args.max_attempts:
            break
        vid, yid = r["vid"], r["yid"]
        start_s, end_s = hms_to_s(r["start"]), hms_to_s(r["end"])
        tmp = tempfile.mkdtemp()
        mp4 = os.path.join(tmp, "clip.mp4")
        ok, why = download_section(args.yt_dlp, yid, start_s, end_s, mp4, args.height,
                                   args.cookies_from_browser)
        if not ok:
            print(f"[skip] {vid}: {why}")
            failures.append({"vid": vid, "yid": yid, "reason": why})
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        try:
            W, H, fps, npkt = probe(args.ffprobe, mp4)
            crop = crop_xywh(vid2bbox[vid], W, H)
            if args.calib:
                stride = max(1, int(npkt // max(1, args.frames_per_signer)) or 1)
                names = extract_frames(args.ffmpeg, mp4, args.out, crop, prefix=f"c_{yid}",
                                       stride=stride, limit=args.frames_per_signer)
                frame_dir = args.out
            else:
                frame_dir = os.path.join(args.out, vid, "frames")
                stride = 1
                names = extract_frames(args.ffmpeg, mp4, frame_dir, crop)
            meta = {"vid": vid, "yid": yid, "crop_xywh": crop, "bbox_norm": vid2bbox[vid],
                    "text": r["raw-text"], "split": split, "fps": fps, "source_wh": [W, H],
                    "n_frames": len(names), "stride": stride,
                    "start": r["start"], "end": r["end"], "duration_s": round(dur, 3)}
            if not args.calib:
                json.dump(meta, open(os.path.join(args.out, vid, "meta.json"), "w"), indent=1)
            done.append(meta)
            print(f"[ok]   {vid}  {W}x{H}@{fps}  crop {crop}  {len(names)} frames")
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {vid}: {e}")
            failures.append({"vid": vid, "yid": yid, "reason": str(e)[:200]})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    index = {"split": split, "mode": "calib" if args.calib else "clips", "requested": want,
             "n_ok": len(done), "n_failed": len(failures), "signers": [m["yid"] for m in done],
             "total_frames": sum(m["n_frames"] for m in done), "clips": done, "failures": failures,
             "crop": "OpenASL bbox scaled to frame and clamped (no square/pad/resize)",
             "height_cap": args.height, "seed": args.seed}
    idx_path = os.path.join(args.out, "index.json" if not args.calib else "calib_index.json")
    json.dump(index, open(idx_path, "w"), indent=1)
    print(f"[done] {len(done)}/{want} from {len(set(index['signers']))} distinct videos, "
          f"{index['total_frames']} frames, {len(failures)} failures -> {idx_path}")
    return 0 if len(done) >= want else 1


if __name__ == "__main__":
    sys.exit(main())
