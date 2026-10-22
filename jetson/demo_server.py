#!/usr/bin/env python3
"""Live demo server: one long-lived process on the board, phone in, English out.

    jetson/run.sh exec-batch python3 jetson/demo_server.py \
        --engine models/rtmw/rtmw-l-m_256x192_fp16.engine \
        --ckpt weights/openasl_pose_only_slt_pruned.pth --mt5 weights/mt5-base \
        --port 8765

Then, from any WireGuard peer:

    curl -s --data-binary @clip.mp4 'http://192.168.1.73:8765/translate?x0=0.36&y0=0&x1=0.78&y1=1'
    curl -s http://192.168.1.73:8765/health

WHY A SERVER AND NOT `unisign/e2e_translate.py`. That script is the M1 end-to-end *measurement*: it
loads both models, runs N sentences from a frames directory, writes JSON and exits. A demo cannot pay
the load cost per sentence -- the pruned LM alone takes 28.6-29.0 s (RESULTS.md L5/§3.3) -- so both
models are loaded once here and held for the process lifetime. The per-sentence pipeline below is
deliberately the same calls in the same order as e2e_translate's `one_sentence`, so the demo cannot
quietly drift from the configuration every number in RESULTS.md describes.

WHY THE PYTORCH LM AND NOT THE TRT ENGINE SET. Measured, not assumed: J10
(`results/lm_sweep_pruned_resident.json`, §2.9E) ran exactly this pairing -- RTMW FP16 engine loaded
BEFORE the LM and held resident for a 15-cell sweep -- and recorded MemFree 5130 -> 4838 after the
pose engine -> 179 after the LM -> 177 after pose warm-up. So it fits, with ~177 MB to spare. The mT5
TRT engine set takes 2.07 GB resident (§L8.2) and has NEVER been run with the pose engine resident;
L8.2 already needed the 1.03 GB PyTorch reference freed and the page cache evicted before those
engines would deserialise. Do not "optimise" this to TRT without re-measuring MemFree first.

WHY THE PHONE SENDS A BOX AND NOT A CROP. There is no person detector in this project
(`task1_rtmpose/03_infer_frames.py`), so the box has to come from the caller -- iOS Vision on the
phone. But the phone must send FULL frames plus the box, not a pre-cropped image, and specifically
NOT OpenASL's 224 square: every pose number we have was extracted from a NATIVE-resolution bbox crop
(`data/clips*/index.json` record `crop_style=native`, "no square/pad/resize") because RTMW does its
own affine (`data/openasl_fetch.py:17`). Cropping to 224 on the phone would resample twice and feed
the affine black padding -- a preprocessing path nothing in RESULTS.md covers.

THE NON-FINITE GUARD IS NOT DEFENSIVE BOILERPLATE. §L28: on frames with no person in the crop the
FP16 engine emits NaN SimCC, and before §L27's fix the decode turned that into a CONSTANT CORNER
COORDINATE that passes every range check. That silently NaN'd a training run which then saved a
570 MB checkpoint and exited 0. A live phone video is full of exactly those frames -- signer out of
shot, hand over the lens, a dark room -- so frames whose pose is not finite are DROPPED here and
counted in the response. Dropping is safe: the sequence is already irregularly sampled by the fps
thinning below.
"""
import argparse
import io
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for p in (ROOT, os.path.join(ROOT, "data"), os.path.join(ROOT, "task1_rtmpose")):
    sys.path.insert(0, p)

MAX_UPLOAD_BYTES = 256 * 1024 * 1024


def _memfree_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemFree"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


class Pipeline:
    """Both models resident, one lock. The GPU and the 15 W budget are not shareable, so requests
    are serialised rather than threaded -- ThreadingHTTPServer is used only so a /health probe is
    not blocked behind a translation in flight."""

    def __init__(self, args):
        import torch
        import cv2
        from common.trt_runner import TrtRunner
        from rtmpose_utils import load_preproc
        from unisign.model import load_model
        self.torch, self.cv2 = torch, cv2
        self.args = args
        self.lock = threading.Lock()
        self.n_served = 0
        self.last = None

        self.pp = load_preproc(args.preproc or os.path.join(
            os.path.dirname(os.path.abspath(args.engine)), "preproc.json"))

        mf0 = _memfree_mb()
        t0 = time.perf_counter()
        self.runner = TrtRunner(args.engine)
        self.load_pose_s = time.perf_counter() - t0
        self.in_name = self.runner.inputs[0]
        mf1 = _memfree_mb()

        t0 = time.perf_counter()
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
        self.model = load_model(args.ckpt, args.mt5, device=args.device, dtype=dtype)
        self.load_lm_s = time.perf_counter() - t0
        mf2 = _memfree_mb()

        # Warm up on synthetic input so the first real request does not pay TRT's first-call cost.
        h, w = self.pp["input_size"][1], self.pp["input_size"][0]
        x0 = np.zeros((1, 3, h, w), dtype=np.float32)
        for _ in range(args.warmup):
            self.runner.infer({self.in_name: x0})
        mf3 = _memfree_mb()

        # WARM THE LM TOO, NOT JUST THE ENGINE. Measured on the board 2026-10-22: the first real
        # request cost gcn 3422 ms / decoder 3526 ms / 11.7 s end to end, against 131-175 ms / 1642 ms
        # / 4.5 s on every request after it -- a 2.6x penalty paid entirely by whoever goes first.
        # In a live demo that is the audience. One synthetic sentence through the same translate()
        # path moves the cost to start-up, where it belongs.
        t0 = time.perf_counter()
        try:
            from common.pose_to_unisign import collate, to_model_inputs
            n = 48
            kps = [np.zeros((1, 133, 2), dtype=np.float32) for _ in range(n)]
            scs = [np.full((1, 133), 0.5, dtype=np.float32) for _ in range(n)]
            inputs, _ = to_model_inputs(kps, scs, args.max_length)
            self.model.translate(collate([inputs], ["warmup"]),
                                 max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            self.warm_lm_s = time.perf_counter() - t0
        except Exception as e:                                        # noqa: BLE001
            self.warm_lm_s = None
            print(f"[demo] LM warm-up skipped ({type(e).__name__}: {e}); the FIRST request will be "
                  f"~2.6x slower than steady state", flush=True)
        mf3 = _memfree_mb()

        self.memfree_mb = {"start": mf0, "after_pose": mf1, "after_lm": mf2, "after_warmup": mf3}
        print(f"[demo] pose engine {self.load_pose_s:.1f}s, LM {self.load_lm_s:.1f}s, "
              f"LM warm-up {self.warm_lm_s if self.warm_lm_s is None else round(self.warm_lm_s, 1)}s",
              flush=True)
        print(f"[demo] MemFree MB {self.memfree_mb}", flush=True)
        if mf3 is not None and mf3 < 120:
            print(f"[demo] WARNING: only {mf3} MB MemFree with both models resident. J10 measured "
                  f"177 MB for this pairing; below that an allocation failure surfaces as "
                  f"'NVML_SUCCESS == r INTERNAL ASSERT FAILED', which means OUT OF MEMORY (§4.3).",
                  flush=True)

    # ---------------------------------------------------------------- per request
    def _decode(self, video_bytes, bbox_norm, max_frames):
        """Video bytes -> (list of cropped BGR frames, src_fps, crop_xywh, source_wh)."""
        cv2 = self.cv2
        from openasl_fetch import crop_xywh
        tmp = os.path.join(self.args.tmpdir, f"upload_{threading.get_ident()}.bin")
        with open(tmp, "wb") as f:
            f.write(video_bytes)
        try:
            cap = cv2.VideoCapture(tmp)
            if not cap.isOpened():
                raise ValueError("could not open the uploaded video (unsupported container/codec?)")
            src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            frames, W, H = [], None, None
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if W is None:
                    H, W = img.shape[:2]
                frames.append(img)
                if max_frames and len(frames) >= max_frames:
                    break
            cap.release()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if not frames:
            raise ValueError("the uploaded video decoded to zero frames")
        # The box is normalised, so it is resolution-independent -- the phone may send any size.
        cx, cy, cw, ch = crop_xywh(bbox_norm, W, H)
        frames = [f[cy:cy + ch, cx:cx + cw] for f in frames]
        return frames, (src_fps if src_fps > 1e-3 else None), [cx, cy, cw, ch], [W, H]

    def _square_norm_fn(self, bbox_norm, crop, source_wh):
        """Same mapping as unisign.e2e_translate.square_norm_fn, but from a caller-supplied box
        rather than a clip's meta.json. Crop pixels -> the authors' square frame (§2.5d), which is
        the coordinate frame the frozen encoder was trained on."""
        from openasl_fetch import openasl_square_crop
        W, H = source_wh
        cx, cy = crop[0], crop[1]
        _, g = openasl_square_crop(bbox_norm, W, H)
        sx, sy, side = g["square_xywh"][0], g["square_xywh"][1], g["square_xywh"][2]

        def f(k):
            out = np.empty_like(k, dtype=np.float32)
            out[..., 0] = (cx + k[..., 0] - sx) / side
            out[..., 1] = (cy + k[..., 1] - sy) / side
            return out
        return f

    def translate(self, video_bytes, bbox_norm, keep_fps, num_beams, max_new_tokens):
        from rtmpose_utils import postprocess, preprocess
        from common.pose_to_unisign import collate, to_model_inputs
        from common.subsample_pkl import keep_idx

        t_req = time.perf_counter()
        frames, src_fps, crop, source_wh = self._decode(video_bytes, bbox_norm,
                                                        self.args.max_frames)
        n_decoded = len(frames)

        # Thin to the operating rate using the video's OWN fps. A hardcoded source rate is the bug
        # that invalidated the first accuracy-vs-rate curve on both tracks (§2.9B / their L7.1).
        # With no fps in the container we do not guess: we keep every frame and say so.
        fps_note = None
        if keep_fps and src_fps:
            idx = keep_idx(len(frames), keep_fps, src_fps)
            frames = [frames[i] for i in idx]
        elif keep_fps and not src_fps:
            fps_note = ("no frame rate in the container, so no thinning was applied -- the clip ran "
                        "at its native rate rather than at the requested one")

        sqf = self._square_norm_fn(bbox_norm, crop, source_wh)

        t_pose0 = time.perf_counter()
        kps, scs, dropped = [], [], []
        for i, img in enumerate(frames):
            x, center, scale = preprocess(img, None, self.pp)
            outs = self.runner.infer({self.in_name: x})
            k, s = postprocess(outs["simcc_x"].cpu().numpy(),
                               outs["simcc_y"].cpu().numpy(), center, scale, self.pp)
            if not (np.isfinite(k).all() and np.isfinite(s).all()):
                dropped.append(i)          # §L28: no person in the crop. Never translate these.
                continue
            kps.append(sqf(k).reshape(1, 133, 2))
            scs.append(np.asarray(s, dtype=np.float32).reshape(1, 133))
        pose_ms = (time.perf_counter() - t_pose0) * 1e3

        if not kps:
            raise ValueError(f"every one of {len(frames)} frames failed the finite check -- there is "
                             f"probably no person inside the box that was sent")

        t_c0 = time.perf_counter()
        inputs, idx = to_model_inputs(kps, scs, self.args.max_length)
        src = collate([inputs], ["live"])
        convert_ms = (time.perf_counter() - t_c0) * 1e3

        r = self.model.translate(src, max_new_tokens=max_new_tokens, num_beams=num_beams)

        out = {
            "text": r["text"],
            "frames_decoded": n_decoded,
            "frames_after_thinning": len(frames),
            "frames_dropped_nonfinite": len(dropped),
            "dropped_frame_indices": dropped[:32],
            "frames_used_by_model": len(idx),
            "src_fps": round(src_fps, 3) if src_fps else None,
            "keep_fps": keep_fps,
            "crop_xywh": crop,
            "source_wh": source_wh,
            "timing_ms": {
                "pose": round(pose_ms, 1),
                "convert": round(convert_ms, 1),
                "lm": {k: round(v, 1) for k, v in r["timing_ms"].items()},
                "request_total": round((time.perf_counter() - t_req) * 1e3, 1),
            },
            "memfree_mb": _memfree_mb(),
        }
        if fps_note:
            out["note"] = fps_note
        if dropped:
            out["dropped_note"] = (f"{len(dropped)} frame(s) produced a non-finite pose and were "
                                   f"dropped rather than translated (§L28).")
        return out


class Handler(BaseHTTPRequestHandler):
    pipeline = None
    server_version = "SLTedge-demo/1.0"

    def log_message(self, fmt, *a):
        print(f"[demo] {self.address_string()} {fmt % a}", flush=True)

    def _json(self, code, payload):
        body = json.dumps(payload, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        p = self.pipeline
        if path == "/health":
            self._json(200, {"ok": True, "served": p.n_served,
                             "load_pose_s": round(p.load_pose_s, 1),
                             "load_lm_s": round(p.load_lm_s, 1),
                             "warm_lm_s": None if p.warm_lm_s is None else round(p.warm_lm_s, 1),
                             "memfree_mb_now": _memfree_mb(),
                             "memfree_mb_at_load": p.memfree_mb,
                             "busy": p.lock.locked()})
        elif path == "/stats":
            self._json(200, p.last or {"note": "nothing translated yet"})
        else:
            self._json(404, {"error": "GET /health or GET /stats"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/translate":
            return self._json(404, {"error": "POST /translate"})
        q = parse_qs(u.query)

        def num(name, default):
            try:
                return float(q[name][0])
            except (KeyError, ValueError, IndexError):
                return default

        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json(400, {"error": "bad Content-Length"})
        if n <= 0:
            return self._json(400, {"error": "empty body; send the video as the raw request body"})
        if n > MAX_UPLOAD_BYTES:
            return self._json(413, {"error": f"{n} bytes exceeds the {MAX_UPLOAD_BYTES} cap"})

        # Default box is the whole frame, which is what 03_infer_frames.py does when given no bbox.
        bbox = [num("x0", 0.0), num("y0", 0.0), num("x1", 1.0), num("y1", 1.0)]
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            return self._json(400, {"error": f"degenerate box {bbox}; expected x0<x1 and y0<y1 "
                                             f"as fractions of the frame"})
        keep_fps = num("fps", self.pipeline.args.keep_fps)
        beams = int(num("beams", self.pipeline.args.num_beams))
        max_new = int(num("max_new_tokens", self.pipeline.args.max_new_tokens))

        body = self.rfile.read(n)
        p = self.pipeline
        if p.lock.locked():
            self.log_message("queued behind a translation in flight")
        with p.lock:
            try:
                out = p.translate(body, bbox, keep_fps, beams, max_new)
                p.n_served += 1
                p.last = out
                self.log_message("-> %r (%.0f ms)", out["text"], out["timing_ms"]["request_total"])
                self._json(200, out)
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:                                    # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="models/rtmw/rtmw-l-m_256x192_fp16.engine")
    ap.add_argument("--preproc", default=None)
    ap.add_argument("--ckpt", default="weights/openasl_pose_only_slt_pruned.pth")
    ap.add_argument("--mt5", default="weights/mt5-base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="fp16 is NOT usable for the LM: §3.5 / L8.2 measured it returning uniform "
                         "logits at -ln(26078). Left configurable only so that stays reproducible.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--keep-fps", type=float, default=24.0,
                    help="operating point from §3.4: free on both metrics against 30 for 21 %% less "
                         "pose energy")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=900,
                    help="hard cap on decoded frames per request, so one long upload cannot stall "
                         "the demo; 900 is ~37 s at 24 fps")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--tmpdir", default="/tmp")
    args = ap.parse_args()

    os.chdir(ROOT)
    Handler.pipeline = Pipeline(args)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[demo] listening on {args.host}:{args.port} -- POST /translate, GET /health", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] shutting down", flush=True)
        srv.shutdown()


if __name__ == "__main__":
    main()
