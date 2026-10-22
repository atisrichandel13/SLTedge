# Running the live demo

`jetson/demo_server.py` — one process on the board holding the pose engine and the language model
resident. Video in, English out. Measured and written up as **§L30**.

## Start it

Two steps, and **the first one is not optional** — skipping it is how the first attempt died with
`NVML_SUCCESS == r INTERNAL ASSERT FAILED`, which is §4.3's signature for out of memory:

```bash
# 1. Reclaim page cache. J10 loaded this pairing from MemFree 5,130; a reclaim to 3,500 is NOT enough.
python3 jetson/drop_file_cache.py --target-free-mb=5400

# 2. Launch, detached, inside the container.
docker exec -d -w /workspace slt-work \
    sh -c "python3 jetson/demo_server.py --port 8765 > /tmp/demo_server.log 2>&1"

# 3. Wait ~55 s (LM load ~49 s + LM warm-up ~6 s), then:
docker exec slt-work grep listening /tmp/demo_server.log
```

It binds `0.0.0.0:8765` on the **host**, because `jetson/run.sh:33` runs `--network host` — no port
mapping, no sudo. Verified reachable from a WireGuard peer: source `10.66.0.9` → `192.168.1.73:8765`.

## Use it

```bash
curl -s http://192.168.1.73:8765/health

curl -s --data-binary @clip.mp4 \
  'http://192.168.1.73:8765/translate?x0=0.245&y0=-0.268&x1=0.773&y1=1.345'
```

The body is the **raw video file**. The four query parameters are the person box as **fractions of
the frame**; they may fall outside 0–1 (OpenASL's boxes routinely do) and are clamped. Omit them and
the whole frame is used. Other knobs: `fps` (default 24), `beams` (default 4),
`max_new_tokens` (default 64).

`GET /health` reports load times and the MemFree ladder; `GET /stats` returns the last response.

## Two things the phone side must get right

1. **Send full frames plus a box. Do not crop on the phone, and do not send OpenASL's 224 square.**
   Every pose number in `RESULTS.md` comes from a *native-resolution* bbox crop — the fetch indices
   record `crop_style=native`, "no square/pad/resize" — because RTMW does its own affine. Cropping
   to 224 first resamples twice and feeds the affine black padding, a path nothing we have measured
   covers. The server does the cropping.
2. **The phone's WireGuard `AllowedIPs` must include `192.168.1.0/24`.** Without it the phone
   resolves nothing on the lab LAN, and the failure looks exactly like a blocked port.

## What to expect

On a 4.5 s clip, thinned to 109 frames at 24 fps, beam 4 — **~4.5–4.9 s per sentence**, of which the
pose stage is ~58 %. Greedy (`beams=1`) takes it to ~4.0 s. Full numbers in §L30.

Frames whose pose comes back non-finite are **dropped and counted**, not translated (§L27/§L28) —
`frames_dropped_nonfinite` in every response. On live phone video this will fire whenever the signer
leaves the shot or a hand covers the lens, and it is the difference between a dropped frame and a
confident translation of nothing.

## Stop it

```bash
docker exec slt-work pkill -f demo_server.py
```

**Please do.** The board is shared scratch and the server holds ~4.7 GB resident; start-up is under a
minute, so run it for the demo rather than leaving it up.
