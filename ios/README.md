# SLTEdge: the phone half of the split pipeline

Pose on the phone, language model on the board. The phone runs RTMW-l-m 256x192 through ONNX
Runtime's CoreML execution provider, from the byte-identical `end2end.onnx` the board's TensorRT
engine was built from, and sends keypoints (not video) to `jetson/demo_server.py`.

```
iPhone 12 Pro (A14)                                   jetson-lpcv-03
  camera 720p -> Vision person box -> crop_xywh         /translate_kp
  -> warpAffine 192x256 -> ORT/CoreML -> SimCC decode   -> square-norm -> ST-GCN -> mT5 -> text
  -> keep_idx thinning -> L28 finite guard  --JSON-->
```

## Build

```bash
brew install xcodegen            # once
cd ios && xcodegen generate      # writes SLTEdge.xcodeproj (not committed)
open SLTEdge.xcodeproj           # first time: Signing & Capabilities -> pick your team
```

Or from the shell once a team is set in `project.yml`:

```bash
xcodebuild -project SLTEdge.xcodeproj -scheme SLTEdge -destination 'id=<device udid>' \
    -allowProvisioningUpdates build
```

The ONNX is read from `../models/rtmw/rtmw-l-m_256x192.onnx` (git-ignored; copy it from the rtmlib
cache or the OpenMMLab SDK zip, sha256 `68db326c...`).

## Files

| file | port of |
|---|---|
| `RTMWPreprocess.swift` | `rtmpose_utils.py` preprocess, `openasl_fetch.py:crop_xywh` |
| `RTMWPose.swift` | ORT session; `rtmpose_utils.py` get_simcc_maximum + postprocess |
| `SentenceRecorder.swift` | `demo_server.Pipeline.translate` up to the keypoints; `subsample_pkl.keep_idx` |
| `BoardClient.swift` | the keypoint payload and `/translate_kp` |
| `ParityTest.swift` | `03_infer_frames.py` output layout, for `07_kpt_agreement.py` |
| `Metrics.swift` | per-sentence JSONL, the phone's half of the C9 protocol |

## Parity gate

Tap "parity test". It runs the 299 frames of the board's reference clip and writes
`Documents/parity_<compute>.json`. Pull and score:

```bash
xcrun devicectl device copy from --device <udid> --domain-type appDataContainer \
    --domain-identifier com.atisri.sltedge --source Documents/parity_all.json --destination results/mobile/phone_all.json
python3 task1_rtmpose/07_kpt_agreement.py --ref results/rtmw_trt_fp32.json --test results/mobile/phone_all.json
```
