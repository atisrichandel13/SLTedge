# SLTEdge: sign language translation entirely on the iPhone

Everything runs on the phone (iPhone 12 Pro, A14). No server, no network.

```
camera 720p -> Vision person box (frozen at press) -> crop_xywh -> warpAffine 192x256
  -> RTMW-l-m (ORT, CoreML EP) -> SimCC decode -> keep_idx thinning to 24 fps -> L28 finite guard
  -> square-norm + load_part_kp -> front.onnx (ST-GCN) -> encoder.onnx -> decoder_init/step (greedy)
  -> pruned-vocab detokenise -> English
```

- **Pose:** the byte-identical `end2end.onnx` the board's TensorRT engine was built from
  (sha256 `68db326c...`), through ONNX Runtime's CoreML execution provider.
- **Translator:** the pruned Uni-Sign pose-only model as four ONNX graphs, FP32 on ORT's CPU
  provider. FP32 because mT5 in FP16 saturates (RESULTS.md L8), which rules out the Neural Engine.
  The Python ONNX chain is token-identical to PyTorch greedy on the reference clip.

## Build and run

```bash
brew install xcodegen
cd ios && xcodegen generate
xcodebuild -project SLTEdge.xcodeproj -scheme SLTEdge -destination 'id=<udid>' -allowProvisioningUpdates build
xcrun devicectl device install app --device <udid> <DerivedData>/Build/Products/Debug-iphoneos/SLTEdge.app
```

Model files (git-ignored, ~1.6 GB):

| file | from |
|---|---|
| `models/rtmw/rtmw-l-m_256x192.onnx` | rtmlib cache or the OpenMMLab SDK zip; bundled at build |
| `models/mt5_pruned_onnx/{encoder,decoder_init,decoder_step}.onnx` | `task3_mt5_onnx/02_export_manual.py` |
| `models/mt5_pruned_onnx/front.onnx` | `task3_mt5_onnx/04_export_front.py --verify` |

The four translator graphs are pushed into the app's Documents once, not bundled:

```bash
for f in front encoder decoder_init decoder_step; do
  xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
    --domain-identifier com.atisri.sltedge --source models/mt5_pruned_onnx/$f.onnx --destination Documents/models/$f.onnx
done
```

## Files

| file | port of |
|---|---|
| `RTMWPreprocess.swift` | `rtmpose_utils.py` preprocess, `openasl_fetch.py:crop_xywh` |
| `RTMWPose.swift` | ORT session; `rtmpose_utils.py` get_simcc_maximum + postprocess |
| `UniSignInputs.swift` | `demo_server._square_norm_fn`, `pose_to_unisign` subsample + `load_part_kp` |
| `Translator.swift` | `PoseOnlyUniSign.translate` (greedy), KV-cache decode loop, SentencePiece detokenise |
| `SentenceRecorder.swift` | hold-to-record; `subsample_pkl.keep_idx` |
| `ParityTest.swift` | 299 reference frames -> `03_infer_frames.py` layout + on-device translation check |
| `Metrics.swift` | per-sentence JSONL: pose and translator stage timings, thermal, battery |

## Parity gate

"parity test" runs the board's 299-frame reference clip, writes `Documents/parity_<compute>.json`,
then translates that clip on the phone and checks the sentence against the Mac/PyTorch greedy result.

```bash
xcrun devicectl device copy from --device <udid> --domain-type appDataContainer \
    --domain-identifier com.atisri.sltedge --source Documents/parity_all.json --destination results/mobile/phone_all.json
python3 task1_rtmpose/07_kpt_agreement.py --ref results/rtmw_trt_fp32.json --test results/mobile/phone_all.json
```
