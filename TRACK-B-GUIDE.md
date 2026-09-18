# Track B guide — Uni-Sign on Colab: reproduce, convert, prune, export

Companion to `WORKSPLIT.md` §2. Everything here runs on Colab (T4 is enough for eval; L4/A100 only
for fine-tuning later). Nothing touches the Jetson. Work in a Colab notebook saved to Drive; keep the
big files on Drive under `MyDrive/unisign/` so a session reset costs minutes, not hours.

Facts pulled from the Uni-Sign repo (`github.com/ZechengLi19/Uni-Sign`, read 2026-09-16):

| Item | Fact |
|---|---|
| OpenASL pose-only checkpoint | `openasl_pose_only_slt.pth`, 1.19 GB, on `huggingface.co/ZechengLi19/Uni-Sign` |
| Language model | mT5-base from `google/mt5-base`, expected at `./pretrained_weight/mt5-base` |
| Labels | `data/OpenASL/labels.{train,dev,test}` are gzip-pickled dicts, `{name.mp4: {text, gloss, video_path}}`, 976 test entries. Our Bitcoin clip `Ads-4j06eJY-00:07:37.233-00:07:47.200.mp4` **is in labels.test** |
| Released poses | `openasl_pose_format.zip.00..07`, ~30 GB total, one pkl per clip: `{"keypoints": [T x (1,133,2)], "scores": [T x (1,133)]}`, xy **normalised by frame [W,H]** |
| Pose extractor in the demo | `rtmlib.Wholebody(mode="lightweight")` = YOLOX-tiny detector + **RTMW-dw-l-m 256x192** (cocktail14). `performance` = YOLOX-m + RTMW-dw-x-l 384x288. **Not** RTMPose-x COCO-WholeBody. Same 133-kpt order. Which one produced the released dataset poses is unverified |
| Keypoint groups (`datasets.py: load_part_kp`) | body `[0,3..10]` (9), left hand `91:112` minus wrist, right hand `112:133` minus wrist, face `[23,25,..,39] + [83..90] + [53]` (18) minus index 53. Body goes through `crop_scale`; the other groups divide by the body scale, clip to [-1,1], zero out conf ≤ 0.3 |
| Eval entry point | `fine_tuning.py --eval --dataset OpenASL --task SLT --finetune <ckpt>` via deepspeed, single GPU |
| `config.py` bug | `pose_dirs["OpenASL"]` points at the WLASL path. Fix to `./dataset/OpenASL/pose_format` |
| Authors' own predictions | `out/eval_openasl_pose_only/test_tmp_{pres,refs}.txt`, usable to check the metric code before running the model |

---

## B1 — reproduce OpenASL pose-only BLEU-4 ≈ 22.67

### B1.1 Colab session setup (every session)

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content
!git clone -q https://github.com/ZechengLi19/Uni-Sign.git
%cd /content/Uni-Sign
# pinned libs the repo needs; skip its requirements.txt (old torch/tensorflow pins fight Colab)
!pip install -q transformers==4.40.0 tokenizers==0.19.1 sentencepiece==0.1.99 deepspeed==0.16.3 \
    sacrebleu==2.2.0 rouge==1.0.1 einops timm==0.9.16 huggingface_hub
!sed -i 's#"OpenASL": "./dataset/WLASL/pose_format"#"OpenASL": "./dataset/OpenASL/pose_format"#' config.py
!grep -n OpenASL config.py
```

### B1.2 One-time downloads to Drive (checkpoint + mT5)

```python
import os; D='/content/drive/MyDrive/unisign'; os.makedirs(D, exist_ok=True)
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download('ZechengLi19/Uni-Sign', 'openasl_pose_only_slt.pth', local_dir=D)          # 1.19 GB
snapshot_download('google/mt5-base', local_dir=f'{D}/mt5-base',
                  allow_patterns=['*.json','*.model','pytorch_model.bin','spiece.model'])   # ~2.3 GB
```

Then link into the repo each session:

```python
!mkdir -p pretrained_weight out/stage3_finetuning dataset/OpenASL
!ln -sfn {D}/mt5-base pretrained_weight/mt5-base
!ln -sfn {D}/openasl_pose_only_slt.pth out/stage3_finetuning/best_checkpoint.pth
```

### B1.3 Metric sanity check, no model needed (5 min)

Score the authors' own predictions with the repo's metric code. This must print ≈ 22.67 BLEU-4. If it
does not, the metric setup is wrong and nothing downstream can be trusted.

```python
from SLRT_metrics import translation_performance
refs = open('out/eval_openasl_pose_only/test_tmp_refs.txt').read().splitlines()
pres = open('out/eval_openasl_pose_only/test_tmp_pres.txt').read().splitlines()
print(len(refs), len(pres)); print(translation_performance(refs, pres))
```

### B1.4 Test-split poses only (one-time, ~45 min on Colab, needs ~35 GB of scratch disk)

The release is one 30 GB zip in 8 parts. Download to Colab local disk (not Drive), join, extract only
the 976 test pkls, copy those to Drive, delete the rest.

```python
import gzip, pickle, subprocess
test = pickle.load(gzip.open('data/OpenASL/labels.test','rb'))
names = [k.replace('.mp4','.pkl') for k in test]            # 976
open('/content/test_pkls.txt','w').write('\n'.join(names))
%cd /content
for i in range(8):
    hf_hub_download('ZechengLi19/Uni-Sign', f'openasl_pose_format.zip.{i:02d}', local_dir='/content/zips')
!cat /content/zips/openasl_pose_format.zip.0* > /content/openasl_pose_format.zip
!unzip -l /content/openasl_pose_format.zip | head -20      # learn the path prefix inside the zip
```

Look at the prefix printed (e.g. `pose_format/` or `openasl_pose_format/`), then:

```python
PREFIX = 'pose_format/'   # <-- set from the listing above
open('/content/test_pkls_full.txt','w').write('\n'.join(PREFIX+n for n in names))
!cd /content && unzip -q openasl_pose_format.zip $(cat /content/test_pkls_full.txt | tr '\n' ' ') -d /content/extract 2>&1 | tail -3
!mkdir -p {D}/openasl_test_pose && cp /content/extract/{PREFIX}*.pkl {D}/openasl_test_pose/
!ls {D}/openasl_test_pose | wc -l                            # expect 976 (fewer if some clips were dropped by the authors; record the number)
!rm -rf /content/zips /content/openasl_pose_format.zip /content/extract
```

If `unzip` rejects the long argument list, loop over `names` in chunks of 200.

Each session after this: `!ln -sfn {D}/openasl_test_pose dataset/OpenASL/pose_format`.

### B1.5 Run the eval

`fine_tuning.py` reads train labels even in `--eval`; if it complains about the missing train poses,
pass a tiny fake train set (edit `config.py` train label path to a 2-entry pickle) — note what you did.

```bash
!deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
   --batch-size 8 --gradient-accumulation-steps 1 --epochs 1 --opt AdamW --lr 3e-4 \
   --output_dir out/eval_openasl_ours --finetune out/stage3_finetuning/best_checkpoint.pth \
   --dataset OpenASL --task SLT --eval
```

Record: BLEU-1/4, ROUGE-L, number of test clips actually evaluated, wall time, GPU type. Target
BLEU-4 **22.67**; anything within ±0.3 is a reproduction (their `load_pose` random-samples frames
above `max_length`, so tiny drift is expected). Save `out/eval_openasl_ours/test_tmp_pres.txt` to Drive.

**Gate:** B1 is done when this number is in `results/RESULTS.md` with the run details.

---

## B2 — pose converter + which extractor made the dataset poses

### B2.1 Which extractor? (the definitive test)

Atisri commits `results/keypoints_fp32_Ads-4j06eJY.json` (Jetson RTMPose-x output for the 299 cropped
frames, pixel coords in the 666x720 crop). The authors' pkl for the same clip is
`{D}/openasl_test_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl` (normalised coords, T frames).

Compare, per frame, after normalising ours by `[666, 720]`:
- frame count T (theirs vs 299 — tells you their fps / trimming),
- mean px distance for hands (91:133) and face subset, on frames where both have conf > 0.3,
- score histograms.

Then run rtmlib `Wholebody(mode="lightweight")` **and** `mode="performance"` on the same 299 frames
(rtmlib is vendored at `demo/rtmlib-main`, `pip install -e .`; the ONNX files download automatically)
and compute the same distances. Whichever of {RTMPose-x, RTMW-dw-l-m, RTMW-dw-x-l} is closest to the
released pkl is the extractor the checkpoint was trained on. Write the answer in `RESULTS.md`.

Why it matters: if it is RTMW-lightweight 256x192, the Jetson "baseline" pose model changes (and gets
cheaper), and the RTMPose-x numbers become the "bigger than needed" row.

### B2.2 `common/pose_to_unisign.py`

Pure numpy + torch, no repo imports. Input: our keypoints JSON (schema in `WORKSPLIT.md` I1) plus crop
size; output: the same dict `load_part_kp` returns (`body`, `left`, `right`, `face_all` tensors) and
optionally the raw pkl format so Uni-Sign can read it unchanged. Lift `load_part_kp` and `crop_scale`
verbatim from `datasets.py` (lines 14–105) and cite them. Unit test: our JSON → pkl → their
`load_pose` gives identical tensors to feeding their pkl directly.

### B2.3 Round-trip on Colab

Feed the Jetson keypoints (converted) through the released checkpoint with `generate()` (see
`models.py:313`, greedy, `max_new_tokens` 64). Expected: something about tweets / donations / Bitcoin.
Also feed the authors' pkl for the same clip. Report both sentences side by side. This is the first
proof that the Jetson pose stage feeds the language model correctly.

---

## B3 — `unisign/unisign_infer.py` (what the Jetson runs)

Plain PyTorch: `--keypoints in.json --crop-wh 666 720 --ckpt openasl_pose_only_slt.pth --mt5 <dir>
--out text.json [--max-new-tokens 64] [--num-beams 1] [--device cuda|cpu]`.

- Build the model from `models.py` (`Uni_Sign` class, pose-only path). Copy the class files you need
  into `unisign/` so the script has **no** dependency on `deformable_attention_2d.py`, deepspeed,
  decord, or the RGB branch. Confirm with `python -c "import unisign.unisign_infer"` in a clean env
  containing only torch, transformers 4.40, sentencepiece, numpy.
- Output JSON: `{"text", "tokens", "token_logprobs", "timing_ms": {"gcn", "encoder", "decoder"}}`.
- Test on Colab CPU once (the Jetson container is Python 3.12 / torch 2.8 — check
  `transformers==4.40.0` installs there; if not, find the newest version that still loads the ckpt).
- Hand-off: script + which transformers version + a sample output for the Bitcoin clip.

---

## B4 — vocabulary pruning

1. Tokenise all OpenASL train+dev+test text with the mT5 tokenizer; collect the used token ids
   (expect ~10–20K of 250,112). Add the special tokens and the prefix-prompt tokens `models.py:269` uses.
2. Slice `shared` embedding and `lm_head` to that id set, build an old→new id map, wrap the tokenizer
   so `decode()` maps back. Save `openasl_pose_only_slt_pruned.pth` + `vocab_map.json`.
3. Re-run B1.5 with the pruned model: BLEU before/after, params before/after, checkpoint MB
   before/after, peak GPU memory before/after. Expect near-zero BLEU change.
4. Hand-off: pruned ckpt + map on Drive, numbers in `RESULTS.md`.

## B5 — mT5 ONNX with KV cache

Use `task3_mt5_onnx/02_export_manual.py --verify` on the pruned model (encoder input = embeddings,
since the encoder consumes projected pose features, not token ids). Verify with ONNX Runtime vs
PyTorch logits over ≥12 decode steps at 3 encoder lengths. Ship `models/mt5_onnx_manual/` + reference
logits `.npz`. Three-day budget; if it does not land, say so and Track A uses the hybrid runtime.

## B6 — data tooling

`data/openasl_fetch.py`: `--split test --n 5 --distinct-signers` and `--split train --frames 300
--calib`. Reuse the yt-dlp section download + bbox crop from `SESSION-LOG.md` §3. Many videos are
private; iterate until N succeed. Outputs to Drive, mirroring `data/`.

---

## Order of work

B1.1 → B1.2 → B1.3 (metric check) → B1.4 → B1.5 (**gate**) → B2.1 → B2.2 → B2.3 (**M1 input**) → B3
(**M1**) → B6 → B4 → B5 → B7/B8.
