# Block 6 adapted weights: which file is which

Four adaptation runs from Colab block 6 (2026-11-02). Each `last.pt` is **not** a full model: it is
the 343 trainable tensors (pose stack, `pose_proj`, `part_para`; 6.89 M params) plus optimizer state.
Every run finished its one epoch (5000/5000 batches, `epoch_done: True`). Checksums in `SHA256SUMS`.

| folder | what it is | trained at | seed | dev BLEU-4 / ROUGE-L (n=967, beam 4) |
|---|---|---|---|---|
| **`fps24_s42/`** | **the 24 fps adapted model: use this one** | 24 fps | 42 | **23.71 / 43.74** |
| `fps24_s43/` | same recipe, second seed (variance check only) | 24 fps | 43 | 23.56 / 43.70 |
| `fps24_s44/` | same recipe, third seed (variance check only) | 24 fps | 44 | 23.38 / 43.24 |
| `ctrl_src_s42/` | control: adapted at the source frame rate, not 24 fps | source | 42 | 23.28 / 43.58 |

For comparison, un-adapted on the same 967 dev clips: 23.38 / 42.72 at 24 fps, 23.13 / 42.93 at
source rate. All scores: `results/block6/eval_dev_*_cap100.json`. Every figure here is **dev split**;
no adapted model has been scored on the 976-clip test split.

## Turning `last.pt` into a loadable model

```bash
python3 -m unisign.rebuild_adapted \
    --base weights/openasl_pose_only_slt_pruned_traindev.pth --mt5 weights/mt5-base \
    --last weights/block6/fps24_s42/last.pt --out weights/block6/fps24_s42_adapted_full.pth
```

Writes the 570 MB `adapted_full.pth`, the same file `train_adapt --full-out` writes, loadable with
`--ckpt` anywhere (`unisign/eval_openasl.py`, `jetson/demo_server.py`). Needs no training data. Checked
bit-identical against `runs/c8_adapt16/adapted_full.pth` rebuilt from its own `last.pt`.

**The base must be `openasl_pose_only_slt_pruned_traindev.pth`** (26,025-token vocabulary, train+dev
census), not the board's `openasl_pose_only_slt_pruned.pth` (26,078 tokens, built over train+dev+test).
The script reads the base name from `last.pt` and refuses the wrong one. The base is 570 MB and is
not on GitHub; sha256 `1818ff6b704e9e895522b4039e4e5bfb32b7337d4bd9cc1be134c187f28787e5`.

Training logs per run: `results/block6/runs/<run>/{log.jsonl,train.log}`.
