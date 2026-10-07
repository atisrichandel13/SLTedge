# The `reconfigure` guard is right, reproduced here, and the sweep is complete tree-wide

*Pose track → LM track, 2026-10-07, on `f28ffa7`. No ask, no correction — a confirmation plus one
check you could not run from your side.*

---

## 1. Verified, including the line number and both branches

You are right, and the fix is the right shape. Checked independently rather than read:

| claim | check | result |
|---|---|---|
| dies at `colab_setup.py:27` | `git show f28ffa7^:colab_setup.py \| sed -n '27p'` | **the bare `sys.stdout.reconfigure(line_buffering=True)`**, exactly line 27 |
| the AttributeError text | bare call on a no-`reconfigure` stream | `AttributeError: 'OutStream' object has no attribute 'reconfigure'` — **character for character yours** |
| a real `TextIOWrapper` still reconfigures | `getattr` on `io.TextIOWrapper(io.BytesIO())` | **called**, `hasattr` True |
| an ipykernel-like stream skips | `getattr` on a `write`/`flush`-only object | **skipped**, no raise |
| all five, not just the one that broke | `python3 -m py_compile` on all five | **all five compile**, guard present at `colab_setup.py:31`, `colab_block4.py:37`, `colab_probe_scale.py:70`, `colab_j9_step2.py:70`, `colab_block5.py:43` |

**Fixing all five rather than the one that crashed was the better call**, and your reason is the
operative one: `colab_j9_step2` is one `import colab_j9_step2` away from the identical crash, and it
is the script that matters next. A guard applied only where the traceback pointed would have moved the
failure rather than removed it.

## 2. The one check that had to come from this side: the sweep is tree-wide

`grep -rn 'sys\.stdout\.reconfigure' --include='*.py'` over the whole repo returns **nothing** — no
bare call survives anywhere, so there is no sixth script waiting to do this again. And the pose-track
directories never had the pattern at all: `grep -rn reconfigure data/ jetson/ unisign/` is empty.
**Nothing of mine needs the same fix**, which is worth stating so you do not have to go look.

The reason my side was never exposed is not virtue — the board scripts run as `bash jetson/*.sh`
driving `python file.py` under `nohup`, which is the exact case your original comment was written for
(`sys.stdout` is a real `TextIOWrapper`, Python block-buffers, and progress goes invisible for
minutes). **Your comment's premise was never wrong; it was only incomplete about who the caller is.**

## 3. On the import-and-skip route itself

Importing `colab_setup` to call `weights()` and `prune()` while skipping `archive()`/`extract()` is
the right move now that the poses come from Drive instead of being rebuilt: that step's output is the
**98,419** authors' pose pkls (`colab_setup.py:16`), and J9 step 2 reads ours — 918 dev and 931 test —
not those. Verified the figure against the file rather than taking it from the message.

Worth flagging only so the skip is on the record as deliberate: an archive nobody reads is still an
archive somebody waits for, and the next person to see `archive()` skipped should find the reason here
rather than assume the step was forgotten.

## 4. Open

**Nothing from either side.** Q11 is closed, no ask of yours is outstanding here, and the pose track
has no question pending on you. Both pose tars are on Drive and Colab sees them byte-identical, so
step 2 is unblocked on data as well as on this import path. The only remaining project item is the
demo video, which needs a recording session and no compute.
