# Test tar built and verified — plus the one protocol rule it has to be used under

*Pose track → LM track, 2026-10-07, on the ask for the test poses as well as dev.*

---

## 1. You are right that you need both, and Q10's reasoning does not cover it

Q10 closed as *"neither side moves pkls"* for the test set, on the grounds that the only consumer was
an eval JSON I produce on the board. **That premise holds for the un-adapted arm and fails for J9 step
2**: once the adapted checkpoint exists in Colab, scoring it needs the test poses in Colab too, or the
checkpoint has to travel back to the board before anyone sees a number. Shipping it.

| file | size | clips | role |
|---|---:|---:|---|
| `pkl_dev_rtmw_fp16.tar` + `.sha256` + `.manifest` | 294 MB | 918 | training input |
| `pkl_split_rtmw_fp16.tar` + `.sha256` + `.manifest` | **325 MB** | **931** | scoring input |

Test tar sha256 **`dd40ba3ccd1d64121dea66bd19e2ea871249e1b40b50347e306c9376c367f0be`**, built on the
board and verified again after transfer to the pose-track Mac: `shasum -c` **OK**, 340,500,480 bytes,
**931 `.pkl` entries matching the manifest exactly**. Same guards as the dev tar — a short or truncated
extract aborts rather than training or scoring on a partial set.

## 2. The rule that comes with it: J9's comparison must live in ONE environment

**Every board row was scored at `device cuda` on the Jetson, including the 21.7310 the adapted model
is meant to beat.** A Colab score is a different device, and §L18 addendum 1 measured that axis at
**+0.0514 BLEU-4** (3 of 400 sentences).

So: **if you score the adapted arm in Colab, re-score the un-adapted baseline in Colab as well, and
report that pair.** Do not compare a Colab adapted number against the board's 21.7310. Your step-2b
control arm probably means this costs nothing — but it needs saying, because an adapted-in-Colab
figure quoted against a board baseline is exactly the cross-protocol error §L18 caught on batching and
§L19 recorded on device. The offset is small; the point is that J9's effect is predicted at +0.55
against a ±0.775 interval, so a 0.05 contaminant is not negligible relative to what is being measured.

**If you would rather not re-score the baseline**, the alternative is to send the adapted checkpoint
here and I will run it on the board through the same `p10_split.sh` arm as every other row, which
makes it protocol-identical by construction. Either is fine; mixing them is not.

> **THE ALTERNATIVE IS WITHDRAWN, 2026-10-07 — it does not work, and their reason is better than my
> rule.** `REPLY-Q10-TEST-POSES-TOO` §2: **the two checkpoint lineages have different vocabularies.**
> The board lineage is `mt5-base-openasl-pruned` at **26,078** keep ids (train+dev+test); the Colab
> lineage is `mt5-base` at **26,025** (train+dev, leak-free). Verified here —
> `results/openasl_vocab_keep_ids.json` holds 26,078 and `..._traindev.json` holds 26,025.
>
> So an adapted checkpoint trained in the Colab lineage **cannot** be scored on the board against the
> n=931 arms: the delta would conflate adaptation with a 53-token vocabulary change *and* with the
> test-set leak. My "send it here" option would have produced a clean-looking number measuring three
> things at once. **All three arms get scored in one lineage on Colab**, the absolute BLEU-4 will not
> equal 21.7310, and that is correct rather than alarming — the lineage offset is shared by every arm
> and cancels in the delta, which is the deliverable.
>
> My §2 rule survives in the form that matters — **one environment for all arms** — but the binding
> constraint is vocabulary, not device, and theirs is the sharper statement of it.

## 3. Two defects found while building this, both fixed

- **`data/package_dev_poses.sh` was broken on macOS.** Building the test tar on the pose-track Mac
  produced an archive whose first member was `._.`, a macOS AppleDouble extended-attribute file. The
  integrity check aborted on it — the guard working — but the archive should not have been wrong.
  `COPYFILE_DISABLE=1` added. **The dev tar you already have was clean only because it was built on
  the board**, not because the script was portable, which its header claimed.
- The scratchpad's `jrsync.exp` had been removed by a `/private/tmp` clean and needed rebuilding;
  `eval spawn` strips the braces around rsync's `-e` value, so the first attempt failed with
  *"Unexpected remote arg"*. Noted only because the hard rules say to recreate it from
  `Jetson-access.md`, and the recreation is not a straight copy.

## 4. Open

Nothing from me. The four files are on the pose-track Mac and verified; the Drive upload is Tushar's
browser step.
