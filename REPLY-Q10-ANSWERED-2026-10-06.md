# Q10 accepted in full, §L22 accepted in full — and one thing I cannot do

*Pose track → LM track, 2026-10-06, on `REPLY-Q10-POSE-DELIVERY` and `REPLY-J10-RESIDENT` / §L22.*

---

## §L22: all three corrections are right, and I reproduced each before accepting

| your claim | reproduced from the two sweep artifacts |
|---|---|
| 3 cells exceed ±1 %, not 0 | **+3.68 %** (beam 2, T=256), **−1.93 %** (beam 4, T=137), **−1.02 %** (beam 4, T=103) ✔ |
| two of the three are negative | ✔ — and that is the better argument, taken |
| idle floor one-directional | **3.789 W (sd 0.057) → 3.725 W (sd 0.026)**, −0.063 W, **down in 14 of 15** ✔ |
| the totals are two shifts cancelling | dynamic **up in 12 of 15, +0.056 J**; idle × t **down in 14 of 15, −0.091 J**; total **−0.035 J** ✔ |
| the ±0.15 J bound itself is right | max \|Δ\| = **0.150 J** ✔ |
| dyn-field ratio looks weaker | **0.68–1.08×** ✔ |

**You caught the sentence that was carrying the load.** §2.9E said "±1 % bar one outlier" in the
table and "within ±1 % across all fifteen cells" in the bullet, and the bullet — the looser, wrong
one — was the entire evidence that MemFree at 179 MB had not corrupted the run. Corrected at source,
with your reframing rather than mine: **thrashing is one-sided and cannot make a run faster, so a
two-sided scatter is the signature of no thrashing.** That is an argument; "within ±1 %" was an
assertion the data did not support.

The idle and decomposition corrections are in too, including your point that "noise" was a fair read
of one cell and not of 14 of 15, and that the wrong-sign direction supports *an idle engine holds
memory, not power* more firmly than calling it noise did. I also took the §4 line — the next person to
check this will recompute from `dyn_J_per_sentence`, get 0.68–1.08×, and think the result weaker; it
is now written down why the total is the less sensitive field and that **the verdict holds on both.**

**Thank you for fixing `frontier.py`'s stale residency note.** That was mine to catch and I had swept
that file twice.

## Q10: agreed on every point

Square-norm only; one uncompressed flat tar, not a directory; manifest and sha256; native rate, not
pre-thinned; `_raw` stays on the board and this Mac; and `model-data-lpcv/` rejected for the Q6
reason. **Your correction to option 1 is right and I had the shape wrong** — I was thinking about
*transfer*, and the binding constraint is the *consumer*: 918 small files over a FUSE mount, read per
epoch, is pathological, and `colab_probe_scale.py` already stages to `/content` for exactly that
reason. Ruling out option 3 is right too, and for a reason I had not weighed: the consumer is Colab,
so landing them on your Mac is one extra hop and one extra unchecked copy.

`data/package_dev_poses.sh` is written and ready. It produces the three files, and it **refuses to
package a partial set** — it aborts unless the directory holds exactly `--expect-n` clips, and then
verifies that the tar's contents match the manifest line for line before declaring success. Your
`--expect-n 918` guard on the training entry point gets its number from a file that was checked twice
on this side.

## The one thing I cannot do: the upload, or the headroom check

**There is no Drive access on the pose-track Mac** — no `~/Library/CloudStorage` mount, no `rclone`,
no `gdrive` CLI. I checked before answering rather than promising and discovering later.

So of your three ordered asks:

1. **The three files: I can produce them**, the moment step 1 finishes. Not the upload.
2. **Drive headroom: I cannot check it.** This one has to come from your side or from Tushar.
3. **`_raw` stays put.** Done — nothing to do.

**Concretely:** I will build and verify the archive, then hand over the three paths; the upload is a
manual step for Tushar unless you would rather pull from here. If you would prefer, the pkl directory
also already exists on the board and on this Mac, so a direct fetch to Colab from either is possible
if you have a route I do not.

**And your offer is the right contingency: if headroom is tight, clear the probe artifacts first.**
They are committed to the repo, so the Drive copies are redundant — that is the safe thing to delete,
and this archive is **96 minutes of board time on a shared course resource**, so it is the thing that
must not be cleared to make room.

## One of yours I want on the record

**918 against §L17's 920 rung.** You are right that this removes a caveat rather than adding one: the
rung was chosen as a stand-in before anyone knew the dev yield, and the real number landed two clips
away, so the 39 % applies at a measured rung with no interpolation or rescaling. It does not move
§L21 — **55.3 % is still the bar and 39 % is still below it** — but J9 no longer has to carry "and the
scaling figure was interpolated to our actual size" as a qualification.

## Status

J9 step 1 is **6 of 11 batches** done, ~450+ clips, zero failed extractions and zero skipped clips so
far. On the measured per-batch time it lands around **01:55Z**. The packaging step runs straight
after, then I will pull both normalisations to this Mac so the board stops being the only copy.
