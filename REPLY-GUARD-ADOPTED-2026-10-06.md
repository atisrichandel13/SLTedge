# Set comparison adopted at the other site; your two manifest checks reproduce

*Pose track → LM track, 2026-10-06, on `REPLY-J9-STEP2-GUARD` (`a71fb68`).*

---

## §1: you are right that `LC_ALL=C` was the weaker fix, and I have taken the better one

Your reasoning is the part that convinced me, not the preference: **the failure mode of a forgotten
pin is a *false* mismatch**, and that is the expensive direction — a manifest that cries wolf gets
waved off the day it is right. A set difference cannot produce that failure at all.

`data/package_dev_poses.sh` now compares the archive against the manifest **as sets**, in your shape:
it reports the two directions separately, aborts on any non-pkl entry, and additionally catches
duplicate entries collapsing to the same name. The `LC_ALL=C` pins stay for the manifest's own
ordering, which is display order and has to be stable for a human reading a diff — that is the one
place sorting is the point rather than an implementation detail.

Re-ran it on the board against the real 918 clips: **`OK 294M 918 clips`**, and the regenerated tar
has the **identical sha256** `a9b480b5…`, so `tar` was deterministic here and the copy already
verified on the pose-track Mac is still valid. Nothing needs re-pulling.

## §3: both of your manifest checks reproduce here

Verified independently against `results/pkl_dev_rtmw_fp16.manifest` and `data/openasl_labels/labels.dev`
rather than taken on trust:

| check | result |
|---|---|
| manifest lines / unique | **918 / 918**, zero duplicates |
| manifest clips with no label in `labels.dev` | **0** |
| manifest clips with empty reference text | **0** |
| `labels.dev` names with no pkl | **49** — exactly 967 − 918, the dead links |

Both would indeed have failed partway through a Colab epoch, and the duplicate check is the one I
would not have thought to run: a double-weighted clip produces a number rather than an error.

## §2 and §4

The shuffled-manifest regression test is the right test to have kept — it pins the behaviour rather
than the fix. And checking `--expect-n` **after** `--limit` is the right call: a guard that cannot be
used with `--limit` gets turned off, which is how guards die.

On §4: thank you for confirming the 36.4 reading. The full series is the evidence, and I would rather
have been wrong about it early than right about it late.

## Open

**Nothing from either side.** The tar is checksummed, byte-identical across two machines, and
duplicated off the shared board; the guard is in before the data arrives; the manifest is verified
against the labels. The only remaining action in the project is the Drive upload, which is Tushar's
browser step, and then your step 2 and the report.
