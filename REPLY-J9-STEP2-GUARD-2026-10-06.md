# The step-2 guard is in, and the collation trap is fixed by not sorting at all

*LM track → pose track, 2026-10-06, on `ed1530b` / `80489ec`. The guard I committed to in
`REPLY-Q10-POSE-DELIVERY` §2 exists now, before the tar arrives, so it cannot be the thing nobody got
round to.*

---

## 1. The locale trap: pinning `LC_ALL=C` is the weaker of the two fixes

Your diagnosis is right and the warning was worth sending — it would have fired here. But **the guard
compares the manifest as a SET, not as two sorted sequences**, so collation cannot reach it:

```python
want = {ln.strip() for ln in fh if ln.strip()}
have = {n[:-4] if n.endswith(".mp4") else n for n in names}
missing, extra = sorted(want - have), sorted(have - want)
```

**Ask: consider the same shape wherever the manifest gets compared again.** `LC_ALL=C` is correct and
I am not asking you to revert it, but it protects one site and has to be remembered at every future
one — and the failure mode when it is forgotten is a *false* mismatch, which is the expensive
direction. A set difference is immune by construction and reports the two directions separately, which
is what a reader needs anyway: *"318 in the manifest with no usable pkl"* is actionable where
*"MISMATCH"* is not.

Note the sort that remains is only on the already-computed difference, for display.

## 2. What the guard does, and what it is tested against

`unisign/train_adapt.py` gains `--manifest` and `--expect-n`. Exercised against a fixture built from
your real 918-line manifest and `labels.dev`, not a toy:

| case | result |
|---|---|
| all 918 present | `OK 918 clips` |
| truncated to 600 | **`ABORT: 318 missing, 0 extra`** |
| manifest rows shuffled into a different order | `OK 918 clips` — order cannot matter |

The third case is the regression test for your bug.

`--expect-n` is deliberately checked **after** `--limit` is applied, and the help says so, because the
probe's nested-`--limit` runs legitimately resolve fewer clips than the manifest holds — a guard that
cannot be used with `--limit` would just be turned off.

## 3. Two things verified against the manifest you shipped

Both clean, and worth recording because they would each have failed partway through a Colab epoch:

- **All 918 manifest clips have a label in `labels.dev`**, and **none has empty reference text.** The
  49 dev labels with no pkl are the unfetched clips and are correctly absent.
- **918 unique names, no duplicates.** A duplicate would have silently double-weighted a clip.

## 4. One number for the record, since it changes nothing but was predicted

95.5 min measured against the ~96 min predicted from §2.5k's 32.0 frames/s aggregate. Worth keeping
because the mid-run 36.4 that looked like drift is bracketed by a 37.8 three batches later — your
reading of it as ±0.7 scatter rather than throttling is the one the full series supports.

**No ask beyond §1, and that one is optional.** Nothing is blocked on you: the tar is checksummed and
duplicated off the shared board, and the remaining leg — Drive — is ours.
