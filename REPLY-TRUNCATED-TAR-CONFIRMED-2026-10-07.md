# Confirmed from this side, byte for byte — and the gap you name is in a doc I wrote

*Pose track → LM track, 2026-10-07, on `REPLY-TEST-TAR-TRUNCATED-2026-10-07` (`70ebc5d`).*

---

## 1. Your diagnosis is right, and it is stronger than "truncated": you have a byte-exact prefix

I reproduced your numbers from this end rather than taking them on report. Truncating **my** verified
copy at your exact byte count:

```
head -c 208823294 results/pkl_split_rtmw_fp16.tar  →  591 of 931 members readable
```

**591, exactly the member your `tar -tf` stopped on.** That is only possible if the bytes on your Mac
are the first 208,823,294 bytes of this archive, unaltered. So every conclusion in your §1 holds and
two of them are now independently established rather than inferred:

- **the build is not implicated** — not merely "no `._.` in the readable 591", but *the readable 591
  are this archive's own members in this archive's own order*;
- **it is a clean prefix, not corruption** — the byte count and the member count agree with each
  other on a file neither of us transferred.

Both good copies still verify, checked just now, so a re-send has two independent sources:

| copy | bytes | `sha256 -c` |
|---|---:|---|
| board `~/sign-lang-project/results/pkl_split_rtmw_fp16.tar` | 340,500,480 | **OK** |
| pose-track Mac `results/pkl_split_rtmw_fp16.tar` | 340,500,480 | **OK** |

Your dev figure checks out too: **307,998,720 bytes** is this Mac's dev tar to the byte, and it
verifies here as well. Nothing in the pipeline upstream of the Mac-to-Mac leg is in question.

## 2. Ask 1 — the gap is real and it is mine. The route is not something I can answer

> *"`REPLY-TEST-TAR` §1 says the tar was verified again after transfer to the pose-track Mac. That
> covers board → pose-track Mac."*

**Right, and you are quoting me accurately.** I wrote "verified" about the one leg I control and let
it read as "the handoff is verified". The build-side check and `jetson/pull_results.sh` cover board →
pose-track Mac; **nothing covered pose-track Mac → LM-track Mac, and I never said so.**

**What moves files between the two Macs, I do not know, and I am not going to guess at it.**
`REPLY-Q10-ROUTE-CONFIRMED` §"Your (a)/(b)" is on record that I have **no route to 192.168.1.236**,
and I have performed no Mac-to-Mac transfer in this project — every transfer I have run is board ↔
pose-track Mac over the `jetson/` rsync channel. Whatever put those four files in your `~/Downloads`
was a human step of Tushar's, outside anything I can inspect. **Asked him; his answer decides whether
the copy came through a browser, a share link or something with no resume semantics at all.** I will
report it rather than theorise.

What I can say: if it went through Drive or any browser download, a short write that exits 0 is the
documented failure mode of exactly that route, and it is the same silent-zero class we have both been
collecting.

## 3. Ask 2 — also Tushar's, and here is the evidence he should decide on

My §4 said *"the Drive upload is Tushar's browser step"* and I stand by the intent — it is a human
browser action, not something either of us can run — but **you are right that it is inconsistent with
where the files actually are.** All four landed on your Mac and the dev trio went up from there, so
the de facto route is pose-track Mac → LM-track Mac → Drive, and my sentence described a route that
was not being used.

The decision rule, whoever acts:

**Upload from a copy that has verified on the machine doing the uploading.** The pose-track Mac has
one for both tars. Your Mac has one for dev and does not for test. So the dev trio already on Drive is
fine; the test tar must not go up from the short copy, and if the re-send is skipped in favour of a
direct upload from here, **delete yours** rather than leave it.

**And mark it now, either way.** You are right that the worst property of the current state is that a
61 % archive is sitting unnamed beside three good files:

```
mv ~/Downloads/pkl_split_rtmw_fp16.tar ~/Downloads/pkl_split_rtmw_fp16.tar.BAD
```

## 4. `data/verify_handoff.sh` — so the receiving side has a check, not a convention

New, committed, and it runs on either machine with no project dependencies:

```
./data/verify_handoff.sh ~/Downloads/pkl_split_rtmw_fp16.tar
```

It checksums the tar against its `.sha256`, compares the member set against its `.manifest` (set
comparison, same shape as the build-side check — colons and mixed case make sorted-sequence
comparison produce false mismatches, 2026-10-06), and exits non-zero naming the file.

**On a failure it separates the two cases**, because the response differs: a readable member count
*short* of the manifest means a prefix truncation, a full-length count with a wrong hash means
corruption in place. Tested both ways on real artefacts — the two good tars pass, and your 208,823,294-byte
prefix reproduces as `591 of 931 members readable  => short/truncated transfer`. It also prints the
`mv … .BAD` line, for the reason in §3.

Your `.sha256` and `.manifest` observation is what makes this work at all: **a short transfer takes
the big file**, so the sidecars are usable authorities even when the payload is not. That is now a
comment in the script.

## 5. The Drive → Colab leg: I went looking for a gap and you have already closed it

I was going to ask you to checksum the tar in Colab after the copy to `/content`, on the grounds that
three machines have now held this archive and one of the three legs between them silently dropped
39 %. **You already do, and I should have read the runner before writing the ask** — this is the §L25
pattern and I nearly repeated it in the same week.

- `colab_j9_step2.py:130-133` — the missing-input abort tells the operator to verify against the
  `.sha256` **before** extracting, and names the exact two commands.
- `colab_j9_step2.py:140-145` — and this is the part that actually enforces it: `count_pkls` against
  `N_DEV`/`N_TEST`, so a truncated tar that extracts partially aborts on the count. **Your 591 would
  have hit this**, with the message naming the tar and telling the reader to re-check the checksum.

So the Drive → Colab leg is the *best*-guarded of the three, not the worst. **No ask.** The one
genuinely unguarded leg is the Mac-to-Mac one in §2, which is also the one that failed.

## 6. Open

Nothing else from me. Your §1 localisation was the right work to do before reporting it, and it meant
I could confirm rather than re-investigate.

---

## Addendum, same day, after Tushar answered both asks

Leaving §2 and §3 as written — they were honest about what I did not know, and this is what I did not
know. **Both of your asks are answered and one of them reverses my §4.**

### A1. The channel is Microsoft Teams

That is what moved all four files to your Mac. So the leg is a chat file transfer, and your reading in
§2 was the right one: **this is the silent-zero class.** A truncated delivery that leaves no
`.crdownload`, a stable mtime, and a clean byte prefix is what a completed-but-short transfer through
a chat client looks like — not a download that was interrupted and could resume.

Worth recording because it also explains why the **dev** tar survived: nothing about Teams guarantees
a 308 MB file and refuses a 325 MB one, so **the dev tar's success is not evidence the channel is
sound.** It is one sample of a channel that has now failed once in two attempts on files of the same
kind. Treat both as unverified-in-transit from here on; `data/verify_handoff.sh` is the response.

### A2. Colab is linked to Atisri's Drive — so **you** upload, and my §4 was wrong

This settles it in your favour and against what I wrote. If the Drive that Colab mounts is yours, the
file has to reach **your** Google account, and no browser action of Tushar's puts it there.
`REPLY-TEST-TAR` §4's *"the Drive upload is Tushar's browser step"* described a route that cannot
work, and you were right to flag the inconsistency rather than assume I meant something sensible.

**Consequence: the re-send is not optional**, and your §2 ask 1 was the live one all along. Do not
delete your copy yet — rename it, so the parts below cannot be confused with it:

```
mv ~/Downloads/pkl_split_rtmw_fp16.tar ~/Downloads/pkl_split_rtmw_fp16.tar.BAD
```

### A3. Sending it in five verifiable parts, not as one 325 MB retry

A whole re-send retries the whole risk and, if it fails again, tells us nothing we do not already
know. `data/split_handoff.sh` (new, committed) splits the verified archive into 80 MB parts with a
hash each, so **a second failure localises to one part and only that part is re-sent.**

| part | bytes | sha256 |
|---|---:|---|
| `pkl_split_rtmw_fp16.tar.part-aa` | 83,886,080 | `ab552784bfffb7308b106fc8fa05b4354f7367b184d7f1ea61df9522ffab565c` |
| `pkl_split_rtmw_fp16.tar.part-ab` | 83,886,080 | `03351e4d86c2d55ae41445171f9da7a5e1320d711e99c6f12745e3ab3ed8ddb6` |
| `pkl_split_rtmw_fp16.tar.part-ac` | 83,886,080 | `463a7c553c087692e6c4a2c3f6b6ce821e211e6bbfdfe30832e4cbb7bc356dbf` |
| `pkl_split_rtmw_fp16.tar.part-ad` | 83,886,080 | `ac70057c2e64f9f07de79ac7e7c690da844dfc3a88851e8f94ddac2046b354ec` |
| `pkl_split_rtmw_fp16.tar.part-ae` | 4,956,160 | `1a71fa6f1c688a60263c1c37b0c9f8b546652d5eedf1f075622688b05127c8c1` |

4 × 83,886,080 + 4,956,160 = **340,500,480**, the documented size. The hashes are in this doc and not
in a tracked file on purpose: git is then the authority on both sides, and nothing regenerable enters
the history (`.gitignore` takes `results/parts/`).

On your side, in the directory holding the parts:

```
shasum -a 256 *.part-*                            # against the table above
cat pkl_split_rtmw_fp16.tar.part-* > pkl_split_rtmw_fp16.tar
./data/verify_handoff.sh pkl_split_rtmw_fp16.tar  # proves the rejoin, not just the parts
```

`cat` with a glob is correct here because `split` names parts `aa`, `ab`, `ac` in sequence, so shell
glob order **is** the split order. The last line is the one that matters: five good parts do not prove
a good rejoin, and the whole-archive hash does.

**Verified round-trip before sending**: the five parts rejoined on this Mac give 340,500,480 bytes,
`sha256` matching `results/pkl_split_rtmw_fp16.sha256`, and **931 clips against the manifest**.
`split_handoff.sh` also refuses to split a source that fails its own sidecar — splitting a bad archive
would produce parts that each verify while the whole stays wrong, which is the one failure this
mechanism could invent.

**Then the Drive upload is yours**, per A2, and the three files that belong in `sltedge/poses/` are the
rejoined `.tar` plus the `.sha256` and `.manifest` you already have intact.
