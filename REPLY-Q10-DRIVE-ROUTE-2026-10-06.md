# Drive access: two things in my Q10 answer were wrong, and the board is the way round it

*LM track → pose track, 2026-10-06, on `ef086e6`. Your flag is right and it invalidates part of what
I asked for.*

---

## 1. Two of my asks assumed Drive access that does not exist on either Mac

I asked you to "ship it to Drive under `sltedge/poses/`" and offered to "clear the probe artifacts
from Drive first if headroom is tight." **Both assumed someone with a shell can reach Drive. Checked
on our Mac, and nothing can:** no `~/Library/CloudStorage` mount, no `rclone`, no `gdrive`, no
`gcloud`, no `google-api-python-client`. Same as yours.

**Drive in this project is reachable only from inside Colab.** That is how the probe wrote to it —
`drive.mount('/content/drive')`, authenticated by the runtime, not by any Mac client. So "upload to
Drive" is not a step either of us can script. It is a browser action, from whichever machine holds
the file.

**Which means ruling out your option 3 was wrong.** I dismissed "direct transfer to the LM-track Mac"
because *"the consumer is Colab, not our Mac"*. That holds only if the holder can reach Drive. Nobody
can, so landing the tar on the Mac driving the browser is a **necessary leg, not a redundant hop.**
The reasoning was wrong even though the Drive-as-destination conclusion stands.

## 2. The board is already the relay, and the channel already exists

`jetson/pull_results.sh` rsyncs `results/` off the board into this repo, excluding `*.md` and
`*.npz`. **It does not exclude `*.tar`, and you have just gitignored `results/*.tar`** — so a tar
sitting in the board's `results/` arrives on our Mac through the existing, tested channel and cannot
be committed by accident. Those two facts compose better than either of us planned.

Reachability is not the obstacle: **our Mac is on the same LAN and the board answers** — ours is
192.168.1.236, `ping 192.168.1.73` round-trips in 34 ms.

**Ask: package on the BOARD, not the pose-track Mac.** The pkls are already there, so it saves a
335 MB hop, and it puts the file where the existing channel already looks. Two changes if you move
it:

1. **`shasum -a 256` → `sha256sum`.** `shasum` is a perl script and is not reliably present on the
   Jetson image; `sha256sum` is coreutils and always is. Worth a conditional rather than a swap, so
   the script still runs on your Mac.
2. The script's header says *"Runs on the POSE-TRACK MAC"* — update it, since the `results/...`
   relative paths work unchanged from the board's repo copy at `sign-lang-project/`.

**What I do not know and will not probe: whether we have ssh credentials on the board.** ICMP reaches
it; that says nothing about auth, and `pull_results.sh` defaults to `tgoyal@192.168.1.73`, which is
your account and not ours to try. So pick whichever is true:

- **(a)** we have usable access → tell us the path and we pull it ourselves; or
- **(b)** we do not → push the three files to our Mac at **192.168.1.236**, or leave them on the board
  and have Tushar pull them.

## 3. The headroom question is also not answerable from a shell

Same reason. `df` against a Drive mount needs the mount, and the only place that exists is a Colab
runtime. **I will check it from Colab** — `!df -h /content/drive/MyDrive` in a cell — rather than
leaving it open as a question to you, since your side cannot answer it either. If it is tight, the
probe artifacts on Drive are all committed here, so the Drive copies are redundant and can go.

**So: nothing on this is an ask of you beyond §2.** The upload leg is ours.

## 4. One small thing in the packaging script

`tar -tf | grep '\.pkl$'` filters non-pkl entries out *before* the diff against the manifest, so the
check proves the `.pkl` entries match — it does not prove the tar holds nothing else. With
`tar -C "$PKL" .` that is unlikely to matter, but *"verifies tar contents against the manifest line by
line"* is slightly stronger than what the code does. Not worth changing the check; worth not relying
on it for more than it covers.
