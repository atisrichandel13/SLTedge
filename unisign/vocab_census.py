"""Token census over chosen OpenASL splits -> keep-id set for MT5Uni.prune_vocab.

Why this exists: the original keep set (results/openasl_vocab_keep_ids.json, 26,078 ids) was built
from a census over train+dev+TEST (98,419 sentences), which leaks test tokens into the deployed
model's vocabulary. A model whose embedding matrix was chosen using the test set has seen the test
set. This rebuilds the keep set from train+dev only, and reports exactly what the leak was worth.

Keep-set composition, matching model.py:
  * pad/eos/unk = 0/1/2 are mandatory (prune_vocab asserts keep[:3] == [0,1,2])
  * every token id appearing in the chosen splits' target sentences
  * every token id in the encoder prefix "Translate sign language video to English: "
"""
import argparse, gzip, json, os, pickle, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--labels-dir", default="data/openasl_labels")
ap.add_argument("--splits", nargs="+", default=["train", "dev"])
ap.add_argument("--mt5", default="weights/mt5-base")
ap.add_argument("--out", default="results/openasl_vocab_keep_ids_traindev.json")
ap.add_argument("--counts-out", default="results/openasl_token_counts_traindev.json")
ap.add_argument("--compare", default="results/openasl_vocab_keep_ids.json")
a = ap.parse_args()

from transformers import T5Tokenizer, AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained(a.mt5, legacy=False)
except Exception:
    tok = T5Tokenizer.from_pretrained(a.mt5)

cnt, n_sent = Counter(), 0
per_split = {}
for sp in a.splits:
    d = pickle.load(gzip.open(os.path.join(a.labels_dir, f"labels.{sp}"), "rb"))
    texts = [v["text"] if isinstance(v, dict) and "text" in v else v for v in d.values()]
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    ids = tok(texts, add_special_tokens=True)["input_ids"]
    c = Counter(i for row in ids for i in row)
    per_split[sp] = {"sentences": len(texts), "distinct_tokens": len(c)}
    cnt.update(c); n_sent += len(texts)
    print(f"[census] {sp}: {len(texts)} sentences, {len(c)} distinct tokens")

prefix_ids = tok(["Translate sign language video to English: "], add_special_tokens=True)["input_ids"][0]
keep = set(cnt) | {0, 1, 2} | set(prefix_ids)
keep = sorted(keep)
print(f"[census] {n_sent} sentences over {a.splits}; keep set = {len(keep)} ids "
      f"({100*len(keep)/len(tok):.2f}% of {len(tok)})")

os.makedirs("results", exist_ok=True)
json.dump(keep, open(a.out, "w"))
json.dump({str(k): v for k, v in cnt.items()}, open(a.counts_out, "w"))

if os.path.exists(a.compare):
    old = set(json.load(open(a.compare)))
    new = set(keep)
    dropped, added = sorted(old - new), sorted(new - old)
    print(f"\n[compare] vs {a.compare} ({len(old)} ids)")
    print(f"  dropped (were in the old set, not justified by train+dev): {len(dropped)}")
    print(f"  added   (in train+dev but somehow absent before):          {len(added)}")
    if dropped:
        occ = sum(cnt.get(i, 0) for i in dropped)
        print(f"  those dropped ids occur {occ} times in train+dev (0 expected if they were test-only)")
        print(f"  examples: {[tok.convert_ids_to_tokens(i) for i in dropped[:15]]}")
    json.dump({"dropped": dropped, "added": added,
               "per_split": per_split, "splits": a.splits,
               "n_keep_new": len(new), "n_keep_old": len(old)},
              open("results/vocab_keep_diff.json", "w"), indent=1)
