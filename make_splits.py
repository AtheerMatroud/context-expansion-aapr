"""
Filter incomplete papers and create train/val/test splits.

The paper removed entries with incomplete information, then took 1000 papers
for validation and 1000 for test (Table I: 31,726 / 1,000 / 1,000).
We keep a paper only if it has a usable abstract and a non-trivial body.

Usage:
  python make_splits.py --inp data/aapr_clean.jsonl --out data/splits --val 1000 --test 1000
  python make_splits.py --inp data/aapr_subset3k.jsonl --out data/splits_small --val 300 --test 300
"""
import json, os, random, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--val", type=int, default=1000)
ap.add_argument("--test", type=int, default=1000)
ap.add_argument("--min_abstract_words", type=int, default=30)
ap.add_argument("--min_body_words", type=int, default=300)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
kept, dropped = [], collections.Counter()
with open(args.inp, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if not r["abstract"]:  # fall back to dataset's pre-cleaned abstract
            r["abstract"] = r["abstract_raw"]
            r["abstract_is_raw"] = True
        if len(r["abstract"].split()) < args.min_abstract_words:
            dropped["short_abstract"] += 1
            continue
        if r.get("n_words", 0) < args.min_body_words:
            dropped["short_body"] += 1
            continue
        if not r["title"]:
            dropped["no_title"] += 1
            continue
        kept.append(r)

random.Random(args.seed).shuffle(kept)
test = kept[: args.test]
val = kept[args.test : args.test + args.val]
train = kept[args.test + args.val :]
for name, rows in [("train", train), ("val", val), ("test", test)]:
    with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(r["label"] for r in rows)
    print(f"{name:5s}: {len(rows):6d}  accepted={pos:6d}  rejected={len(rows)-pos:6d}")
print("dropped:", dict(dropped))
