"""
Overlap between the sentence selections of two summarisers.

Each .sum.jsonl row has a four-part `summary` ("Research problem: ... Method: ...").
The summary is split back into its sentences and, for every paper present in both
files, the Jaccard overlap |A ∩ B| / |A ∪ B| of the two sentence sets is computed.
The mean over papers is reported together with the share of A's sentences that B
also selected.

Usage:
  python overlap.py --a data/splits/val.sum.jsonl --b data/splits_graph_embed_indegree/val.sum.jsonl
  python overlap.py --a A1.jsonl A2.jsonl --b B1.jsonl B2.jsonl      # several splits pooled
"""
import argparse, json, re

_PREFIX = re.compile(r"^(Research problem|Method|Key results|Contributions):\s*")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def sentence_set(summary):
    out = set()
    for part in summary.split("\n"):
        part = _PREFIX.sub("", part.strip())
        for s in _SPLIT.split(part):
            s = re.sub(r"\s+", " ", s).strip().lower()
            if len(s.split()) >= 6:
                out.add(s)
    return out


def load(paths):
    d = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d[r["id"]] = sentence_set(r.get("summary", ""))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    jac, rec = [], []
    for i in ids:
        a, b = A[i], B[i]
        if not a or not b:
            continue
        jac.append(len(a & b) / len(a | b))
        rec.append(len(a & b) / len(a))
    n = len(jac)
    print(f"{args.name_a} vs {args.name_b}: {n} papers; mean Jaccard overlap {100*sum(jac)/n:.1f} %; "
          f"share of {args.name_a} sentences also selected by {args.name_b} {100*sum(rec)/n:.1f} %")


if __name__ == "__main__":
    main()
