"""
Enhancement #3 (conclusion, "fine-tuning with expert-annotated data").

Free source of real reviewer annotations: PeerRead (Kang et al., NAACL 2018,
github.com/allenai/PeerRead). ICLR 2017 has 427 papers with accept/reject and
per-review scores (RECOMMENDATION 1-10, ORIGINALITY, CLARITY, SOUNDNESS, IMPACT...).

This converts PeerRead into our JSONL schema so the same pipeline
(summarize.py -> train.py) can (a) fine-tune on expert-scored papers and
(b) learn an auxiliary "reviewer recommendation" regression target.

  git clone --depth 1 --filter=blob:none --sparse https://github.com/allenai/PeerRead
  cd PeerRead && git sparse-checkout set data/iclr_2017
  python enhance/peerread.py --root ../PeerRead/data/iclr_2017 --out data/peerread_iclr17
"""
import os, sys, json, glob, argparse, re, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCORE_KEYS = ["RECOMMENDATION", "ORIGINALITY", "CLARITY", "SOUNDNESS_CORRECTNESS", "IMPACT", "SUBSTANCE", "MEANINGFUL_COMPARISON", "REVIEWER_CONFIDENCE"]
SEC_MAP = [
    ("intro", re.compile(r"introduction|overview|motivation", re.I)),
    ("related", re.compile(r"related|background|prior work|previous work", re.I)),
    ("experiment", re.compile(r"experiment|evaluation|result|empirical|analysis", re.I)),
    ("conclusion", re.compile(r"conclu|discussion|future work", re.I)),
]


def sections_from_parsed(pdf_json):
    secs = {}
    for s in pdf_json.get("metadata", {}).get("sections", []) or []:
        head = s.get("heading") or ""
        key = "method"
        for k, rx in SEC_MAP:
            if rx.search(head):
                key = k
                break
        secs[key] = (secs.get(key, "") + "\n" + (s.get("text") or "")).strip()
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "dev", "test"):
        rows = []
        for f in sorted(glob.glob(os.path.join(args.root, split, "reviews", "*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            pid = os.path.basename(f).replace(".json", "")
            pdf = os.path.join(args.root, split, "parsed_pdfs", pid + ".pdf.json")
            secs = sections_from_parsed(json.load(open(pdf, encoding="utf-8"))) if os.path.exists(pdf) else {}
            scores = {}
            for k in SCORE_KEYS:
                vals = [float(r[k]) for r in d["reviews"] if k in r and str(r[k]).strip() not in ("", "None")]
                if vals:
                    scores["score_" + k.lower()] = statistics.mean(vals)
            n_words = sum(len(v.split()) for v in secs.values())
            rows.append({
                "id": f"iclr17_{pid}", "title": d.get("title", ""), "authors": ", ".join(d.get("authors", [])),
                "category": "cs.LG", "venue": "ICLR" if d.get("accepted") else "No", "label": int(bool(d.get("accepted"))),
                "abstract_raw": d.get("abstract", ""), "abstract": d.get("abstract", ""), "sections": secs,
                "n_words": n_words, "n_reviews": len([r for r in d["reviews"] if not r.get("IS_META_REVIEW")]),
                **scores, "source": "peerread_iclr2017",
            })
        name = {"dev": "val"}.get(split, split)
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        acc = sum(r["label"] for r in rows)
        withrec = sum(1 for r in rows if "score_recommendation" in r)
        print(f"{name}: {len(rows)} papers, accepted={acc}, with recommendation score={withrec}")


if __name__ == "__main__":
    main()
