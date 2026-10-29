"""
Classical, download-free baselines (TF-IDF + logistic regression) on the same splits.
Useful as a sanity check of the data/labels and as a lower bound for the BERT models.

  python baselines_classic.py --data data/splits_small
"""
import os, json, argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix
from train import text_for, feature_names, feature_vector

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
args = ap.parse_args()


def load(s):
    p = os.path.join(args.data, f"{s}.sum.jsonl")
    p = p if os.path.exists(p) else os.path.join(args.data, f"{s}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")]


tr, va, te = load("train"), load("val"), load("test")
ytr, yva, yte = [np.array([r["label"] for r in d]) for d in (tr, va, te)]
FEATS = feature_names("struct,novelty")
_mu = _sd = None


def struct(d):
    global _mu, _sd
    X = np.array([feature_vector(r, FEATS) for r in d], dtype=float)
    if _mu is None:
        _mu, _sd = X.mean(0), X.std(0) + 1e-6
    return csr_matrix((X - _mu) / _sd)


results = {}
for mode in ["abstract", "title_abstract", "summary", "summary_captions", "sections", "abstract_summary"]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform([text_for(r, mode) for r in tr])
    Xva = vec.transform([text_for(r, mode) for r in va])
    Xte = vec.transform([text_for(r, mode) for r in te])
    for use_struct in (False, True):
        A, B, C = (Xtr, Xva, Xte) if not use_struct else (hstack([Xtr, struct(tr)]), hstack([Xva, struct(va)]), hstack([Xte, struct(te)]))
        best = None
        for Cval in (0.3, 1, 3, 10):
            clf = LogisticRegression(C=Cval, max_iter=2000).fit(A, ytr)
            acc = accuracy_score(yva, clf.predict(B))
            if best is None or acc > best[0]:
                best = (acc, Cval, clf)
        clf = best[2]
        p = clf.predict_proba(C)[:, 1]
        name = f"tfidf[{mode}]" + ("+struct+novelty" if use_struct else "")
        results[name] = {"val_acc": best[0], "C": best[1], "test_acc": accuracy_score(yte, (p >= 0.5).astype(int)), "test_f1": f1_score(yte, (p >= 0.5).astype(int)), "test_auc": roc_auc_score(yte, p)}
        r = results[name]
        print(f"{name:32s} val_acc={r['val_acc']:.3f}  test_acc={r['test_acc']:.3f}  f1={r['test_f1']:.3f}  auc={r['test_auc']:.3f}")

# structure-only
clf = LogisticRegression(max_iter=2000).fit(struct(tr), ytr)
p = clf.predict_proba(struct(te))[:, 1]
print(f"{'struct+novelty-only':32s} test_acc={accuracy_score(yte, (p>=0.5).astype(int)):.3f}  auc={roc_auc_score(yte, p):.3f}")
print(f"{'majority class':32s} test_acc={max(yte.mean(), 1-yte.mean()):.3f}")
os.makedirs("runs", exist_ok=True)
json.dump(results, open(os.path.join("runs", f"classic_{os.path.basename(args.data)}.json"), "w", encoding="utf-8"), indent=1)
