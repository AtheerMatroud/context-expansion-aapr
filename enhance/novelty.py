"""
Enhancement #1 (conclusion, "external knowledge bases to assess novelty").

Without paid APIs we build the knowledge base from the corpus itself: the
training papers form a retrieval index; for every paper we compute how close it
is to existing work.  Features added to each record (prefix `nov_`):

  nov_max_sim        cosine similarity to the most similar *other* training paper
  nov_top5_mean      mean similarity of the 5 nearest training papers
  nov_prior_max_sim  max similarity to training papers with an EARLIER arXiv id
                     (arXiv ids are yymm.nnnnn, so they order by submission month)
  nov_cat_centroid   similarity to the centroid of the paper's own primary category
  nov_density        number of training papers with similarity > 0.3

Low similarity to prior work = high novelty. Two embedding backends:
  tfidf (default, no downloads)   |   st (sentence-transformers, e.g. allenai-specter)

Optional: --s2 enriches records with Semantic Scholar citation counts of the
paper (free API, no key needed at low rate) — only meaningful for analysis,
NOT as a training feature for acceptance prediction (it leaks the future).

Usage:
  python enhance/novelty.py --data data/splits_small            # adds nov_* to *.sum.jsonl in place
"""
import os, sys, json, argparse, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def doc_text(r):
    return r["title"] + ". " + r["abstract"] + " " + (r.get("summary") or "")


def arxiv_order(pid):
    """Sortable key from arXiv id (new style yymm.nnnnn or old style cat/yymmnnn)."""
    m = re.match(r"(\d{2})(\d{2})\.(\d+)", pid)
    if m:
        return int(m.group(1)) * 100 + int(m.group(2))
    m = re.search(r"/(\d{2})(\d{2})\d+", pid)
    if m:
        return int(m.group(1)) * 100 + int(m.group(2))
    return 0


def embed(texts, backend, model):
    if backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True, stop_words="english")
        return vec, vec.fit_transform(texts)
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model)
        return st, st.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)


def transform(vec, texts, backend):
    if backend == "tfidf":
        return vec.transform(texts)
    return vec.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)


def sims(A, B, backend):
    if backend == "tfidf":
        return (A @ B.T).toarray()  # tf-idf rows are l2-normalised -> cosine
    return A @ B.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", default="tfidf", choices=["tfidf", "st"])
    ap.add_argument("--model", default="allenai-specter")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    def path(s):
        p = os.path.join(args.data, f"{s}.sum.jsonl")
        return p if os.path.exists(p) else os.path.join(args.data, f"{s}.jsonl")

    splits = {s: [json.loads(l) for l in open(path(s), encoding="utf-8")] for s in ("train", "val", "test")}
    train = splits["train"]
    vec, Xtr = embed([doc_text(r) for r in train], args.backend, args.model)
    tr_order = np.array([arxiv_order(r["id"]) for r in train])
    tr_cat = [r["category"].split(",")[0] for r in train]
    cats = sorted(set(tr_cat))
    cat_idx = {c: [i for i, cc in enumerate(tr_cat) if cc == c] for c in cats}
    # category centroids (dense)
    Xtr_dense_centroids = {}
    for c, idx in cat_idx.items():
        m = np.asarray(Xtr[idx].mean(axis=0)).ravel()
        n = np.linalg.norm(m)
        Xtr_dense_centroids[c] = m / n if n > 0 else m

    for s, rows in splits.items():
        X = Xtr if s == "train" else transform(vec, [doc_text(r) for r in rows], args.backend)
        for start in range(0, len(rows), args.chunk):
            S = sims(X[start:start + args.chunk], Xtr, args.backend)
            for j in range(S.shape[0]):
                i = start + j
                r = rows[i]
                row = S[j].copy()
                if s == "train":
                    row[i] = -1.0  # exclude self
                order = np.argsort(-row)
                top = row[order[:5]]
                earlier = row[tr_order < arxiv_order(r["id"])] if arxiv_order(r["id"]) else np.array([])
                c = r["category"].split(",")[0]
                cen = Xtr_dense_centroids.get(c)
                x = X[start + j]
                xv = np.asarray(x.todense()).ravel() if args.backend == "tfidf" else x
                r["nov_max_sim"] = float(top[0])
                r["nov_top5_mean"] = float(top.mean())
                r["nov_prior_max_sim"] = float(earlier.max()) if earlier.size else float(top[0])
                r["nov_cat_centroid"] = float(xv @ cen) if cen is not None else 0.0
                r["nov_density"] = int((row > 0.3).sum())
        with open(path(s), "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        v = np.array([[r["nov_max_sim"], r["nov_prior_max_sim"], r["nov_density"]] for r in rows])
        print(f"{s}: n={len(rows)} mean nov_max_sim={v[:,0].mean():.3f} prior={v[:,1].mean():.3f} density={v[:,2].mean():.1f}")


if __name__ == "__main__":
    main()
