"""
Seed-replicated comparison of two configurations with significance tests.

  python stats_compare.py --a "runs/graph_tfidf_indegree_bert_summary_s*" --b "runs/graph_embed_indegree_bert_summary_s*" \
                          --name_a "TF-IDF + Indegree" --name_b "MiniLM + Indegree"

For each configuration it collects every matching run (one per seed) and reports
  * accuracy, F1, AUC as mean +/- SD over seeds (and each seed's value),
  * Welch's t-test across seeds for each metric (independent samples, n = #seeds),
  * a paired test on the 1,000 test papers, which is far more powerful than n = 3 seeds:
      - McNemar's exact test on the per-paper correctness of the seed-averaged
        (ensemble) predictions of A vs B,
      - a paired bootstrap (10,000 resamples of test papers) for the difference in
        accuracy, F1 and AUC between the seed-averaged predictions, giving a 95 % CI
        and a two-sided p-value.
Writes runs/stats_<slug>.md and prints it.
"""
import argparse, glob, json, os, re
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load(pattern):
    runs = []
    for d in sorted(glob.glob(pattern)):
        rj, pj = os.path.join(d, "result.json"), os.path.join(d, "preds.json")
        if os.path.exists(rj) and os.path.exists(pj):
            r, p = json.load(open(rj, encoding="utf-8")), json.load(open(pj, encoding="utf-8"))
            runs.append({"name": r["name"], "seed": r["args"].get("seed"), "test": r["test"], "ids": p["ids"], "y": np.array(p["y"]), "p": np.array(p["p"])})
    return runs


def mean_sd(v):
    v = np.array(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred), "auc": roc_auc_score(y, p)}


def mcnemar_exact(b, c):
    """Exact McNemar test: b = A right & B wrong, c = A wrong & B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return min(1.0, p)


def paired_bootstrap(y, pa, pb, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = {"acc": [], "f1": [], "auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        ma, mb = metrics(y[idx], pa[idx]), metrics(y[idx], pb[idx])
        for k in diffs:
            diffs[k].append(mb[k] - ma[k])
    out = {}
    for k, v in diffs.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        pval = 2 * min((v <= 0).mean(), (v >= 0).mean())  # two-sided, H0: no difference
        out[k] = (v.mean(), lo, hi, min(1.0, pval))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="glob for configuration A run dirs")
    ap.add_argument("--b", required=True, help="glob for configuration B run dirs")
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    assert A and B, f"no complete runs found: A={len(A)} B={len(B)}"
    lines = [f"# {args.name_a} vs {args.name_b}", ""]
    lines.append(f"Seeds: {args.name_a}: {[r['seed'] for r in A]} | {args.name_b}: {[r['seed'] for r in B]}")
    lines.append("")
    lines.append("## Per-seed test results")
    lines.append("| configuration | seed | acc | F1 | AUC |")
    lines.append("|---|---|---|---|---|")
    for nm, runs in ((args.name_a, A), (args.name_b, B)):
        for r in runs:
            t = r["test"]
            lines.append(f"| {nm} | {r['seed']} | {100*t['acc']:.1f} | {100*t['f1']:.1f} | {100*t['auc']:.1f} |")
    lines.append("")
    lines.append("## Mean +/- SD over seeds and Welch's t-test (independent, n = #seeds)")
    lines.append("| metric | " + args.name_a + " | " + args.name_b + " | diff (B-A) | t | p (Welch) |")
    lines.append("|---|---|---|---|---|---|")
    for k in ("acc", "f1", "auc"):
        va, vb = [100 * r["test"][k] for r in A], [100 * r["test"][k] for r in B]
        ma, sa = mean_sd(va); mb, sb = mean_sd(vb)
        if len(va) > 1 and len(vb) > 1:
            t, p = stats.ttest_ind(vb, va, equal_var=False)
        else:
            t, p = float("nan"), float("nan")
        lines.append(f"| {k.upper()} | {ma:.2f} +/- {sa:.2f} | {mb:.2f} +/- {sb:.2f} | {mb-ma:+.2f} | {t:.2f} | {p:.3f} |")
    # paired tests on the test papers, using seed-averaged probabilities per configuration
    ids = A[0]["ids"]
    assert all(r["ids"] == ids for r in A + B), "test sets differ between runs"
    y = A[0]["y"]
    pa = np.mean([r["p"] for r in A], axis=0); pb = np.mean([r["p"] for r in B], axis=0)
    ca, cb = ((pa >= 0.5).astype(int) == y), ((pb >= 0.5).astype(int) == y)
    b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_mc = mcnemar_exact(b_, c_)
    ma, mb = metrics(y, pa), metrics(y, pb)
    boot = paired_bootstrap(y, pa, pb)
    lines.append("")
    lines.append(f"## Paired tests on the {len(y)} test papers (seed-averaged predictions)")
    lines.append(f"Seed-averaged: {args.name_a}: acc {100*ma['acc']:.1f}, F1 {100*ma['f1']:.1f}, AUC {100*ma['auc']:.1f} | "
                 f"{args.name_b}: acc {100*mb['acc']:.1f}, F1 {100*mb['f1']:.1f}, AUC {100*mb['auc']:.1f}")
    lines.append(f"McNemar (exact): {args.name_a} right / {args.name_b} wrong = {b_}; {args.name_a} wrong / {args.name_b} right = {c_}; p = {p_mc:.4f}")
    lines.append("")
    lines.append("| metric | diff (B-A) | 95% CI | p (paired bootstrap) |")
    lines.append("|---|---|---|---|")
    for k in ("acc", "f1", "auc"):
        m, lo, hi, p = boot[k]
        lines.append(f"| {k.upper()} | {100*m:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {p:.4f} |")
    txt = "\n".join(lines)
    slug = re.sub(r"[^a-z0-9]+", "_", (args.name_a + "_vs_" + args.name_b).lower()).strip("_")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"stats_{slug}.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
