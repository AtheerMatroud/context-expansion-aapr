"""
Holm adjustment over a family of comparisons produced by stats_compare.py.

  python stats_family.py --name "controls" --primary mcnemar runs/stats_random_k*.json runs/stats_lead_k*.json ...

Primary hypothesis per comparison (one per comparison, so the family has as many hypotheses as comparisons):
  --primary mcnemar : exact McNemar test on the accuracy of the seed-averaged ensembles (equal ensemble sizes required)
  --primary boot_acc: paired bootstrap on ensemble accuracy
  --primary paired_acc: paired t-test on per-seed accuracy differences (matched seeds)
All other metrics and tests are reported as secondary, unadjusted.
"""
import argparse, json, os
import numpy as np


def holm(pvals):
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", default="family")
    ap.add_argument("--primary", default="mcnemar", choices=["mcnemar", "boot_acc", "paired_acc"])
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    S = [json.load(open(f, encoding="utf-8")) for f in args.files]
    ps = []
    for s in S:
        if args.primary == "mcnemar":
            assert s["n_a"] == s["n_b"], f"unequal ensemble sizes in {s['name_a']} vs {s['name_b']}"
            ps.append(s["ensemble"]["mcnemar_p"])
        elif args.primary == "boot_acc":
            ps.append(s["ensemble"]["boot"]["acc"]["p"])
        else:
            ps.append(s["paired_runs"]["acc"]["p"])
    adj = holm(np.array(ps, dtype=float))
    lines = [f"# Family '{args.name}': {len(S)} comparisons, primary test = {args.primary} on accuracy, Holm adjustment", "",
             "| comparison (B vs A) | ensemble sizes | ensemble acc A -> B | McNemar p | boot acc diff [95% CI] | paired per-seed acc diff mean +/- SD | primary p | Holm-adjusted p |",
             "|---|---|---|---|---|---|---|---|"]
    for s, p, q in zip(S, ps, adj):
        e = s["ensemble"]; b = e["boot"]["acc"]; pr = s.get("paired_runs", {}).get("acc")
        prs = f"{pr['mean']:+.2f} +/- {pr['sd']:.2f} (n={len(pr['diffs'])})" if pr else "no matched seeds"
        lines.append(f"| {s['name_b']} vs {s['name_a']} | {s['n_a']} / {s['n_b']} | {100*e['a']['acc']:.1f} -> {100*e['b']['acc']:.1f} | {e['mcnemar_p']:.4f} | {100*b['diff']:+.2f} [{100*b['lo']:+.2f}, {100*b['hi']:+.2f}] | {prs} | {p:.4f} | {q:.4f} |")
    txt = "\n".join(lines)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"family_{args.name}.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
