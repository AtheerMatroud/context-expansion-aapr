"""Collect runs/*/result.json into a results table (markdown + csv)."""
import glob, json, os

rows = []
for f in sorted(glob.glob("runs/*/result.json")):
    r = json.load(open(f, encoding="utf-8"))
    t = r["test"]
    rows.append((r["name"], r["args"]["input"], r["args"]["contrastive"], r["args"]["lam"], r["args"].get("features", ""), r["best_val_acc"], t["acc"], t["f1"], t["auc"], t["n"]))
hdr = ["run", "input", "contrastive", "lambda", "features", "val_acc", "test_acc", "test_f1", "test_auc", "n_test"]
with open("runs/results.md", "w", encoding="utf-8") as fo:
    fo.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for r in rows:
        fo.write("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |\n")
with open("runs/results.csv", "w", encoding="utf-8") as fo:
    fo.write(",".join(hdr) + "\n")
    for r in rows:
        fo.write(",".join(str(x) for x in r) + "\n")
print(open("runs/results.md", encoding="utf-8").read())
