import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
names = ["BERT\nabstract", "LMCBert\n$\\lambda$=0.1", "LMCBert\n$\\lambda$=0.5", "BERT\nlexical\nsummary", "BERT\nCEA\nsummary"]
acc = [61.9, 64.6, 65.5, 65.6, 66.5]; auc = [68.3, 69.2, 70.2, 69.5, 72.2]
x = np.arange(len(names)); w = 0.36
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#999999"); ax.spines["bottom"].set_color("#999999")
b1 = ax.bar(x - w/2 - 0.01, acc, w, color="#3b6ea8", label="Accuracy", zorder=3)
b2 = ax.bar(x + w/2 + 0.01, auc, w, color="#e08a3c", label="AUC", zorder=3)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.25, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=6, color="#222222")
# highlight the proposed configuration
ax.axvspan(x[-1] - 0.48, x[-1] + 0.48, color="#e9f4ea", zorder=0)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.3)
ax.set_ylim(55, 75); ax.set_yticks([55, 60, 65, 70, 75]); ax.set_ylabel("test score (%)", fontsize=7.5)
ax.tick_params(axis="y", labelsize=6.5, colors="#444444", length=2); ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
ax.legend(frameon=False, loc="upper left", fontsize=6.5, ncol=2, handlelength=1.2, columnspacing=1.0)
fig.tight_layout(pad=0.3)
fig.savefig("fig_results.pdf"); plt.close(fig); print("ok")
