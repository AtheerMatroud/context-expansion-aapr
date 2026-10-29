import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
BLUE, ORANGE, GREY = "#3b6ea8", "#e08a3c", "#8c8c8c"
fig, ax = plt.subplots(figsize=(3.4, 2.1))
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#999999"); ax.spines["bottom"].set_color("#999999")
pipes = ["BERT\nabstract", "BERT\nlexical\nsummary", "LMCBert\nlexical\nsummary", "BERT\nCEA\nsummary", "LMCBert\nGPT-4o\nsummary"]
acc = [61.9, 65.6, 65.5, 66.5, 68.0]; cols = [GREY, GREY, GREY, BLUE, ORANGE]
b = ax.bar(pipes, acc, width=0.6, color=cols, zorder=3)
for r, v in zip(b, acc): ax.text(r.get_x() + r.get_width()/2, v + 0.2, f"{v}", ha="center", va="bottom", fontsize=6.5)
ax.set_ylim(55, 72); ax.set_yticks([55, 60, 65, 70]); ax.set_ylabel("test accuracy (%)", fontsize=7.5)
ax.tick_params(axis="y", labelsize=6.5, colors="#444444", length=2); ax.tick_params(axis="x", labelsize=6.2, length=0)
ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
fig.tight_layout(pad=0.3); fig.savefig("fig_cost_acc.pdf"); plt.close(fig); print("ok")
