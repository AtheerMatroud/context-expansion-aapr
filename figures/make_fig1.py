import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch, Rectangle
sys.path.insert(0, "..")
from summarize import sentences, GraphRanker
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})

# real paper from the validation subset
rows = [json.loads(l) for l in open("../data/splits_small/val.jsonl", encoding="utf-8")]
r = next(c for c in rows if c["id"] == "1104.1601")
secs = {"Introduction": "intro", "Method": "method", "Experiments": "experiment", "Conclusion": "conclusion"}
rk = GraphRanker(sim="tfidf", centrality="indegree")
data = {}
for name, key in secs.items():
    txt = r["sections"].get(key, "") or r["abstract"]
    s = sentences(txt)[:6]
    if len(s) < 3: s = sentences(r["abstract"])[:6]
    S = rk.similarity_matrix(s); deg = S.sum(1)
    k = 2
    data[name] = (s, S, set(np.argsort(-deg)[:k].tolist()))



# ---- monochrome styling: shades and hatches instead of colours
HDR  = {"paper": "#555555", "graph": "#333333", "summary": "#777777", "clf": "#222222"}
HATCH= {"paper": "", "graph": "", "summary": "", "clf": ""}
FILL = {"paper": "#f2f2f2", "graph": "#ffffff", "summary": "#f2f2f2", "clf": "#ffffff"}
PH   = {"paper": "....", "graph": "", "summary": "////", "clf": "----"}   # panel background hatch per stage
plt.rcParams["hatch.linewidth"] = 0.4
fig = plt.figure(figsize=(7.16, 3.85)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
Y0, H = 0.115, 0.66
TOP = Y0 + H

def panel(x, y, w, h, title, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc="white", ec="black", lw=0.9))
    if PH[kind]:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc="none", ec="none", hatch=PH[kind], alpha=0.13, lw=0))
    ax.add_patch(FancyBboxPatch((x + 0.008, y + h - 0.072), w - 0.016, 0.058, boxstyle="round,pad=0.003,rounding_size=0.008", fc=HDR[kind], ec="none"))
    ax.text(x + w / 2, y + h - 0.043, title, ha="center", va="center", fontsize=6.3, color="white", weight="bold")

def step(x0, x1, num, text):
    y = 0.915
    ax.add_patch(FancyBboxPatch((x0, y - 0.03), x1 - x0, 0.072, boxstyle="round,pad=0.002,rounding_size=0.006", fc="white", ec="black", lw=0.7))
    ax.add_patch(Ellipse((x0 + 0.02, y + 0.006), 0.024, 0.045, fc="black", ec="none"))
    ax.text(x0 + 0.02, y + 0.006, str(num), ha="center", va="center", fontsize=6, color="white", weight="bold")
    ax.text(x0 + 0.038, y + 0.006, text, ha="left", va="center", fontsize=5.4, color="black", linespacing=1.1)

# ---- geometry: wider gaps so every box is separated and joined by arrows
PX, PW = 0.010, 0.112
GAP = 0.024
gx0, gw, ggap = PX + PW + GAP, 0.104, 0.024
GX1 = gx0 + 4 * gw + 3 * ggap
SX, SW = GX1 + GAP, 0.155
CX = SX + SW + GAP; CW = 1 - CX - 0.010

step(PX, gx0 - 0.006, 1, "Sections and\nsentences")
step(gx0, gx0 + 2 * gw + ggap, 2, "Sentence-level similarity\n= weighted edges")
step(gx0 + 2 * gw + 2 * ggap, GX1, 3, "Centrality ranking\n(Indegree or PageRank)")
step(SX, SX + SW, 4, "Four-part core-content\nassembly")
step(CX, CX + CW, 5, "Acceptance\nprediction")

# ---- Panel A: original paper
panel(PX, Y0, PW, H, "Original paper", "paper")
px, py = PX + 0.024, Y0 + 0.075
ax.add_patch(Rectangle((px + 0.010, py + 0.010), 0.070, 0.44, fc="white", ec="#444444", lw=0.7, zorder=0))
ax.add_patch(Rectangle((px, py), 0.070, 0.44, fc="white", ec="#444444", lw=0.7))
for lab, yy in zip(["Abstract", "Introduction", "Method", "Experiments", "Conclusion"], [Y0 + 0.44, Y0 + 0.365, Y0 + 0.29, Y0 + 0.215, Y0 + 0.14]):
    ax.text(px + 0.005, yy + 0.02, lab, fontsize=4.5, color="black", weight="bold")
    for j in range(2):
        ax.plot([px + 0.005, px + 0.062 - 0.012 * j], [yy + 0.006 - 0.011 * j] * 2, color="#999999", lw=0.5)
ax.text(PX + PW / 2, Y0 + 0.028, "LaTeX source", ha="center", fontsize=5.2, style="italic", color="black")

# ---- Panels B: per-section sentence graphs
for i, (name, (s, S, top)) in enumerate(data.items()):
    x = gx0 + i * (gw + ggap)
    panel(x, Y0, gw, H, name, "graph")
    n = len(s); cx, cy, Rx, Ry = x + gw / 2, Y0 + 0.345, 0.034, 0.17
    pts = [(cx + Rx * np.cos(2 * np.pi * t / n + np.pi / 2), cy + Ry * np.sin(2 * np.pi * t / n + np.pi / 2)) for t in range(n)]
    mx = S.max() if S.max() > 0 else 1
    for a in range(n):
        for b in range(a + 1, n):
            w = S[a, b] / mx
            if w > 0.08:
                ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="black", lw=0.35 + 2.0 * w, alpha=0.45 + 0.55 * w, zorder=1, solid_capstyle="round")
            else:
                ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="#888888", lw=0.5, ls=(0, (1.5, 2)), zorder=1)
    deg = S.sum(1)
    for t in range(n):
        sel = t in top
        ax.add_patch(Ellipse(pts[t], 0.037, 0.060, fc="#3a3a3a" if sel else "white", ec="black", lw=1.0 if sel else 0.7, zorder=3))
        ax.text(pts[t][0], pts[t][1], f"$s_{t+1}$", ha="center", va="center", fontsize=6, color="white" if sel else "black", zorder=4, weight="bold" if sel else "normal")
    order = np.argsort(-deg)
    ax.text(x + gw / 2, Y0 + 0.062, "rank: " + " > ".join(f"$s_{t+1}$" for t in order[:3]) + " > …", ha="center", va="center", fontsize=4.9, color="black")
    ax.text(x + gw / 2, Y0 + 0.028, "keep top-$k$: " + ", ".join(f"$s_{t+1}$" for t in sorted(top)), ha="center", va="center", fontsize=4.9, color="black", weight="bold")

def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, color="black", lw=0.9, zorder=5))
YM = Y0 + 0.345
# paper -> distribution bus above the four section graphs -> one arrow down into each
bus_y = TOP + 0.022
ax.plot([PX + PW + 0.004, PX + PW + GAP / 2], [YM, YM], color="black", lw=0.9)
ax.plot([PX + PW + GAP / 2, PX + PW + GAP / 2], [YM, bus_y], color="black", lw=0.9)
ax.plot([PX + PW + GAP / 2, GX1 - gw / 2], [bus_y, bus_y], color="black", lw=0.9)
for i in range(4):
    xc = gx0 + i * (gw + ggap) + gw / 2
    ax.add_patch(FancyArrowPatch((xc, bus_y), (xc, TOP + 0.005), arrowstyle="-|>", mutation_scale=7, color="black", lw=0.9, zorder=5))
ax.text(gx0 + (GX1 - gx0) / 2, bus_y + 0.012, "one sentence graph per section", fontsize=4.8, ha="center", va="bottom", color="black", style="italic")
# short arrows between the section boxes (parallel graphs, processed in the same way)
for i in range(3):
    xa = gx0 + i * (gw + ggap) + gw + 0.003; arrow(xa, xa + ggap - 0.006, YM)
# graphs -> core content -> classifier
arrow(GX1 + 0.004, SX - 0.004, YM)
arrow(SX + SW + 0.004, CX - 0.004, YM)

# ---- Panel C: core content
panel(SX, Y0, SW, H, "Core content", "summary")
rows_y = [Y0 + 0.505, Y0 + 0.395, Y0 + 0.285, Y0 + 0.175]
labels = ["Research problem", "Method", "Key results", "Contributions"]
for i, (lab, yy) in enumerate(zip(labels, rows_y)):
    name = list(data.keys())[i]; s, S, top = data[name]
    ax.add_patch(FancyBboxPatch((SX + 0.012, yy - 0.040), SW - 0.024, 0.086, boxstyle="round,pad=0.003,rounding_size=0.006", fc="white", ec="black", lw=0.6))
    ax.text(SX + 0.02, yy + 0.026, lab + ":", fontsize=5.4, weight="bold", color="black")
    sel = ", ".join(f"$s_{t+1}$" for t in sorted(top))
    ax.text(SX + 0.02, yy - 0.006, f"{sel} of {name.lower()}", fontsize=5.0, color="black")
    for j in range(2):
        ax.plot([SX + 0.02, SX + SW - 0.03 - 0.03 * j], [yy - 0.022 - 0.010 * j] * 2, color="#999999", lw=0.5)
ax.text(SX + SW / 2, Y0 + 0.03, "20 to 30 % of the paper", ha="center", fontsize=4.9, style="italic", color="black")

# ---- Panel D: classifier
panel(CX, Y0, CW, H, "Classifier", "clf")
bx, by, bw, bh = CX + 0.014, Y0 + 0.31, CW - 0.028, 0.19
ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.003,rounding_size=0.008", fc="white", ec="black", lw=0.8))
ax.text(bx + bw / 2, by + bh - 0.045, "BERT", ha="center", va="center", fontsize=6.4, weight="bold", color="black")
ax.text(bx + bw / 2, by + 0.095, "fine-tuned on\ncore content", ha="center", va="center", fontsize=4.8, color="black", linespacing=1.1)
ax.text(bx + bw / 2, by + 0.035, "(+ MoCo branch\nin LMCBert)", ha="center", va="center", fontsize=4.3, color="#444444", style="italic")
ax.add_patch(FancyArrowPatch((bx + bw / 2, by - 0.005), (bx + bw / 2, by - 0.055), arrowstyle="-|>", mutation_scale=7, color="black", lw=0.8))
ax.add_patch(FancyBboxPatch((bx, Y0 + 0.135), bw, 0.11, boxstyle="round,pad=0.003,rounding_size=0.008", fc="#eeeeee", ec="black", lw=0.7))
ax.text(bx + bw / 2, Y0 + 0.21, "Accept", ha="center", va="center", fontsize=5.8, weight="bold", color="black")
ax.text(bx + bw / 2, Y0 + 0.16, "Reject", ha="center", va="center", fontsize=5.8, weight="bold", color="black")
ax.text(bx + bw / 2, Y0 + 0.555, "input: 512 tokens", ha="center", va="center", fontsize=5.0, color="black")

# ---- legend
lx = 0.010; ly = 0.045
ax.add_patch(Ellipse((lx + 0.012, ly), 0.02, 0.034, fc="white", ec="black", lw=0.7)); ax.text(lx + 0.028, ly, "sentence node", va="center", fontsize=5.2)
ax.add_patch(Ellipse((lx + 0.125, ly), 0.02, 0.034, fc="#3a3a3a", ec="black", lw=0.9)); ax.text(lx + 0.141, ly, "sentence selected by centrality", va="center", fontsize=5.2)
ax.plot([lx + 0.335, lx + 0.365], [ly, ly], color="black", lw=2.0); ax.text(lx + 0.372, ly, "sentence-level similarity (line width = weight)", va="center", fontsize=5.2)
ax.plot([lx + 0.625, lx + 0.655], [ly, ly], color="#888888", lw=0.7, ls=(0, (1.5, 2))); ax.text(lx + 0.662, ly, "no connectivity (below threshold)", va="center", fontsize=5.2)
fig.savefig("fig_pipeline.pdf"); plt.close(fig)
print({k: (len(v[0]), sorted(v[2])) for k, v in data.items()})
