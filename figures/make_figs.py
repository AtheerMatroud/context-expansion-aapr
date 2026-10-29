import json, re, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, "..")
from summarize import sentences, GraphRanker
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
C = {"blue": "#1f4e79", "orange": "#c55a11", "green": "#548235", "grey": "#7f7f7f", "light": "#dbe5f1", "red": "#a50f15"}

# ---------------------------------------------------------------- Fig 1: pipeline
fig, ax = plt.subplots(figsize=(7.0, 1.9)); ax.axis("off")
boxes = [("Paper\n(LaTeX source)", C["grey"]), ("Section split\n+ sentence\nsegmentation", C["grey"]),
         ("Sentence graph\nnodes = sentences\nedges = semantic\nsimilarity", C["blue"]),
         ("Centrality\nranking\n(Indegree /\nPageRank)", C["blue"]),
         ("Four-part core\ncontent\n(problem, method,\nresults, contributions)", C["orange"]),
         ("BERT / LMCBert\nclassifier", C["green"]), ("Accept /\nReject", C["green"])]
n = len(boxes); w = 0.125; gap = (1 - n * w) / (n - 1)
for i, (t, col) in enumerate(boxes):
    x = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.15), w, 0.7, boxstyle="round,pad=0.01", fc="white", ec=col, lw=1.4))
    ax.text(x + w / 2, 0.5, t, ha="center", va="center", fontsize=6.6, color="black")
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + w, 0.5), (x + w + gap, 0.5), arrowstyle="-|>", mutation_scale=9, color="black", lw=0.9))
ax.text(0.5, 0.0, "core-content extraction (this work, adapted from the graph-based WSD method of Abdalgader and Al Shibli)",
        ha="center", va="bottom", fontsize=6.5, style="italic", color=C["blue"])
ax.plot([0.13, 0.75], [0.05, 0.05], color=C["blue"], lw=0.8)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.05, 0.95)
fig.savefig("fig_pipeline_old.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- data for graph / heatmap: one real paper
rows = [json.loads(l) for l in open("../data/splits_small/val.jsonl", encoding="utf-8")]
r = None
for cand in rows:
    s = sentences(cand.get("sections", {}).get("method", ""))
    if 14 <= len(s) <= 22:
        r = cand; break
sents = sentences(r["sections"]["method"])[:16]
rk = GraphRanker(sim="tfidf", centrality="indegree")
S = rk.similarity_matrix(sents)
deg = S.sum(1)
# word-level overlap (Jaccard on content words) for contrast
from summarize import _content_words
W = np.zeros_like(S)
cw = [set(_content_words(s)) for s in sents]
for i in range(len(sents)):
    for j in range(len(sents)):
        if i != j and cw[i] and cw[j]:
            W[i, j] = len(cw[i] & cw[j]) / len(cw[i] | cw[j])
print("paper", r["id"], r["title"][:80], "| sentences", len(sents))

# ---------------------------------------------------------------- Fig 2: sentence graph
import networkx as nx
G = nx.Graph()
for i in range(len(sents)): G.add_node(i)
thr = np.percentile(S[S > 0], 60)
for i in range(len(sents)):
    for j in range(i + 1, len(sents)):
        if S[i, j] > thr: G.add_edge(i, j, weight=S[i, j])
pos = nx.spring_layout(G, seed=3, weight="weight", k=0.9)
top = set(np.argsort(-deg)[:4])
fig, ax = plt.subplots(figsize=(3.4, 2.9)); ax.axis("off")
ew = [G[u][v]["weight"] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, ax=ax, width=[0.4 + 4 * w for w in ew], edge_color="#9dbbd9", alpha=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[120 + 900 * d / deg.max() for d in deg],
                       node_color=[C["orange"] if i in top else C["light"] for i in G.nodes()],
                       edgecolors=C["blue"], linewidths=0.8)
nx.draw_networkx_labels(G, pos, ax=ax, labels={i: f"s{i+1}" for i in G.nodes()}, font_size=6)
ax.set_title("Method section of an arXiv paper: 16 sentence nodes;\nnode size = weighted Indegree, orange = 4 selected sentences", fontsize=7)
fig.savefig("fig_graph.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: heatmaps word-level vs sentence-level
k = 10
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, M, t in zip(axes, [W[:k, :k], S[:k, :k]], ["(a) Word-level overlap (Jaccard)", "(b) Sentence-level similarity (graph edge weights)"]):
    M = M.copy(); np.fill_diagonal(M, 1.0)
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([f"s{i+1}" for i in range(k)], rotation=45); ax.set_yticklabels([f"s{i+1}" for i in range(k)])
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{M[i,j]:.2f}" if i != j else "1", ha="center", va="center", fontsize=5.2, color="black" if M[i, j] < 0.6 else "white")
    ax.set_title(t, fontsize=7.5)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
fig.savefig("fig_heatmap.pdf", bbox_inches="tight"); plt.close(fig)
print("mean off-diag word-level %.3f sentence-level %.3f; zero edges word %.0f%% sent %.0f%%" % (
    W[:k,:k][~np.eye(k, dtype=bool)].mean(), S[:k,:k][~np.eye(k, dtype=bool)].mean(),
    100*(W[:k,:k][~np.eye(k,dtype=bool)]==0).mean(), 100*(S[:k,:k][~np.eye(k,dtype=bool)]==0).mean()))

# ---------------------------------------------------------------- Fig 4: validation curves (real logs)
curves = {
 "BERT, abstract (baseline)":            [0.625, 0.611, 0.618, 0.612],
 "BERT, lexical extractive summary":     [0.626, 0.618, 0.609, 0.588],
 "BERT, graph-centrality summary (ours)":[0.605, 0.616, 0.620, 0.630, 0.606, 0.594, 0.594],
 "LMCBert $\\lambda$=0.5, extractive":   [0.593, 0.602, 0.596, 0.605, 0.612, 0.603, 0.602, 0.583],
}
styles = {"BERT, abstract (baseline)": (C["grey"], "--", "s"), "BERT, lexical extractive summary": (C["orange"], "-", "o"),
          "BERT, graph-centrality summary (ours)": (C["blue"], "-", "D"), "LMCBert $\\lambda$=0.5, extractive": (C["green"], ":", "^")}
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for name, ys in curves.items():
    col, ls, mk = styles[name]
    ax.plot(range(1, len(ys) + 1), [100 * y for y in ys], ls, marker=mk, ms=3.5, lw=1.2, color=col, label=name)
    b = int(np.argmax(ys)); ax.plot(b + 1, 100 * ys[b], marker=mk, ms=7, color=col, mfc="none", mew=1.2)
ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (%)"); ax.set_ylim(57.5, 64.5); ax.grid(alpha=0.3)
ax.legend(loc="lower left", frameon=False)
fig.savefig("fig_curves.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: results bars
names = ["BERT\nabstract", "LMCBert\n$\\lambda$=0.1", "LMCBert\n$\\lambda$=0.5", "BERT\nextractive\nsummary", "BERT\nCEA\nsummary"]
acc = [61.9, 64.6, 65.5, 65.6, 66.5]; auc = [68.3, 69.2, 70.2, 69.5, 72.2]
x = np.arange(len(names)); w = 0.36
fig, ax = plt.subplots(figsize=(3.4, 2.4))
b1 = ax.bar(x - w/2, acc, w, color=C["blue"], label="accuracy")
b2 = ax.bar(x + w/2, auc, w, color=C["orange"], label="AUC")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.5); ax.set_ylim(58, 75); ax.set_ylabel("test score (%)")
ax.axhline(61.9, color=C["grey"], lw=0.6, ls="--"); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", alpha=0.3)
fig.savefig("fig_results.pdf", bbox_inches="tight"); plt.close(fig)
json.dump({"paper_id": r["id"], "title": r["title"], "sents": sents, "deg": deg.tolist(), "top": sorted(int(i) for i in top)}, open("case.json", "w"), indent=1)
print("done")
