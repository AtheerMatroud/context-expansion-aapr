"""
Step 3 — Extract the "core content" of each paper (LMCBert Sec. III-A) at zero cost.

The paper used GPT-4o with a prompt asking for the research problem, method,
key experimental results and contributions, compressed to 20-30 % of the paper.
Three free backends are provided:

  extractive  (default)  no model download, runs anywhere. Structured
                          TF-IDF/centrality extractive summary, section-aware,
                          same four-part structure as the paper's prompt.
  hf                      any open instruct LLM via transformers, e.g.
                          Qwen/Qwen2.5-1.5B-Instruct (needs GPU + HF access).
  ollama                  local Ollama server (e.g. `ollama run qwen2.5:3b`).
  graph                   sentence-graph + centrality ranking (Abdalgader & Al Shibli 2021
                          adapted to sentences); --sim tfidf|wordnet|embed, --centrality indegree|pagerank

Output: the input JSONL with a new field `summary` (and `summary_backend`).

Usage:
  python summarize.py --inp data/splits_small/train.jsonl --out data/splits_small/train.sum.jsonl
  python summarize.py --inp ... --out ... --backend hf --model Qwen/Qwen2.5-3B-Instruct
"""
import json, re, argparse, sys, os, math, time
import numpy as np

# --------------------------------------------------------------------------- prompt
PROMPT = (
    "You are an expert reviewer. Read the following academic paper and extract its core "
    "content as a structured summary with exactly these four parts:\n"
    "1. Research problem: what problem the paper addresses and why it matters.\n"
    "2. Method: the proposed approach and its key technical ideas.\n"
    "3. Key experimental results: datasets, baselines, main quantitative findings.\n"
    "4. Contributions: the novel contributions relative to prior work.\n"
    "Ignore generic background, notation and boilerplate. Be concrete and faithful to the "
    "paper. Target length: about {target_words} words (20-30% of the original).\n\n"
    "Title: {title}\n\nPaper:\n{body}\n\nStructured summary:"
)

SECTION_ORDER = ["intro", "related", "method", "experiment", "conclusion"]


def paper_body(r, max_words=None):
    parts = [f"Abstract: {r['abstract']}"]
    for s in SECTION_ORDER:
        t = r.get("sections", {}).get(s, "")
        if t:
            parts.append(f"[{s.upper()}]\n{t}")
    body = "\n\n".join(parts)
    if max_words:
        w = body.split()
        if len(w) > max_words:
            body = " ".join(w[:max_words])
    return body


# --------------------------------------------------------------------------- extractive
_sent_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in _sent_split.split(text) if s.strip()]
    # drop junk sentences: too short, mostly math placeholders, no letters
    good = []
    for s in sents:
        w = s.split()
        if len(w) < 6 or len(w) > 80:
            continue
        if s.count("[MATH]") > max(2, len(w) // 4):
            continue
        if sum(c.isalpha() for c in s) < 0.5 * len(s):
            continue
        good.append(s)
    return good


CUE_WORDS = {
    "intro": ["we propose", "in this paper", "this paper", "we present", "we introduce", "we show", "our contribution", "problem", "we study", "we address", "goal"],
    "method": ["we propose", "our method", "our approach", "the key idea", "we define", "algorithm", "model", "framework", "we formulate"],
    "experiment": ["outperform", "achieve", "accuracy", "improve", "compared", "baseline", "results show", "dataset", "%", "significant", "state-of-the-art"],
    "conclusion": ["we proposed", "we presented", "we have", "contribution", "in this paper", "future work", "we showed", "novel"],
}


def rank_sentences(sents, section, k):
    """TF-IDF centrality + positional prior + cue-phrase bonus; returns top-k in document order."""
    if not sents or k <= 0:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        X = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(sents)
        sim = (X @ X.T).toarray()
        centrality = sim.sum(1) / max(1, len(sents))
    except ValueError:
        centrality = np.ones(len(sents))
    pos = np.array([1.0 / (1 + 0.15 * i) for i in range(len(sents))])  # lead bias
    cues = np.array([1.0 + 0.5 * sum(c in s.lower() for c in CUE_WORDS.get(section, [])) for s in sents])
    score = centrality * pos * cues
    idx = np.argsort(-score)[:k]
    return [sents[i] for i in sorted(idx)]


def lead_ranker(sents, section, k):
    """Control: the first k sentences of the section (lead-k), same budget as the other extractors."""
    return sents[:k] if k > 0 else []


def lead_order(sents, section):
    return list(range(len(sents)))


def lexical_order(sents, section):
    """Ranking order (best first) of the lexical TF-IDF/positional/cue ranker."""
    if not sents:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        X = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(sents)
        centrality = (X @ X.T).toarray().sum(1) / max(1, len(sents))
    except ValueError:
        centrality = np.ones(len(sents))
    pos = np.array([1.0 / (1 + 0.15 * i) for i in range(len(sents))])
    cues = np.array([1.0 + 0.5 * sum(c in s.lower() for c in CUE_WORDS.get(section, [])) for s in sents])
    return list(np.argsort(-(centrality * pos * cues)))


class RandomRanker:
    """Control: k sentences drawn at random from the section (document order kept), same budget.
    The draw is seeded by the paper id so that the summaries are reproducible."""

    def __init__(self, seed=0):
        self.seed = seed
        self.paper_id = ""

    def __call__(self, sents, section, k):
        if not sents or k <= 0:
            return []
        idx = self.order(sents, section)[:k]
        return [sents[i] for i in sorted(idx)]

    def order(self, sents, section):
        import zlib
        rng = np.random.RandomState(zlib.crc32(f"{self.paper_id}|{section}|{self.seed}".encode()) % (2 ** 31))
        return list(rng.permutation(len(sents)))


_TOK = None

# Explicit publication-status statements (the masking rule of the leakage audit; applied identically to train/val/test).
# Ordinary venue names and references to previous publications are NOT masked.
SELF = r"(?:this|the present|our|the current) (?:paper|work|article|manuscript|submission|study|version|draft)"
STATUS_VERBS = r"(?:accepted|to appear|appears|appeared|appearing|published|presented|forthcoming|in press|under review|under submission|submitted|rejected)"
STATUS_PATTERNS = [
    # self-referential status statements: "this paper was accepted at ...", "an extended version of this work appeared in ..."
    rf"\b{SELF}\b[^.;]{{0,40}}?\b(?:is|was|has been|will be|will appear|appears|appeared)\b[^.;]{{0,10}}?\b{STATUS_VERBS}\b[^.;]{{0,120}}",
    rf"\b(?:a |an )?(?:preliminary|shorter?|conference|extended|earlier|journal|full|longer) version of {SELF}\b[^.;]{{0,120}}",
    # title-page style notes without a cited subject
    r"(?:^|(?<=[.!?]\s))(?-i:Accepted (?:at|to|for|in)|To appear in|In Proceedings of|Published in|Presented at|Submitted to|Under review (?:at|for))\b[^.;]{0,120}",
    r"\bcamera[- ]ready\b[^.;]{0,60}",
    r"\bcopyright\b[^.;]{0,40}\b(?:ieee|acm|springer|elsevier|aaai|association for computational linguistics|iclr|icml|neurips|nips)\b[^.;]{0,60}",
    r"\b(?:personal use of this material is permitted|permission from ieee must be obtained)\b[^.;]{0,120}",
]
# venue mentions that are NOT masked but counted by the audit (references to other work, named venues)
VENUE_MENTION = re.compile(r"\b(?:nips|neurips|icml|iclr|cvpr|iccv|eccv|acl|emnlp|naacl|coling|aaai|ijcai|kdd|sigir|www|icra|iros|interspeech|icassp|siggraph|chi|uai|aistats|colt|proceedings of|in press|to appear|accepted|camera[- ]ready|under review|submitted)\b", re.I)
_STATUS_RE = re.compile("|".join(STATUS_PATTERNS), re.I)


def mask_status(text, token="[STATUS]"):
    """Replace explicit publication-status statements by a placeholder; returns (text, number of replacements)."""
    return _STATUS_RE.subn(token, text)


def _bert_tokens(text):
    """Number of BERT (bert-base-uncased) word pieces in `text`, without special tokens."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("bert-base-uncased")
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


LABELS = {"intro": "Research problem:", "method": "Method:", "experiment": "Key results:", "conclusion": "Contributions:"}


def extractive_summary_tokens(r, order_fn, token_budget=510, ratio=0.25, return_stats=False):
    """Four-part summary with the budget enforced directly in BERT word pieces (bert-base-uncased).

    L_tok   = word pieces of the abstract plus the five section fields (intro, related, method, experiment, conclusion).
    budget  = floor(min(token_budget, max(150, ratio * L_tok))) - (word pieces of the four labels).
              token_budget = 510 leaves room for [CLS] and [SEP] in a 512-token input.
    share_k = floor(alloc_k * budget) with alloc = 25 / 30 / 25 / 20 % for intro / method / experiment / conclusion.
    Pass 1 (first-fit by rank): the sentences of each part are considered in ranking order and each is added if it
              fits the remaining share of its part (a long high-ranked sentence can be skipped in favour of shorter
              lower-ranked ones).
    Pass 2 (round-robin redistribution): the unused remainder of the whole budget is offered to the parts in the fixed
              order intro, method, experiment, conclusion, one sentence per part per round, by rank, until no sentence fits.
    A part that is still empty after pass 2 is omitted together with its label; its label reservation is NOT reclaimed
              (the budget is simply under-used), so the result can only be shorter than planned.
    Within each part the chosen sentences are emitted in document order. The returned text never exceeds
    `token_budget` word pieces; the caller verifies this with the same tokenizer.
    """
    secs = r.get("sections", {})
    alloc = {"intro": 0.25, "method": 0.30, "experiment": 0.25, "conclusion": 0.20}
    texts = {"intro": secs.get("intro", "") or r["abstract"], "method": secs.get("method", ""),
             "experiment": secs.get("experiment", ""), "conclusion": secs.get("conclusion", "") or r["abstract"]}
    sents = {k: sentences(v) for k, v in texts.items()}
    paper_tokens = sum(_bert_tokens(secs.get(k, "")) for k in SECTION_ORDER) + _bert_tokens(r["abstract"])
    label_tokens = sum(_bert_tokens(v) for v in LABELS.values())
    budget = int(math.floor(min(token_budget, max(150, ratio * paper_tokens)))) - label_tokens
    keys = ("intro", "method", "experiment", "conclusion")
    orders = {k: list(order_fn(sents[k], k)) for k in keys}
    ntok = {k: [_bert_tokens(x) for x in sents[k]] for k in keys}
    chosen = {k: [] for k in keys}
    used = {k: 0 for k in keys}
    # pass 1: first-fit by rank within each part's share
    for k in keys:
        share = int(math.floor(alloc[k] * budget))
        for i in orders[k]:
            if used[k] + ntok[k][i] <= share:
                chosen[k].append(i); used[k] += ntok[k][i]
    # pass 2: round-robin redistribution of the unused remainder
    leftover = budget - sum(used.values())
    progress = True
    while leftover > 0 and progress:
        progress = False
        for k in keys:
            for i in orders[k]:
                if i not in chosen[k] and ntok[k][i] <= leftover:
                    chosen[k].append(i); used[k] += ntok[k][i]; leftover -= ntok[k][i]; progress = True
                    break
    parts = [LABELS[k] + " " + " ".join(sents[k][i] for i in sorted(chosen[k])) for k in keys if chosen[k]]
    text = "\n".join(parts)
    if return_stats:
        return text, {"L_tok": paper_tokens, "budget": budget + label_tokens, "content_tokens": dict(used),
                      "n_sents": {k: len(v) for k, v in chosen.items()}, "total_tokens": _bert_tokens(text)}
    return text


def global_summary(r, ranker, ratio=0.25, max_words=420, token_budget=0):
    """Control: one sentence graph over the four sections together (same text sources and abstract fallbacks as the
    four-part summary), ranked once, selected under the same total budget, emitted in document order, no labels.
    Word budget: k = max(1, floor(B / 25)) sentences with B = min(max_words, max(120, ratio * L)); then cut to max_words.
    Token budget: sentences added first-fit by rank while they fit floor(min(token_budget, max(150, ratio * L_tok)))."""
    secs = r.get("sections", {})
    texts = [secs.get("intro", "") or r["abstract"], secs.get("method", ""), secs.get("experiment", ""),
             secs.get("conclusion", "") or r["abstract"]]
    sents = [s for t in texts for s in sentences(t)]
    if not sents:
        return ""
    order = list(ranker.order(sents, "all"))
    if token_budget:
        paper_tokens = sum(_bert_tokens(secs.get(k, "")) for k in SECTION_ORDER) + _bert_tokens(r["abstract"])
        budget = int(math.floor(min(token_budget, max(150, ratio * paper_tokens))))
        ntok = [_bert_tokens(s) for s in sents]
        chosen, used = [], 0
        for i in order:
            if used + ntok[i] <= budget:
                chosen.append(i); used += ntok[i]
        return " ".join(sents[i] for i in sorted(chosen))
    total = sum(len(secs.get(s, "").split()) for s in SECTION_ORDER) + len(r["abstract"].split())
    budget = int(min(max_words, max(120, ratio * total)))
    k = max(1, budget // 25)
    text = " ".join(sents[i] for i in sorted(order[:k]))
    w = text.split()
    return " ".join(w[:max_words]) if len(w) > max_words else text


def extractive_summary(r, ratio=0.25, max_words=420, ranker=None, return_uncut=False):
    """Four-part structured summary. `ranker(sents, section, k)` picks k sentences; default = TF-IDF/positional."""
    rank = ranker or rank_sentences
    secs = r.get("sections", {})
    total = sum(len(secs.get(s, "").split()) for s in SECTION_ORDER) + len(r["abstract"].split())
    budget = int(min(max_words, max(120, ratio * total)))
    # budget split mirrors the four-part prompt
    alloc = {"intro": 0.25, "method": 0.30, "experiment": 0.25, "conclusion": 0.20}
    out = []
    out.append("Research problem: " + " ".join(rank(sentences(secs.get("intro", "") or r["abstract"]), "intro", max(1, int(alloc["intro"] * budget / 25)))))
    out.append("Method: " + " ".join(rank(sentences(secs.get("method", "")), "method", max(1, int(alloc["method"] * budget / 25)))))
    out.append("Key results: " + " ".join(rank(sentences(secs.get("experiment", "")), "experiment", max(1, int(alloc["experiment"] * budget / 25)))))
    concl = secs.get("conclusion", "") or r["abstract"]
    out.append("Contributions: " + " ".join(rank(sentences(concl), "conclusion", max(1, int(alloc["conclusion"] * budget / 25)))))
    text = "\n".join(o for o in out if len(o.split()) > 2)
    uncut = text
    w = text.split()
    if len(w) > max_words:
        text = " ".join(w[:max_words])
    if return_uncut:
        return text, uncut
    return text



# --------------------------------------------------------------------------- graph-centrality ranker
# Adaptation of Abdalgader & Al Shibli (2021), "Context expansion approach for graph-based
# word sense disambiguation", ESWA 168:114313, to core-content extraction: nodes are the
# sentences of a section, edges carry a *sentence-level* semantic similarity, and a graph
# centrality measure (Indegree / PageRank, their two best) ranks the nodes.  Three edge
# similarities are offered:
#   tfidf    lexical cosine (baseline; equals our default extractor without priors)
#   wordnet  the paper's measure: Li et al. (2006) sentence similarity over WordNet path
#            similarity (Rada et al. 1989), computed on content words with a word-pair cache
#   embed    cosine of sentence embeddings from a small open model (default all-MiniLM-L6-v2);
#            same graph, but edge weights that know scientific vocabulary
_WN_CACHE = {}
_STOP = None


def _content_words(sent):
    global _STOP
    if _STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _STOP = set(ENGLISH_STOP_WORDS)
    return [w for w in re.findall(r"[a-z][a-z\-]+", sent.lower()) if w not in _STOP and len(w) > 2 and w != "math"]


def _wn_word_sim(a, b):
    """Rada et al. path similarity between the best-matching synsets of two words (cached)."""
    if a == b:
        return 1.0
    key = (a, b) if a < b else (b, a)
    v = _WN_CACHE.get(key)
    if v is not None:
        return v
    from nltk.corpus import wordnet as wn
    best = 0.0
    sa, sb = wn.synsets(a)[:3], wn.synsets(b)[:3]  # first senses = most frequent (as in the paper)
    for x in sa:
        for y in sb:
            if x.pos() != y.pos():
                continue
            p = x.path_similarity(y)
            if p and p > best:
                best = p
    _WN_CACHE[key] = best
    return best


def _li_sentence_sim(wa, wb):
    """Li, McLean, Bandar et al. (2006) semantic-vector cosine, as used in Sec. 4.3 of the paper."""
    if not wa or not wb:
        return 0.0
    union = sorted(set(wa) | set(wb))
    sa, sb = set(wa), set(wb)
    va = np.array([1.0 if w in sa else max((_wn_word_sim(w, x) for x in sa), default=0.0) for w in union])
    vb = np.array([1.0 if w in sb else max((_wn_word_sim(w, x) for x in sb), default=0.0) for w in union])
    va[va < 0.2] = 0.0; vb[vb < 0.2] = 0.0  # small path similarities are noise
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na > 0 and nb > 0 else 0.0


class GraphRanker:
    def __init__(self, sim="embed", centrality="indegree", model="sentence-transformers/all-MiniLM-L6-v2",
                 damping=0.8, use_prior=False, max_words_per_sent=25, max_sents=0,
                 pacsum_l1=-2.0, pacsum_l2=1.0, pacsum_beta=0.6, sim_words=0, stable_ties=False):
        """sim_words > 0: for tfidf/embed, the similarity is computed on the first `sim_words` content words of each
        sentence (the full sentence is still passed to the classifier); matches the WordNet pruning.
        stable_ties: break equal scores by document order (stable sort) instead of numpy's unstable default."""
        self.sim_words, self.stable_ties = sim_words, stable_ties
        self.sim, self.centrality, self.damping, self.use_prior = sim, centrality, damping, use_prior
        self.pacsum_l1, self.pacsum_l2, self.pacsum_beta = pacsum_l1, pacsum_l2, pacsum_beta
        self.max_words_per_sent, self.max_sents = max_words_per_sent, max_sents
        self.encoder = None
        if sim == "embed":
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model)
        elif sim == "wordnet":
            import nltk
            try:
                from nltk.corpus import wordnet as wn; wn.synsets("test")
            except LookupError:
                nltk.download("wordnet", quiet=True)

    def similarity_matrix(self, sents):
        n = len(sents)
        if self.sim_words and self.sim in ("tfidf", "embed"):
            sents = [" ".join(_content_words(s)[: self.sim_words]) or s for s in sents]
        if self.sim == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                X = TfidfVectorizer(stop_words="english").fit_transform(sents)
                S = (X @ X.T).toarray()
            except ValueError:
                S = np.eye(n)
        elif self.sim == "embed":
            E = self.encoder.encode(sents, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            S = E @ E.T
        else:  # wordnet
            words = [_content_words(s)[: self.max_words_per_sent] for s in sents]
            S = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    S[i, j] = S[j, i] = _li_sentence_sim(words[i], words[j])
        np.fill_diagonal(S, 0.0)
        S = np.clip(S, 0.0, None)
        return S

    def scores(self, S):
        n = S.shape[0]
        if self.centrality == "pacsum":
            # position-aware directed centrality of Zheng and Lapata (2019), eqs. (2), (5), (6):
            # edge weights are shifted by min + beta (max - min) and clipped at zero, then
            # score_i = l1 * sum_{j<i} E_ij + l2 * sum_{j>i} E_ij  (l1 negative penalises similarity to earlier sentences)
            off = S[~np.eye(n, dtype=bool)]
            if off.size == 0:
                return np.zeros(n)
            thr = off.min() + self.pacsum_beta * (off.max() - off.min())
            E = np.clip(S - thr, 0.0, None)
            np.fill_diagonal(E, 0.0)
            lower = np.tril(E, -1).sum(1)   # j < i
            upper = np.triu(E, 1).sum(1)    # j > i
            return self.pacsum_l1 * lower + self.pacsum_l2 * upper
        if self.centrality == "pagerank":  # weighted PageRank, eq. (9) of the paper
            colsum = S.sum(0); colsum[colsum == 0] = 1.0
            M = S / colsum  # column-stochastic transition matrix
            pr = np.ones(n) / n
            for _ in range(100):
                new = (1 - self.damping) / n + self.damping * (M @ pr)
                if np.abs(new - pr).sum() < 1e-6:
                    pr = new; break
                pr = new
            return pr
        return S.sum(1)  # weighted Indegree, eq. (10)

    def __call__(self, sents, section, k):
        if not sents or k <= 0:
            return []
        if len(sents) == 1:
            return sents
        if self.max_sents and len(sents) > self.max_sents:  # cap graph size (wordnet mode is O(n^2 w^2))
            sents = sents[: self.max_sents]
        S = self.similarity_matrix(sents)
        sc = self.scores(S)
        if self.use_prior:  # optional: keep the positional / cue-phrase priors of the default extractor
            pos = np.array([1.0 / (1 + 0.15 * i) for i in range(len(sents))])
            cues = np.array([1.0 + 0.5 * sum(c in s.lower() for c in CUE_WORDS.get(section, [])) for s in sents])
            sc = sc * pos * cues
        idx = self._argsort(sc)[:k]
        return [sents[i] for i in sorted(idx)]

    def _argsort(self, sc):
        # descending; with stable_ties equal scores keep document order, otherwise numpy's default (unstable) order
        return np.argsort(-sc, kind="stable") if self.stable_ties else np.argsort(-sc)

    def order(self, sents, section):
        """Ranking order (best first) over all sentences of the section."""
        if not sents:
            return []
        if len(sents) == 1:
            return [0]
        if self.max_sents and len(sents) > self.max_sents:
            sents = sents[: self.max_sents]
        S = self.similarity_matrix(sents)
        sc = self.scores(S)
        if self.use_prior:
            pos = np.array([1.0 / (1 + 0.15 * i) for i in range(len(sents))])
            cues = np.array([1.0 + 0.5 * sum(c in s.lower() for c in CUE_WORDS.get(section, [])) for s in sents])
            sc = sc * pos * cues
        return list(self._argsort(sc))

# --------------------------------------------------------------------------- LLM backends
class HFBackend:
    def __init__(self, model, max_new_tokens=512, max_input_words=6000):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.max_input_words = max_input_words

    def __call__(self, r, ratio):
        body = paper_body(r, self.max_input_words)
        target = int(ratio * len(body.split()))
        msgs = [{"role": "user", "content": PROMPT.format(target_words=target, title=r["title"], body=body)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


class OllamaBackend:
    def __init__(self, model, host="http://localhost:11434", max_input_words=6000):
        self.model, self.host, self.max_input_words = model, host, max_input_words

    def __call__(self, r, ratio):
        import urllib.request

        body = paper_body(r, self.max_input_words)
        target = int(ratio * len(body.split()))
        payload = json.dumps({"model": self.model, "prompt": PROMPT.format(target_words=target, title=r["title"], body=body), "stream": False, "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["response"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="extractive", choices=["extractive", "graph", "lead", "random", "hf", "ollama"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ratio", type=float, default=0.25, help="compression ratio (paper: 20-30%% best)")
    ap.add_argument("--max_words", type=int, default=420, help="cap for extractive (BERT sees 512 tokens)")
    ap.add_argument("--resume", action="store_true", help="skip ids already present in --out")
    ap.add_argument("--sim", default="embed", choices=["tfidf", "wordnet", "embed"], help="graph backend: edge similarity")
    ap.add_argument("--centrality", default="indegree", choices=["indegree", "pagerank", "pacsum"], help="graph backend: node scoring")
    ap.add_argument("--pacsum_l1", type=float, default=-2.0, help="pacsum: weight of edges to earlier sentences")
    ap.add_argument("--pacsum_l2", type=float, default=1.0, help="pacsum: weight of edges to later sentences")
    ap.add_argument("--pacsum_beta", type=float, default=0.6, help="pacsum: relative threshold in [0,1]")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--prior", action="store_true", help="graph backend: also apply positional/cue priors")
    ap.add_argument("--max_sents", type=int, default=0, help="graph backend: cap sentences per section (0 = all)")
    ap.add_argument("--max_words_per_sent", type=int, default=25, help="graph/wordnet: content words per sentence")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N input papers (0 = all)")
    ap.add_argument("--token_budget", type=int, default=0, help="enforce the four-part budget in BERT tokens (e.g. 510); 0 = word-based budget")
    ap.add_argument("--store_uncut", action="store_true", help="word budget only: also store the assembled text before the max_words cut as summary_uncut")
    ap.add_argument("--sim_words", type=int, default=0, help="graph tfidf/embed: compute similarity on the first N content words only (0 = full sentence); matches --max_words_per_sent of wordnet")
    ap.add_argument("--stable_ties", action="store_true", help="graph backend: break equal centrality scores by document order")
    ap.add_argument("--global_graph", action="store_true", help="graph backend control: one graph over the four sections, same total budget, no section labels")
    ap.add_argument("--mask_status", action="store_true", help="mask explicit publication-status statements in the sections and abstract before summarising (STATUS_PATTERNS)")
    ap.add_argument("--timing", action="store_true", help="print per-stage timings (clean+split, similarity/ranking, assembly) to stderr; GPU work synchronised")
    args = ap.parse_args()

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["id"])
    backend = None
    ranker = None
    if args.backend == "lead":
        ranker = lead_ranker
    if args.backend == "random":
        ranker = RandomRanker(seed=0)
    if args.backend == "graph":
        ranker = GraphRanker(args.sim, args.centrality, args.embed_model, use_prior=args.prior,
                             max_words_per_sent=args.max_words_per_sent, max_sents=args.max_sents,
                             pacsum_l1=args.pacsum_l1, pacsum_l2=args.pacsum_l2, pacsum_beta=args.pacsum_beta,
                             sim_words=args.sim_words, stable_ties=args.stable_ties)
    if args.backend == "hf":
        backend = HFBackend(args.model)
    elif args.backend == "ollama":
        backend = OllamaBackend(args.model)

    n = 0
    seen = 0
    timings = []
    with open(args.inp, encoding="utf-8") as f, open(args.out, "a" if args.resume else "w", encoding="utf-8") as fo:
        for line in f:
            seen += 1
            if args.limit and seen > args.limit:
                break
            r = json.loads(line)
            if r["id"] in done:
                continue
            if args.mask_status:
                nm = 0
                r["abstract"], c = mask_status(r["abstract"]); nm += c
                for k in list((r.get("sections") or {}).keys()):
                    r["sections"][k], c = mask_status(r["sections"][k]); nm += c
                r["n_status_masked"] = nm
            t_start = time.perf_counter()
            if backend is None:
                if isinstance(ranker, RandomRanker):
                    ranker.paper_id = str(r.get("id", ""))
                if args.global_graph:
                    assert args.backend == "graph", "--global_graph needs --backend graph"
                    r["summary"] = global_summary(r, ranker, args.ratio, args.max_words, args.token_budget)
                elif args.token_budget:
                    if ranker is None:
                        order_fn = lexical_order
                    elif ranker is lead_ranker:
                        order_fn = lead_order
                    else:
                        order_fn = ranker.order
                    r["summary"], st = extractive_summary_tokens(r, order_fn, args.token_budget, args.ratio, return_stats=True)
                    assert st["total_tokens"] <= args.token_budget, (r["id"], st["total_tokens"])
                    r["summary_tokens"] = st["total_tokens"]
                    r["summary_budget"] = st["budget"]
                elif args.store_uncut:
                    r["summary"], r["summary_uncut"] = extractive_summary(r, args.ratio, args.max_words, ranker, return_uncut=True)
                else:
                    r["summary"] = extractive_summary(r, args.ratio, args.max_words, ranker)
            else:
                try:
                    r["summary"] = backend(r, args.ratio)
                except Exception as e:
                    print(f"[warn] {r['id']}: {e}; falling back to extractive", file=sys.stderr)
                    r["summary"] = extractive_summary(r, args.ratio, args.max_words)
            r["summary_backend"] = (((f"graph:{args.sim}:{args.centrality}" if args.backend == "graph" else args.backend) if ranker else args.backend) + (f":tok{args.token_budget}" if args.token_budget else "") + (":global" if args.global_graph else "") + (":masked" if args.mask_status else "")) if backend is None else f"{args.backend}:{args.model}"
            if args.timing:
                try:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                except Exception:
                    pass
                timings.append(time.perf_counter() - t_start)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()
            n += 1
            if n % 200 == 0:
                print(f"{n} summarised", file=sys.stderr, flush=True)
    print(f"done: {n}", file=sys.stderr)
    if args.timing and timings:
        warm = timings[:50]; steady = timings[50:] or timings
        print(f"timing: per-paper wall time incl. cleaning, sentence splitting, similarity, ranking and assembly "
              f"(GPU synchronised); warm-up (first {len(warm)}): mean {np.mean(warm)*1000:.1f} ms; "
              f"steady state ({len(steady)}): mean {np.mean(steady)*1000:.1f} ms, median {np.median(steady)*1000:.1f} ms, "
              f"total {sum(timings):.1f} s for {len(timings)} papers", file=sys.stderr)


if __name__ == "__main__":
    main()
