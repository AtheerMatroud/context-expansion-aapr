"""
Core-content extraction for automatic academic paper rating.

The paper used GPT-4o with a prompt asking for the research problem, method,
key experimental results and contributions, compressed to 20-30 % of the paper.
Four backends are provided; `graph` is the context expansion approach used in the paper:

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
import json, re, argparse, sys, os, math
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


def extractive_summary(r, ratio=0.25, max_words=420, ranker=None):
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
    w = text.split()
    if len(w) > max_words:
        text = " ".join(w[:max_words])
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
                 damping=0.8, use_prior=False, max_words_per_sent=25, max_sents=0):
        self.sim, self.centrality, self.damping, self.use_prior = sim, centrality, damping, use_prior
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
        idx = np.argsort(-sc)[:k]
        return [sents[i] for i in sorted(idx)]

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
    ap.add_argument("--backend", default="extractive", choices=["extractive", "graph", "hf", "ollama"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ratio", type=float, default=0.25, help="compression ratio (paper: 20-30%% best)")
    ap.add_argument("--max_words", type=int, default=420, help="cap for extractive (BERT sees 512 tokens)")
    ap.add_argument("--resume", action="store_true", help="skip ids already present in --out")
    ap.add_argument("--sim", default="embed", choices=["tfidf", "wordnet", "embed"], help="graph backend: edge similarity")
    ap.add_argument("--centrality", default="indegree", choices=["indegree", "pagerank"], help="graph backend: node scoring")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--prior", action="store_true", help="graph backend: also apply positional/cue priors")
    ap.add_argument("--max_sents", type=int, default=0, help="graph backend: cap sentences per section (0 = all)")
    ap.add_argument("--max_words_per_sent", type=int, default=25, help="graph/wordnet: content words per sentence")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N input papers (0 = all)")
    args = ap.parse_args()

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["id"])
    backend = None
    ranker = None
    if args.backend == "graph":
        ranker = GraphRanker(args.sim, args.centrality, args.embed_model, use_prior=args.prior,
                             max_words_per_sent=args.max_words_per_sent, max_sents=args.max_sents)
    if args.backend == "hf":
        backend = HFBackend(args.model)
    elif args.backend == "ollama":
        backend = OllamaBackend(args.model)

    n = 0
    seen = 0
    with open(args.inp, encoding="utf-8") as f, open(args.out, "a" if args.resume else "w", encoding="utf-8") as fo:
        for line in f:
            seen += 1
            if args.limit and seen > args.limit:
                break
            r = json.loads(line)
            if r["id"] in done:
                continue
            if backend is None:
                r["summary"] = extractive_summary(r, args.ratio, args.max_words, ranker)
            else:
                try:
                    r["summary"] = backend(r, args.ratio)
                except Exception as e:
                    print(f"[warn] {r['id']}: {e}; falling back to extractive", file=sys.stderr)
                    r["summary"] = extractive_summary(r, args.ratio, args.max_words)
            r["summary_backend"] = (f"graph:{args.sim}:{args.centrality}" if ranker else args.backend) if backend is None else f"{args.backend}:{args.model}"
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()
            n += 1
            if n % 200 == 0:
                print(f"{n} summarised", file=sys.stderr, flush=True)
    print(f"done: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
