"""
Stream the raw AAPR dataset (Yang et al. 2018) and produce a compact,
readable JSONL with one record per paper.

Fields written per paper:
  id, title, authors, category, venue, label (1 = accepted, 0 = rejected),
  abstract_raw (the dataset's pre-cleaned, stopword-stripped abstract),
  abstract (readable abstract recovered from the LaTeX where possible),
  sections: {intro, related, method, experiment, conclusion}  (readable text),
  n_figures, n_tables, n_equations, n_citations, n_words (cheap structural features)

Label rule follows the original AAPR code: venue in {'CoRR', 'No'} -> rejected.
"""
import ijson, json, re, sys, os, argparse, collections

RAW_DIR = os.environ.get("AAPR_RAW", "data/AAPR_Dataset")
FILES = ["data1", "data2", "data3", "data4"]

# ---------------------------------------------------------------- LaTeX cleaning
_cmd_keep_arg = re.compile(r"\\(?:textbf|textit|emph|texttt|underline|textsc|mbox|hbox|text|textrm)\s*\{")
_cite = re.compile(r"~?\\(?:cite|citep|citet|citealp|citeauthor|citeyear|newcite|ref|eqref|autoref|cref|Cref|label|footnote|url|href|footnotemark|index)\*?(?:\[[^\]]*\])?\{[^{}]*\}")
_env_drop = re.compile(r"\\begin\{(equation|equation\*|align|align\*|eqnarray|eqnarray\*|figure|figure\*|table|table\*|tabular|tabular\*|algorithm|algorithmic|verbatim|lstlisting|tikzpicture|thebibliography|multline|gather|displaymath|subfigure|wrapfigure|minipage)\}.*?\\end\{\1\}", re.S)
_display_math = re.compile(r"\\\[.*?\\\]|\$\$.*?\$\$", re.S)
_inline_math = re.compile(r"\$[^$]*\$")
_comment = re.compile(r"(?<!\\)%.*")
_generic_cmd = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?")
_ws = re.compile(r"[ \t\r\f\v]+")
_multi_nl = re.compile(r"\n\s*\n+")


def strip_comments(tex: str) -> str:
    return "\n".join(_comment.sub("", ln) for ln in tex.split("\n"))


def clean_tex(tex: str) -> str:
    """Turn a LaTeX fragment into readable prose (formulas -> [MATH], envs dropped)."""
    tex = _env_drop.sub(" ", tex)
    tex = _display_math.sub(" [MATH] ", tex)
    tex = _inline_math.sub(" [MATH] ", tex)
    tex = _cite.sub("", tex)
    tex = _cmd_keep_arg.sub("{", tex)
    tex = _generic_cmd.sub(" ", tex)
    tex = tex.replace("{", "").replace("}", "").replace("~", " ")
    tex = tex.replace("\\\\", " ").replace("\\", "")
    tex = _ws.sub(" ", tex)
    tex = _multi_nl.sub("\n", tex)
    return tex.strip()


_section_re = re.compile(r"\\(?:section|chapter)\*?\s*\{([^}]*)\}", re.I)
_caption = re.compile(r"\\caption\*?\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_eq_env = re.compile(r"\\begin\{(?:equation|align|eqnarray|multline|gather)\*?\}(.*?)\\end\{(?:equation|align|eqnarray|multline|gather)\*?\}", re.S)

SECTION_MAP = [
    ("intro", re.compile(r"introduction|overview|motivation", re.I)),
    ("related", re.compile(r"related|background|prior work|previous work|literature", re.I)),
    ("experiment", re.compile(r"experiment|evaluation|result|empirical|analysis|case stud|benchmark|simulation", re.I)),
    ("conclusion", re.compile(r"conclu|discussion|future work|summary", re.I)),
]


def split_sections(body: str):
    """Return dict of readable text per coarse section; unmatched sections -> 'method'."""
    out = collections.defaultdict(list)
    matches = list(_section_re.finditer(body))
    if not matches:
        out["method"].append(body)
    else:
        for i, m in enumerate(matches):
            title = m.group(1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            chunk = body[start:end]
            key = "method"
            for name, rx in SECTION_MAP:
                if rx.search(title):
                    key = name
                    break
            out[key].append(chunk)
    return {k: clean_tex("\n".join(v)) for k, v in out.items()}


_abstract_env = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S | re.I)
_abstract_cmd = re.compile(r"\\abstract\s*\{(.*?)\}\s*\n", re.S | re.I)
_begin_doc = re.compile(r"\\begin\{document\}", re.I)
_end_doc = re.compile(r"\\end\{document\}", re.I)
_bib_start = re.compile(r"\\begin\{thebibliography\}|\\bibliography\{", re.I)


def process_tex(tex_lines):
    tex = "".join(tex_lines) if isinstance(tex_lines, list) else str(tex_lines)
    tex = strip_comments(tex)
    feats = {
        "n_figures": len(re.findall(r"\\begin\{figure", tex)),
        "n_tables": len(re.findall(r"\\begin\{table", tex)),
        "n_equations": len(re.findall(r"\\begin\{(?:equation|align|eqnarray|multline|gather)", tex)) + len(re.findall(r"\$\$|\\\[", tex)),
        "n_citations": len(re.findall(r"\\cite", tex)),
    }
    # --- multimodal-lite: captions of figures/tables and formula statistics (conclusion rec. #2)
    captions = [clean_tex(c) for c in _caption.findall(tex)]
    captions = [c for c in captions if len(c.split()) >= 3][:40]
    eqs = _eq_env.findall(tex) + [m for m in _display_math.findall(tex)]
    eq_lens = [len(e) for e in eqs]
    feats.update({
        "n_captions": len(captions),
        "eq_mean_len": float(sum(eq_lens) / len(eq_lens)) if eq_lens else 0.0,
        "eq_max_len": max(eq_lens) if eq_lens else 0,
        "n_inline_math": len(_inline_math.findall(tex)),
        "n_theorems": len(re.findall(r"\\begin\{(?:theorem|lemma|proposition|corollary|definition)\}", tex)),
        "n_algorithms": len(re.findall(r"\\begin\{algorithm", tex)),
        "n_sections": len(_section_re.findall(tex)),
        "n_subfigures": len(re.findall(r"\\includegraphics", tex)),
    })
    m = _abstract_env.search(tex) or _abstract_cmd.search(tex)
    abstract = clean_tex(m.group(1)) if m else ""
    body = tex
    bd = _begin_doc.search(body)
    if bd:
        body = body[bd.end():]
    ed = _end_doc.search(body)
    if ed:
        body = body[: ed.start()]
    bs = _bib_start.search(body)
    if bs:
        body = body[: bs.start()]
    body = _abstract_env.sub(" ", body)
    sections = split_sections(body)
    feats["n_words"] = sum(len(v.split()) for v in sections.values())
    feats["captions"] = captions
    return abstract, sections, feats


def label_from_venue(venue: str) -> int:
    return 0 if venue.strip() in ("CoRR", "No", "") else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/aapr_clean.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="stop after N papers (debug)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    venues = collections.Counter()
    labels = collections.Counter()
    with open(args.out, "w", encoding="utf-8") as fo:
        for fn in FILES:
            path = os.path.join(RAW_DIR, fn)
            with open(path, "rb") as f:
                for pid, rec in ijson.kvitems(f, "", use_float=True):
                    venue = str(rec.get("venue", ""))
                    lab = label_from_venue(venue)
                    try:
                        abstract, sections, feats = process_tex(rec.get("tex_data", []))
                    except Exception as e:  # keep going on pathological LaTeX
                        abstract, sections, feats = "", {}, {}
                    out = {
                        "id": pid,
                        "title": rec.get("title", ""),
                        "authors": rec.get("authors", ""),
                        "category": rec.get("category", ""),
                        "venue": venue,
                        "label": lab,
                        "abstract_raw": rec.get("abstract", ""),
                        "abstract": abstract,
                        "sections": sections,
                        **feats,
                    }
                    fo.write(json.dumps(out, ensure_ascii=False) + "\n")
                    venues[venue] += 1
                    labels[lab] += 1
                    n += 1
                    if n % 2000 == 0:
                        print(f"{n} papers...", file=sys.stderr, flush=True)
                    if args.limit and n >= args.limit:
                        break
            if args.limit and n >= args.limit:
                break
    print(f"done: {n} papers; labels={dict(labels)}", file=sys.stderr)
    print("top venues:", venues.most_common(25), file=sys.stderr)


if __name__ == "__main__":
    main()
