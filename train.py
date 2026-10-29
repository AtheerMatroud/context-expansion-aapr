"""
Train and evaluate the BERT classifier (with or without the MoCo branch of LMCBert) on a summary variant.

Paper hyper-parameters (Sec. IV-B): batch 32, 80 epochs, lr 2e-5, 5000 warmup
steps, Adam, tau 0.07, queue 4096, momentum 0.99, lambda by grid search.

Examples
  # full LMCBert on the small split (GPU machine):
  python train.py --data data/splits_small --input summary --contrastive moco --lam 0.5 --epochs 5

  # ablations
  python train.py ... --input sections            # w/o LLM  (raw sections instead of summary)
  python train.py ... --contrastive inbatch       # w/o queue
  python train.py ... --contrastive none          # w/o contrastive (plain BERT on summaries)
  python train.py ... --input abstract --contrastive none   # BERT baseline (Table II)
  python train.py ... --encoder allenai/scibert_scivocab_uncased --input abstract --contrastive none  # SciBERT

  # offline smoke test with a random tiny BERT (no downloads):
  python train.py --data data/splits_small --tiny --epochs 1 --max_train 64 --batch 8
"""
import os, json, argparse, time, random, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertConfig, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lmcbert import LMCBert

SECTION_ORDER = ["intro", "related", "method", "experiment", "conclusion"]

# side-feature groups (enhancements from the paper's conclusion)
FEATURE_GROUPS = {
    "struct": ["n_figures", "n_tables", "n_equations", "n_citations", "n_words", "n_captions", "eq_mean_len",
               "eq_max_len", "n_inline_math", "n_theorems", "n_algorithms", "n_sections", "n_subfigures"],
    "novelty": ["nov_max_sim", "nov_top5_mean", "nov_prior_max_sim", "nov_cat_centroid", "nov_density"],
}
LOG_FEATURES = set(FEATURE_GROUPS["struct"]) | {"nov_density"}


def feature_names(spec):
    names = []
    for g in [s for s in spec.split(",") if s]:
        names += FEATURE_GROUPS[g]
    return names


def feature_vector(r, names):
    v = []
    for n in names:
        x = float(r.get(n, 0.0) or 0.0)
        v.append(np.log1p(max(x, 0.0)) if n in LOG_FEATURES else x)
    return v


def text_for(r, mode):
    if mode == "summary":
        return r.get("summary") or r["abstract"]
    if mode == "abstract":
        return r["abstract"]
    if mode == "title_abstract":
        return r["title"] + ". " + r["abstract"]
    if mode == "sections":  # "w/o LLM": intro + method + experiments from the original paper
        return " ".join(r.get("sections", {}).get(s, "") for s in ("intro", "method", "experiment"))
    if mode == "abstract_summary":
        return r["abstract"] + " " + (r.get("summary") or "")
    if mode == "summary_captions":  # multimodal-lite: figure/table captions + summary
        # captions go FIRST (capped at ~80 words) so they survive max_len truncation;
        # the summary alone already fills ~500 tokens, so appending would drop them.
        caps = " ".join(" ".join(r.get("captions", [])[:10]).split()[:80])
        head = ("Figures and tables: " + caps + " ") if caps else ""
        return head + (r.get("summary") or r["abstract"])
    raise ValueError(mode)


class PaperDS(Dataset):
    def __init__(self, path, mode, limit=0, feats=(), scaler=None, aux_key=None):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
                if limit and len(self.rows) >= limit:
                    break
        self.mode, self.feats, self.aux_key = mode, list(feats), aux_key
        X = np.array([feature_vector(r, self.feats) for r in self.rows], dtype=np.float32) if self.feats else np.zeros((len(self.rows), 0), np.float32)
        if self.feats:
            if scaler is None:  # fit on train only
                scaler = (X.mean(0), X.std(0) + 1e-6)
            X = (X - scaler[0]) / scaler[1]
        self.scaler, self.X = scaler, X

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        aux = float(r.get(self.aux_key, float("nan"))) if self.aux_key else float("nan")
        return text_for(r, self.mode), r["abstract"], r["label"], self.X[i], aux


def make_collate(tok, max_len):
    def collate(batch):
        s, a, y, x, aux = zip(*batch)
        si = tok(list(s), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        ai = tok(list(a), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return si, ai, torch.tensor(y), torch.tensor(np.stack(x)), torch.tensor(aux, dtype=torch.float32)
    return collate


@torch.no_grad()
def evaluate(model, loader, device, return_preds=False):
    model.eval()
    ps, ys = [], []
    for si, ai, y, x, aux in loader:
        si = {k: v.to(device) for k, v in si.items()}
        logit = model(si, extra=x.to(device) if x.shape[1] else None)["logit"]
        ps.extend(torch.sigmoid(logit).cpu().tolist())
        ys.extend(y.tolist())
    ps, ys = np.array(ps), np.array(ys)
    pred = (ps >= 0.5).astype(int)
    m = {"acc": float(accuracy_score(ys, pred)), "f1": float(f1_score(ys, pred)), "auc": float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else float("nan"), "n": int(len(ys))}
    return (m, ps, ys) if return_preds else m


def build_tiny(tok_dir, train_path):
    """Random tiny BERT + WordPiece tokenizer trained on the corpus. Offline smoke test only."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertTokenizerFast

    os.makedirs(tok_dir, exist_ok=True)
    if not os.path.exists(os.path.join(tok_dir, "vocab.txt")):
        wp = BertWordPieceTokenizer(lowercase=True)
        texts = [json.loads(l)["abstract"] for l in open(train_path, encoding="utf-8")]
        wp.train_from_iterator(texts, vocab_size=8000, min_frequency=2)
        wp.save_model(tok_dir)
    tok = BertTokenizerFast(vocab_file=os.path.join(tok_dir, "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, max_position_embeddings=512)
    return tok, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train/val/test .sum.jsonl (or .jsonl)")
    ap.add_argument("--encoder", default="bert-base-uncased")
    ap.add_argument("--input", default="summary", choices=["summary", "abstract", "title_abstract", "sections", "abstract_summary", "summary_captions"])
    ap.add_argument("--features", default="", help="side-feature groups, comma separated: struct,novelty")
    ap.add_argument("--aux", default="", help="aux regression target field, e.g. score_recommendation (PeerRead)")
    ap.add_argument("--aux_weight", type=float, default=0.1)
    ap.add_argument("--init_from", default="", help="warm-start from a checkpoint (e.g. PeerRead-tuned)")
    ap.add_argument("--contrastive", default="moco", choices=["moco", "inbatch", "none"])
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--queue", type=int, default=4096)
    ap.add_argument("--momentum", type=float, default=0.99)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay")
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10, help="early stop on val acc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tiny", action="store_true", help="offline smoke test with random tiny BERT")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--resume", action="store_true", help="save last.pt every epoch and resume from it if present")
    ap.add_argument("--skip_done", action="store_true", help="exit immediately if result.json already exists")
    ap.add_argument("--progress", type=int, default=0, help="print a progress line every N batches")
    ap.add_argument("--clean", action="store_true", help="delete best.pt/last.pt after the final test (saves disk)")
    args = ap.parse_args()
    if args.skip_done and os.path.exists(os.path.join(args.out, args.name or "", "result.json")):
        print(f"{args.name}: result.json exists, skipping"); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = feature_names(args.features)
    name = args.name or f"{'tiny' if args.tiny else os.path.basename(args.encoder)}_{args.input}_{args.contrastive}_lam{args.lam}" + (f"_f[{args.features}]" if args.features else "") + (f"_aux" if args.aux else "")
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)

    def split_path(s):
        p = os.path.join(args.data, f"{s}.sum.jsonl")
        return p if os.path.exists(p) else os.path.join(args.data, f"{s}.jsonl")

    if args.tiny:
        tok, cfg = build_tiny(os.path.join(args.out, "tiny_tok"), split_path("train"))
        model = LMCBert(config=cfg, queue_size=args.queue, momentum=args.momentum, tau=args.tau, contrastive=args.contrastive, extra_dim=len(feats), aux_head=bool(args.aux))
    else:
        tok = AutoTokenizer.from_pretrained(args.encoder)
        model = LMCBert(args.encoder, queue_size=args.queue, momentum=args.momentum, tau=args.tau, contrastive=args.contrastive, extra_dim=len(feats), aux_head=bool(args.aux))
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm-start from {args.init_from}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)

    collate = make_collate(tok, args.max_len)
    tr = PaperDS(split_path("train"), args.input, args.max_train, feats, None, args.aux or None)
    va = PaperDS(split_path("val"), args.input, 0, feats, tr.scaler, args.aux or None)
    te = PaperDS(split_path("test"), args.input, 0, feats, tr.scaler, args.aux or None)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=(args.contrastive != "none"))
    vl = DataLoader(va, batch_size=args.batch * 2, collate_fn=collate)
    tel = DataLoader(te, batch_size=args.batch * 2, collate_fn=collate)
    print(f"train {len(tr)} val {len(va)} test {len(te)} | device {device} | {name}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    total = (len(tl) // args.accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, min(args.warmup, total // 10), total)
    scaler = torch.amp.GradScaler(enabled=(args.fp16 and device == "cuda"))

    best, bad, log, start_ep = -1, 0, [], 0
    last_pt = os.path.join(run_dir, "last.pt")
    if args.resume and os.path.exists(last_pt):  # epoch-level resume (Colab disconnects)
        ck = torch.load(last_pt, map_location=device, weights_only=False)  # our own checkpoint (contains optimizer/log)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); best, bad, log, start_ep = ck["best"], ck["bad"], ck["log"], ck["epoch"]
        print(f"resumed from {last_pt}: {start_ep} epochs done, best val_acc {best:.4f}")
    for ep in range(start_ep, args.epochs):
        model.train()
        t0, tot, totc, totcls, nb = time.time(), 0.0, 0.0, 0.0, 0
        for si, ai, y, x, aux in tl:
            if args.progress and nb % args.progress == 0 and nb:
                print(f"  ep{ep+1} batch {nb}/{len(tl)} loss {tot/nb:.4f} {time.time()-t0:.0f}s", flush=True)
            si = {k: v.to(device) for k, v in si.items()}
            ai = {k: v.to(device) for k, v in ai.items()}
            y = y.to(device)
            with torch.autocast(device_type="cuda", enabled=(args.fp16 and device == "cuda")):
                out = model(si, ai, y, lam=args.lam, extra=x.to(device) if x.shape[1] else None,
                            aux_target=aux.to(device) if args.aux else None, aux_weight=args.aux_weight)
            scaler.scale(out["loss"] / args.accum).backward()
            nb += 1
            if nb % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            tot += out["loss"].item(); totcls += out["loss_cls"].item(); totc += out.get("loss_con", torch.tensor(0.0)).item()
        val = evaluate(model, vl, device)
        rec = {"epoch": ep + 1, "loss": tot / nb, "loss_cls": totcls / nb, "loss_con": totc / nb, **{f"val_{k}": v for k, v in val.items()}, "sec": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec))
        if val["acc"] > best:
            best, bad = val["acc"], 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
        if args.resume:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "best": best, "bad": bad, "log": log, "epoch": ep + 1}, last_pt)
        if bad >= args.patience:
            print("early stop"); break

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    test, ps, ys = evaluate(model, tel, device, return_preds=True)
    with open(os.path.join(run_dir, "preds.json"), "w", encoding="utf-8") as f:  # per-paper test predictions (for paired significance tests)
        json.dump({"ids": [r["id"] for r in te.rows], "y": [int(v) for v in ys], "p": [round(float(v), 6) for v in ps]}, f)
    result = {"name": name, "args": vars(args), "best_val_acc": best, "test": test, "log": log}
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("TEST", json.dumps(test))
    if args.clean:
        for fn in ("best.pt", "last.pt"):
            fp = os.path.join(run_dir, fn)
            if os.path.exists(fp): os.remove(fp)
    elif os.path.exists(last_pt):
        os.remove(last_pt)


if __name__ == "__main__":
    main()
