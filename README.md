# Context expansion approach to core-content extraction for automatic academic paper rating

Code, notebooks and figure scripts for the paper *A context expansion approach to core-content extraction in automatic academic paper rating* (Matroud, Abdalgader and Tajvidi, submitted to Expert Systems with Applications).

The pipeline replaces the large-language-model summarisation step of LMCBert (Liu et al., IEEE Trans. Cybernetics, 2025) with a deterministic extractor adapted from the context expansion approach of Abdalgader and Al Shibli (Expert Systems with Applications, 2021): each section of a paper becomes a graph whose nodes are sentences and whose edges carry a sentence-level semantic similarity, a centrality measure (weighted Indegree or PageRank) ranks the sentences, and the top-ranked sentences of the introduction, method, experiments and conclusion form a four-part summary that a fine-tuned BERT classifier rates. Everything runs on open-source components; the full benchmark was processed on a single Google Colab T4.

## Layout

```
prepare_data.py         raw AAPR JSON -> cleaned sections, captions, structural statistics (JSONL)
make_splits.py          filtering and train / validation / test split (fixed seed)
summarize.py            core-content extraction: --backend graph (--sim tfidf | wordnet | embed,
                        --centrality indegree | pagerank), extractive (lexical baseline), hf, ollama
lmcbert/model.py        BERT classifier with optional MoCo branch (LMCBert re-implementation)
train.py                training and evaluation; writes runs/<name>/result.json and preds.json
stats_compare.py        mean +/- SD over seeds, Welch's t-test, exact McNemar, paired bootstrap
collect_results.py      gathers all runs into runs/results.md and runs/results.csv
baselines_classic.py    TF-IDF + logistic regression sanity baselines
enhance/novelty.py      retrieval-based novelty features (optional, --features novelty)
enhance/peerread.py     PeerRead conversion (optional)
notebooks/              the three Colab notebooks that produced every number in the paper
figures/                scripts that draw the paper's figures
```

## Reproducing the paper

1. Download the AAPR dataset of Yang et al. (2018) (`AAPR_Dataset.zip`, see https://github.com/lancopku/AAPR) and set `AAPR_RAW` to the unzipped folder.
2. `python prepare_data.py --out data/aapr_clean.jsonl` then `python make_splits.py --inp data/aapr_clean.jsonl --out data/splits --val 1000 --test 1000`.
3. Build the summaries, for example the main configuration:
   `python summarize.py --inp data/splits/train.jsonl --out data/splits_graph_embed_indegree/train.sum.jsonl --backend graph --sim embed --centrality indegree` (and likewise for `val` and `test`).
4. Train and evaluate:
   `python train.py --data data/splits_graph_embed_indegree --input summary --contrastive none --epochs 10 --patience 3 --batch 16 --accum 2 --max_len 512 --lr 2e-5 --fp16 --seed 42 --name graph_embed_indegree_bert`.
5. Seed-replicated comparisons: run step 4 with several `--seed` values and then
   `python stats_compare.py --a "runs/<config A>_s*" --b "runs/<config B>_s*" --name_a "A" --name_b "B"`.

The notebooks in `notebooks/` run exactly these steps on Google Colab with all intermediate files cached on Google Drive (`MyDrive/lmcbert_colab/`): `01` prepares the data and the graph summaries and trains the main configurations; `02` runs the five-seed TF-IDF versus MiniLM comparison; `03` runs the WordNet-based similarity on the reduced split and the learning-rate / weight-decay sensitivity study. Each notebook expects this repository zipped as `lmcbert_code.zip` in the Drive folder.

## Requirements

Python 3.10 or later, PyTorch, `transformers`, `sentence-transformers` (for `--sim embed`), `nltk` with the WordNet corpus (for `--sim wordnet`), `scikit-learn`, `scipy`, `numpy`. See `requirements.txt`.

## Citation

Matroud, A. A., Abdalgader, K., & Tajvidi, M. A context expansion approach to core-content extraction in automatic academic paper rating. Submitted.

## Licence

MIT (see LICENSE).
