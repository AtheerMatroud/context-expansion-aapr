# Context expansion approach to core-content extraction for automatic academic paper rating

Code, Colab notebooks, figure scripts, split identifiers, statistical reports and supplementary material for the paper *A context expansion approach to core-content extraction in automatic academic paper rating* (Matroud, Abdalgader and Tajvidi, submitted to Expert Systems with Applications).

The pipeline replaces the large-language-model summarisation step of LMCBert (Liu et al., IEEE Trans. Cybernetics, 2025) with a deterministic extractor adapted from the context expansion approach of Abdalgader and Al Shibli (Expert Systems with Applications, 2021): each section of a paper becomes a graph whose nodes are sentences and whose edges carry a sentence-level semantic similarity, a centrality measure (weighted degree, "Indegree", or PageRank) ranks the sentences, and the top-ranked sentences of the introduction, method, experiments and conclusion form a four-part summary that a fine-tuned BERT classifier rates. Everything runs on open-source components; all experiments were run on a single Google Colab T4.

## Data

The AAPR dataset of Yang et al. (2018) is public. The archive used in the paper is the one distributed with the dataset (four JSON files, 37,464 records): https://github.com/lancopku/AAPR, Google Drive file identifier `1ttKdXUjaFi3eS6zeHITuZjsTGrh6M4ij` (`gdown 1ttKdXUjaFi3eS6zeHITuZjsTGrh6M4ij`).

`data/aapr_split_ids.csv` lists the arXiv identifier, split (train / val / test) and label of all 36,113 papers kept after filtering. `make_splits.py` regenerates exactly these splits from the archive (fixed seed 42; notebook 10 verifies this).

## Layout

```
prepare_data.py         raw AAPR JSON -> cleaned sections, captions, structural statistics (JSONL)
make_splits.py          filtering and train / validation / test split (fixed seed)
summarize.py            core-content extraction: --backend graph (--sim tfidf | wordnet | embed,
                        --centrality indegree | pagerank), extractive (lexical baseline), random, lead;
                        options --token_budget, --global_graph, --max_sents, --sim_words, --stable_ties,
                        --store_uncut, --timing
lmcbert/model.py        BERT classifier with optional MoCo branch (LMCBert re-implementation)
train.py                training and evaluation; writes runs/<name>/result.json and preds.json
stats_compare.py        per-seed results, macro-F1 and per-class precision / recall, paired per-seed
                        differences, Welch's t-test, exact McNemar and paired bootstrap on ensembles
stats_family.py         Holm adjustment over a family of comparisons
overlap.py              overlap between the sentences selected by two extractors
collect_results.py      gathers all runs into runs/results.md and runs/results.csv
baselines_classic.py    TF-IDF + logistic regression sanity baselines
enhance/                optional novelty features and PeerRead conversion (not used in the paper)
notebooks/              the Colab notebooks that produced every number in the paper
figures/                scripts that draw the paper's figures
data/                   split identifiers
results/stats/          statistical reports of the comparisons reported in the paper
supplementary/          supplementary material of the paper
```

## Notebooks

Each notebook runs on Google Colab with intermediate files cached on Google Drive (`MyDrive/lmcbert_colab/`) and expects this repository zipped as `lmcbert_code.zip` in that folder.

| Notebook | Content |
|---|---|
| `01_full_pipeline_and_graph_summaries` | data preparation, splits, graph summaries, main configurations |
| `02_seed_replicated_edge_similarity` | TF-IDF versus MiniLM edges, five seeds |
| `03_wordnet_subset_and_sensitivity` | WordNet edges on the reduced split; learning-rate and weight-decay sensitivity |
| `04_control_baselines` | majority class, random-k, lead-k, raw sections, MoCo classifier on the summary |
| `05_class_balance_and_extractor_overlap` | class balance; overlap of each extractor with the lexical extractor |
| `06_sentence_filter_statistics` | sentences removed by each filter rule |
| `07_section_detection_rates` | detection rates of the section rule; abstract fall-back |
| `08_case_study_similarity_matrices` | similarity matrices of the worked example (Figure 2) |
| `09_truncation_audit_and_token_budget` | audit of the text that reaches BERT; token-budgeted retraining |
| `10_data_integrity_audit` | provenance and split reproduction; duplicates; publication-status clues |
| `11_abstract_global_pruning_timing` | abstract-only seeds, global-selection control, matched pruning, timing, repeatability, statistics |

## Reproducing the main configuration from the command line

1. Download and unzip the AAPR archive (see Data) and set `AAPR_RAW` to the unzipped folder.
2. `python prepare_data.py --out data/aapr_clean.jsonl`, then `python make_splits.py --inp data/aapr_clean.jsonl --out data/splits --val 1000 --test 1000`.
3. Build the summaries, for example the main configuration:
   `python summarize.py --inp data/splits/train.jsonl --out data/splits_graph_embed_indegree/train.sum.jsonl --backend graph --sim embed --centrality indegree` (and likewise for `val` and `test`).
4. Train and evaluate:
   `python train.py --data data/splits_graph_embed_indegree --input summary --contrastive none --epochs 10 --patience 3 --batch 16 --accum 2 --max_len 512 --lr 2e-5 --fp16 --seed 42 --name graph_embed_indegree_bert`.
5. Seed-replicated comparisons: run step 4 with several `--seed` values, then
   `python stats_compare.py --a "runs/<config A>_s*" --b "runs/<config B>_s*" --name_a "A" --name_b "B"`.

## Requirements

Python 3.10 or later, PyTorch, `transformers`, `sentence-transformers` (for `--sim embed`), `nltk` with the WordNet corpus (for `--sim wordnet`), `scikit-learn`, `scipy`, `numpy`. See `requirements.txt`.

## Citation

Matroud, A. A., Abdalgader, K., & Tajvidi, M. A context expansion approach to core-content extraction in automatic academic paper rating. Submitted.

## Licence

MIT (see LICENSE).
