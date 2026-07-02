# Jupyter Notebooks

Every example in this manual is available as a runnable Jupyter notebook.
Launch the full environment in your browser — no installation required:

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab)

Or run locally:

```bash
git clone https://github.com/lorenzoamabili/ds_manual
cd ds_manual
pip install -r requirements.txt jupyter
jupyter lab
```

---

## Project notebooks

End-to-end workflows with narrative markdown and executable code cells.
Each notebook mirrors the corresponding `.py` project script with added context.

| Project | Topic | Launch |
|---|---|---|
| P1 · Supervised Classification | No-leakage Pipeline, calibration, permutation importance | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p1_supervised_learning/notebook.ipynb) |
| P2 · Time-Series Forecasting | Rolling-origin backtest, ETS vs naive | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p2_time_series_forecasting/notebook.ipynb) |
| P3 · Causal Inference | Doubly-robust ATE, confounding, propensity scores | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p3_causal_inference/notebook.ipynb) |
| P4 · Unsupervised Learning | KMeans + PCA on wine, silhouette k-selection | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p4_unsupervised_learning/notebook.ipynb) |
| P5 · Survival Analysis | Cox PH on synthetic churn, censoring, Kaplan-Meier | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p5_survival_analysis/notebook.ipynb) |
| P6 · Fairness Audit | Disparate impact, equalised odds, threshold fix | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p6_fairness_audit/notebook.ipynb) |
| P7 · Anomaly Detection | Isolation Forest vs GBM, PR-AUC on 1% fraud | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p7_anomaly_detection/notebook.ipynb) |
| P8 · Recommender System | MovieLens SVD, coverage vs RMSE trade-off | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p8_recommender/notebook.ipynb) |
| P9 · NLP Classification | TF-IDF + LinearSVC on 20 Newsgroups, top features | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p9_nlp_classification/notebook.ipynb) |
| P10 · Optimisation | LP budget allocation, MILP knapsack, Hungarian staffing | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p10_optimization/notebook.ipynb) |
| P11 · Multi-Armed Bandits | Thompson sampling vs UCB1 vs epsilon-greedy, LinUCB | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p11_bandit/notebook.ipynb) |
| P12 · RAG Pipeline | Chunking, TF-IDF retrieval, Recall@K, MRR | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/projects/p12_rag/notebook.ipynb) |

**Case study:**

| Notebook | Topic | Launch |
|---|---|---|
| Churn + uplift modelling | Persuadables vs risk, T-learner CATE, ROI comparison | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/case_study_churn_uplift/notebook.ipynb) |

---

## Topic notebooks

Standalone concept notebooks — one key idea, fully runnable, ~10 minutes each.

| Topic | Concepts | Launch |
|---|---|---|
| Statistics fundamentals | Bootstrap CI, power analysis, p-values, BH correction | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/notebooks/01-statistics-fundamentals.ipynb) |
| Model evaluation | ROC vs PR-AUC, calibration curves, cost-based thresholds | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/notebooks/02-model-evaluation.ipynb) |
| A/B testing & causal inference | Peeking problem, CUPED, T-learner CATE | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/notebooks/03-ab-testing.ipynb) |
| NLP text classification | TF-IDF, feature importance, cosine similarity | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/notebooks/04-nlp-text-classification.ipynb) |
| Feature engineering | Leakage guard, shuffle-label test, target encoding | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/notebooks/05-feature-engineering.ipynb) |

---

## Tips

**First Binder load takes 1-3 minutes** — the environment is built once and
cached. Subsequent loads are instant.

**Run a specific notebook:**
click any badge above, or use the full URL pattern:

```
https://mybinder.org/v2/gh/lorenzoamabili/ds_manual/main?urlpath=lab/tree/PATH/TO/notebook.ipynb
```

**Execute all notebooks locally** (CI-style):

```bash
make notebooks   # requires: pip install jupyter
```

**VS Code:** open any `.ipynb` directly — the built-in notebook editor
uses your local Python kernel with no extra setup.
