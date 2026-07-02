# Jupyter Notebooks

Every example in this manual is available as a runnable Jupyter notebook.
Launch any notebook in Google Colab — no installation required, free GPU available:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/01-statistics-fundamentals.ipynb)

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
| P1 · Supervised Classification | No-leakage Pipeline, calibration, permutation importance | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p1_supervised_learning/notebook.ipynb) |
| P2 · Time-Series Forecasting | Rolling-origin backtest, ETS vs naive | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p2_time_series_forecasting/notebook.ipynb) |
| P3 · Causal Inference | Doubly-robust ATE, confounding, propensity scores | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p3_causal_inference/notebook.ipynb) |
| P4 · Unsupervised Learning | KMeans + PCA on wine, silhouette k-selection | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p4_unsupervised_learning/notebook.ipynb) |
| P5 · Survival Analysis | Cox PH on synthetic churn, censoring, Kaplan-Meier | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p5_survival_analysis/notebook.ipynb) |
| P6 · Fairness Audit | Disparate impact, equalised odds, threshold fix | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p6_fairness_audit/notebook.ipynb) |
| P7 · Anomaly Detection | Isolation Forest vs GBM, PR-AUC on 1% fraud | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p7_anomaly_detection/notebook.ipynb) |
| P8 · Recommender System | MovieLens SVD, coverage vs RMSE trade-off | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p8_recommender/notebook.ipynb) |
| P9 · NLP Classification | TF-IDF + LinearSVC on 20 Newsgroups, top features | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p9_nlp_classification/notebook.ipynb) |
| P10 · Optimisation | LP budget allocation, MILP knapsack, Hungarian staffing | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p10_optimization/notebook.ipynb) |
| P11 · Multi-Armed Bandits | Thompson sampling vs UCB1 vs epsilon-greedy, LinUCB | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p11_bandit/notebook.ipynb) |
| P12 · RAG Pipeline | Chunking, TF-IDF retrieval, Recall@K, MRR | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/projects/p12_rag/notebook.ipynb) |

**Case study:**

| Notebook | Topic | Launch |
|---|---|---|
| Churn + uplift modelling | Persuadables vs risk, T-learner CATE, ROI comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/case_study_churn_uplift/notebook.ipynb) |

---

## Topic notebooks

Standalone concept notebooks — one key idea, fully runnable, ~10 minutes each.

| Topic | Concepts | Launch |
|---|---|---|
| Statistics fundamentals | Bootstrap CI, power analysis, p-values, BH correction | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/01-statistics-fundamentals.ipynb) |
| Model evaluation | ROC vs PR-AUC, calibration curves, cost-based thresholds | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/02-model-evaluation.ipynb) |
| A/B testing & causal inference | Peeking problem, CUPED, T-learner CATE | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/03-ab-testing.ipynb) |
| NLP text classification | TF-IDF, feature importance, cosine similarity | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/04-nlp-text-classification.ipynb) |
| Feature engineering | Leakage guard, shuffle-label test, target encoding | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/05-feature-engineering.ipynb) |
| Time-series forecasting | Decomposition, rolling-origin backtest, prediction intervals | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/06-time-series-forecasting.ipynb) |
| Clustering & dimensionality reduction | KMeans elbow/silhouette, PCA biplot, DBSCAN | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/07-clustering.ipynb) |
| Survival analysis | Kaplan-Meier, Cox PH hazard ratios, censoring | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/notebooks/08-survival-analysis.ipynb) |

---

## Tips

**Google Colab loads instantly** — no environment build wait. Sign in with
a Google account to save changes and use free GPU/TPU.

**Run a specific notebook:**
click any badge above, or use the URL pattern:

```
https://colab.research.google.com/github/lorenzoamabili/ds_manual/blob/main/PATH/TO/notebook.ipynb
```

**Execute all notebooks locally** (CI-style):

```bash
make notebooks   # requires: pip install jupyter
```

**VS Code:** open any `.ipynb` directly — the built-in notebook editor
uses your local Python kernel with no extra setup.
