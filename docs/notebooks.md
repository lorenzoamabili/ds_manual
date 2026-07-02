# Jupyter Notebooks

All examples in this manual are available as runnable Jupyter notebooks.
Clone the repo and run locally:

```bash
git clone https://github.com/lorenzoamabili/ds_manual
cd ds_manual
pip install -r requirements.txt jupyter
make notebooks        # execute all notebooks top-to-bottom
jupyter lab           # browse interactively
```

---

## Project notebooks

Each project is a standalone end-to-end workflow with narrative markdown cells
and executable code cells. Open in Jupyter Lab or VS Code.

| Project | Topic | Notebook |
|---|---|---|
| P1 · Supervised Classification | Breast cancer, no-leakage Pipeline, calibration | `projects/p1_supervised_learning/notebook.ipynb` |
| P2 · Time-Series Forecasting | Airline passengers, rolling-origin backtest | `projects/p2_time_series_forecasting/notebook.ipynb` |
| P3 · Causal Inference | Doubly-robust ATE estimation, confounding | `projects/p3_causal_inference/notebook.ipynb` |
| P4 · Unsupervised Learning | KMeans + PCA on wine, silhouette selection | `projects/p4_unsupervised_learning/notebook.ipynb` |
| P5 · Survival Analysis | Cox PH on synthetic churn, censoring | `projects/p5_survival_analysis/notebook.ipynb` |
| P6 · Fairness Audit | Credit scoring, disparate impact, threshold fix | `projects/p6_fairness_audit/notebook.ipynb` |
| P7 · Anomaly Detection | Isolation Forest vs GBM, PR-AUC on fraud | `projects/p7_anomaly_detection/notebook.ipynb` |
| P8 · Recommender System | MovieLens SVD, coverage vs RMSE | `projects/p8_recommender/notebook.ipynb` |
| P9 · NLP Classification | TF-IDF + LinearSVC on 20 Newsgroups | `projects/p9_nlp_classification/notebook.ipynb` |
| P10 · Optimisation | LP budget allocation, MILP knapsack, Hungarian staffing | `projects/p10_optimization/notebook.ipynb` |

**Case study:**

| Case Study | Notebook |
|---|---|
| Churn prediction with uplift modelling | `case_study_churn_uplift/notebook.ipynb` |

---

## Topic notebooks

Standalone concept notebooks — each demonstrates one foundational DS topic with
runnable, self-contained examples.

| Topic | Key concepts | Notebook |
|---|---|---|
| [Statistics fundamentals](02-statistics-that-matter.md) | Bootstrap CI, power analysis, BH correction, p-values | `notebooks/01-statistics-fundamentals.ipynb` |
| [Model evaluation](04-evaluation-and-validation.md) | ROC vs PR-AUC, calibration curves, cost-based thresholds | `notebooks/02-model-evaluation.ipynb` |
| [A/B testing & causal inference](09-causal-inference-and-experimentation.md) | Peeking problem, CUPED, T-learner CATE | `notebooks/03-ab-testing.ipynb` |
| [NLP text classification](10-nlp-and-llms.md) | TF-IDF, top features, cosine similarity | `notebooks/04-nlp-text-classification.ipynb` |
| [Feature engineering](03-data-and-feature-engineering.md) | Leakage guard, shuffle-label test, target encoding | `notebooks/05-feature-engineering.ipynb` |

---

## Tips for running notebooks

**Execute all notebooks** (CI-style check):

```bash
make notebooks
```

**Run one notebook interactively:**

```bash
jupyter lab projects/p1_supervised_learning/notebook.ipynb
```

**Run one notebook headlessly:**

```bash
jupyter nbconvert --to notebook --execute projects/p1_supervised_learning/notebook.ipynb \
    --output projects/p1_supervised_learning/notebook_executed.ipynb
```

**VS Code:** open any `.ipynb` file directly — VS Code has a built-in notebook editor.
No Jupyter installation needed; it uses the Python kernel from your virtual environment.
