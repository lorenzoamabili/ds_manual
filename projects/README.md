# Projects

Nine self-contained, **verified** end-to-end scripts plus a case study. Each loads
real open data (or reproducible seeded simulations), runs start to finish, and writes
`metrics.md` plus figures. Small on purpose — each teaches *one* best practice
beginners get wrong.

| # | Run | Teaches | Paired doc |
|---|-----|---------|------------|
| **P1** | `python p1_supervised_learning/train.py` | Leak-free pipelines; simple models often win; calibration; permutation importance | [05](../docs/05-supervised-learning.md) |
| **P2** | `python p2_time_series_forecasting/forecast.py` | Rolling-origin backtest; always beat the seasonal-naive baseline | [07](../docs/07-time-series-forecasting.md) |
| **P3** | `python p3_causal_inference/causal.py` | Naive comparison can flip the sign; doubly-robust estimators | [09](../docs/09-causal-inference-and-experimentation.md) |
| **P4** | `python p4_unsupervised_learning/cluster.py` | Scale before distance; pick k with silhouette; validate with ARI | [06](../docs/06-unsupervised-learning.md) |
| **P5** | `python p5_survival_analysis/survival.py` | Censoring is signal; Cox hazard ratios | [16](../docs/16-survival-analysis.md) |
| **P6** | `python p6_fairness_audit/fairness.py` | Omitting a protected attribute ≠ fair; the 80% rule | [19](../docs/19-responsible-ai-and-fairness.md) |
| **P7** | `python p7_anomaly_detection/detect.py` | 99.5% accuracy catches zero fraud — use PR-AUC | [13](../docs/13-anomaly-detection.md) |
| **P8** | `python p8_recommender/recommend.py` | RMSE hides popularity bias; evaluate coverage + precision@K | [08](../docs/08-recommendation-systems.md) |
| **P9** | `python p9_nlp_classification/classify.py` | TF-IDF + LR beats Naive Bayes; beat the linear baseline first | [10](../docs/10-nlp-and-llms.md) |

## Verified results (re-run `make projects` to regenerate)

| # | Headline result |
|---|-----------------|
| P1 | LogReg beats RF & GBM; test ROC-AUC ≈ 0.996 |
| P2 | ETS MAPE ≈ 3.6% vs. 8.1% seasonal-naive |
| P3 | True ATE = 3.0; naive = −1.5; DR ≈ 3.16 |
| P4 | Recovered k=3 clusters; ARI ≈ 0.90 |
| P5 | Cox recovers known hazard ratios within CI |
| P6 | Apparent parity hides 80%-rule violation |
| P7 | Naive accuracy = 99.5%; GBM PR-AUC ≈ 0.8+ |
| P8 | Popularity coverage < 5%; SVD covers 30%+ at same RMSE |
| P9 | LR (TF-IDF) F1-macro > NB by 3–5 points |

## Suggested extensions

- **P1:** add SHAP force plots; nested CV for hyperparameter tuning; cost-based threshold.
- **P2:** add exogenous regressors; GBM-on-lags model; conformal prediction intervals.
- **P3:** DoWhy refutation; CATE with EconML; bootstrap CIs.
- **P4:** DBSCAN / HDBSCAN / GMM comparison; UMAP embedding.
- **P7:** autoencoder anomaly detection; time-series anomaly on real sensor data.
- **P8:** LightFM hybrid model; session-based recommendations.
- **P9:** fine-tune a BERT classifier; compare with zero-shot GPT prompting.
