# Model & Data Cards

Short, standardised documentation for models and datasets — the "nutrition labels"
that responsible-AI practice ([docs/19](../docs/19-responsible-ai-and-fairness.md))
and the EU AI Act increasingly expect. They record intended use, training data,
evaluation *by subgroup*, and known limitations, so a model's assumptions travel
with it instead of living in someone's head.

- [`TEMPLATE_model_card.md`](TEMPLATE_model_card.md) — copy for any new model.
- [`TEMPLATE_data_card.md`](TEMPLATE_data_card.md) — copy for any dataset.

Filled exemplars (one per project + case study):

| Card | Project | Model type |
|------|---------|------------|
| [`p1_breast_cancer_model_card.md`](p1_breast_cancer_model_card.md) | P1 supervised | Logistic regression |
| [`p2_forecasting_model_card.md`](p2_forecasting_model_card.md) | P2 forecasting | ETS / Holt-Winters |
| [`p3_causal_model_card.md`](p3_causal_model_card.md) | P3 causal inference | Doubly-robust AIPW |
| [`p4_clustering_model_card.md`](p4_clustering_model_card.md) | P4 clustering | KMeans |
| [`p5_survival_model_card.md`](p5_survival_model_card.md) | P5 survival | Cox PH |
| [`p6_fairness_model_card.md`](p6_fairness_model_card.md) | P6 fairness | GBM + audit |
| [`p7_anomaly_model_card.md`](p7_anomaly_model_card.md) | P7 anomaly | Isolation Forest |
| [`p8_recommender_model_card.md`](p8_recommender_model_card.md) | P8 recommender | SVD matrix factorisation |
| [`p9_nlp_model_card.md`](p9_nlp_model_card.md) | P9 NLP | TF-IDF + logistic regression |
| [`case_study_uplift_model_card.md`](case_study_uplift_model_card.md) | Case study | Uplift / causal forest |

Data cards:
- [`p1_breast_cancer_data_card.md`](p1_breast_cancer_data_card.md)

Every consequential model in a real project should carry a card; treating it as a
required deliverable (like a README) is the cheap habit that pays off under audit.
