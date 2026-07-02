# The Applied Data Science Manual

![ci](https://github.com/lorenzo-amabili/ds-manual/actions/workflows/ci.yml/badge.svg)

A personal knowledge base **and** portfolio: a structured, opinionated reference for
how data science is actually practised — the methods, the models, the best
practices, the pitfalls — backed by **runnable, verified code on real open data**,
and wired together with the engineering hygiene the docs preach (tests, CI,
one-command reproducibility, model cards).

It serves two purposes at once:

1. **A manual** — where I return for the technique, the assumption, the "which model
   when", and the gotcha I always forget.
2. **A portfolio** — evidence of how I think, structure work, and *validate* results
   — not just which libraries I can import. Every project demonstrates a best
   practice beginners get wrong, and the repo itself demonstrates that I ship.

```bash
make setup       # install deps
make reproduce   # lint -> tests -> every project -> case study, from source
```

---

## Reproduce everything
`make reproduce` runs the linter, the unit tests, all six projects, and the case
study end-to-end. Continuous integration ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
does the same on every push across Python 3.10 and 3.12.

| Command | What it does |
|---------|--------------|
| `make test` | Unit tests for the shared `dsmanual` package + a leakage guard |
| `make lint` | `ruff` static checks |
| `make projects` | Run P1–P9 (writes metrics + figures) |
| `make case-study` | Run the churn + uplift case study |

---

## Start here: the case study
[**`case_study_churn_uplift/`**](case_study_churn_uplift) — the piece that shows
judgment rather than technique. A retention-offer problem taken from framing →
data → model → **causal uplift** → an explicit budget recommendation. The punchline,
quantified on a simulated pilot RCT: targeting the *persuadable* (by predicted
uplift) nets **£55.7k**; targeting the *high-risk* — the intuitive move — nets only
**£6.6k**; treating everyone **loses £34k**. Risk ≠ uplift, and the difference is
the whole job.

---

## How this is organised

Data science splits along **two axes**. *Domains* (FinTech, Health, Retail…) are
where work is applied; *functions* (forecasting, causal inference, NLP…) are the
transferable skills that move with you. This manual is organised by **function**,
because that is what compounds. The domain map
([`docs/00`](docs/00-map-of-data-science.md)) shows how functions light up across
industries.

### Foundations — the part everyone skips and later regrets
| Doc | Topic |
|-----|-------|
| [01](docs/01-lifecycle-and-reproducibility.md) | Project lifecycle, structure, reproducibility |
| [02](docs/02-statistics-that-matter.md) | The statistics that actually come up |
| [03](docs/03-data-and-feature-engineering.md) | Cleaning, EDA, feature engineering, **leakage** |
| [04](docs/04-evaluation-and-validation.md) | Metrics, cross-validation, calibration |

### Methods (by function)
| Doc | Function | Paired project |
|-----|----------|----------------|
| [05](docs/05-supervised-learning.md) | Supervised learning | [P1](projects/p1_supervised_learning) |
| [06](docs/06-unsupervised-learning.md) | Clustering & dimensionality reduction | [P4](projects/p4_unsupervised_learning) |
| [07](docs/07-time-series-forecasting.md) | Forecasting | [P2](projects/p2_time_series_forecasting) |
| [08](docs/08-recommendation-systems.md) | Recommenders | [P8](projects/p8_recommender) |
| [09](docs/09-causal-inference-and-experimentation.md) | Causal inference & A/B testing | [P3](projects/p3_causal_inference) · [interactive](interactive/confounding-explainer.html) |
| [10](docs/10-nlp-and-llms.md) | NLP & LLM applications | [P9](projects/p9_nlp_classification) |
| [11](docs/11-computer-vision.md) | Computer vision | — |
| [12](docs/12-optimization.md) | Optimisation & operations research | — |
| [13](docs/13-anomaly-detection.md) | Anomaly detection | [P7](projects/p7_anomaly_detection) |
| [16](docs/16-survival-analysis.md) | Survival / time-to-event | [P5](projects/p5_survival_analysis) |
| [17](docs/17-experimentation-advanced.md) | Experimentation beyond A/B (CUPED, sequential, bandits) | — |
| [20](docs/20-bayesian-and-probabilistic.md) | Bayesian & probabilistic modelling | — |
| [21](docs/21-graph-and-network-analysis.md) | Graph & network analysis | — |
| [22](docs/22-geospatial.md) | Geospatial analysis | — |

### Engineering, production & responsibility
| Doc | Topic |
|-----|-------|
| [14](docs/14-mlops-and-productionization.md) | MLOps, monitoring, reproducible serving |
| [15](docs/15-communication-and-visualization.md) | Turning results into decisions |
| [18](docs/18-sql-and-data-engineering.md) | SQL fluency & the modern data stack |
| [19](docs/19-responsible-ai-and-fairness.md) | Fairness, privacy, GDPR / EU AI Act | [P6](projects/p6_fairness_audit) |

### Domain guides (where the functions are applied)
| Doc | Domain | Signature problems |
|-----|--------|--------------------|
| [30](docs/30-product-analytics.md) | Product Analytics | Funnels, retention, A/B testing |
| [31](docs/31-fintech.md) | FinTech | Fraud detection, credit scoring, risk |
| [32](docs/32-retail-ecommerce.md) | Retail / E-commerce | Demand forecasting, basket analysis, RFM |
| [33](docs/33-healthtech.md) | HealthTech / Clinical | Risk prediction, readmission, clinical trials |
| [34](docs/34-manufacturing.md) | Manufacturing / Industry 4.0 | Predictive maintenance, RUL, anomaly |
| [35](docs/35-martech.md) | MarTech | Segmentation, uplift modelling, attribution |
| [36](docs/36-energy.md) | Energy & Utilities | Load forecasting, grid anomalies |
| [37](docs/37-cybersecurity.md) | Cybersecurity | Intrusion detection, UEBA, anomaly |
| [38](docs/38-hrtech.md) | HRTech / People Analytics | Attrition, pay equity, workforce planning |
| [39](docs/39-gaming-media.md) | Gaming & Media | Player churn, matchmaking, LTV, toxicity |
| [40](docs/40-mobility-logistics.md) | Mobility & Logistics | ETA, VRP routing, surge pricing |
| [41](docs/41-climate-environment.md) | Climate & Environment | Air quality, species modelling, emissions |
| [42](docs/42-govtech-public.md) | GovTech & Public Sector | Policy evaluation (RDD), fraud, fairness |

---

## The projects

Self-contained scripts that run end-to-end, load **real open data** (or reproducible
seeded simulations where a known ground truth makes the lesson checkable), and write
metrics + figures. Small enough to read in one sitting; each one teaches a *best
practice beginners get wrong*.

| # | Project | Real lesson it teaches | Data |
|---|---------|------------------------|------|
| P1 | Supervised classification | No-leakage pipelines; simple models often win | Wisconsin Breast Cancer (sklearn) |
| P2 | Forecasting | Rolling-origin backtest; always beat a naive baseline | Airline passengers (GitHub) |
| P3 | Causal inference | Naive comparison can flip the sign; DR estimators | Semi-synthetic (known ATE) |
| P4 | Unsupervised | Scale before distance; pick k with silhouette | Wine (sklearn) |
| P5 | Survival analysis | Censoring is signal; recover hazard ratios with Cox | Semi-synthetic (known HRs) |
| P6 | Fairness audit | Omitting a protected attribute ≠ fair; the 80% rule | Semi-synthetic (equal base rates) |
| P7 | Anomaly detection | Accuracy is 99.5% and catches zero fraud — use PR-AUC | Synthetic (seeded, 0.5% fraud rate) |
| P8 | Recommender system | RMSE hides popularity bias; evaluate coverage + precision@K | MovieLens 100K |
| P9 | NLP classification | TF-IDF + logistic regression beats Naive Bayes; beat the linear baseline first | 20 Newsgroups (4 categories) |
| P10 | Optimisation | Predictions → actions under constraints; LP/MILP/assignment | Synthetic (scipy/PuLP) |
| — | **Case study** | Target uplift, not risk; model → decision → £ | Simulated pilot RCT |

---

## Engineering (the "show, don't tell")
The docs argue for tested, importable transformation code and reproducible
pipelines — so the repo is built that way:

- [`src/dsmanual/`](src/dsmanual) — small, **tested** utilities (forecasting metrics,
  standardised mean difference, propensity clipping) imported by P2/P3.
- [`tests/`](tests) — 15 tests including a shuffled-label leakage guard.
- [`cards/`](cards) — model & data cards for the flagship models, with templates.
- `Makefile`, `.github/workflows/ci.yml`, `.devcontainer/` — one-command repro, CI,
  reproducible dev environment.
- [`interactive/confounding-explainer.html`](interactive/confounding-explainer.html)
  — a D3 explainer where dragging a confounder live-reverses a regression line
  (Simpson's paradox), tying visually to P3's sign flip.

---

## The one idea behind the whole thing

> A data scientist is not someone who can fit a model. It is someone who can tell
> whether a fitted model should be **trusted** — and can say *why* in a sentence a
> decision-maker understands.

Almost every "best practice" here is a defence against one of three ways of fooling
yourself: **leakage** (the target sneaks into training), **overfitting** (memorising
noise), and **confounding** (mistaking correlation for effect). Keep those three in
view and most of the discipline follows.
#   d s _ m a n u a l  
 