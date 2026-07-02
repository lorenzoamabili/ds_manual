# 00 · The Map of Data Science

Data science is not one thing. It is a set of **functional skills** applied across
**domains**. Understanding both axes is the difference between "I know sklearn"
and "I know where my skills transfer and what a new industry will actually ask of
me."

---

## Axis 1 — Domains (where the work lives)

| Domain | Signature problems | Dominant functions |
|--------|--------------------|--------------------|
| **MarTech / marketing** | Segmentation, attribution, campaign lift, churn | Causal/experimentation, recommenders, classification |
| **FinTech** | Fraud, credit scoring, risk, pricing, algo trading | Classification, anomaly detection, time series |
| **Retail / e-commerce** | Demand forecasting, pricing, basket analysis | Forecasting, recommenders, optimisation |
| **Manufacturing (Industry 4.0)** | Predictive maintenance, quality control, digital twins | Time series, anomaly detection, computer vision |
| **HealthTech / clinical** | Risk prediction, imaging, genomics, trial analysis | Classification, CV, causal inference, survival analysis |
| **Mobility / logistics** | Routing, fleet, ETA, autonomous perception | Optimisation, forecasting, CV/RL |
| **Energy / utilities** | Load forecasting, grid balancing, maintenance | Forecasting, optimisation, anomaly detection |
| **Product analytics (SaaS)** | Funnels, retention, experimentation, feature impact | Causal/experimentation, forecasting, classification |
| **Cybersecurity** | Intrusion & fraud detection, behavioural biometrics | Anomaly detection, classification, sequence models |
| **Gaming / media** | Player modelling, monetisation, matchmaking, recs | Recommenders, causal, classification |
| **GovTech / public** | Fraud, policy modelling, smart cities | Causal inference, forecasting, optimisation |
| **HRTech / people** | Attrition, hiring, workforce planning | Classification, causal, forecasting |
| **Climate / environment** | Emissions, remote sensing, monitoring | Time series, CV, spatial modelling |

**The insight most people miss:** in big-tech and SaaS companies, "Data Scientist"
usually means **product analytics + experimentation** (causal inference, A/B
testing, metrics), *not* deep ML. "Machine Learning Engineer" is the deep-model
role. Read job titles through this lens.

---

## Axis 2 — Functions (what actually transfers)

These are the reusable capabilities. Pick 1–2 to go deep on; be literate in the
rest.

| Function | Core question | This manual |
|----------|---------------|-------------|
| **Supervised learning** | Predict a labelled outcome | [05](05-supervised-learning.md) |
| **Unsupervised learning** | Find structure without labels | [06](06-unsupervised-learning.md) |
| **Forecasting** | Predict the future of a series | [07](07-time-series-forecasting.md) |
| **Recommendation** | What will this user want next | [08](08-recommendation-systems.md) |
| **Causal inference / experimentation** | Did X *cause* Y | [09](09-causal-inference-and-experimentation.md) |
| **NLP / language** | Understand & generate text | [10](10-nlp-and-llms.md) |
| **Computer vision** | Understand images & video | [11](11-computer-vision.md) |
| **Optimisation / OR** | Choose the best action under constraints | [12](12-optimization.md) |
| **Anomaly detection** | Flag the abnormal | [13](13-anomaly-detection.md) |

---

## The domain × function matrix

The same function reappears everywhere. This is *why* functional depth compounds:
learn forecasting once, apply it in energy, finance, retail, and ops.

|                         | Class./Regr. | Forecast | Recsys | Causal | NLP | CV | Optim | Anomaly |
|-------------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MarTech                 | ● | ○ | ● | ● | ○ |   | ○ | ○ |
| FinTech                 | ● | ● |   | ○ | ○ |   | ○ | ● |
| Retail                  | ● | ● | ● | ○ |   |   | ● | ○ |
| Manufacturing           | ○ | ● |   | ○ |   | ● | ● | ● |
| Health                  | ● | ○ |   | ● | ○ | ● |   | ○ |
| Mobility                | ○ | ● |   | ○ |   | ● | ● | ○ |
| Energy                  |   | ● |   | ○ |   |   | ● | ● |
| Product analytics       | ● | ● | ○ | ● | ○ |   |   | ○ |
| Cybersecurity           | ● |   |   |   | ○ |   |   | ● |

● primary   ○ common

---

## How to choose your specialisation

A practical heuristic, in priority order:

1. **Follow a function you enjoy the *reasoning* of**, not the hype. If you like
   arguing about whether an effect is real, causal inference/experimentation is a
   durable, undersupplied skill. If you like squeezing signal from messy sensors,
   forecasting/anomaly detection.
2. **Pair it with a domain you have credibility in.** Domain knowledge is a moat;
   it is what lets you spot the leakage nobody else sees.
3. **Stay literate everywhere else.** The 20% of each other function that you can
   learn in a week covers 80% of conversations.

> "An industry (FinTech, Health, Retail) **+** a modelling skillset (forecasting,
> NLP, causal inference)" — that pairing is what a specialised data scientist
> actually is.

---

## Domain guides

Function docs (05–22) teach *how*. Domain docs (30–38) teach *where* and *why*:
the business context, the data reality, and the modelling traps specific to each
industry.

| Doc | Domain | Signature problems |
|-----|--------|--------------------|
| [30](30-product-analytics.md) | Product Analytics (SaaS / big tech) | Funnels, retention cohorts, A/B testing, DAU |
| [31](31-fintech.md) | FinTech | Fraud detection, credit scoring, risk, calibration |
| [32](32-retail-ecommerce.md) | Retail / E-commerce | Demand forecasting, market basket, RFM |
| [33](33-healthtech.md) | HealthTech / Clinical | Risk prediction, readmission, clinical trial analysis |
| [34](34-manufacturing.md) | Manufacturing / Industry 4.0 | Predictive maintenance, RUL, anomaly in sensors |
| [35](35-martech.md) | MarTech | Segmentation, campaign uplift, attribution |
| [36](36-energy.md) | Energy & Utilities | Load forecasting, renewable intermittency, meter anomaly |
| [37](37-cybersecurity.md) | Cybersecurity | Intrusion detection, UEBA, phishing, log anomaly |
| [38](38-hrtech.md) | HRTech / People Analytics | Attrition, hiring, pay equity, workforce planning |
