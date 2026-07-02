# FinTech Case Studies

## PayPal — fraud detection at 40 million transactions/day

**Problem:** Real-time fraud detection at sub-100ms latency, on a dataset where fraud rate ≈ 0.1-0.5% (severe imbalance), with adversaries who adapt to the model as soon as it deploys.

**Approach:** A two-stage pipeline: (1) **rule engine** catches known fraud patterns instantly (zero ML overhead); (2) an **ML scoring layer** handles novel patterns. The ML layer uses [gradient boosting](05-supervised-learning.md) on 200+ features: transaction amount, velocity features (transactions in last 1/5/15 minutes), device fingerprint, IP geolocation deviation from history, graph features (shared device/card networks).

**Key technical decisions:**
- **Optimise for precision at a fixed recall**, not AUC. Blocking a legitimate transaction has a known cost (customer complaint, lost sale). Letting fraud through has a known cost (chargeback, liability). The threshold is set by the cost ratio, not by default 0.5. See [doc 04](../04-evaluation-and-validation.md) cost-based threshold.
- **Graph features**: fraudsters often reuse devices, IPs, or cards. A transaction's graph neighbourhood (is this device connected to 50 accounts?) is a strong signal that raw transaction features miss. See [doc 21](../21-graph-and-network-analysis.md).
- **Online learning / model refresh**: fraud patterns shift weekly. Static models decay. PayPal retrains on a rolling window and monitors score distributions in real-time for [drift](14-mlops-and-productionization.md).

**What failed first:** The first gradient boosting model had excellent AUC (0.97) but its [calibration](04-evaluation-and-validation.md) was poor — probabilities were not reliable. A threshold of 0.5 gave mediocre precision/recall. Once they calibrated (Platt scaling, isotonic regression) and set the threshold from the cost ratio, operational performance improved significantly. See [doc 04](../04-evaluation-and-validation.md).

**Transferable lesson:** AUC is not an operational metric. Fraud teams need precision and recall at *the threshold they'll actually use*, and that threshold comes from the business cost ratio, not from the ROC curve.

---

## Stripe Radar — ML credit card fraud with federated signals

**Problem:** Stripe processes payments for millions of merchants. Each merchant has too little data to train a fraud model. But Stripe sees the same card across merchants — information no single merchant can see.

**Approach:** **Cross-merchant [feature engineering](03-data-and-feature-engineering.md)**: Stripe builds features that aggregate a card's behaviour across all Stripe merchants (e.g. "this card was declined 3 times in the last hour across different merchants") without exposing any single merchant's data. These network features are the most predictive signals and are the moat that individual merchants' own fraud systems can't replicate.

**Key technical decisions:**
- **Two-sided causal problem**: the fraud model's decision affects the label. If the model blocks a transaction, we never observe whether it was fraud (counterfactual). Stripe uses a *randomised holdout* (a small % of transactions score-blocked anyway) to estimate counterfactual fraud rates. This is [causal inference](09-causal-inference-and-experimentation.md) applied to ML evaluation. See [doc 09](../09-causal-inference-and-experimentation.md).
- **Feedback delay**: chargebacks arrive 30-90 days after the transaction. Model training must account for this label delay — training on incomplete labels biases toward false negatives.
- **Adversarial robustness**: Stripe publishes a feature importance list? Fraudsters test against it. Features that are easily observable by adversaries are deliberately de-weighted.

**What failed first:** Early versions trained on historical data accumulated *survivor bias* — the historical data only contained transactions the old rules *allowed through*, so the model never saw patterns the rules blocked. Resolving this required careful causal framing of the training set.

**Transferable lesson:** When your model's decisions affect your training labels, you have a causal feedback loop. Estimate counterfactuals via randomised holdouts, or you will perpetually train on a biased sample.

---

## Monzo — real-time spending insights and proactive alerts

**Problem:** Monzo's differentiator is the instant push notification with spending context (e.g. "You spent £4.50 at Costa Coffee — that's your 3rd coffee this week, your weekly total is £14.50"). This requires real-time transaction categorisation at millisecond latency.

**Approach:** **Merchant categorisation via NLP**: merchant descriptor strings (the raw string from card networks like "COSTA COFFEE LONDON 1234") are messy, truncated, and inconsistent. Monzo trained a multi-class text classifier (character-level [CNN](11-computer-vision.md), later [fine-tuned](11-computer-vision.md) [BERT](10-nlp-and-llms.md)) on a human-labelled corpus of merchant strings → category.

**Key technical decisions:**
- **Active learning** loop: the classifier surfaces uncertain predictions (low max-class probability) for human review, creating a labelling queue prioritised by model uncertainty. This focuses human effort on the hardest cases, not random samples.
- **Spend intelligence features**: rolling aggregations per category per user (week, month, year) are computed as a streaming pipeline on every transaction event. Materializating these in real time requires a stream processor (Kafka + Flink), not a batch job.
- **Personalised [anomaly detection](13-anomaly-detection.md)**: "you spent £200 more than usual this month" requires a per-user baseline, not a population baseline. This is a personalized z-score over recent monthly spend. See [doc 13](../13-anomaly-detection.md).

**What failed first:** The first categorisation model was trained on UK data, released in the US, and failed on US merchant descriptor formats — "STARBUCKS #12345" vs "STARBUCKS STORE 1234 LONDON". Domain shift. They now train country-specific models with country-specific active learning queues.

**Transferable lesson:** NLP on operational text (merchant strings, log messages, addresses) looks simple — it's actually a domain adaptation problem. Train and evaluate on data from the deployment domain, not from a convenient proxy.

---

## Cross-cutting lessons

1. **Precision over recall** in high-cost-of-false-positive settings. Set thresholds from costs, not from ROC curves.
2. **Causal feedback loops**: fraud and credit scoring both suffer from "the model affects the labels" — always have randomised holdouts.
3. **Label delay** is endemic in finance. Build pipelines that handle incomplete labels at training time; don't ignore them.
4. **Network features** (graph-level, cross-entity aggregations) are the moat that per-entity ML can't replicate.
