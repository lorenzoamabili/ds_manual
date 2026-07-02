# Model Card — P9 NLP Text Classifier (Logistic Regression + TF-IDF)

## Overview
- **Purpose / intended use:** Multi-class text classification on news articles. Demonstrates that a TF-IDF + linear model is a strong NLP baseline that should be beaten before reaching for transformers.
- **Out-of-scope uses:** Not for semantic similarity, generation, or entity extraction. Performance degrades on short texts or highly specialised domains.
- **Owner / contact:** ds-manual portfolio project
- **Version / date:** 1.0 · 2024

## Model details
- **Type / architecture:** Three models compared — (1) Naive Bayes + CountVectorizer, (2) Logistic Regression + TF-IDF (bigrams, sublinear TF), (3) Linear SVM + TF-IDF.
- **Inputs:** Raw news article text (headers, footers, and quotes stripped to avoid trivial pattern matching).
- **Output:** One of 4 class labels: `sci.med`, `sci.space`, `rec.sport.baseball`, `talk.politics.guns`.
- **Training data:** 20 Newsgroups (sklearn built-in), 4-category subset. ~2 400 train / ~1 600 test documents.

## Evaluation
- **Validation scheme:** Scikit-learn's fixed 20 Newsgroups train/test split.
- **Headline metrics (see metrics.md for current run):**
  - LR (TF-IDF) typically achieves F1-macro ≈ 0.92–0.95
  - Linear SVM is often within 1 point
  - Naive Bayes is 3–5 points lower
- **Subgroup performance:** Accuracy varies by class. `sci.space` vs `sci.med` is harder to separate than `rec.sport.baseball` vs `talk.politics.guns`. See confusion matrix.
- **Calibration:** LogisticRegression outputs calibrated probabilities. LinearSVC does not output probabilities by default.

## Limitations & ethical considerations
- **Domain specificity:** 20 Newsgroups is a clean, balanced, well-separated dataset. Real-world text classification is messier (class overlap, label noise, domain shift).
- **Bag-of-words loses order:** TF-IDF ignores word order. "Not good" and "good" are treated similarly if "not" is a stop word. Use transformer models when sequence matters.
- **Out-of-vocabulary:** Words not in the training vocabulary are ignored. Subword tokenisation (transformers) handles this better.
- **Fairness assessment:** News classification may carry latent demographic biases (political alignment, jargon tied to demographics). Audit vocabulary for protected-group proxies before deploying in any moderation context.
- **Human oversight:** Any content moderation use case should keep a human review layer above the classifier.

## Maintenance
- **Monitoring:** Track per-class F1 on incoming labelled samples. Watch for vocabulary drift (new slang, new topics).
- **Retraining trigger:** When F1-macro drops >3 points on a weekly evaluation holdout, or when a new topic category emerges.
