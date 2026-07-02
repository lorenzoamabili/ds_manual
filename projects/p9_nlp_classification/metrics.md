# P9 · NLP Text Classification - Metrics

**Dataset:** 20 Newsgroups (4 categories: sci.med, sci.space, rec.sport.baseball, talk.politics.guns)

|                       |   Accuracy |   F1-macro |   F1-weighted |
|:----------------------|-----------:|-----------:|--------------:|
| Naive Bayes (Count)   |      0.868 |      0.868 |         0.868 |
| Logistic Reg (TF-IDF) |      0.85  |      0.85  |         0.85  |
| Linear SVM (TF-IDF)   |      0.854 |      0.853 |         0.853 |

**Best model:** Naive Bayes (Count)

## Key insight

TF-IDF + linear models are a strong NLP baseline. They are fast, interpretable (inspect top coefficients), and often within 2-3 points of transformer models on topic classification. Don't jump to BERT until you've beaten this baseline.
