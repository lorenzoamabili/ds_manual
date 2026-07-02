# Project 1 results — LogReg (L2) selected

## 5-fold cross-validated (training set)

| model                |   ROC_AUC |   PR_AUC |   Brier |
|:---------------------|----------:|---------:|--------:|
| LogReg (L2)          |    0.9914 |   0.9909 |  0.0217 |
| RandomForest         |    0.9877 |   0.9854 |  0.0348 |
| HistGradientBoosting |    0.9903 |   0.9875 |  0.0316 |

## Held-out test set

|         |   value |
|:--------|--------:|
| ROC_AUC |  0.9962 |
| PR_AUC  |  0.9943 |
| Brier   |  0.0224 |

Interpretation: PR-AUC is the headline metric because the positive (malignant) class is the minority and the cost of a false negative is high. The calibration plot confirms whether the probabilities can be trusted as probabilities, not just as a ranking.
