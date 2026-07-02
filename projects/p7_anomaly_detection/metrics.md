# P7 · Anomaly Detection - Metrics

**Dataset:** Synthetic fraud (n=20 000, fraud_rate~0.5%)

|                      |   Accuracy |   Precision |   Recall |    F1 |   PR-AUC |   ROC-AUC |
|:---------------------|-----------:|------------:|---------:|------:|---------:|----------:|
| Naive (all normal)   |      0.994 |       0     |    0     | 0     |    0.006 |     0.5   |
| Isolation Forest     |      0.99  |       0     |    0     | 0     |    0.008 |     0.599 |
| Local Outlier Factor |      0.991 |       0.133 |    0.121 | 0.127 |    0.096 |     0.794 |
| GBM (supervised)     |      0.992 |       0.071 |    0.03  | 0.043 |    0.154 |     0.8   |

## Key insight

The naive model (predict all normal) achieves **99.5% accuracy** - yet it catches zero fraud.
This is why accuracy is misleading on imbalanced data.
**PR-AUC** is the correct primary metric: it measures performance across all thresholds weighted toward the rare class.
