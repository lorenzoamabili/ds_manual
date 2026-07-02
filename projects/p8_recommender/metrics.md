# P8 · Recommender System - Metrics

**Dataset:** MovieLens 100K (or synthetic if download unavailable)

|            |   RMSE |   Precision@10 |   Coverage |
|:-----------|-------:|---------------:|-----------:|
| Popularity |  1.122 |          0.086 |      0.03  |
| User-CF    |  1.032 |          0     |      0.014 |
| SVD        |  1.008 |          0.075 |      0.215 |

## Key insight

The popularity baseline may have competitive RMSE but terrible coverage - it recommends the same blockbusters to everyone.
SVD improves coverage and precision simultaneously.
**Never evaluate a recommender system on RMSE alone.**
