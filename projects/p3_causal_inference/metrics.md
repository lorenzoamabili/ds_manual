# Project 3 results - true ATE = 3.0

|                       |   estimate |
|:----------------------|-----------:|
| TRUE ATE              |      3     |
| Naive diff-in-means   |     -1.525 |
| Regression adjustment |      3.195 |
| IPW                   |      3.292 |
| PSM (ATT)             |      3.272 |
| Doubly-robust (AIPW)  |      3.162 |

The naive comparison is badly biased (confounders push treated units' baseline outcomes down). Regression, IPW, matching and AIPW all recover the planted effect. AIPW is *doubly robust*: consistent if EITHER the outcome model OR the propensity model is correctly specified.

## Covariate balance |SMD|

|       |   before |   after_IPW |
|:------|---------:|------------:|
| age   |    0.337 |       0.016 |
| educ  |    0.387 |       0.018 |
| prior |    0.481 |       0.026 |
