# Case study results

|   budget_% |   target_by_uplift |   target_by_risk |   random |
|-----------:|-------------------:|-----------------:|---------:|
|          5 |              14240 |            -2040 |    -5747 |
|         25 |              51569 |             3451 |    -6779 |
|         50 |              35277 |            -9059 |    -5019 |
|         75 |               9931 |           -18729 |   -18152 |
|        100 |             -33936 |           -33936 |   -33936 |

- Optimal policy: **treat top 30% by predicted uplift** -> net **£55,664** on this test cohort.
- Treat-everyone: £-33,936. Best risk-targeting: £6,602.
- Uplift-model ranking corr with ground truth: 0.649.
