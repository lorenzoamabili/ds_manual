# Project 5 - survival analysis (56% events, rest censored)

## Cox proportional-hazards: estimated vs. planted effects

|              |   true_log_HR |   est_log_HR |   hazard_ratio |   p_value |
|:-------------|--------------:|-------------:|---------------:|----------:|
| premium_plan |          -0.7 |       -0.708 |          0.493 |         0 |
| high_charges |           0.5 |        0.491 |          1.635 |         0 |
| n_tickets    |           0.3 |        0.33  |          1.39  |         0 |

Log-rank (premium vs standard): chi2=163.5, p=0.0e+00.

A hazard ratio of 0.5 means half the instantaneous churn rate. The Cox model recovers the planted log-hazard-ratios despite ~half the data being censored - which is exactly the information ordinary regression throws away.
