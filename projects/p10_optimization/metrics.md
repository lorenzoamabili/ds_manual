# P10 · Optimisation - Metrics

## Problem 1: LP Budget Allocation

- Optimal revenue: **£207.0k** vs. naive equal-split: £160.9k
- Uplift: **£46.1k (28.7%)**

## Problem 2: MILP Campaign Selection

- Solver: Greedy (ROI order)
- Selected 5/10 campaigns | ROI: 36.33x

## Problem 3: Staffing Assignment

- Hungarian algorithm total dissatisfaction: **5**
- Random baseline average: 15.1 | improvement: **67.0%**

## Key lesson

A model's predictions only create value when turned into *actions* under constraints. LP/MILP/assignment solvers are the bridge from prediction to decision.
