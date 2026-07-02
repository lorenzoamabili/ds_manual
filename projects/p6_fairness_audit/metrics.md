# Project 6 — fairness audit

## 'Unaware' model, single threshold

|    |   selection_rate |   TPR |   FPR |   accuracy |
|---:|-----------------:|------:|------:|-----------:|
|  0 |            0.401 | 0.839 | 0.176 |      0.829 |
|  1 |            0.234 | 0.577 | 0.039 |      0.822 |

Disparate-impact ratio = **0.58** — fails the 80% rule despite the model never seeing the protected attribute.

## After equalising opportunity (group-specific thresholds)

|    |   selection_rate |   TPR |   FPR |   accuracy |
|---:|-----------------:|------:|------:|-----------:|
|  0 |            0.401 | 0.839 | 0.176 |      0.829 |
|  1 |            0.411 | 0.845 | 0.166 |      0.838 |

Disparate-impact ratio = **1.03**; overall accuracy moves 0.825 -> 0.833. The gap closes at a small accuracy cost — the fairness/accuracy trade-off made explicit rather than hidden.
