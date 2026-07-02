# Project 2 results - rolling-origin backtest

MAPE (%) across three forecast origins, 12-month horizon each:

|                |   SeasonalNaive |   ETS |   SARIMA |
|:---------------|----------------:|------:|---------:|
| origin@1957-12 |            3.14 |  4.58 |     5.06 |
| origin@1958-12 |           11.06 |  4.79 |    10.84 |
| origin@1959-12 |            9.99 |  2.21 |     3.76 |
| MEAN           |            8.06 |  3.86 |     6.55 |

**Winner: ETS.** Note that both statistical models must be compared against the seasonal-naive baseline - reporting an absolute error without that reference is meaningless.
