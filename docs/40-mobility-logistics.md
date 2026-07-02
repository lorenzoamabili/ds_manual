# 40 · Mobility & Logistics

## Signature problems

| Problem | Approach |
|---------|----------|
| ETA prediction | [Gradient boosting](05-supervised-learning.md) + routing API, survival models |
| Demand [forecasting](07-time-series-forecasting.md) (ride-hail, delivery) | Spatial-temporal models, [DeepAR](07-time-series-forecasting.md) (see [07](07-time-series-forecasting.md)) |
| Route optimisation | [Vehicle routing](12-optimization.md) problem (VRP), OR-Tools (see [12](12-optimization.md)) |
| Dynamic pricing / surge | Causal elasticity estimation (see [09](09-causal-inference-and-experimentation.md)) |
| Driver/courier allocation | Assignment optimisation, bandit algorithms |
| [Anomaly detection](13-anomaly-detection.md) (fleet) | [Isolation Forest](13-anomaly-detection.md), [time-series](07-time-series-forecasting.md) anomaly (see [13](13-anomaly-detection.md)) |
| Churn of drivers/couriers | [Survival analysis](16-survival-analysis.md) (see [16](16-survival-analysis.md)) |
| [Geospatial](22-geospatial.md) demand heatmaps | [H3](22-geospatial.md) spatial aggregation, kriging (see [22](22-geospatial.md)) |

## Domain characteristics

- **Real-time requirements**: surge pricing and driver dispatch must complete in <200ms. Model complexity is constrained by latency, not accuracy.
- **Spatial-temporal coupling**: demand is correlated in space (nearby hexes) and time (rush hour spreads). Ignoring spatial correlation wastes signal.
- **Two-sided marketplace**: optimising for riders and drivers simultaneously. Increasing driver utilisation can reduce rider wait times — but only up to a point (supply saturation).
- **Weather and events dominate short-term demand**: a single sports event can 5-10× normal demand in a geographic cell.

## ETA: multi-stage prediction

Delivery/ride ETA = segment sum with propagated uncertainty:

```python
"""
Multi-stage ETA with asymmetric loss.
Each stage: restaurant prep + courier pickup + travel.
Final ETA uses 80th-percentile to err on the conservative side.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
n = 2000

# Synthetic delivery features
distance_km  = rng.uniform(0.5, 8, n)
hour_of_day  = rng.integers(0, 24, n)
rain         = rng.binomial(1, 0.15, n)
items_count  = rng.integers(1, 8, n)

# True prep time (minutes): base 8 + items + rain penalty
prep_true = 8 + items_count * 1.5 + rain * 4 + rng.normal(0, 3, n)
# True travel time: distance × speed (slower in rush hour + rain)
speed = 15 - 4 * ((hour_of_day >= 8) & (hour_of_day <= 9)).astype(float) \
           - 3 * ((hour_of_day >= 17) & (hour_of_day <= 19)).astype(float) \
           - 2 * rain
travel_true = distance_km / speed * 60 + rng.normal(0, 2, n)
total_true = prep_true + travel_true

X = np.column_stack([distance_km, hour_of_day, rain, items_count])
X_tr, X_te, y_tr, y_te = train_test_split(
    X, total_true, test_size=0.25, random_state=42)

# Point forecast (symmetric MSE loss)
gbm_point = GradientBoostingRegressor(n_estimators=200, random_state=42)
gbm_point.fit(X_tr, y_tr)
mae_point = mean_absolute_error(y_te, gbm_point.predict(X_te))

# Conservative forecast (80th-percentile quantile regression)
gbm_q80 = GradientBoostingRegressor(
    n_estimators=200, loss="quantile", alpha=0.80, random_state=42)
gbm_q80.fit(X_tr, y_tr)
eta_q80 = gbm_q80.predict(X_te)

# % of deliveries arriving on time (before q80 estimate)
on_time_rate = (y_te <= eta_q80).mean()

print(f"Point forecast MAE: {mae_point:.1f} min")
print(f"80th-pct ETA: {on_time_rate:.1%} of deliveries arrive on/before ETA")
print("Lesson: conservative ETAs (high quantile) reduce customer frustration")
print("even if the stated ETA is slightly later than the point forecast.")
```

## Vehicle routing: nearest-neighbour heuristic

```python
"""
Vehicle Routing Problem (VRP) — nearest-neighbour heuristic.
For production: use Google OR-Tools for optimal solutions.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
depot    = np.array([0.0, 0.0])
stops    = rng.uniform(-5, 5, (12, 2))
all_pts  = np.vstack([depot, stops])

def nearest_neighbour_route(points, start=0):
    unvisited = list(range(1, len(points)))
    route = [start]
    while unvisited:
        last = route[-1]
        dists = [np.linalg.norm(points[last] - points[j]) for j in unvisited]
        nxt = unvisited[np.argmin(dists)]
        route.append(nxt); unvisited.remove(nxt)
    route.append(start)  # return to depot
    return route

route = nearest_neighbour_route(all_pts)
total_dist = sum(np.linalg.norm(all_pts[route[i]] - all_pts[route[i+1]])
                 for i in range(len(route)-1))

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(*stops.T, s=80, color="steelblue", zorder=3, label="Stops")
ax.scatter(*depot, s=200, color="red", zorder=4, marker="*", label="Depot")
for i in range(len(route)-1):
    a, b = all_pts[route[i]], all_pts[route[i+1]]
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
ax.set(title=f"NN route — total distance: {total_dist:.1f} units", aspect="equal")
ax.legend(); plt.tight_layout()
plt.savefig("vrp_route.png", dpi=120); plt.close()
print(f"Route: {route}\nTotal distance: {total_dist:.2f}")
print("For optimal solution: pip install ortools and use CP-SAT solver")
```

## Cross-references

- [07](07-time-series-forecasting.md) — demand forecasting (spatial-temporal)
- [09](09-causal-inference-and-experimentation.md) — surge pricing elasticity (Uber case study)
- [12](12-optimization.md) — vehicle routing, assignment problems
- [22](22-geospatial.md) — H3 hex demand aggregation
- [case studies](case_studies/cs-forecasting.md) — Uber demand forecasting, DoorDash ETA
