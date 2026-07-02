# 12 · Optimisation & Operations Research

Prediction tells you what *will* happen; optimisation tells you what to *do* about
it. The two together are where data science creates operational value —
[forecasting](07-time-series-forecasting.md) demand is nice, but *deciding inventory/staffing/routing* given that
forecast is the deliverable.

## Problem types
| Type | Looks like | Method |
|------|-----------|--------|
| **Linear programming (LP)** | Continuous decisions, linear objective & constraints | Simplex / interior-point (`PuLP`, `OR-Tools`, Gurobi) |
| **Mixed-integer programming (MILP)** | Some decisions are integer/binary (yes-no, counts) | Branch-and-bound solvers; the workhorse of real OR |
| **Constraint programming (CP)** | Scheduling, assignment, feasibility-heavy | `OR-Tools` CP-SAT |
| **Convex optimisation** | Smooth convex objective | `cvxpy`; guaranteed global optimum |
| **Non-convex / combinatorial** | Routing, layout, huge discrete spaces | Metaheuristics: genetic algorithms, simulated annealing, ALNS |
| **Sequential decisions under uncertainty** | Actions now affect future states | Reinforcement learning, stochastic/dynamic programming |

## Classic applications
Vehicle routing (VRP) and last-mile logistics, production scheduling, workforce/
shift rostering, portfolio optimisation, network design, blending/cutting-stock,
price and assortment optimisation, facility location.

## How to actually model a problem
1. **Decision variables** — what you get to choose.
2. **Objective** — the single scalar to maximise/minimise (cost, profit, time).
3. **Constraints** — the rules reality imposes (capacity, budget, demand,
   logic/either-or via binary variables).
4. **Solve, then interrogate.** Check the solution is sane, run **sensitivity
   analysis** (how does the optimum move as a parameter changes?), and look at
   **shadow prices** (what a unit of extra capacity is worth) — often the most
   valuable business insight in the whole model.

## Where it meets ML — "predict-then-optimise"
The common pipeline: an ML model predicts parameters (demand, travel time,
prices), an optimiser then chooses actions. **Pitfall:** minimising prediction
error is *not* the same as making good decisions — a small forecast error in the
wrong place can be very costly. "Decision-focused" / end-to-end learning trains
the predictor against downstream *decision* quality, not raw accuracy.

## Practical notes
- **Start with an LP/MILP formulation** before reaching for metaheuristics; exact
  solvers are astonishingly good and give you an optimality gap.
- **Model size explodes fast.** Watch the number of integer variables; that's what
  makes MILP hard. Good formulation (tight constraints, symmetry breaking) beats a
  bigger solver.
- `OR-Tools` (free, excellent) covers routing, scheduling, and CP-SAT out of the
  box and is the best default entry point.

---

## Python example — LP for budget allocation (scipy) + knapsack MILP (PuLP)

```python
"""
Two canonical optimisation problems:
  1. LP: marketing budget allocation across channels (scipy linprog)
  2. MILP: 0-1 knapsack — which projects to fund? (PuLP, or scipy as fallback)
"""
import numpy as np
from scipy.optimize import linprog

# ── Problem 1: Budget allocation LP ──────────────────────────────────────────
# Allocate £1M across 4 channels to maximise expected revenue.
# Each £1 in channel i yields r[i] revenue; channels have min/max spend constraints.
r = np.array([3.2, 2.8, 4.1, 1.9])   # revenue per £ by channel (SEO, SEM, TV, Print)
budget = 1_000_000
min_spend = np.array([50_000,  50_000,  100_000, 0])
max_spend = np.array([400_000, 300_000, 600_000, 200_000])

# linprog minimises, so negate revenue for maximisation
result = linprog(
    c=-r,
    A_ub=[[1,1,1,1]],   # total spend ≤ budget
    b_ub=[budget],
    bounds=list(zip(min_spend, max_spend)),
    method="highs",
)

if result.success:
    allocation = result.x
    print("Budget allocation (LP):")
    channels = ["SEO","SEM","TV","Print"]
    for ch, alloc in zip(channels, allocation):
        print(f"  {ch}: £{alloc:>10,.0f}")
    print(f"  Total: £{allocation.sum():>10,.0f}")
    print(f"  Expected revenue: £{-result.fun:,.0f}")
    print(f"\n  Insight: TV (highest ROI=4.1) gets max spend; Print (1.9) gets min.")
else:
    print("LP failed:", result.message)

# ── Problem 2: 0-1 Knapsack (project selection) ──────────────────────────────
# 8 projects; limited budget of £500k. Choose projects to maximise total NPV.
rng = np.random.default_rng(42)
n_proj = 8
costs  = rng.integers(50, 200, n_proj) * 1000   # project cost in £
npvs   = rng.integers(80, 400, n_proj) * 1000   # expected NPV in £
capacity = 500_000

print(f"\nProject selection knapsack (budget=£{capacity:,}):")
for i, (c, v) in enumerate(zip(costs, npvs)):
    print(f"  P{i+1}: cost=£{c:,}  NPV=£{v:,}  ratio={v/c:.2f}")

try:
    import pulp
    prob = pulp.LpProblem("project_selection", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n_proj)]
    prob += pulp.lpSum(npvs[i] * x[i] for i in range(n_proj))   # maximise NPV
    prob += pulp.lpSum(costs[i] * x[i] for i in range(n_proj)) <= capacity
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    selected = [i for i in range(n_proj) if x[i].value() > 0.5]
    total_cost = sum(costs[i] for i in selected)
    total_npv  = sum(npvs[i]  for i in selected)
    print(f"\n  Optimal selection: projects {[i+1 for i in selected]}")
    print(f"  Total cost: £{total_cost:,}  Total NPV: £{total_npv:,}")
    print(f"  ROI: {total_npv/total_cost:.2f}x")
except ImportError:
    print("\n  Install pulp for MILP: pip install pulp")
    # Greedy fallback for illustration
    ratios = npvs / costs
    order  = np.argsort(ratios)[::-1]
    selected, spent, value = [], 0, 0
    for i in order:
        if spent + costs[i] <= capacity:
            selected.append(i); spent += costs[i]; value += npvs[i]
    print(f"  Greedy selection (suboptimal): projects {[i+1 for i in selected]}")
    print(f"  Total cost: £{spent:,}  Total NPV: £{value:,}")
```

---

## Cross-references

- [32](32-retail-ecommerce.md) — pricing and inventory optimisation
- [36](36-energy.md) — energy dispatch optimisation
- [35](35-martech.md) — budget allocation across marketing channels
