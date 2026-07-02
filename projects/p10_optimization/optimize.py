"""
Project 10 - Optimisation & Operations Research
================================================
Three classic optimisation problems, each paired with a business framing:

1. LP - marketing budget allocation (scipy linprog)
2. MILP - 0/1 knapsack (PuLP with CBC solver)
3. Assignment - staffing schedule (scipy linear_sum_assignment)

Real lesson: Optimisation turns a model's outputs (predicted revenue, predicted
risk) into *actions* under constraints. A perfect model with no optimisation layer
produces suboptimal decisions; a mediocre model with a good optimisation layer
often beats it.

Run:  python optimize.py
Outputs: metrics.md, budget_allocation.png, knapsack_items.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog, linear_sum_assignment

OUT = Path(__file__).parent
rng = np.random.default_rng(42)

# ===============================================================================
# Problem 1: Marketing budget allocation (LP)
# ===============================================================================
print("-" * 60)
print("Problem 1: Marketing budget allocation (LP)")
print("-" * 60)

# 5 channels: Paid Search, Social, Display, Email, Affiliate
channels = ["Paid Search", "Social", "Display", "Email", "Affiliate"]
# Revenue per £1 spent (estimated by a media mix model)
revenue_per_pound = np.array([2.8, 1.9, 1.2, 3.5, 2.1])
# Min spend per channel (contractual obligations, £k)
min_spend = np.array([5, 2, 0, 1, 0])
# Max spend per channel (audience saturation limit, £k)
max_spend = np.array([40, 25, 20, 10, 15])
# Total budget: £80k
total_budget = 80

# LP: maximise sum(revenue_per_pound * x)
# linprog minimises, so negate objective
c = -revenue_per_pound

# Inequality constraints: sum(x) <= total_budget
A_ub = [np.ones(len(channels))]
b_ub = [total_budget]

# Bounds: min_spend <= x <= max_spend
bounds = list(zip(min_spend, max_spend))

result_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

if result_lp.success:
    allocation = result_lp.x
    total_revenue = -result_lp.fun
    # Naive baseline: equal allocation (£16k each, capped at max)
    naive_alloc = np.clip(np.full(5, total_budget / 5), min_spend, max_spend)
    naive_rev   = (naive_alloc * revenue_per_pound).sum()

    print(f"\nLP optimal allocation (total budget: £{total_budget}k):")
    for ch, alloc, rev_pp in zip(channels, allocation, revenue_per_pound):
        print(f"  {ch:<15} £{alloc:5.1f}k  (£{alloc*rev_pp:.1f}k revenue)")
    print(f"\nOptimal revenue:   £{total_revenue:.1f}k")
    print(f"Naive (equal):     £{naive_rev:.1f}k")
    print(f"Uplift from LP:    £{total_revenue - naive_rev:.1f}k ({(total_revenue/naive_rev-1):.1%})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x_pos = np.arange(len(channels))
    axes[0].bar(x_pos - 0.2, allocation, 0.4, label="LP optimal", color="#3498db")
    axes[0].bar(x_pos + 0.2, naive_alloc, 0.4, label="Equal split", color="#95a5a6")
    axes[0].set(xticks=x_pos, xticklabels=channels, ylabel="Spend (£k)",
                title="Budget allocation: LP vs. equal split")
    axes[0].legend(); axes[0].tick_params(axis="x", rotation=20)

    roi_vals = allocation * revenue_per_pound
    axes[1].bar(channels, roi_vals, color="#2ecc71", alpha=0.8)
    axes[1].set(ylabel="Revenue (£k)", title=f"Revenue by channel\n(total: £{total_revenue:.0f}k)")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(OUT / "budget_allocation.png", dpi=120)
    plt.close()
    print("\nPlot saved: budget_allocation.png")
else:
    print("LP solver failed:", result_lp.message)
    allocation = np.full(5, total_budget / 5)
    total_revenue = (allocation * revenue_per_pound).sum()
    naive_rev = total_revenue


# ===============================================================================
# Problem 2: 0/1 Knapsack (campaign selection - MILP)
# ===============================================================================
print("\n" + "-" * 60)
print("Problem 2: Campaign selection (0/1 Knapsack / MILP)")
print("-" * 60)

# 10 potential campaigns: each has cost (£k) and expected revenue (£k)
n_items = 10
costs    = rng.integers(5, 25, n_items).astype(float)
revenues = costs * rng.uniform(1.2, 3.5, n_items)  # each has positive ROI
budget   = 60.0

# Try PuLP; fall back to greedy if not installed
try:
    import pulp
    prob = pulp.LpProblem("campaign_selection", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n_items)]
    prob += pulp.lpSum(revenues[i] * x[i] for i in range(n_items))     # objective
    prob += pulp.lpSum(costs[i]    * x[i] for i in range(n_items)) <= budget  # budget
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    selected = np.array([int(pulp.value(x[i])) for i in range(n_items)])
    solver_name = "MILP (PuLP/CBC)"
except ImportError:
    # Greedy fallback: sort by ROI and fill budget
    print("  (PuLP not installed - using greedy ROI heuristic)")
    roi = revenues / costs
    order = np.argsort(-roi)
    selected = np.zeros(n_items, int)
    spend = 0.0
    for i in order:
        if spend + costs[i] <= budget:
            selected[i] = 1
            spend += costs[i]
    solver_name = "Greedy (ROI order)"

total_cost = (selected * costs).sum()
total_rev  = (selected * revenues).sum()
print(f"\n{solver_name}: selected {selected.sum()} of {n_items} campaigns")
print(f"  Total spend:   £{total_cost:.0f}k  (budget: £{budget:.0f}k)")
print(f"  Total revenue: £{total_rev:.0f}k")
print(f"  Overall ROI:   {total_rev/total_cost:.2f}x")

# Visualise campaign portfolio
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2ecc71" if s else "#e74c3c" for s in selected]
x_pos = np.arange(n_items)
ax.bar(x_pos, revenues, color=colors, alpha=0.8, label="Revenue")
ax.bar(x_pos, -costs,   color=colors, alpha=0.4)
ax.axhline(0, color="black", lw=0.5)
ax.set(xticks=x_pos, xticklabels=[f"C{i+1}" for i in range(n_items)],
       ylabel="£k  (green=selected, red=skipped)",
       title=f"Campaign selection - {solver_name}\n"
             f"Selected: {selected.sum()} campaigns, revenue: £{total_rev:.0f}k")
plt.tight_layout()
plt.savefig(OUT / "knapsack_items.png", dpi=120)
plt.close()
print("Plot saved: knapsack_items.png")


# ===============================================================================
# Problem 3: Staffing assignment (Hungarian algorithm)
# ===============================================================================
print("\n" + "-" * 60)
print("Problem 3: Staff-to-shift assignment (Hungarian algorithm)")
print("-" * 60)

# 5 staff x 5 shifts: cost matrix = predicted dissatisfaction (lower = better)
# Each row: staff member; each col: shift slot
n_staff = 5
staff_names = ["Alice", "Bob", "Carlos", "Diana", "Eve"]
shift_names = ["Mon AM", "Mon PM", "Tue AM", "Tue PM", "Wed AM"]

# Preference cost: lower = staff prefers this shift
cost_matrix = np.array([
    [1, 4, 2, 5, 3],   # Alice
    [3, 1, 4, 2, 5],   # Bob
    [5, 3, 1, 4, 2],   # Carlos
    [2, 5, 3, 1, 4],   # Diana
    [4, 2, 5, 3, 1],   # Eve
])

row_ind, col_ind = linear_sum_assignment(cost_matrix)
total_cost = cost_matrix[row_ind, col_ind].sum()
# Random baseline: average cost of random assignment
random_costs = [cost_matrix[range(n_staff), rng.permutation(n_staff)].sum()
                for _ in range(1000)]
random_avg = np.mean(random_costs)

print(f"\nOptimal assignment (Hungarian / linear_sum_assignment):")
for r, c in zip(row_ind, col_ind):
    print(f"  {staff_names[r]:<8} -> {shift_names[c]}  (dissatisfaction: {cost_matrix[r,c]})")
print(f"\nTotal dissatisfaction: {total_cost}  (random average: {random_avg:.1f})")
print(f"Improvement over random: {(random_avg - total_cost)/random_avg:.1%}")


# ===============================================================================
# Write metrics.md
# ===============================================================================
with open(OUT / "metrics.md", "w") as f:
    f.write("# P10 · Optimisation - Metrics\n\n")
    f.write("## Problem 1: LP Budget Allocation\n\n")
    f.write(f"- Optimal revenue: **£{total_revenue:.1f}k** vs. naive equal-split: £{naive_rev:.1f}k\n")
    f.write(f"- Uplift: **£{total_revenue - naive_rev:.1f}k ({(total_revenue/naive_rev-1):.1%})**\n\n")
    f.write("## Problem 2: MILP Campaign Selection\n\n")
    f.write(f"- Solver: {solver_name}\n")
    f.write(f"- Selected {selected.sum()}/{n_items} campaigns | ROI: {total_rev/total_cost:.2f}x\n\n")
    f.write("## Problem 3: Staffing Assignment\n\n")
    f.write(f"- Hungarian algorithm total dissatisfaction: **{total_cost}**\n")
    f.write(f"- Random baseline average: {random_avg:.1f} | "
            f"improvement: **{(random_avg - total_cost)/random_avg:.1%}**\n\n")
    f.write("## Key lesson\n\n")
    f.write("A model's predictions only create value when turned into *actions* under constraints. "
            "LP/MILP/assignment solvers are the bridge from prediction to decision.\n")

print(f"\nOutputs written: metrics.md, budget_allocation.png, knapsack_items.png")
print("Lesson: predictions -> decisions requires an optimisation layer.")
print(f"LP gave {(total_revenue/naive_rev-1):.1%} more revenue than naive equal-split budget.")
