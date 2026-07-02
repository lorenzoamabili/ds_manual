"""
P11 · Multi-armed bandit - epsilon-greedy vs UCB1 vs Thompson sampling.

Real lesson: explore/exploit trade-off. Thompson sampling empirically
dominates on cumulative reward with sub-linear regret. Contextual bandits
(LinUCB) condition arm selection on features - the production standard
for personalisation systems.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# -- Problem setup -------------------------------------------------------------
# 5 marketing message variants; true conversion rates unknown to the algorithm
TRUE_P = [0.10, 0.15, 0.20, 0.35, 0.25]
K      = len(TRUE_P)
T      = 5_000   # total impressions

print(f"Bandit problem: K={K} arms, T={T:,} pulls")
print(f"True rates: {TRUE_P}  (arm 3 is best at {max(TRUE_P):.0%})")
print(f"Oracle reward: {int(max(TRUE_P) * T):,}\n")


def pull(arm: int) -> int:
    return int(rng.random() < TRUE_P[arm])


# -- Epsilon-greedy ------------------------------------------------------------
def epsilon_greedy(eps: float = 0.10) -> list[int]:
    counts, rewards = np.zeros(K), np.zeros(K)
    cumulative, history = 0, []
    for _ in range(T):
        if rng.random() < eps or counts.min() == 0:
            arm = int(rng.integers(K))
        else:
            arm = int(np.argmax(rewards / np.maximum(counts, 1)))
        r = pull(arm)
        counts[arm] += 1
        rewards[arm] += r
        cumulative += r
        history.append(cumulative)
    return history, counts


# -- UCB1 ----------------------------------------------------------------------
def ucb1() -> list[int]:
    counts, rewards = np.zeros(K), np.zeros(K)
    cumulative, history = 0, []
    for t in range(1, T + 1):
        if counts.min() == 0:
            arm = int(counts.argmin())
        else:
            ucb = rewards / counts + np.sqrt(2 * np.log(t) / counts)
            arm = int(np.argmax(ucb))
        r = pull(arm)
        counts[arm] += 1
        rewards[arm] += r
        cumulative += r
        history.append(cumulative)
    return history, counts


# -- Thompson sampling ---------------------------------------------------------
def thompson() -> list[int]:
    alpha = np.ones(K)
    beta  = np.ones(K)
    cumulative, history = 0, []
    for _ in range(T):
        arm = int(np.argmax(rng.beta(alpha, beta)))
        r   = pull(arm)
        alpha[arm] += r
        beta[arm]  += (1 - r)
        cumulative += r
        history.append(cumulative)
    return history, alpha - 1  # successes = alpha - 1


# -- Run all -------------------------------------------------------------------
eg_hist, eg_counts   = epsilon_greedy(eps=0.10)
ub_hist, ub_counts   = ucb1()
th_hist, th_alpha    = thompson()
oracle               = np.cumsum([max(TRUE_P)] * T)

# -- Contextual bandit (LinUCB) ------------------------------------------------
D     = 5   # context features: age-group, device, hour-bucket, recency, region
TRUE_W = rng.normal(0, 1, (K, D))   # true reward weights (unknown to algorithm)
ALPHA = 1.0

A_ctx = [np.eye(D) for _ in range(K)]
b_ctx = [np.zeros(D) for _ in range(K)]
ctx_cumulative, ctx_history = 0, []

for _ in range(T):
    x = rng.normal(0, 1, D)
    x = x / np.linalg.norm(x)
    scores = []
    for k in range(K):
        A_inv = np.linalg.inv(A_ctx[k])
        theta = A_inv @ b_ctx[k]
        scores.append(theta @ x + ALPHA * np.sqrt(x @ A_inv @ x))
    arm = int(np.argmax(scores))
    r   = float(TRUE_W[arm] @ x + rng.normal(0, 0.1))
    A_ctx[arm] += np.outer(x, x)
    b_ctx[arm] += r * x
    ctx_cumulative += r
    ctx_history.append(ctx_cumulative)

# -- Results -------------------------------------------------------------------
print("-" * 48)
print(f"{'Algorithm':<20} {'Reward':>8} {'Regret':>8}")
print("-" * 48)
for name, hist in [
    ("Epsilon-greedy", eg_hist),
    ("UCB1",          ub_hist),
    ("Thompson",      th_hist),
]:
    regret = int(oracle[-1]) - hist[-1]
    print(f"{name:<20} {hist[-1]:>8,} {regret:>8,}")
print(f"{'Oracle':20} {int(oracle[-1]):>8,} {'0':>8}")
print("-" * 48)

print("\nArm selection count (which arms were pulled):")
for name, counts in [("Epsilon-greedy", eg_counts), ("UCB1", ub_counts)]:
    print(f"  {name}: {counts.astype(int)}")
print(f"  Thompson successes: {th_alpha.astype(int)}")

# -- Plot ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

ax = axes[0]
ax.plot(eg_hist, label="ε-greedy (0.10)", alpha=0.85)
ax.plot(ub_hist, label="UCB1",            alpha=0.85)
ax.plot(th_hist, label="Thompson",        alpha=0.85)
ax.plot(oracle, "--k", alpha=0.3, label="Oracle")
ax.set(xlabel="Pull", ylabel="Cumulative reward", title="Cumulative reward")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(oracle - np.array(eg_hist), label="ε-greedy")
ax.plot(oracle - np.array(ub_hist), label="UCB1")
ax.plot(oracle - np.array(th_hist), label="Thompson")
ax.set(xlabel="Pull", ylabel="Cumulative regret", title="Regret (lower = better)")
ax.legend(fontsize=9)

ax = axes[2]
ax.plot(ctx_history, color="purple", label="LinUCB (contextual)")
ax.set(xlabel="Round", ylabel="Cumulative reward", title="Contextual bandit (LinUCB)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("bandit_results.png", dpi=120)
print("\nSaved bandit_results.png")
