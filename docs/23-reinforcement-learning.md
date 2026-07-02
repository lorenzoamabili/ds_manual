# Reinforcement Learning & Bandits

Reinforcement learning (RL) sits at the intersection of statistics, control theory,
and optimisation. Most DS practitioners encounter it through **multi-armed bandits**
(online decisions under uncertainty) before full RL (sequential decisions with delayed
rewards). This doc covers the practical subset that appears in production DS systems.

---

## The bandit problem

A slot machine (bandit) with K arms. Each pull returns a stochastic reward. Goal:
maximise cumulative reward. Core tension: **exploration** (try uncertain arms to learn
their payoff) vs. **exploitation** (pull the best arm known so far).

Real examples: which ad copy to show, which recommendation to surface, which
push notification to send, which price to charge.

**Regret** = reward you'd have earned with perfect knowledge − reward you actually earned.
A good bandit policy sub-linear regret (it catches up to optimal over time).

---

## Strategies

### Epsilon-greedy

With probability ε, explore (random arm). With probability 1−ε, exploit (best arm so far).

Simple, robust, widely used. Downside: exploration is random — it wastes budget on
clearly inferior arms.

### UCB1 (Upper Confidence Bound)

Pull the arm with the highest `mean_reward + C * sqrt(log(t) / n_pulls)`.

The bonus term is an uncertainty penalty: arms with fewer pulls get inflated scores.
UCB1 achieves `O(log T)` regret — optimal for stationary reward distributions.

### Thompson sampling

Maintain a Beta(α, β) posterior per arm where α = successes, β = failures.
At each step: sample θ_k ∼ Beta(α_k, β_k) for all arms, pull the arm with max θ_k.

Bayesian, naturally balances explore/exploit. Empirically the best performer in most
A/B platform comparisons.

---

## Python example — epsilon-greedy vs UCB1 vs Thompson sampling

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# True conversion rates for 5 arms — arm 3 is best (0.35)
TRUE_P = [0.10, 0.15, 0.20, 0.35, 0.25]
K = len(TRUE_P)
T = 5000  # total pulls


def pull(arm: int) -> int:
    return int(rng.random() < TRUE_P[arm])


# ── Epsilon-greedy ────────────────────────────────────────────────────────────
def epsilon_greedy(eps: float = 0.10):
    counts   = np.zeros(K)
    rewards  = np.zeros(K)
    total    = 0
    history  = []
    for _ in range(T):
        if rng.random() < eps or counts.min() == 0:
            arm = rng.integers(K)
        else:
            arm = np.argmax(rewards / counts)
        r = pull(arm)
        counts[arm]  += 1
        rewards[arm] += r
        total        += r
        history.append(total)
    return history


# ── UCB1 ──────────────────────────────────────────────────────────────────────
def ucb1():
    counts   = np.zeros(K)
    rewards  = np.zeros(K)
    total    = 0
    history  = []
    for t in range(1, T + 1):
        if counts.min() == 0:
            arm = counts.argmin()
        else:
            ucb = rewards / counts + np.sqrt(2 * np.log(t) / counts)
            arm = np.argmax(ucb)
        r = pull(arm)
        counts[arm]  += 1
        rewards[arm] += r
        total        += r
        history.append(total)
    return history


# ── Thompson sampling ─────────────────────────────────────────────────────────
def thompson():
    alpha = np.ones(K)   # successes + 1
    beta  = np.ones(K)   # failures + 1
    total   = 0
    history = []
    for _ in range(T):
        samples = rng.beta(alpha, beta)
        arm     = np.argmax(samples)
        r       = pull(arm)
        alpha[arm] += r
        beta[arm]  += (1 - r)
        total      += r
        history.append(total)
    return history


optimal = np.array([max(TRUE_P)] * T).cumsum()

eg  = epsilon_greedy()
ub  = ucb1()
th  = thompson()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(eg, label="ε-greedy (ε=0.10)")
ax1.plot(ub, label="UCB1")
ax1.plot(th, label="Thompson")
ax1.plot(optimal, "--k", alpha=0.4, label="Optimal (oracle)")
ax1.set(xlabel="Pull", ylabel="Cumulative reward", title="Cumulative reward")
ax1.legend()

regret_eg = np.array(optimal) - np.array(eg)
regret_ub = np.array(optimal) - np.array(ub)
regret_th = np.array(optimal) - np.array(th)

ax2.plot(regret_eg, label="ε-greedy")
ax2.plot(regret_ub, label="UCB1")
ax2.plot(regret_th, label="Thompson")
ax2.set(xlabel="Pull", ylabel="Cumulative regret", title="Regret (lower = better)")
ax2.legend()

plt.tight_layout()
plt.savefig("bandit_comparison.png", dpi=120)
plt.show()

print(f"\nFinal cumulative reward after {T} pulls:")
print(f"  ε-greedy:   {eg[-1]}")
print(f"  UCB1:       {ub[-1]}")
print(f"  Thompson:   {th[-1]}")
print(f"  Oracle:     {int(max(TRUE_P) * T)}")
print(f"\nRegret:")
print(f"  ε-greedy:   {int(regret_eg[-1])}")
print(f"  UCB1:       {int(regret_ub[-1])}")
print(f"  Thompson:   {int(regret_th[-1])}")
```

---

## Contextual bandits

Standard bandit ignores context. A **contextual bandit** conditions the arm selection
on observable features: `π(arm | context)`.

Real examples:
- Push notification: arm = message variant; context = user features (age, recency, hour)
- Ad serving: arm = ad creative; context = page content + user profile
- Price optimisation: arm = price tier; context = demand signals

### LinUCB

A popular contextual bandit: assumes reward is linear in context features.
Each arm maintains a ridge regression ridge `A_k, b_k`. UCB score:
`θ_k^T x + α * sqrt(x^T A_k^{-1} x)`.

The second term is a Mahalanobis-distance uncertainty bonus — arms with less
data in the direction of the current context are explored more.

```python
import numpy as np

rng = np.random.default_rng(42)

# Contextual bandit: 3 arms, 5 context features, 2000 rounds
K, D, T = 3, 5, 2000
TRUE_W = rng.normal(0, 1, (K, D))   # true reward weights (unknown to algorithm)

def pull_contextual(arm: int, x: np.ndarray) -> float:
    return TRUE_W[arm] @ x + rng.normal(0, 0.1)

alpha = 1.0   # exploration parameter
A  = [np.eye(D)    for _ in range(K)]
b  = [np.zeros(D)  for _ in range(K)]

cumulative = 0
for t in range(T):
    x = rng.normal(0, 1, D)
    x = x / np.linalg.norm(x)   # unit-normalise context

    # UCB score per arm
    scores = []
    for k in range(K):
        A_inv = np.linalg.inv(A[k])
        theta = A_inv @ b[k]
        ucb   = theta @ x + alpha * np.sqrt(x @ A_inv @ x)
        scores.append(ucb)

    arm  = int(np.argmax(scores))
    r    = pull_contextual(arm, x)
    A[arm] += np.outer(x, x)
    b[arm] += r * x
    cumulative += r

oracle_reward = sum(
    max(TRUE_W[k] @ rng.normal(0, 1, D) for k in range(K))
    for _ in range(T)
)
print(f"LinUCB cumulative reward: {cumulative:.1f}")
print(f"LinUCB explored all arms: {[int(np.linalg.det(A[k])) for k in range(K)]}")
```

---

## Q-learning — tabular RL

When decisions are **sequential** (each action changes the environment state),
bandits are insufficient. Full RL models the **Markov Decision Process** (MDP):

- **State** S: current environment observation
- **Action** A: decision taken
- **Reward** R: immediate feedback
- **Transition** P(s'|s,a): how the environment changes

**Q-learning** learns `Q(s,a)` = expected discounted future reward of taking
action a in state s. Update rule:

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
```

```python
import numpy as np

# Simple grid world: 4x4, agent navigates from (0,0) to (3,3)
# Actions: 0=up, 1=down, 2=left, 3=right
SIZE  = 4
GOAL  = (3, 3)
TRAPS = [(1, 1), (2, 2)]

Q = np.zeros((SIZE, SIZE, 4))

def step(state, action):
    r, c = state
    if   action == 0: r = max(r - 1, 0)
    elif action == 1: r = min(r + 1, SIZE - 1)
    elif action == 2: c = max(c - 1, 0)
    elif action == 3: c = min(c + 1, SIZE - 1)
    next_state = (r, c)
    if next_state == GOAL:   return next_state, +10.0, True
    if next_state in TRAPS:  return next_state,  -5.0, True
    return next_state, -0.1, False

alpha, gamma, eps = 0.1, 0.9, 0.2
rng = np.random.default_rng(42)

for episode in range(3000):
    state = (0, 0)
    eps   = max(0.05, eps * 0.999)   # decay exploration
    for _ in range(100):
        if rng.random() < eps:
            action = rng.integers(4)
        else:
            action = int(np.argmax(Q[state]))
        next_state, reward, done = step(state, action)
        td_target = reward + gamma * np.max(Q[next_state]) * (not done)
        Q[state][action] += alpha * (td_target - Q[state][action])
        state = next_state
        if done:
            break

# Greedy policy
state, path = (0, 0), [(0, 0)]
for _ in range(20):
    action = int(np.argmax(Q[state]))
    state, _, done = step(state, action)
    path.append(state)
    if done:
        break

print("Greedy path from (0,0) to goal:", path)
print(f"Reached goal: {path[-1] == GOAL}")
print(f"\nQ-values at start: {Q[0,0].round(2)}")
print("Action: 0=up 1=down 2=left 3=right")
```

---

## When to use RL/bandits

| Scenario | Recommended |
|---|---|
| One-shot personalisation (ad, email) | Thompson sampling / LinUCB |
| Sequential decisions, sparse rewards | Q-learning / policy gradient |
| Known reward function, complex constraints | Optimisation (LP/MILP, see doc 12) |
| A/B test with fixed budget | Bayesian A/B (see doc 20) |

RL is rarely the right first tool for tabular business data. Reach for it when:
(1) feedback is delayed and sequential, (2) exploration budget matters, or
(3) the decision policy must adapt online.

---

## Pitfalls

- **Reward hacking**: the agent finds an unexpected way to maximise the specified
  reward that violates the intent. Define reward carefully.
- **Non-stationary rewards**: UCB1 and Thompson assume stationary arms. Use
  discounted/sliding-window variants for drifting reward distributions.
- **Evaluation**: cannot use standard train/test split. Use replay evaluation
  or importance-weighted counterfactual estimators on logged data.
- **Off-policy evaluation**: most production systems can't run random exploration.
  Use doubly-robust off-policy estimators to evaluate policies from logged data.
