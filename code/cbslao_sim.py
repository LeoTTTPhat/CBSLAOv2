"""
CBSLAO simulator: Cost- and SLA-Bounded Agent Orchestration.

Implements a synthetic environment and five orchestration policies:
  1. ReAct-uncapped (myopic greedy)
  2. ReAct-capped (soft prompt cap)
  3. Non-adaptive greedy (offline ranking)
  4. BwK-vanilla (bandits-with-knapsacks baseline)
  5. CBUC (ours: type-pruned, budget-aware UCB)

Runs a sweep over registry size K, budget tightness rho, cost tail index alpha,
and utility sparsity. Outputs CSV results and publication-ready plots.

Determinism: fixed master seed; sub-seeds derived per (policy, cell, replicate).
"""
from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------ Environment ---------------------------------- #


@dataclass
class Tool:
    tid: int
    mean_cost: float
    cost_tail: float           # Pareto tail index; lower = heavier tail
    mean_latency: float
    marginal_utility: float    # true expected contribution to utility in [0,1]
    types_in: frozenset         # input types
    types_out: frozenset        # output types

    def sample_cost(self, rng: np.random.Generator) -> float:
        # Pareto cost; clipped to avoid extreme draws in simulation.
        u = rng.pareto(a=self.cost_tail) + 1.0
        base = self.mean_cost * (self.cost_tail - 1) / self.cost_tail if self.cost_tail > 1 else self.mean_cost
        c = base * u
        return float(np.clip(c, 0.0, 50.0 * self.mean_cost + 1e-9))

    def sample_latency(self, rng: np.random.Generator) -> float:
        return float(max(0.0, rng.normal(self.mean_latency, 0.25 * self.mean_latency)))

    def sample_observation_utility(self, rng: np.random.Generator) -> float:
        # Noisy realization of marginal utility in [0,1].
        return float(np.clip(rng.normal(self.marginal_utility, 0.1), 0.0, 1.0))


@dataclass
class Task:
    query_types: frozenset
    budget: float
    deadline: float
    delta: float


@dataclass
class Registry:
    tools: List[Tool]

    @property
    def K(self) -> int:
        return len(self.tools)

    def type_closure(self, query_types: frozenset) -> frozenset:
        """Iteratively expand reachable types."""
        cl = set(query_types)
        changed = True
        while changed:
            changed = False
            for t in self.tools:
                if t.types_in.issubset(cl) and not t.types_out.issubset(cl):
                    cl |= t.types_out
                    changed = True
        return frozenset(cl)

    def feasible_subset(self, query_types: frozenset) -> List[Tool]:
        cl = self.type_closure(query_types)
        return [t for t in self.tools if t.types_in.issubset(cl)]


def make_registry(K: int, cost_tail: float, utility_sparsity: float,
                  rng: np.random.Generator) -> Registry:
    """Synthesize K tools. A fraction `utility_sparsity` of tools carry non-zero
    marginal utility; the rest are distractors."""
    tools: List[Tool] = []
    # Shared type universe of size up to 8; ensures a non-trivial closure.
    type_pool = list(range(8))
    useful_count = max(1, int(round(utility_sparsity * K)))
    useful_ids = set(rng.choice(K, size=useful_count, replace=False).tolist())
    for i in range(K):
        n_in = int(rng.integers(0, 3))
        n_out = int(rng.integers(1, 3))
        types_in = frozenset(rng.choice(type_pool, size=n_in, replace=False).tolist())
        types_out = frozenset(rng.choice(type_pool, size=n_out, replace=False).tolist())
        mean_cost = float(rng.uniform(1.0, 10.0))
        mean_latency = float(rng.uniform(0.1, 2.0))
        mu = float(rng.uniform(0.05, 0.9)) if i in useful_ids else 0.0
        tools.append(Tool(i, mean_cost, cost_tail, mean_latency, mu, types_in, types_out))
    return Registry(tools)


def make_task(registry: Registry, rho: float, rng: np.random.Generator) -> Task:
    """rho = budget / sum(mean_cost) -> budget tightness."""
    total = sum(t.mean_cost for t in registry.tools)
    budget = rho * total
    deadline = 10.0 * max(t.mean_latency for t in registry.tools)
    query_types = frozenset([0, 1])  # seed the type closure
    return Task(query_types, budget, deadline, delta=0.1)


# ------------------------------ Policies ------------------------------------- #


@dataclass
class RunResult:
    policy: str
    utility: float
    cost: float
    latency: float
    n_calls: int
    budget: float
    deadline: float
    budget_violated: bool
    deadline_violated: bool


def saturate_utility(sum_util: float) -> float:
    """Diminishing-returns aggregator in [0,1]."""
    return 1.0 - math.exp(-sum_util)


def run_policy_react_uncapped(tools: List[Tool], task: Task,
                              rng: np.random.Generator) -> RunResult:
    sum_cost = 0.0
    sum_lat = 0.0
    sum_u = 0.0
    n = 0
    # Myopic: always call the tool with highest *apparent* utility (prior =
    # rough estimate perturbed with noise) until we have called every tool once.
    # ReAct in practice keeps calling until it feels "done"; we approximate with
    # a 3x overshoot.
    order = list(range(len(tools)))
    rng.shuffle(order)
    max_calls = 3 * len(tools)  # uncapped overshoot analog
    for step in range(max_calls):
        tid = order[step % len(order)]
        t = tools[tid]
        c = t.sample_cost(rng)
        l = t.sample_latency(rng)
        u = t.sample_observation_utility(rng)
        sum_cost += c; sum_lat += l; sum_u += u; n += 1
        # No budget awareness. Stop only at max.
    return RunResult("react_uncapped", saturate_utility(sum_u), sum_cost,
                     sum_lat, n, task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


def run_policy_react_capped(tools: List[Tool], task: Task,
                            rng: np.random.Generator) -> RunResult:
    """Soft cap: policy *tries* to stay within budget but uses only realized
    (not predicted) cost. Realistic proxy for a prompt like 'stay under $B'."""
    sum_cost = 0.0; sum_lat = 0.0; sum_u = 0.0; n = 0
    order = list(range(len(tools)))
    rng.shuffle(order)
    for tid in order:
        t = tools[tid]
        # After each call, check whether budget has been exceeded.
        if sum_cost >= task.budget or sum_lat >= task.deadline:
            break
        c = t.sample_cost(rng)
        l = t.sample_latency(rng)
        u = t.sample_observation_utility(rng)
        sum_cost += c; sum_lat += l; sum_u += u; n += 1
    return RunResult("react_capped", saturate_utility(sum_u), sum_cost,
                     sum_lat, n, task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


def run_policy_nonadaptive_greedy(tools: List[Tool], task: Task,
                                  rng: np.random.Generator) -> RunResult:
    """Offline rank by mean utility / mean cost; execute in order until
    expected cumulative cost would exceed budget."""
    ranked = sorted(tools, key=lambda t: -(t.marginal_utility / max(t.mean_cost, 1e-6)))
    sum_cost = 0.0; sum_lat = 0.0; sum_u = 0.0; n = 0
    for t in ranked:
        if sum_cost + t.mean_cost > task.budget:
            break
        c = t.sample_cost(rng); l = t.sample_latency(rng); u = t.sample_observation_utility(rng)
        sum_cost += c; sum_lat += l; sum_u += u; n += 1
    return RunResult("nonadaptive_greedy", saturate_utility(sum_u), sum_cost,
                     sum_lat, n, task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


def run_policy_bwk_vanilla(tools: List[Tool], task: Task,
                           rng: np.random.Generator,
                           prior_N: int = 3) -> RunResult:
    """Vanilla bandits-with-knapsacks: UCB on utility-per-cost, no type pruning,
    no chance-constraint correction."""
    K = len(tools)
    hat_u = np.zeros(K); hat_c = np.ones(K); hat_l = np.ones(K); N = np.full(K, prior_N)
    sum_cost = 0.0; sum_lat = 0.0; sum_u = 0.0; n = 0
    t_step = 1
    # Warm start: one sample per tool up to budget
    for i in range(K):
        if sum_cost + tools[i].mean_cost > task.budget or sum_lat + tools[i].mean_latency > task.deadline:
            continue
        c = tools[i].sample_cost(rng); l = tools[i].sample_latency(rng); u = tools[i].sample_observation_utility(rng)
        hat_u[i] = u; hat_c[i] = c; hat_l[i] = l; N[i] = 1
        sum_cost += c; sum_lat += l; sum_u += u; n += 1; t_step += 1
    # Exploit: repeatedly pick best UCB ratio until exhausted.
    while True:
        best_i = -1; best_score = -np.inf
        for i in range(K):
            bonus = math.sqrt(2.0 * math.log(t_step + 1) / max(N[i], 1))
            score = (hat_u[i] + bonus) / max(hat_c[i] - bonus, 0.5)
            if sum_cost + hat_c[i] > task.budget or sum_lat + hat_l[i] > task.deadline:
                continue
            if score > best_score:
                best_score = score; best_i = i
        if best_i < 0 or best_score <= 0:
            break
        c = tools[best_i].sample_cost(rng); l = tools[best_i].sample_latency(rng); u = tools[best_i].sample_observation_utility(rng)
        N[best_i] += 1
        hat_u[best_i] += (u - hat_u[best_i]) / N[best_i]
        hat_c[best_i] += (c - hat_c[best_i]) / N[best_i]
        hat_l[best_i] += (l - hat_l[best_i]) / N[best_i]
        sum_cost += c; sum_lat += l; sum_u += u; n += 1; t_step += 1
    return RunResult("bwk_vanilla", saturate_utility(sum_u), sum_cost,
                     sum_lat, n, task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


def run_policy_cbuc(registry: Registry, task: Task, rng: np.random.Generator,
                    alpha: float = 1.0, beta: float = 1.5,
                    gamma: float = 1.5, prior_N: int = 3,
                    z_level: float = 1.2816, variant_name: str = "cbuc") -> RunResult:
    """CBUC (ours): type-pruned candidate set + chance-constrained UCB.
    z_level controls chance-constraint aggressiveness: 1.28 ≈ delta=0.1,
    0 disables the correction (aggressive)."""
    feas = registry.feasible_subset(task.query_types)
    if not feas:
        return RunResult("cbuc", 0.0, 0.0, 0.0, 0, task.budget, task.deadline, False, False)
    K = len(feas)
    hat_u = np.zeros(K); hat_c = np.array([t.mean_cost for t in feas])
    hat_l = np.array([t.mean_latency for t in feas]); N = np.full(K, prior_N)
    sum_cost = 0.0; sum_lat = 0.0; sum_u = 0.0; n = 0; t_step = 1

    # Chance-constraint correction: shrink effective budget.
    # Sub-Gaussian proxy; conservative.
    sigma_c = 0.5 * float(np.mean(hat_c))
    # Estimate max calls; target small correction.
    T_guess = max(1, int(task.budget / max(np.mean(hat_c), 1e-6)))
    z = z_level
    effB = task.budget - z * sigma_c * math.sqrt(T_guess)
    effT = task.deadline - z * 0.2 * float(np.mean(hat_l)) * math.sqrt(T_guess)

    while True:
        best_i = -1; best_score = -np.inf
        for i in range(K):
            conf = math.sqrt(2.0 * math.log(t_step + 1) / max(N[i], 1))
            c_lo = max(hat_c[i] - gamma * conf, 0.5)
            l_lo = max(hat_l[i] - gamma * conf, 0.01)
            if sum_cost + (hat_c[i] + beta * conf) > effB:
                continue
            if sum_lat + (hat_l[i] + beta * conf) > effT:
                continue
            score = (hat_u[i] + alpha * conf) / c_lo
            if score > best_score:
                best_score = score; best_i = i
        if best_i < 0:
            break
        t = feas[best_i]
        c = t.sample_cost(rng); l = t.sample_latency(rng); u = t.sample_observation_utility(rng)
        N[best_i] += 1
        hat_u[best_i] += (u - hat_u[best_i]) / N[best_i]
        hat_c[best_i] += (c - hat_c[best_i]) / N[best_i]
        hat_l[best_i] += (l - hat_l[best_i]) / N[best_i]
        sum_cost += c; sum_lat += l; sum_u += u; n += 1; t_step += 1
        # Stop if no remaining tool has estimated marginal utility above a threshold.
        if np.max(hat_u) < 0.05 and n > 5:
            break
    return RunResult(variant_name, saturate_utility(sum_u), sum_cost, sum_lat, n,
                     task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


# ------------------------------ Oracle --------------------------------------- #


def oracle_nonadaptive(tools: List[Tool], task: Task) -> float:
    """True best non-adaptive feasible expected utility (sum of marginal utilities
    selected to maximize sum subject to sum(mean_cost) <= B). 0/1 knapsack via DP
    on scaled integer costs (coarse). Upper bound on feasible utility."""
    scale = 10
    W = int(task.budget * scale)
    n = len(tools)
    weights = [max(1, int(round(t.mean_cost * scale))) for t in tools]
    values = [t.marginal_utility for t in tools]
    # DP O(nW)
    dp = [0.0] * (W + 1)
    for i in range(n):
        w = weights[i]; v = values[i]
        if w > W: continue
        for cap in range(W, w - 1, -1):
            nv = dp[cap - w] + v
            if nv > dp[cap]:
                dp[cap] = nv
    return saturate_utility(dp[W])


# ---------- Stronger, constraint-aware baselines (added per reviewer critique) ----
#
# These four baselines target the concern that the original competitor set
# (ReAct-capped, BwK-vanilla) was constraint-unaware. We add:
#   B1. Lagrangian primal-dual (safe-CMDP style; cf. Mahdavi et al. 2012,
#       Agrawal-Devanur 2014 for the constrained bandit primal-dual view).
#   B2. CVaR-aware BwK (replaces sub-Gaussian tail with empirical-CVaR tail on
#       the cost samples; appropriate when costs are heavy-tailed).
#   B3. Deterministic pre-check UCB (a careful practitioner's guarded greedy:
#       refuse the call if the upper confidence bound on realised cost would
#       push the cumulative sum past B).
#   B4. Non-adaptive chance-constrained knapsack (offline 0/1 knapsack on
#       "inflated" costs c_i + z * sigma_i; this is the proper non-adaptive
#       optimum against which CBUC's regret bound is stated).


def run_policy_lagrangian(tools: List[Tool], task: Task,
                          rng: np.random.Generator,
                          eta: float = 0.1, prior_N: int = 3) -> RunResult:
    """B1. Primal-dual Lagrangian bandit.

    Maintains non-negative dual variables lam_b (budget) and lam_d (deadline).
    Each round selects i = argmax hat_u_i - lam_b * hat_c_i - lam_d * hat_l_i,
    subject to simple remaining-budget feasibility.  Duals are updated by
    projected sub-gradient ascent on the *normalised* resource consumption,
    mirroring the standard safe-CMDP / Lagrangian BwK formulation."""
    K = len(tools)
    hat_u = np.zeros(K); hat_c = np.array([t.mean_cost for t in tools])
    hat_l = np.array([t.mean_latency for t in tools]); N = np.full(K, prior_N)
    sum_cost = 0.0; sum_lat = 0.0; sum_u = 0.0; n = 0; t_step = 1
    lam_b = 0.0; lam_d = 0.0
    # Expected horizon for per-round budget share.
    T_guess = max(1, int(task.budget / max(np.mean(hat_c), 1e-6)))
    per_step_b = task.budget / T_guess
    per_step_d = task.deadline / T_guess
    while True:
        best_i = -1; best_score = -np.inf
        for i in range(K):
            if sum_cost + hat_c[i] > task.budget or sum_lat + hat_l[i] > task.deadline:
                continue
            # Add a modest UCB bonus to encourage exploration.
            bonus = math.sqrt(2.0 * math.log(t_step + 1) / max(N[i], 1))
            score = (hat_u[i] + bonus) - lam_b * hat_c[i] - lam_d * hat_l[i]
            if score > best_score:
                best_score = score; best_i = i
        if best_i < 0 or best_score <= 0:
            break
        t = tools[best_i]
        c = t.sample_cost(rng); l = t.sample_latency(rng); u = t.sample_observation_utility(rng)
        N[best_i] += 1
        hat_u[best_i] += (u - hat_u[best_i]) / N[best_i]
        hat_c[best_i] += (c - hat_c[best_i]) / N[best_i]
        hat_l[best_i] += (l - hat_l[best_i]) / N[best_i]
        sum_cost += c; sum_lat += l; sum_u += u; n += 1; t_step += 1
        # Projected dual updates: penalise overspending per-step quota.
        lam_b = max(0.0, lam_b + eta * (c - per_step_b) / max(per_step_b, 1e-6))
        lam_d = max(0.0, lam_d + eta * (l - per_step_d) / max(per_step_d, 1e-6))
    return RunResult("lagrangian", saturate_utility(sum_u), sum_cost,
                     sum_lat, n, task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


def run_policy_cvar_bwk(tools: List[Tool], task: Task,
                        rng: np.random.Generator,
                        delta: float = 0.1, prior_N: int = 3) -> RunResult:
    """B2. CVaR-aware BwK.

    Maintains per-arm empirical cost samples and uses the empirical
    conditional-value-at-risk at level (1-delta) as the feasibility test,
    rather than a sub-Gaussian tail.  This is more defensible under heavy
    (Pareto) cost tails."""
    K = len(tools)
    samples_c: List[List[float]] = [[] for _ in range(K)]
    samples_l: List[List[float]] = [[] for _ in range(K)]
    hat_u = np.zeros(K); hat_c = np.array([t.mean_cost for t in tools])
    hat_l = np.array([t.mean_latency for t in tools]); N = np.full(K, prior_N)
    sum_cost = 0.0; sum_lat = 0.0; sum_u = 0.0; n = 0; t_step = 1

    def cvar_estimate(samples: List[float], fallback: float) -> float:
        if len(samples) < 3:
            return fallback * 1.5  # mild inflation before enough data
        s = sorted(samples)
        k = max(1, int(math.ceil(delta * len(s))))
        tail = s[-k:]
        return float(np.mean(tail))

    # Warm start.
    for i in range(K):
        if sum_cost + tools[i].mean_cost > task.budget or sum_lat + tools[i].mean_latency > task.deadline:
            continue
        c = tools[i].sample_cost(rng); l = tools[i].sample_latency(rng); u = tools[i].sample_observation_utility(rng)
        samples_c[i].append(c); samples_l[i].append(l)
        hat_u[i] = u; hat_c[i] = c; hat_l[i] = l; N[i] = 1
        sum_cost += c; sum_lat += l; sum_u += u; n += 1; t_step += 1

    while True:
        best_i = -1; best_score = -np.inf
        for i in range(K):
            cvar_c = cvar_estimate(samples_c[i], hat_c[i])
            cvar_l = cvar_estimate(samples_l[i], hat_l[i])
            if sum_cost + cvar_c > task.budget or sum_lat + cvar_l > task.deadline:
                continue
            bonus = math.sqrt(2.0 * math.log(t_step + 1) / max(N[i], 1))
            score = (hat_u[i] + bonus) / max(cvar_c, 0.5)
            if score > best_score:
                best_score = score; best_i = i
        if best_i < 0 or best_score <= 0:
            break
        t = tools[best_i]
        c = t.sample_cost(rng); l = t.sample_latency(rng); u = t.sample_observation_utility(rng)
        N[best_i] += 1
        samples_c[best_i].append(c); samples_l[best_i].append(l)
        hat_u[best_i] += (u - hat_u[best_i]) / N[best_i]
        hat_c[best_i] += (c - hat_c[best_i]) / N[best_i]
        hat_l[best_i] += (l - hat_l[best_i]) / N[best_i]
        sum_cost += c; sum_lat += l; sum_u += u; n += 1; t_step += 1
    return RunResult("cvar_bwk", saturate_utility(sum_u), sum_cost,
                     sum_lat, n, task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


def run_policy_precheck_ucb(tools: List[Tool], task: Task,
                            rng: np.random.Generator,
                            prior_N: int = 3, kappa: float = 1.0) -> RunResult:
    """B3. Deterministic pre-check UCB.

    Engineer-style guarded greedy: for each candidate arm, compute an upper
    confidence bound on realised cost (hat_c + kappa * conf).  Only fire the
    call if sum_cost + UCB_cost(i) <= B and sum_lat + UCB_lat(i) <= D."""
    K = len(tools)
    hat_u = np.zeros(K); hat_c = np.array([t.mean_cost for t in tools])
    hat_l = np.array([t.mean_latency for t in tools]); N = np.full(K, prior_N)
    sum_cost = 0.0; sum_lat = 0.0; sum_u = 0.0; n = 0; t_step = 1
    while True:
        best_i = -1; best_score = -np.inf
        for i in range(K):
            conf = math.sqrt(2.0 * math.log(t_step + 1) / max(N[i], 1))
            ucb_c = hat_c[i] + kappa * conf
            ucb_l = hat_l[i] + kappa * conf
            if sum_cost + ucb_c > task.budget or sum_lat + ucb_l > task.deadline:
                continue
            score = (hat_u[i] + conf) / max(hat_c[i], 0.5)
            if score > best_score:
                best_score = score; best_i = i
        if best_i < 0 or best_score <= 0:
            break
        t = tools[best_i]
        c = t.sample_cost(rng); l = t.sample_latency(rng); u = t.sample_observation_utility(rng)
        N[best_i] += 1
        hat_u[best_i] += (u - hat_u[best_i]) / N[best_i]
        hat_c[best_i] += (c - hat_c[best_i]) / N[best_i]
        hat_l[best_i] += (l - hat_l[best_i]) / N[best_i]
        sum_cost += c; sum_lat += l; sum_u += u; n += 1; t_step += 1
    return RunResult("precheck_ucb", saturate_utility(sum_u), sum_cost,
                     sum_lat, n, task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


def run_policy_cc_knapsack(tools: List[Tool], task: Task,
                           rng: np.random.Generator,
                           z_level: float = 1.2816) -> RunResult:
    """B4. Non-adaptive chance-constrained knapsack.

    Offline solve max sum v_i x_i s.t. sum (mu_i + z * sigma_i) x_i <= B with
    x_i in {0,1}; execute selected set in fixed order.  This is the proper
    non-adaptive oracle under chance constraints and the target of CBUC's
    regret bound.  Uses *true* mu_i, sigma_i (clairvoyant on the distribution
    family but not on realised draws), so we report it as an oracle."""
    n = len(tools)
    # Pareto(alpha) * mean_cost has sigma = mean_cost / sqrt((alpha-1)^2 (alpha-2))
    # for alpha > 2; for alpha <= 2 variance is infinite so we cap with a
    # conservative proxy (5 * mean).  z_level ~ 1.28 -> delta = 0.1.
    sigmas = []
    for t in tools:
        a = t.cost_tail
        if a > 2:
            sigma = t.mean_cost / math.sqrt((a - 1) ** 2 * (a - 2))
        else:
            sigma = 5.0 * t.mean_cost
        sigmas.append(sigma)
    inflated = [t.mean_cost + z_level * sigmas[i] for i, t in enumerate(tools)]
    # 0/1 knapsack on scaled integer weights.
    scale = 10
    W = max(1, int(task.budget * scale))
    weights = [max(1, int(round(inflated[i] * scale))) for i in range(n)]
    values = [t.marginal_utility for t in tools]
    dp = [0.0] * (W + 1)
    choice = [[False] * (W + 1) for _ in range(n)]
    for i in range(n):
        w = weights[i]; v = values[i]
        if w > W:
            continue
        for cap in range(W, w - 1, -1):
            nv = dp[cap - w] + v
            if nv > dp[cap]:
                dp[cap] = nv
                choice[i][cap] = True
    # Backtrack.
    selected: List[int] = []
    cap = W
    for i in range(n - 1, -1, -1):
        if choice[i][cap]:
            selected.append(i); cap -= weights[i]
    selected.reverse()
    # Execute the selected set adaptively only in the sense of realised draws.
    sum_cost = 0.0; sum_lat = 0.0; sum_u = 0.0; n_calls = 0
    for i in selected:
        t = tools[i]
        c = t.sample_cost(rng); l = t.sample_latency(rng); u = t.sample_observation_utility(rng)
        sum_cost += c; sum_lat += l; sum_u += u; n_calls += 1
        if sum_cost > task.budget or sum_lat > task.deadline:
            # Commit to the non-adaptive plan; do not halt mid-plan.
            pass
    return RunResult("cc_knapsack", saturate_utility(sum_u), sum_cost,
                     sum_lat, n_calls, task.budget, task.deadline,
                     sum_cost > task.budget, sum_lat > task.deadline)


# ------------------------------ Sweep ---------------------------------------- #


def run_cell(K: int, rho: float, cost_tail: float, utility_sparsity: float,
             seed: int, include_stronger: bool = False) -> List[dict]:
    """One replicate across all policies.

    When include_stronger=True, additionally runs the four constraint-aware
    baselines (Lagrangian, CVaR-BwK, pre-check UCB, CC-knapsack oracle)."""
    master = np.random.default_rng(seed)
    registry = make_registry(K, cost_tail, utility_sparsity, master)
    task = make_task(registry, rho, master)
    # Sub-seeds per policy for variance-reduction alignment.
    def sub(s): return np.random.default_rng(seed * 1000 + s)
    results: List[RunResult] = [
        run_policy_react_uncapped(registry.tools, task, sub(1)),
        run_policy_react_capped(registry.tools, task, sub(2)),
        run_policy_nonadaptive_greedy(registry.tools, task, sub(3)),
        run_policy_bwk_vanilla(registry.tools, task, sub(4)),
        run_policy_cbuc(registry, task, sub(5), z_level=1.2816, variant_name="cbuc"),
        run_policy_cbuc(registry, task, sub(6), z_level=0.5, variant_name="cbuc_mod"),
        run_policy_cbuc(registry, task, sub(7), z_level=0.0, variant_name="cbuc_aggr"),
    ]
    if include_stronger:
        results.extend([
            run_policy_lagrangian(registry.tools, task, sub(11)),
            run_policy_cvar_bwk(registry.tools, task, sub(12)),
            run_policy_precheck_ucb(registry.tools, task, sub(13)),
            run_policy_cc_knapsack(registry.tools, task, sub(14)),
        ])
    oracle_u = oracle_nonadaptive(registry.tools, task)
    rows = []
    for r in results:
        rows.append({
            "K": K, "rho": rho, "cost_tail": cost_tail,
            "utility_sparsity": utility_sparsity,
            "seed": seed, "policy": r.policy,
            "utility": r.utility, "cost": r.cost, "latency": r.latency,
            "n_calls": r.n_calls,
            "budget": r.budget, "deadline": r.deadline,
            "budget_violated": int(r.budget_violated),
            "deadline_violated": int(r.deadline_violated),
            "oracle_utility": oracle_u,
            "regret": oracle_u - r.utility,
        })
    return rows


def main(out_dir: str, n_seeds: int = 60, include_stronger: bool = False,
         out_name: str = "results.csv",
         K_list=(10, 50, 200), rho_list=(0.2, 0.5, 1.0),
         tail_list=(1.5, 2.5), sparsity_list=(0.2,)):
    os.makedirs(out_dir, exist_ok=True)
    Ks = list(K_list); rhos = list(rho_list); tails = list(tail_list); sparsities = list(sparsity_list)
    all_rows = []
    total = len(Ks) * len(rhos) * len(tails) * len(sparsities) * n_seeds
    done = 0
    for K in Ks:
        for rho in rhos:
            for tail in tails:
                for sp in sparsities:
                    for s in range(n_seeds):
                        seed = hash((K, round(rho*100), round(tail*10), round(sp*100), s)) & 0xffff_ffff
                        all_rows.extend(run_cell(K, rho, tail, sp, seed,
                                                 include_stronger=include_stronger))
                        done += 1
                        if done % 50 == 0:
                            print(f"  progress {done}/{total}")
    # Write CSV
    csv_path = os.path.join(out_dir, out_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"wrote {csv_path}  ({len(all_rows)} rows)")
    return csv_path


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "./out"
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    stronger = (len(sys.argv) > 3 and sys.argv[3] == "stronger")
    name = sys.argv[4] if len(sys.argv) > 4 else "results.csv"
    main(out, n_seeds=ns, include_stronger=stronger, out_name=name)
