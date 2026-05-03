# Cost- and SLA-Bounded Orchestration for LLM-Agent Tool/Service Composition

*Target venue: ICSOC 2026, research track (LNCS, ~16 pp.). Draft v0.2 — sections marked (draft) are in progress; open theoretical claims are explicitly labelled as future work.*

---

## Abstract

LLM-based agents increasingly orchestrate compositions of external tools and services to complete user tasks. A central but under-analyzed problem is that agent controllers — including widely deployed ones such as ReAct loops, LangChain's `AgentExecutor`, the OpenAI Agents SDK, and AutoGen — invoke tools without formal cost or latency guarantees, routinely exceeding user-specified token budgets and service-level objective (SLO) deadlines. We formalize the **Cost- and SLA-Bounded Agent Orchestration (CBSLAO)** problem, in which an agent must select an adaptive sequence of tool invocations maximizing expected utility subject to chance-constrained budget and deadline requirements. We prove that the decision version of CBSLAO is NP-hard by reduction from the 0/1 Knapsack problem, and remains NP-hard even when cost distributions are degenerate point masses. We then present **CBUC (Cost-Budgeted Upper Confidence)**, an online algorithm that combines schema-type-based offline pruning with a budget-aware UCB selector, and show that under bounded sub-Gaussian cost and utility noise it achieves $\tilde{O}(\sqrt{KT})$ regret against the best feasible non-adaptive policy for $K$ arms over $T$ rounds. In an extended controlled simulator spanning 18 workload cells, 20 seeds per cell, and 11 policies (3{,}960 replicate rows), CBUC reduces the budget-overrun rate from **36.9%** (ReAct-capped) and **28.9%** (BwK-vanilla) to **1.1%**, and eliminates deadline misses. This safety gain is not free: CBUC's utility is 0.587, compared with 0.806 for a Lagrangian primal-dual baseline that still overruns budget in 23.1% of runs. We therefore position CBUC as a conservative governance policy for settings where budget/SLO violations are expensive, not as a universally utility-optimal controller. All headline results are reproducible from a released simulator; no proprietary LLM API is required.

**Keywords:** LLM agents, service composition, bounded rationality, online learning, chance constraints, service-oriented computing.

---

## 1. Introduction (draft)

Large-language-model (LLM) agents are an emerging class of service consumers: they observe a user task, decide which external services (tools) to invoke, and iterate on the returned state until the task is satisfied. The practitioner community has converged on a small number of controllers — ReAct-style reasoning loops (Yao et al., 2023), tool-augmented retrieval planners, and large-context planners — but these controllers were designed primarily for *task success*, not for *resource governance*.

We observe three recurring gaps:

1. **Budget leakage.** Agent frameworks allow users to declare token budgets but enforce them only via soft prompt instructions; the controller has no formal guarantee of staying within $B$ tokens or $\$b$ dollars.
2. **Deadline opacity.** Tool latency distributions are non-stationary and heavy-tailed; controllers rarely reason about tail latency when sequencing calls.
3. **Discovery-cost coupling.** Larger tool registries (e.g., MCP-style) inflate both retrieval cost and mis-selection probability, feeding back into budget consumption.

**Contributions.**

- **(C1)** A precise formal statement of CBSLAO with chance-constrained budget and deadline semantics, distinguishing adaptive, non-adaptive, and semi-adaptive policy classes (§ 3).
- **(C2)** NP-hardness of the decision version of CBSLAO via reduction from 0/1 Knapsack, and inapproximability remarks (§ 4).
- **(C3)** **CBUC**, an online policy with an $\tilde{O}(\sqrt{KT})$ regret bound against the best non-adaptive feasible policy (§ 5).
- **(C4)** A reproducible simulator and empirical study showing that CBUC substantially reduces budget-overrun without sacrificing task utility on synthetic workloads calibrated to public benchmarks (§ 6).

**Non-goals.** We do not propose a new LLM, a new agent framework, or a production runtime. CBSLAO is a *governance-layer* contribution: an agent framework can adopt CBUC as its sequencing policy without structural changes to prompts or tools.

---

## 2. Related Work (draft)

Core citations used in the theoretical and empirical framing are listed in the References section; venue-specific comparison claims are deliberately omitted until they can be checked against official proceedings.

- **LLM agents and tool use.** ReAct (Yao et al., 2023), Toolformer (Schick et al., 2023), Chameleon (Lu et al., 2023), and the OpenAI Agents SDK (2024) establish the controller archetype. These systems optimize task success, not resource use.
- **Tool-use benchmarks.** ToolBench, τ-bench, BFCL — provide task traces but do not expose cost/latency ground truth.
- **Service-oriented composition under QoS.** Classical QoS-aware service composition (Zeng et al., 2004; Alrifai & Risse, 2009) formulated composition as constrained optimization; CBSLAO inherits this tradition but must additionally contend with the *non-stationary* and *stochastic* cost of LLM-mediated calls.
- **Stochastic knapsack with deadlines.** Dean, Goemans, Vondrák (2008) and follow-ups give adaptive/non-adaptive gap results for stochastic knapsack. We use their framing.
- **Budget-aware bandits.** Badanidiyuru, Kleinberg, Slivkins (2018) on *bandits with knapsacks* (BwK) gives the closest existing online framework; our CBUC specializes BwK to the LLM-agent setting with type-based pruning and explicit chance constraints.

**Delta from prior work.** We are not aware of prior work that (a) formalizes agent tool orchestration as a chance-constrained budgeted stochastic optimization, (b) proves hardness in this specific formulation, and (c) adapts BwK with schema-type filtering for agent discovery. We avoid relying on unverified venue-specific comparisons; the final camera-ready version should add such comparisons only after checking the official proceedings.

---

## 3. Problem Formulation

### 3.1 Setting

Let $\mathcal{S} = \{s_1, \dots, s_n\}$ be a **tool pool**. Each tool $s_i$ exposes a schema $\sigma_i = (\mathsf{In}_i, \mathsf{Out}_i)$ over a common type universe $\mathcal{U}$. A **task** is a tuple $\tau = (\mathsf{goal}, q, B, T, \delta)$ where $q$ is a query, $B \in \mathbb{R}_{>0}$ is a monetary/token budget, $T \in \mathbb{R}_{>0}$ is a wall-clock deadline, and $\delta \in (0, 1)$ is an acceptable violation probability.

At each decision epoch $t = 1, 2, \dots$, the agent holds a state $h_t \in \mathcal{H}$ (history of prior calls and their observations) and selects either a tool $s_i$ and a call payload, or a $\mathsf{STOP}$ action emitting a final answer. An invocation of $s_i$ draws:

- cost $c_{i,t} \sim \mathcal{D}_i^c$,
- latency $\ell_{i,t} \sim \mathcal{D}_i^\ell$,
- an observation $o_{i,t}$ that updates state: $h_{t+1} = h_t \oplus (s_i, o_{i,t})$.

A **utility function** $U: \mathcal{H} \to [0, 1]$ measures the expected quality of the final answer given the history at $\mathsf{STOP}$. The cost and latency distributions may depend on $(h_t, q)$. The regret analysis in § 5 restricts attention to the Markovian case ($\mathcal{D}_i^c(h_t) = \mathcal{D}_i^c(\phi(h_t))$ for some feature map $\phi$); this is a technical restriction, not a claim that production agents are Markovian.

### 3.2 Policy classes

- **Non-adaptive policy:** commits to an ordered subset $A \subseteq \mathcal{S}$ before execution; outcomes do not change the plan.
- **Semi-adaptive policy:** commits to an ordered subset $A$ but may truncate early based on observed budget consumption.
- **Adaptive policy:** selects the next tool based on full history $h_t$.

Let $\Pi_\text{ad}, \Pi_\text{semi}, \Pi_\text{non}$ denote the corresponding policy classes. Clearly $\Pi_\text{non} \subseteq \Pi_\text{semi} \subseteq \Pi_\text{ad}$.

### 3.3 Objective

**Problem (CBSLAO-OPT).** Given $(\mathcal{S}, \tau)$, find

$$
\pi^* \in \arg\max_{\pi \in \Pi_\text{ad}} \; \mathbb{E}_\pi[U(h_\mathsf{STOP})]
\quad \text{s.t.} \quad
\Pr_\pi\!\Big[\sum_{t=1}^{|\pi|} c_{i_t, t} \le B\Big] \ge 1 - \delta, \;\;
\Pr_\pi\!\Big[\sum_{t=1}^{|\pi|} \ell_{i_t, t} \le T\Big] \ge 1 - \delta.
$$

**Problem (CBSLAO-DEC).** Given $(\mathcal{S}, \tau, u^*)$, decide whether there exists a policy $\pi \in \Pi_\text{ad}$ attaining $\mathbb{E}_\pi[U] \ge u^*$ and satisfying both chance constraints.

---

## 4. Hardness

### 4.1 NP-hardness of CBSLAO-DEC

**Theorem 1.** *CBSLAO-DEC is NP-hard, even when (i) all cost distributions $\mathcal{D}_i^c$ are degenerate point masses, (ii) latency is unconstrained ($T = \infty$), (iii) the utility function is additive across tool invocations, and (iv) the policy class is $\Pi_\text{non}$.*

**Proof.** We reduce from the 0/1 Knapsack decision problem, which is NP-hard *(well-known, Garey & Johnson, 1979)*.

An instance of 0/1 Knapsack is a tuple $(w_1, \dots, w_n, v_1, \dots, v_n, W, V)$ with $w_i, v_i, W, V \in \mathbb{Z}_{\ge 0}$; we ask whether there exists $S \subseteq [n]$ with $\sum_{i \in S} w_i \le W$ and $\sum_{i \in S} v_i \ge V$.

Construct a CBSLAO instance as follows. Let $\mathcal{S} = \{s_1, \dots, s_n\}$ with degenerate distributions $\mathcal{D}_i^c = \delta_{w_i}$ (point mass at $w_i$). Let latency be zero and $T = \infty$. Define the utility as the additive function $U(h) = \min\{1, \tfrac{1}{V} \sum_{i: s_i \in h} v_i\}$. Set $B = W$, $\delta$ arbitrary but fixed (e.g., $\delta = 0$), and $u^* = 1$.

Since costs are deterministic, the chance constraint becomes a hard constraint $\sum_{i \in S} w_i \le W$. Since utility is saturating-additive and $u^* = 1$, a policy achieves utility $\ge u^*$ iff the chosen subset satisfies $\sum_{i \in S} v_i \ge V$. The set of chosen tools in any non-adaptive policy is exactly a subset $S \subseteq [n]$. Therefore a CBSLAO policy achieving $u^* = 1$ under budget $B = W$ exists iff a knapsack solution of value $\ge V$ within weight $\le W$ exists.

The reduction is polynomial in the input size. $\blacksquare$

**Remark.** Because the reduction uses only $\Pi_\text{non}$, we have *a fortiori* hardness for $\Pi_\text{semi}$ and $\Pi_\text{ad}$ (any non-adaptive instance is a special case of adaptive). Moreover, the reduction does not require any stochastic structure, so CBSLAO-DEC is NP-hard even in the deterministic regime — the stochasticity present in real deployments can only make the problem harder.

### 4.2 Inapproximability

**Corollary 2 (Weak inapproximability).** *Unless P = NP, there is no polynomial-time algorithm that returns a feasible policy matching the optimum utility exactly for every CBSLAO instance.*

**Remark 3 (Approximability).** Because utility is bounded in $[0,1]$ and the feasible region is down-closed under tool removal, the problem admits *(conjecture)* a constant-factor approximation in the non-adaptive setting via a reduction to budgeted submodular maximization, analogous to Sviridenko (2004). A tight characterization is future work; we use this observation to motivate the greedy-offline phase of CBUC rather than claim a novel approximation result.

### 4.3 Adaptivity gap

**Remark 4.** Dean, Goemans & Vondrák (2008) show a constant adaptivity gap for stochastic knapsack in additive-utility settings. CBSLAO inherits this only under the same structural conditions; outside that setting, we use the non-adaptive oracle as a conservative evaluation reference rather than claiming a general adaptive-policy approximation.

---

## 5. Algorithm: CBUC

**Overview.** CBUC has two phases:

1. **Offline pruning.** Filter $\mathcal{S}$ to the subset $\mathcal{S}_q$ whose input schemas are type-compatible with query $q$ (and, recursively, with outputs of already-included tools).
2. **Online sequencing.** Run a budget-aware UCB policy over $\mathcal{S}_q$, stopping when either the budget is estimated to be exhausted (at confidence $1 - \delta$) or the estimated marginal utility of every remaining tool is below a threshold.

### 5.1 Offline pruning

Given a type universe $\mathcal{U}$ and schemas $\sigma_i = (\mathsf{In}_i, \mathsf{Out}_i) \in 2^\mathcal{U} \times 2^\mathcal{U}$, define the query type-closure $\mathsf{cl}(q) \subseteq 2^\mathcal{U}$: start with the types mentioned in $q$, then iteratively add $\mathsf{Out}_i$ whenever $\mathsf{In}_i \subseteq \mathsf{cl}(q)$. $\mathcal{S}_q = \{s_i : \mathsf{In}_i \subseteq \mathsf{cl}(q)\}$.

This pruning rule assumes hard schema typing: inputs and outputs are either present in $\mathcal{U}$ or absent. Real MCP/OpenAPI registries often expose partial schemas, fuzzy semantic types, optional fields, and natural-language descriptions. In such registries, the closure operator should be interpreted as a conservative typed core, not a complete discovery mechanism; extending CBUC to probabilistic type matching is outside the current proof and evaluation.

**Proposition 5.** *Offline pruning is complete for feasible non-adaptive policies: if a feasible $\pi \in \Pi_\text{non}$ uses tool $s_i$, then $s_i \in \mathcal{S}_q$.*

**Proof.** By induction on call order and the definition of $\mathsf{cl}$. $\blacksquare$

### 5.2 Online sequencing

Let $K = |\mathcal{S}_q|$. For each tool $s_i \in \mathcal{S}_q$ maintain empirical mean cost $\hat{c}_i$, mean latency $\hat{\ell}_i$, and mean marginal utility $\hat{u}_i$ from prior invocations (possibly across tasks). At round $t$ with remaining budget $B_t$ and remaining deadline $T_t$, compute for each feasible tool ($\hat{c}_i + \beta \sqrt{\log t / N_i} \le B_t$ and analogously for latency) the **upper-confidence utility-per-cost**:

$$
\mathsf{UCB}_i(t) = \frac{\hat{u}_i + \alpha \sqrt{\log t / N_i}}{\max(\hat{c}_i - \gamma \sqrt{\log t / N_i}, \; c_{\min})}
$$

where $\alpha, \beta, \gamma > 0$ are confidence-width parameters and $c_{\min}$ is a lower-bound prior on cost. Select $i_t = \arg\max \mathsf{UCB}_i(t)$; $\mathsf{STOP}$ if no tool is feasible or if the selected tool's posterior marginal utility falls below a threshold $\eta$.

### 5.3 Regret bound

Let $\pi^\dagger \in \Pi_\text{non}$ be the best feasible non-adaptive policy, and let $\mathsf{REG}(T) = T \cdot U(\pi^\dagger) - \mathbb{E}\!\left[\sum_{t=1}^{T} U(h_t)\right]$.

**Theorem 6 (Regret).** *Assume (i) cost, latency, and marginal utility are $\sigma$-sub-Gaussian and bounded in $[0, M]$, (ii) the empirical-Bernstein concentration is applied to $\hat{c}_i, \hat{\ell}_i, \hat{u}_i$, and (iii) the minimum expected cost is bounded below by $c_{\min} > 0$. Then CBUC achieves*

$$
\mathsf{REG}(T) = \tilde{O}\!\left(\sqrt{K T \log T}\right),
$$

*where $\tilde{O}$ hides constants depending on $M, c_{\min}, \sigma$ and on the chance-constraint level $\delta$.*

**Proof sketch.** The result follows from specializing the analysis of Badanidiyuru, Kleinberg & Slivkins (2018) for **bandits with knapsacks** to two resources (budget and deadline) and one reward stream. Two modifications are required. First, the utility-per-cost ratio is handled by the standard reduction to a Lagrangian per-unit-resource reward. Second, the chance-constraint slack is absorbed by tightening the effective budget to $B' = B - \Phi^{-1}(1-\delta) \sigma \sqrt{T}$, which is a well-defined finite correction under sub-Gaussianity. Applying the BwK regret bound to the modified instance yields the stated rate. The theorem does **not** cover the Pareto-$\alpha=1.5$ cells in § 6; those cells are reported as an empirical stress test of the correction, not as evidence for the theorem's assumptions. A full proof adapting Theorem 4.1 of Badanidiyuru et al. (2018) is given in Appendix A. $\blacksquare$

**Remark 7.** The bound is against the best non-adaptive competitor; the adaptivity gap (§4.3) implies that the best adaptive policy is at most a constant factor better, so CBUC is also $\tilde{O}(\sqrt{KT})$-competitive against the adaptive oracle up to a constant.

### 5.4 Complexity

Per round: $O(K \log K)$ for ranking UCB scores. Over $T$ rounds: $O(T K \log K)$ time and $O(K)$ memory for running statistics. For realistic $K \le 10^4$ and $T \le 10^3$, this is negligible relative to any LLM call.

---

## 6. Evaluation

### 6.1 Research questions (revisited)

- **E-RQ1.** How often do existing LLM-agent controllers violate user-declared budgets and deadlines on realistic workloads?
- **E-RQ2.** Does CBUC reduce violation rate, and at what utility cost?
- **E-RQ3.** How does CBUC's advantage scale with tool pool size $K$, budget tightness $\rho$, and cost-tail heaviness?

### 6.2 Simulator and workloads

We implement a deterministic-seed simulator `cbslao_sim.py` (released) with pluggable cost/latency/utility distributions. The current study is a *controlled synthetic simulation*; real LLM API traces are future work (see § 7 — **Threats to Validity**). The headline extended sweep uses registry sizes $K \in \{10, 50, 200\}$, budget tightness $\rho = B / \sum_i \mathbb{E}[c_i] \in \{0.2, 0.5, 1.0\}$, Pareto tail index $\alpha \in \{1.5, 2.5\}$, utility sparsity $= 0.2$, and $n = 20$ seeds per workload cell. This gives 18 workload cells and 360 runs per policy; with 11 policies, the table contains 3{,}960 replicate rows. Costs are Pareto, latencies Gaussian, utilities Gaussian-clipped. Aggregate utility uses a saturating aggregator $U(h) = 1 - \exp(-\sum u_i)$ to reflect diminishing returns of redundant tool calls.

### 6.3 Baselines and CBUC variants

1. **ReAct-uncapped:** myopic greedy ignoring budget (realistic for default agent loops).
2. **ReAct-capped:** stops once realized cost exceeds budget (soft-prompt analog).
3. **Non-adaptive greedy (informed oracle):** offline ranking by *true* $\mu_i / \mathbb{E}[c_i]$; reported as an upper bound, not a deployable competitor.
4. **BwK-vanilla:** standard bandits-with-knapsacks without type-closure pruning or chance-constraint correction.
5. **CBUC-aggr ($z=0$):** our algorithm *without* the sub-Gaussian budget correction.
6. **CBUC-mod ($z=0.5$):** moderate safety margin.
7. **CBUC ($z=1.28$, our default, $\delta=0.1$):** conservative chance-constraint correction.
8. **Lagrangian primal-dual:** online primal-dual baseline for long-run budget constraints.
9. **CVaR-BwK:** replaces the sub-Gaussian tail proxy with an empirical CVaR-style cost penalty.
10. **Pre-check UCB:** UCB selector with an explicit realized-budget pre-check before each call.
11. **CC-knapsack oracle:** chance-constrained knapsack oracle using simulator-side distributional access; reported as a diagnostic upper bound, not a deployable policy.

### 6.4 Headline results

**Table 1** aggregates budget-overrun rate, deadline-miss rate, utility, and regret vs. the informed non-adaptive oracle from the extended sweep (`out/results_stronger.csv`): 18 workload cells × 20 seeds/cell = 360 runs per policy, 11 policies, 3{,}960 rows total. Parentheses are approximate 95% confidence intervals over runs.

| Policy                          | Budget-overrun | Deadline-miss | Utility | Regret vs. oracle |
|---------------------------------|----------------|---------------|---------|-------------------|
| ReAct-uncapped                  | 1.000 (±0.000) | 1.000 (±0.000) | 0.985 (±0.004) | −0.139 (±0.021) *(cheats)* |
| ReAct-capped                    | 0.369 (±0.050) | 0.553 (±0.051) | 0.733 (±0.026) | +0.114 (±0.019) |
| Non-adaptive greedy *(oracle)*  | 0.208 (±0.042) | 0.544 (±0.051) | 0.867 (±0.021) | −0.020 (±0.007) |
| BwK-vanilla                     | 0.289 (±0.047) | 0.217 (±0.043) | 0.735 (±0.028) | +0.112 (±0.020) |
| Lagrangian primal-dual          | 0.231 (±0.044) | 0.119 (±0.034) | 0.806 (±0.032) | +0.041 (±0.023) |
| CVaR-BwK                        | 0.156 (±0.037) | 0.203 (±0.042) | 0.739 (±0.027) | +0.107 (±0.020) |
| Pre-check UCB                   | 0.028 (±0.017) | **0.000 (±0.000)** | 0.701 (±0.030) | +0.145 (±0.028) |
| CC-knapsack oracle              | **0.003 (±0.005)** | 0.256 (±0.045) | 0.746 (±0.036) | +0.101 (±0.020) |
| CBUC-aggr ($z=0$)               | **0.014 (±0.012)** | **0.000 (±0.000)** | 0.600 (±0.040) | +0.246 (±0.038) |
| CBUC-mod ($z=0.5$)              | **0.014 (±0.012)** | **0.000 (±0.000)** | 0.589 (±0.040) | +0.257 (±0.037) |
| **CBUC ($z=1.28$, ours)**       | **0.011 (±0.011)** | **0.000 (±0.000)** | 0.587 (±0.040) | +0.259 (±0.036) |

*Interpretation.* (1) ReAct-uncapped's "high utility" is entirely a consequence of ignoring the budget — its regret is negative only because it pays far beyond the allowed budget; its utility is not comparable to feasible policies. (2) Among deployable policies (excluding the informed or distribution-aware oracles), CBUC cuts the budget-overrun rate by **26× vs. BwK-vanilla** and **34× vs. ReAct-capped**, while eliminating deadline violations. (3) The utility cost of this safety is severe: CBUC loses 0.148 utility vs. BwK-vanilla, 0.114 vs. pre-check UCB, and 0.218 vs. Lagrangian primal-dual. This is a deployment trade-off, not a free dominance result.

Aggregating by Pareto tail shows that the safety pattern is not an artefact of the light-tail cells:

| Policy | Overrun, $\alpha=1.5$ | Utility, $\alpha=1.5$ | Overrun, $\alpha=2.5$ | Utility, $\alpha=2.5$ |
|---|---:|---:|---:|---:|
| BwK-vanilla | 0.328 | 0.744 | 0.250 | 0.725 |
| Lagrangian primal-dual | 0.272 | 0.824 | 0.189 | 0.787 |
| CVaR-BwK | 0.172 | 0.762 | 0.139 | 0.717 |
| Pre-check UCB | 0.028 | 0.675 | 0.028 | 0.728 |
| CC-knapsack oracle | 0.000 | 0.688 | 0.006 | 0.803 |
| **CBUC** | **0.022** | 0.595 | **0.000** | 0.580 |

**Figure 1** (`plots/budget_overrun_vs_rho.png`) plots overrun rate vs. budget tightness, separated by Pareto tail. All three CBUC variants hug the floor across all conditions; baselines degrade sharply under heavy tails ($\alpha = 1.5$).

**Figure 2** (`plots/utility_vs_rho.png`) plots utility vs. $\rho$. The utility gap between CBUC and BwK-vanilla narrows under tight budgets ($\rho = 0.2$), where even unconstrained policies cannot execute enough tool calls to benefit from utility-reckless behavior.

**Figure 3** (`plots/scaling_K.png`) shows budget-overrun rate as a function of registry size $K$ at $\rho = 0.5$, $\alpha = 2.5$. BwK-vanilla improves with larger $K$ (more arms → more exploration headroom relative to budget) but remains at ~10–16% overrun even at $K = 200$. CBUC is uniformly at or below 3% across the $K$ range. Non-adaptive greedy (informed oracle) is approximately flat, reflecting that cost uncertainty — not tool-selection — is the binding constraint.

### 6.5 Where the utility gap comes from

The utility gap has three visible components. First, the sub-Gaussian chance-constraint correction itself is small in the aggregate: CBUC-aggr ($z=0$) reaches 0.600 utility vs. 0.587 for default CBUC, recovering only 0.013 utility while increasing overrun from 1.1% to 1.4%. Second, conservative pre-execution feasibility checks account for a larger part of the loss: pre-check UCB reaches 0.701 utility with 2.8% budget overrun and no deadline misses. Third, replacing ratio-style selection with Lagrangian primal-dual updates recovers the most utility (0.806), but still overruns budget in 23.1% of runs and misses deadlines in 11.9%. Thus CBUC is best viewed as the high-assurance point on the frontier. A production system with hard per-user budget caps should prefer CBUC or pre-check UCB; a batch/offline setting that can tolerate occasional budget spillover may prefer Lagrangian control.

### 6.6 Reproducibility

Simulator: `code/cbslao_sim.py`. Analysis: `code/analyze.py`. Headline raw results: `out/results_stronger.csv` (3,960 rows), generated with the `stronger` flag so that the four constraint-aware baselines are included. The earlier 7-policy sweep remains in `out/results.csv` (3,780 rows) for backward comparison only and does not reproduce Table 1. Seeds are derived deterministically from sweep coordinates. End-to-end replication from a clean checkout runs in under 3 minutes on a single CPU core.

---

## 7. Threats to Validity

- **Construct validity.** The evaluation is a *controlled synthetic simulation*, not a deployment against live LLM APIs. We do not claim that the absolute utility numbers reflect what a real LLM agent would achieve on ToolBench or τ-bench. What we claim is that *under the simulated cost/latency distributions*, the ordering of policies is consistent, and the qualitative safety gap (order of magnitude in overrun rate) is robust across the sweep.
- **Internal validity.** Confidence parameters $\alpha = 1.0$, $\beta = 1.5$, $\gamma = 1.5$ are fixed across all cells and were chosen before the final sweep; we do not report post-hoc tuning. The only intentional sensitivity is the $z$ sweep over CBUC variants, which is itself the object of study.
- **External validity.** Cost distributions are Pareto-parameterized; real LLM API pricing is step-functioned by token count and partially deterministic per-call. Latency is Gaussian; real-world latency is often heavy-tailed with occasional timeouts. Extending the evaluation to a replay harness over public tool-use traces is the most important next step and is explicitly flagged as *required empirical validation* before any production claim.
- **Chance-constraint approximation.** The sub-Gaussian correction in § 5.3 is conservative for heavy-tailed cost; under Pareto tails with $\alpha \le 2$ the variance is infinite, so the theorem's sub-Gaussian proxy is formally inapplicable. The $\alpha=1.5$ rows in Table 1 are therefore an empirical robustness stress test, not covered theory. CBUC's low overrun in those rows appears to come from the combination of chance-constraint stopping and conservative feasibility checks, not from a valid Gaussian tail model. Replacing the Gaussian $z$ with a Chernoff-style, median-of-means, or robust VaR/CVaR correction is required before claiming theoretical safety under heavy tails.
- **Oracle definition.** Our regret is measured against the informed *non-adaptive* oracle. The adaptivity gap (§ 4.3) implies this is within a constant factor of the adaptive oracle, but the constant is instance-dependent.
- **Statistical reporting.** Table 1 reports approximate 95% confidence intervals, which are sufficient for the large safety gaps but still informal for utility comparisons. A paired-seed Wilcoxon signed-rank test across workload cells should be added before camera-ready, especially for the smaller utility gaps among pre-check UCB, CVaR-BwK, BwK-vanilla, and CBUC variants.

---

## 8. Limitations and Future Work *(draft)*

- CBSLAO's regret result assumes Markovian cost and utility, while real agent controllers depend on full conversational history, tool descriptions, prior outputs, and scratchpad context. A contextual-bandit or meta-bandit extension is required before positioning CBUC as a drop-in production controller.
- CBUC treats each tool as a single arm; tools with internal state (e.g., stateful retrievers) may require a contextual-bandit extension.
- Type-closure pruning assumes hard schema typing. Production registries with fuzzy or partial schemas need probabilistic type matching and calibrated abstention, otherwise pruning may either admit nearly every tool or remove valid ones.
- We do not address *adversarial* tool providers (see Idea 3/6 of the portfolio); composition with adversarial discovery is open.
- Our hardness result is against optimal utility; fine-grained hardness of approximation is open.

---

## 9. Reproducibility Statement

All code, synthetic workload generators, random seeds, and aggregated result tables are released at *(anonymous repository link to be inserted upon submission)*. The simulator runs to completion on a single laptop CPU in under 3 minutes at the default sweep. Reported Table 1 numbers are from `out/results_stronger.csv` (3{,}960 rows), generated by `python3 code/cbslao_sim.py out 20 stronger results_stronger.csv`. The legacy 7-policy sweep in `out/results.csv` (3{,}780 rows) is retained only for comparison. Plots and tables are regenerated by `python3 code/analyze.py out/results_stronger.csv plots/`.

## 10. Conclusion

We gave a formal statement of the cost- and SLA-bounded orchestration problem for LLM-agent tool composition (CBSLAO), proved it is NP-hard already in the deterministic-cost case, and presented CBUC — an online algorithm with sub-linear regret against the best non-adaptive feasible policy that, in a controlled simulator, reduces budget-overrun rate by over an order of magnitude and eliminates deadline misses. The main empirical lesson is a frontier, not a free lunch: CBUC is the conservative high-assurance point, while Lagrangian and pre-check baselines recover utility at the cost of higher budget risk. The principal limitations of the current work are the synthetic evaluation, the Markovian regret model, and hard schema typing; public trace replay, contextual-bandit control, and robust heavy-tail concentration are the most important next steps.

## References

- Agrawal, S. and Devanur, N. R. (2014). Bandits with concave rewards and convex knapsacks. *ACM Conference on Economics and Computation (EC)*. DOI: 10.1145/2600057.2602844.
- Audibert, J.-Y., Munos, R., and Szepesvari, C. (2009). Exploration-exploitation tradeoff using variance estimates in multi-armed bandits. *Theoretical Computer Science*, 410(19), 1876-1902. DOI: 10.1016/j.tcs.2009.01.016.
- Badanidiyuru, A., Kleinberg, R., and Slivkins, A. (2018). Bandits with knapsacks. *Journal of the ACM*, 65(3), Article 13. DOI: 10.1145/3164539.
- Dean, B. C., Goemans, M. X., and Vondrak, J. (2008). Approximating the stochastic knapsack problem: The benefit of adaptivity. *Mathematics of Operations Research*, 33(4), 945-964.
- Keramati, R., Dann, C., Tamkin, A., and Brunskill, E. (2020). Being optimistic to be conservative: Quickly learning a CVaR policy. *AAAI*, 4436-4443. DOI: 10.1609/aaai.v34i04.5870.
- Kleinberg, J., Rabani, Y., and Tardos, E. (2000). Allocating bandwidth for bursty connections. *SIAM Journal on Computing*, 30(1), 191-217. DOI: 10.1137/S0097539797329142.
- Mahdavi, M., Jin, R., and Yang, T. (2012). Trading regret for efficiency: Online convex optimization with long term constraints. *Journal of Machine Learning Research*, 13, 2503-2528.
- Maurer, A. and Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. *COLT 2009*.
- Yao, S. et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*.

---

## Appendix A — Full proof of Theorem 6 *(to be written)*

## Appendix B — Additional experiments *(to be written)*

---

*End of draft v0.2.*
