# Cost- and SLA-Bounded Orchestration for LLM-Agent Tool/Service Composition

**ICSOC 2026 submission** · Research Track · LNCS format

> **CBSLAO** formalizes the problem of orchestrating LLM-agent tool calls under hard budget and SLA constraints, proves it NP-hard, and presents **CBUC** — an online algorithm that reduces budget-overrun rates by **34×** while eliminating deadline misses.

---

## Introduction

LLM-based agents are increasingly deployed as *service consumers*: they receive a user task, decide which external tools or APIs to invoke, and iterate until the task is satisfied. Frameworks like ReAct, LangChain `AgentExecutor`, the OpenAI Agents SDK, and AutoGen have become the de-facto standard for this pattern — but they were designed for *task success*, not *resource governance*.

In practice, three recurring gaps undermine safe deployment:

| Gap | Description |
|---|---|
| **Budget leakage** | Token budgets are enforced via soft prompt instructions only; there is no formal guarantee of staying within a declared dollar or token cap. |
| **Deadline opacity** | Tool latency distributions are non-stationary and heavy-tailed; controllers rarely reason about tail latency when sequencing calls. |
| **Discovery-cost coupling** | Larger tool registries (e.g., MCP-style) inflate both retrieval cost and mis-selection probability, feeding back into budget consumption. |

This work introduces the **Cost- and SLA-Bounded Agent Orchestration (CBSLAO)** problem and the **CBUC (Cost-Budgeted Upper Confidence)** algorithm to address these gaps at the governance layer — without requiring changes to prompts, tools, or the underlying LLM.

### Contributions

- **(C1)** Formal problem statement of CBSLAO with **chance-constrained** budget and deadline semantics, distinguishing adaptive, non-adaptive, and semi-adaptive policy classes.
- **(C2)** **NP-hardness** of CBSLAO-DEC via reduction from 0/1 Knapsack, holding even when cost distributions are degenerate point masses.
- **(C3)** **CBUC**, an online algorithm with an $\tilde{O}(\sqrt{KT})$ regret bound against the best feasible non-adaptive policy.
- **(C4)** A **reproducible simulator** and empirical study across 3,960 replicate runs showing that CBUC reduces budget-overrun by over an order of magnitude while eliminating deadline misses.

---

## Installation

No external LLM API is required. The simulator runs entirely on CPU.

### Requirements

- Python ≥ 3.9
- `numpy`, `pandas`, `matplotlib`, `scipy`

```bash
pip install numpy pandas matplotlib scipy
```

### Clone & Run

```bash
git clone https://github.com/LeoTTTPhat/CBSLAO.git
cd CBSLAO

# Run the full extended sweep (18 workload cells × 20 seeds × 11 policies = 3,960 rows)
# Completes in ~3 minutes on a single CPU core
python3 code/cbslao_sim.py out 20 stronger results_stronger.csv

# Regenerate all plots and tables
python3 code/analyze.py out/results_stronger.csv plots/
```

### Repository Structure

```
CBSLAO/
├── code/
│   ├── cbslao_sim.py       # Main simulator (pluggable distributions)
│   ├── analyze.py          # Analysis + plotting
│   └── ablations.py        # Hyperparameter & replay ablations
├── out/
│   ├── results_stronger.csv        # 3,960-row headline results
│   ├── results.csv                 # Legacy 7-policy sweep (backward compat.)
│   ├── ablation_abg.csv            # α/β/γ hyperparameter sweep
│   ├── ablation_distractors.csv    # Distractor tool ablation
│   └── ablation_replay.csv         # Calibrated-trace replay (BFCL / τ-bench / ToolBench)
├── plots/
│   ├── budget_overrun_vs_rho.png   # Figure 1
│   ├── utility_vs_rho.png          # Figure 2
│   ├── scaling_K.png               # Figure 3
│   ├── ablation_abg.png
│   ├── ablation_distractors.png
│   └── ablation_replay.png
└── paper/
    └── CBSLAO_paper_draft.md       # Full manuscript draft
```

---

## Summary of Main Results

### Workload Setup

The headline evaluation uses a **controlled synthetic simulator** with:
- Registry sizes $K \in \{10, 50, 200\}$
- Budget tightness $\rho = B / \sum_i \mathbb{E}[c_i] \in \{0.2, 0.5, 1.0\}$
- Pareto tail index $\alpha \in \{1.5, 2.5\}$
- **18 workload cells × 20 seeds × 11 policies = 3,960 replicate rows**

### Headline Results (Table 1)

Results from `out/results_stronger.csv`. Parentheses are approximate 95% CI over runs.

| Policy | Budget-overrun | Deadline-miss | Utility | Regret vs. oracle |
|---|:---:|:---:|:---:|:---:|
| ReAct (uncapped) | 1.000 (±0.000) | 1.000 (±0.000) | 0.985 (±0.004) | −0.139 *(cheats)* |
| ReAct (capped) | 0.369 (±0.050) | 0.553 (±0.051) | 0.733 (±0.026) | +0.114 |
| Non-adaptive greedy *(oracle)* | 0.208 (±0.042) | 0.544 (±0.051) | 0.867 (±0.021) | −0.020 |
| BwK-vanilla | 0.289 (±0.047) | 0.217 (±0.043) | 0.735 (±0.028) | +0.112 |
| Lagrangian primal-dual | 0.231 (±0.044) | 0.119 (±0.034) | 0.806 (±0.032) | +0.041 |
| CVaR-BwK | 0.156 (±0.037) | 0.203 (±0.042) | 0.739 (±0.027) | +0.107 |
| Pre-check UCB | 0.028 (±0.017) | **0.000** | 0.701 (±0.030) | +0.145 |
| CC-knapsack oracle | **0.003** (±0.005) | 0.256 (±0.045) | 0.746 (±0.036) | +0.101 |
| CBUC-aggr (*z*=0) | **0.014** (±0.012) | **0.000** | 0.600 (±0.040) | +0.246 |
| CBUC-mod (*z*=0.5) | **0.014** (±0.012) | **0.000** | 0.589 (±0.040) | +0.257 |
| **CBUC (*z*=1.28, ours)** | **0.011** (±0.011) | **0.000** | 0.587 (±0.040) | +0.259 |

> **Key takeaway:** Among deployable policies, CBUC cuts budget-overrun **34× vs. ReAct-capped** and **26× vs. BwK-vanilla**, while achieving **zero deadline misses** across all 360 runs. This safety guarantee comes at a utility cost (~0.15 vs. BwK-vanilla), positioning CBUC as the conservative high-assurance point on the safety–utility frontier.

---

### Figure 1 — Budget-Overrun Rate vs. Budget Tightness (ρ)

![Budget-overrun rate vs. ρ, separated by Pareto tail index](plots/budget_overrun_vs_rho.png)

All three CBUC variants remain near zero across all tightness levels and tail conditions. Baselines degrade sharply under heavy tails (α = 1.5).

---

### Figure 2 — Utility vs. Budget Tightness (ρ)

![Utility vs. ρ for all policies](plots/utility_vs_rho.png)

The utility gap between CBUC and BwK-vanilla **narrows under tight budgets** (ρ = 0.2), where even unconstrained policies cannot execute enough tool calls to benefit from utility-reckless behavior.

---

### Figure 3 — Scaling with Registry Size K

![Budget-overrun rate as a function of K at ρ=0.5, α=2.5](plots/scaling_K.png)

BwK-vanilla improves with larger K (more arms → more exploration headroom) but plateaus at 10–16% overrun. **CBUC is uniformly ≤ 3% across the entire K range.**

---

### Ablation: Replay on Calibrated Traces

CBUC is also evaluated on traces calibrated to three public tool-use benchmarks:

| Profile | Policy | Overrun | Utility |
|---|---|:---:|:---:|
| BFCL subset | ReAct (capped) | 1.000 | 0.387 |
| BFCL subset | BwK-vanilla | 0.580 | 0.374 |
| BFCL subset | **CBUC** | **0.020** | **0.492** |
| τ-bench | ReAct (capped) | 1.000 | 0.802 |
| τ-bench | BwK-vanilla | 0.560 | 0.810 |
| τ-bench | **CBUC** | **0.000** | 0.728 |
| ToolBench subset | ReAct (capped) | 0.620 | 0.646 |
| ToolBench subset | BwK-vanilla | 0.440 | 0.653 |
| ToolBench subset | **CBUC** | **0.000** | 0.469 |

![Replay ablation across calibrated trace profiles](plots/ablation_replay.png)

---

### Ablation: Hyperparameter Sensitivity (α, β, γ)

![Hyperparameter sensitivity across α, β, γ settings](plots/ablation_abg.png)

CBUC achieves **zero budget-overrun** across all tested hyperparameter values, with utility ranging from 0.519 to 0.852. The default settings (α=1.0, β=1.5, γ=1.5) were fixed before the final sweep.

---

### Ablation: Distractor Tools

![Distractor tool ablation](plots/ablation_distractors.png)

The type-closure offline pruning phase of CBUC is robust to the presence of irrelevant ("distractor") tools in the registry, maintaining low overrun rates even as the fraction of distractors increases.

---

## Algorithm Overview: CBUC

CBUC has two phases:

1. **Offline pruning** — Filter the tool pool $\mathcal{S}$ to the subset $\mathcal{S}_q$ whose input schemas are type-compatible with the query. This is complete for feasible non-adaptive policies (Proposition 5) and runs in $O(n |\mathcal{U}|)$ time.

2. **Online sequencing** — Run a budget-aware UCB policy over $\mathcal{S}_q$. At each round, select the tool maximizing the upper-confidence utility-per-cost ratio, while enforcing a chance-constraint feasibility check before each invocation.

**Regret bound:** Under bounded sub-Gaussian cost, latency, and utility noise, CBUC achieves $\tilde{O}(\sqrt{KT \log T})$ regret against the best feasible non-adaptive policy over $T$ rounds and $K$ candidate tools.

**Complexity:** $O(K \log K)$ per round; negligible relative to any LLM API call.

---

## Reproducibility

All results are fully reproducible from a clean checkout in under **3 minutes** on a single CPU core:

```bash
# Headline Table 1 (results_stronger.csv — 3,960 rows)
python3 code/cbslao_sim.py out 20 stronger results_stronger.csv

# Plots and tables
python3 code/analyze.py out/results_stronger.csv plots/

# Ablations
python3 code/ablations.py out plots/
```

Seeds are derived deterministically from sweep coordinates. No proprietary LLM API is required.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{cbslao2026,
  title     = {Cost- and SLA-Bounded Orchestration for LLM-Agent Tool/Service Composition},
  booktitle = {Proceedings of the 24th International Conference on Service-Oriented Computing (ICSOC)},
  year      = {2026},
  note      = {To appear}
}
```

---

## License

This repository is released for research reproducibility purposes. See `LICENSE` for details.
