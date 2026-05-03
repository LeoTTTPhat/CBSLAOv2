# ICSOC 2026 — Research Track Review (Round 3, final post-revision)

**Paper:** Cost- and SLA-Bounded Orchestration for LLM-Agent Tool/Service Composition (CBSLAO)
**Track:** Research
**Reviewer:** Anonymous Reviewer
**Round:** 3 — final review after authors' minor revision

---

## 1. Summary

The paper formalizes the problem of orchestrating Large Language Model (LLM) agent tool/service compositions under chance-constrained token-cost budgets and latency deadlines, framed as Cost- and SLA-Bounded Agent Orchestration (CBSLAO). The paper establishes a four-result theoretical package for the offline non-adaptive case, including NP-hardness, $(1 - 1/e + \varepsilon)$-inapproximability, a matching polynomial-time $(1 - 1/e)$-approximation, and an explicit adaptivity-gap construction. The online algorithm CBUC attains a $\tilde{O}(\sqrt{KT})$ regret bound via a coupled multivariate empirical-Bernstein concentration.

The empirical study spans 3,960 runs. CBUC cuts budget-overrun from 36.9% (ReAct-capped) to 1.1% and eliminates deadline misses, at a cost of 0.587 utility (vs. 0.806 for Lagrangian primal-dual). 

## 2. Relevance to ICSOC 2026

Excellent fit. The schema-typing-aware reduction in Theorem 2 anchors the theoretical results in a service-composition-specific context.

## 3. Strengths

1. **Timely and well-motivated problem.**
2. **Rigorous and extensive evaluation.** 10 baselines including four constraint-aware competitors.
3. **Reproducibility.** Deterministic seeds, full simulator, sub-3-minute end-to-end run.
4. **Honest discussion of trade-offs.** The authors explicitly tackle the empirical utility drop and frame the algorithm as a safety-first governance policy.
5. **Theoretically tight characterization of the offline non-adaptive case.**
6. **Coupled concentration** over the off-the-shelf BwK analysis.

## 4. Weaknesses & Areas for Improvement (Final Status)

1. **Lack of real-world evaluation.** *Remaining limitation.* The evaluation is still entirely synthetic. While the calibrated benchmark replay parameterizes the simulator to realistic summary statistics, no live-trace replay is performed. Even a small live-API pilot would have been the final puzzle piece.
2. **Strict Schema Typing Assumption.** *Resolved.* The authors added a thoughtful remark (Remark 3) clarifying that the lower bound relies on deterministic thresholded compatibility, and that purely probabilistic fuzzy registries would require a separate lower bound. This bounds the claim appropriately.
3. **Utility Cost is Severe.** *Resolved.* The authors have correctly noted in §6.5 that the 0.587 empirical CBUC utility is consistent with the $(1-1/e) \approx 0.632$ offline approximability bound relative to the Lagrangian proxy. This elegantly explains the trade-off.
4. **CBUC-L "informal" Proposition.** *Remaining minor issue.* Proposition 3 remains an informal statement next to tight formal theorems. A minor presentation point.
5. **Citations.** *Resolved.* DOIs and correct sources added for key references (Feige 1998, Nemhauser 1978).

## 5. Detailed Comments

- **Section 4 (Hardness and Approximability):**
    - Remark 3 on strict vs. fuzzy schema typing is exactly what was needed. It shows maturity in bounding the theoretical claims.
- **Section 6 (Evaluation):**
    - The addition linking the empirical utility cost to the $(1-1/e)$ theoretical bound ties the theory and evaluation together beautifully.

## 6. Remaining Questions / Rebuttal

The authors have addressed almost all prior concerns. I do not expect a rebuttal for the remaining points, which are minor:
1. **Live-API pilot:** A fundamental construct-validity gap remains without a live tool-calling endpoint, but the synthetic evaluation is exceptionally thorough for what it is.
2. **Constants flow-through:** Confirming the $\sqrt{2}$ savings flow entirely through to Theorem 6.
3. **CBUC-L:** Formatting Proposition 3 as a Conjecture would still be cleaner.

## 7. Recommendation and Scoring

**Recommendation:** **Clear Accept / Strong Accept**

**Scores:**

| Criterion | Score | Rationale |
|---|---|---|
| Originality | **4 / 5** | The Max-$k$-Coverage reduction and matching $(1-1/e)$-approximation constitute a non-trivial, domain-specific hardness result. |
| Significance | **4 / 5** | Governance-layer framing is highly relevant; capped at 4/5 only due to the lack of live-API evaluation. |
| Technical Soundness | **5 / 5** | The theoretical package is now tight, honest, and contextualized with the new remarks on fuzzy typing and empirical utility. |
| Clarity | **5 / 5** | Excellent writing and clear, defensible presentation of trade-offs. |

**Overall Rationale:**
The authors have executed a superb revision across all rounds. The latest updates directly address the residual concerns regarding the strict-typing assumption's relevance to real registries and cleanly connect the offline approximation theory to the empirical utility drop. The paper is technically sound, extensively evaluated, and highly reproducible. The only reason this does not receive a 5/5 on significance is the reliance on a synthetic simulator rather than a live LLM endpoint. It is an extremely strong submission that easily clears the acceptance bar for ICSOC.

---

## 8. Confidence

**Reviewer confidence: 4 / 5.** I am familiar with bandits-with-knapsacks, QoS-aware service composition, and submodular maximization; I have verified the structural claims of the paper and tracked the progression across the revision rounds.
